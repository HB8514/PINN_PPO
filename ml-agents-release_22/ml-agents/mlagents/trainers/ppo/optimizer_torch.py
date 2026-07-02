from typing import Dict, cast, Optional
import math
import attr

from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.networks import ValueNetwork
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


@attr.s(auto_attribs=True)
class PPOSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    shared_critic: bool = False
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR

    # ===== Physics Loss =====
    # 0.0 → baseline PPO (physics 비활성)
    # 0.01 → yaw consistency 활성
    physics_loss_strength: float = 0.0

    # ===== Vehicle Constants (Unity 실측) =====
    # Prometeo maxSpeed=40 km/h → 40/3.6 m/s
    physics_velocity_scale_mps: float = 40.0 / 3.6
    physics_yaw_rate_norm: float = 5.0
    physics_wheel_base: float = 3.102          # WheelCollider 실측
    physics_max_steer_angle_deg: float = 45.0  # Inspector 확인

    # ===== Yaw Consistency Hyperparameters =====
    # 저속에서 kinematic 모델이 부정확 → speed gate로 약화
    physics_yaw_min_speed_mps: float = 1.0    # gate 시작 속도
    physics_yaw_speed_temperature: float = 0.5

    # ===== Understeer 보정 (고속 yaw 과대예측 교정) =====
    # yaml에서 반드시 명시할 것:
    #   physics_understeer_k: 0.0    → 기존 무슬립 kinematic yaw residual
    #   physics_understeer_k: 0.003  → 약한 understeer 보정 후보
    #   physics_understeer_k: 0.01   → 중간 understeer 보정 후보
    #   physics_understeer_k: 0.02   → 강한 understeer 보정 후보
    #
    # None은 "yaml에 값이 없음"을 의미한다. physics_loss_strength > 0인데
    # 이 값이 None이면 __init__에서 에러를 내서 실험값 누락을 막는다.
    physics_understeer_k: Optional[float] = None


class TorchPPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
        )

        if (
            self.hyperparameters.physics_loss_strength > 0.0
            and self.hyperparameters.physics_understeer_k is None
        ):
            raise ValueError(
                "physics_understeer_k is missing in yaml. "
                "Set physics_understeer_k: 0.0 for the original kinematic PI-PPO, "
                "or a positive value such as 0.003 / 0.01 / 0.02 for understeer PI-PPO."
            )

        params = list(self.policy.actor.parameters())
        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            self._critic.to(default_device())
            params += list(self._critic.parameters())

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
            "Losses/Physics Yaw Loss": "physics_yaw_loss",
        }

        self.stream_names = list(self.reward_signals.keys())
        self._debug_printed = False
        self._obs_diagnostic_printed = False

    @property
    def critic(self):
        return self._critic

    def _zero(self, current_obs):
        if len(current_obs) > 0:
            return current_obs[0].sum() * 0.0
        return torch.tensor(0.0, device=default_device())

    def _get_vector_obs(self, current_obs):
        """
        CarAgent.cs vector observation [batch, 5]:
          obs[0] = forwardVelocity / velocityObservationScaleMps
          obs[1] = lateralVelocity / velocityObservationScaleMps
          obs[2] = yawRate / yawRateNorm
          obs[3] = last raw steer command
          obs[4] = last raw throttle command
        """
        for obs in current_obs:
            if obs.dim() == 2 and obs.shape[1] >= 5:
                return obs
        return None

    def _get_deterministic_action(self, current_obs, act_masks, memories):
        """현재 policy의 deterministic continuous action 반환."""
        if not hasattr(self.policy.actor, "network_body"):
            return None
        if not hasattr(self.policy.actor, "action_model"):
            return None

        actor_encoding, _ = self.policy.actor.network_body(
            current_obs,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        action_outputs = self.policy.actor.action_model.get_action_out(
            actor_encoding, act_masks,
        )
        # index 3 = deterministic continuous action
        return action_outputs[3]

    def _calculate_yaw_loss(self, current_obs, act_masks, memories, loss_masks):
        """
        Quasi-static kinematic yaw consistency loss.

        물리적 근거:
          Decision Period=5, fixedDeltaTime=0.02s → 같은 action이 0.1s 유지.
          그 구간에서 kinematic bicycle 모델의 quasi-static 조건:
            r_obs ≈ (vx / L) × tan(δ_policy)
          이를 self-consistency constraint로 사용.

        gradient 경로:
          θ → π(s_t) → steer_now → yaw_rate_pred → yaw_loss → ∇θ
          (steer action으로만 gradient 흐름)

        반환: yaw_loss (scalar)
        """
        zero = self._zero(current_obs)

        if self.hyperparameters.physics_loss_strength <= 0.0:
            return zero

        vector_obs = self._get_vector_obs(current_obs)
        if vector_obs is None:
            return zero
        if vector_obs.shape[1] < 5:
            return zero

        det_action = self._get_deterministic_action(current_obs, act_masks, memories)
        if det_action is None:
            return zero
        if det_action.shape[-1] < 1:
            return zero

        # 상수
        vel_scale = max(float(self.hyperparameters.physics_velocity_scale_mps), 1e-6)
        yaw_norm = max(float(self.hyperparameters.physics_yaw_rate_norm), 1e-6)
        L = max(float(self.hyperparameters.physics_wheel_base), 1e-6)
        max_steer_rad = float(self.hyperparameters.physics_max_steer_angle_deg) * math.pi / 180.0
        v_min = float(self.hyperparameters.physics_yaw_min_speed_mps)
        v_temp = max(float(self.hyperparameters.physics_yaw_speed_temperature), 1e-6)
        # physics_loss_strength > 0이면 __init__에서 None을 이미 차단한다.
        # baseline PPO(physics_loss_strength <= 0)에서는 이 함수가 앞에서 zero를 반환한다.
        understeer_k = max(float(self.hyperparameters.physics_understeer_k), 0.0)

        # state (buffer 상수)
        vx = vector_obs[:, 0] * vel_scale          # 전진 속도 (m/s)
        yaw_obs = vector_obs[:, 2] * yaw_norm       # 관측 yaw rate (rad/s)

        # policy action (미분가능)
        steer_now = det_action[:, 0]
        steer_angle = steer_now * max_steer_rad

        # kinematic bicycle 예측 yaw rate
        #   무슬립: yaw = (vx / L) * tan(δ)
        # understeer 보정 (K>0):
        #   yaw = ((vx / L) * tan(δ)) / (1 + K * vx²)
        #   고속에서 실제보다 회전을 과대예측하는 문제를 정상상태 요레이트로 교정.
        #   K=0 이면 기존 무슬립 식과 동일 (하위호환).
        yaw_pred_kinematic = (vx / L) * torch.tan(steer_angle)
        understeer_denom = 1.0 + understeer_k * vx * vx
        yaw_pred = yaw_pred_kinematic / understeer_denom

        # 정규화 잔차
        yaw_residual = (yaw_obs - yaw_pred) / yaw_norm

        # 저속 gate: vx < 1m/s에서 kinematic 모델 부정확 → 자연스럽게 약화
        speed_gate = torch.sigmoid(
            (torch.abs(vx) - v_min) / v_temp
        )

        # smooth_l1: L2보다 outlier에 robust
        yaw_loss_each = speed_gate * torch.nn.functional.smooth_l1_loss(
            yaw_residual,
            torch.zeros_like(yaw_residual),
            reduction="none",
        )

        return ModelUtils.masked_mean(yaw_loss_each, loss_masks)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        if not self._debug_printed:
            print("CUSTOM PPO OPTIMIZER (YAW CONSISTENCY) RUNNING")
            print("optimizer_torch.py path:", __file__)
            print(f"physics_loss_strength: {self.hyperparameters.physics_loss_strength}")
            print(f"physics_understeer_k: {self.hyperparameters.physics_understeer_k}")
            self._debug_printed = True

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())

        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        # ===== 관측(observation) 진단: 카메라 입력 여부 확정 =====
        if not self._obs_diagnostic_printed:
            print("=" * 60)
            print("[OBS DIAGNOSTIC] 관측 개수(n_obs):", n_obs)
            for i, obs in enumerate(current_obs):
                print(f"  current_obs[{i}] shape = {tuple(obs.shape)}  (dim={obs.dim()})")
            try:
                specs = self.policy.behavior_spec.observation_specs
                print("[OBS DIAGNOSTIC] behavior_spec observation_specs:")
                for i, sp in enumerate(specs):
                    name = getattr(sp, "name", f"obs{i}")
                    shape = getattr(sp, "shape", None)
                    print(f"  spec[{i}] name='{name}' shape={shape}")
            except Exception as e:
                print("[OBS DIAGNOSTIC] specs 출력 실패:", e)
            has_image = any(o.dim() >= 4 for o in current_obs)
            print("[OBS DIAGNOSTIC] 이미지(4D) 관측 존재?:", has_image,
                  "->", "카메라 입력 사용 중" if has_image else "벡터 관측만(카메라 미사용)")
            print("=" * 60)
            self._obs_diagnostic_printed = True
        # ========================================================

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"]
        entropy = run_out["entropy"]

        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )

        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        log_probs = log_probs.flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )

        yaw_loss = self._calculate_yaw_loss(
            current_obs=current_obs,
            act_masks=act_masks,
            memories=memories,
            loss_masks=loss_masks,
        )

        loss = (
            policy_loss
            + 0.5 * value_loss
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
            + self.hyperparameters.physics_loss_strength * yaw_loss
        )

        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        update_stats = {
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Physics Yaw Loss": yaw_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
            "Policy/Physics Loss Strength": self.hyperparameters.physics_loss_strength,
            "Policy/Physics Understeer K": (
                0.0
                if self.hyperparameters.physics_understeer_k is None
                else float(self.hyperparameters.physics_understeer_k)
            ),
        }

        return update_stats

    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules