from typing import Dict, cast
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

    # =============================
    # Physics loss coefficient
    # =============================
    # base PPO: 0.0
    # PINN-PPO: 0.001, 0.01, 0.05 등으로 실험
    physics_loss_strength: float = 0.0


class TorchPPOOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        """
        Takes a Policy and a Dict of trainer parameters and creates an Optimizer around the policy.
        The PPO optimizer has a value estimator and a loss function.
        :param policy: A TorchPolicy object that will be updated by this PPO Optimizer.
        :param trainer_params: Trainer parameters dictionary that specifies the
        properties of the trainer.
        """
        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: PPOSettings = cast(
            PPOSettings, trainer_settings.hyperparameters
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
            "Losses/Physics Loss": "physics_loss",
            "Losses/Physics Friction Loss": "physics_friction_loss",
            "Losses/Physics Steer Smoothness Loss": "physics_steer_rate_loss",
            "Losses/Physics Throttle Smoothness Loss": "physics_throttle_rate_loss",
        }

        self.stream_names = list(self.reward_signals.keys())
        self._debug_update_printed = False

    @property
    def critic(self):
        return self._critic

    def _zero_physics_loss(self, current_obs):
        if len(current_obs) > 0:
            return current_obs[0].sum() * 0.0

        return torch.tensor(0.0, device=default_device())

    def _get_vector_observation(self, current_obs):
        """
        Camera observation은 보통 4D tensor이고,
        CarAgent.cs의 vector observation은 [batch, 5] 형태의 2D tensor임.
        여기서는 5개 이상 들어있는 2D observation을 차량 상태 vector로 사용.
        """
        for obs in current_obs:
            if obs.dim() == 2 and obs.shape[1] >= 5:
                return obs

        return None

    def _get_policy_deterministic_continuous_action(
        self,
        current_obs,
        act_masks,
        memories,
    ):
        """
        현재 policy network가 현재 observation에서 내는 deterministic continuous action을 가져옴.
        action[0] = steer
        action[1] = throttle
        """
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
            actor_encoding,
            act_masks,
        )

        deterministic_continuous_action = action_outputs[3]

        return deterministic_continuous_action

    def _calculate_physics_loss(self, current_obs, act_masks, memories, loss_masks):
        """
        [최종 재설계 / 방식 B] 현재 관측 state(buffer 상수) + 현재 policy action
        (미분가능)으로 즉각적 물리 타당성을 제약한다. gradient 경로:
            theta -> pi(s_t) -> physics_loss -> grad theta

        세 항 (옵션 A):
          1. friction-circle : 코너에서 미끄러지지 않기.
             종/횡 가속도 명령이 타이어 마찰 원 안에 있도록.
             steer, throttle 두 action 모두에 gradient.
          2. steering-rate   : 핸들 떨림 방지.
             현재 steer가 직전 steer(obs[3]) 대비 급변하지 않도록. 항상 active.
          3. throttle-rate   : throttle 떨림 방지.
             현재 throttle이 직전 throttle(obs[4]) 대비 급변하지 않도록. 항상 active.

        kinematic bicycle의 yaw transition 항은 제거했다. 그 항은 action->다음상태
        전이 관계라서 방식 B(현재상태 즉각 제약)와 시간축이 맞지 않기 때문이다.

        CarAgent.cs vector observation 순서:
        obs[0] = forwardVelocity / velocityObservationScaleMps
        obs[1] = lateralVelocity / velocityObservationScaleMps
        obs[2] = yawRate / yawRateNorm
        obs[3] = lastSteerAction      (직전 적용 steer)
        obs[4] = lastThrottleAction   (직전 적용 throttle)

        반환: (physics_loss, friction_loss, steer_rate_loss, throttle_rate_loss)
        """
        zero = self._zero_physics_loss(current_obs)

        if self.hyperparameters.physics_loss_strength <= 0.0:
            return zero, zero, zero, zero

        vector_obs = self._get_vector_observation(current_obs)

        if vector_obs is None:
            return zero, zero, zero, zero

        if vector_obs.shape[1] < 5:
            return zero, zero, zero, zero

        deterministic_action = self._get_policy_deterministic_continuous_action(
            current_obs,
            act_masks,
            memories,
        )

        if deterministic_action is None:
            return zero, zero, zero, zero

        if deterministic_action.shape[-1] < 2:
            return zero, zero, zero, zero

        # ===== Unity / CarAgent와 맞춘 실측·식별 상수 =====
        # Prometeo maxSpeed=40 km/h -> 40/3.6 m/s.
        velocity_scale_mps = 40.0 / 3.6
        wheel_base = 3.102
        max_steer_angle_rad = 45.0 * math.pi / 180.0

        # 종/횡 가속도 한계: mu_effective * g.
        # mu_effective는 WheelCollider 실측 마찰계수가 아니라 보수적으로 설정한
        # 가속도 한계(feasibility) 파라미터다 (논문 limitation).
        # ax_limit != ay_limit으로 두면 엄밀히는 friction ellipse가 되지만,
        # 현재는 동일 값으로 보수적 원(circle)을 사용한다.
        mu_effective = 0.9
        gravity = 9.81
        ay_limit = max(mu_effective * gravity, 1e-6)
        ax_limit = max(mu_effective * gravity, 1e-6)

        # throttle -> 종가속도(m/s^2) 매핑 gain. throttle sweep으로 식별 필요.
        # 주의: ax_limit과 분리해야 gain이 약분되지 않고 실제 의미를 갖는다.
        # (예: ax_cmd/ax_limit = gain*throttle / (mu*g) -> gain이 살아있음)
        throttle_accel_gain = 3.3

        # friction utilization을 smooth hinge에 넣을 때의 temperature.
        # 값이 작을수록 max(usage-1, 0)에 가까워져 마찰 원 안쪽 처벌이 사라진다.
        friction_temperature = 0.05

        # ===== 내부 가중치 (옵션 A: 항별 독립 lambda) =====
        # 세 항의 스케일이 달라 그대로 더하면 rate 항이 friction을 압도할 수 있다.
        # steer 떨림을 throttle 떨림보다 더 강하게 억제(핸들 안정 우선).
        lambda_steer_rate = 0.1
        lambda_throttle_rate = 0.05

        # ===== state (buffer 상수) =====
        forward_velocity = vector_obs[:, 0] * velocity_scale_mps  # vx (m/s)
        last_steer = vector_obs[:, 3]      # 직전 steer action (상수, 기준점)
        last_throttle = vector_obs[:, 4]   # 직전 throttle action (상수, 기준점)

        # ===== 현재 policy action (미분가능) =====
        # clamp하지 않는다: clamp 포화 구간에서 gradient가 0으로 죽기 때문.
        steer_now = deterministic_action[:, 0]
        throttle_now = deterministic_action[:, 1]
        steer_angle = steer_now * max_steer_angle_rad

        # ===== 1. combined-acceleration feasibility (friction circle) =====
        # a_y = v^2/R, R = L/tan(d)  =>  a_y = (v^2/L) tan(d).  steer 연결.
        ay_cmd = (forward_velocity ** 2 / wheel_base) * torch.tan(steer_angle)
        # a_x = gain * throttle.  throttle 연결. (gain은 ax_limit과 분리되어 살아있음)
        ax_cmd = throttle_accel_gain * throttle_now

        # utilization: sqrt norm. <1 허용, =1 한계, >1 초과 로 직접 해석 가능.
        # 제곱합을 그대로 쓰면 바깥에서 4제곱으로 폭발하므로 sqrt를 취한다.
        friction_utilization = torch.sqrt(
            (ax_cmd / ax_limit) ** 2
            + (ay_cmd / ay_limit) ** 2
            + 1e-8
        )

        # smooth hinge ~ max(utilization - 1, 0). temperature가 작을수록
        # 마찰 원 안쪽(util<1)에서는 거의 0, 경계 초과 시 부드럽게 증가.
        # (기존 softplus(util-1)은 util=0에서도 ~0.098을 줘서 정지/저속을
        #  부당하게 처벌 → 차가 출발을 꺼리는 원인이었음)
        friction_excess = friction_temperature * torch.nn.functional.softplus(
            (friction_utilization - 1.0) / friction_temperature
        )
        friction_loss_each = friction_excess ** 2

        # ===== 2. steering command smoothness =====
        # (steer_now - last_steer)^2. Δt로 나누지 않으므로 물리적 rate(rad/s)가
        # 아니라 "명령 부드러움(command smoothness)" regularizer다. last_steer는
        # obs[3]의 직전 raw steer command (CarAgent에서 raw 기준으로 통일).
        steer_rate_loss_each = (steer_now - last_steer) ** 2

        # ===== 3. throttle command smoothness =====
        # 위와 동일하게 raw-to-raw smoothness. last_throttle은 obs[4]의 직전 raw
        # throttle command. (raw_now - raw_last로 일관 → 방식 B 정합성)
        throttle_rate_loss_each = (throttle_now - last_throttle) ** 2

        # ===== masked mean =====
        friction_loss = ModelUtils.masked_mean(friction_loss_each, loss_masks)
        steer_rate_loss = ModelUtils.masked_mean(steer_rate_loss_each, loss_masks)
        throttle_rate_loss = ModelUtils.masked_mean(throttle_rate_loss_each, loss_masks)

        # ===== 합산 (항별 내부 가중치 적용) =====
        physics_loss = (
            friction_loss
            + lambda_steer_rate * steer_rate_loss
            + lambda_throttle_rate * throttle_rate_loss
        )

        return physics_loss, friction_loss, steer_rate_loss, throttle_rate_loss

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Performs update on model.
        :param batch: Batch of experiences.
        :param num_sequences: Number of sequences to process.
        :return: Results of update.
        """
        if not self._debug_update_printed:
            print("CUSTOM PPO OPTIMIZER UPDATE RUNNING")
            print("optimizer_torch.py path:", __file__)
            self._debug_update_printed = True

        # Get decayed parameters
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

        # Convert to tensors
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # Get value memories
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

        physics_loss, friction_loss, steer_rate_loss, throttle_rate_loss = (
            self._calculate_physics_loss(
                current_obs=current_obs,
                act_masks=act_masks,
                memories=memories,
                loss_masks=loss_masks,
            )
        )

        loss = (
            policy_loss
            + 0.5 * value_loss
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
            + self.hyperparameters.physics_loss_strength * physics_loss
        )

        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Losses/Physics Loss": physics_loss.item(),
            "Losses/Physics Friction Loss": friction_loss.item(),
            "Losses/Physics Steer Smoothness Loss": steer_rate_loss.item(),
            "Losses/Physics Throttle Smoothness Loss": throttle_rate_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
            "Policy/Physics Loss Strength": self.hyperparameters.physics_loss_strength,
        }

        return update_stats

    # TODO move module update into TorchOptimizer for reward_provider
    def get_modules(self):
        modules = {
            "Optimizer:value_optimizer": self.optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules