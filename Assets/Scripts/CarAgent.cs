using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

public class CarAgent : Agent
{
    [Header("References")]
    public PrometeoCarController car;
    public Transform spawnPoint;

    private Rigidbody rb;

    [Header("Vehicle Observation")]
    [Tooltip("전진 속도 정규화 기준 (m/s). Prometeo maxSpeed=40 km/h → 40/3.6.")]
    public float velocityObservationScaleMps = 40f / 3.6f;
    [Tooltip("후진 속도 정규화 기준 (m/s). Prometeo maxReverseSpeed=30 km/h → 30/3.6.")]
    public float reverseVelocityObservationScaleMps = 30f / 3.6f;
    public float yawRateNorm = 5f;

    private float lastSteerAction = 0f;
    private float lastThrottleAction = 0f;

    // 분석용: 실제 환경에 적용된 throttle(속도제한/브레이크 통과 후).
    // Inspector에서 실시간으로 볼 수 있도록 [SerializeField]. throttle sweep 때 유용.
    [SerializeField] private float lastAppliedThrottleDebug = 0f;

    // 0 나누기 방어 (Inspector에서 실수로 0을 입력했을 때 대비)
    private float VelocityScaleMps => Mathf.Max(velocityObservationScaleMps, 1e-3f);
    private float ReverseVelocityScaleMps => Mathf.Max(reverseVelocityObservationScaleMps, 1e-3f);

    [Header("Rewards / Penalties")]
    public float timePenalty = -0.001f;
    public float forwardRewardScale = 0.08f;
    public float backwardPenaltyScale = -0.01f;

    public float idlePenalty = -0.002f;
    public float idleEndPenalty = -1.0f;
    private int idleSteps = 0;
    public int maxIdleSteps = 300;

    public float fallPenalty = -1.0f;

    [Header("Line Rewards / Penalties")]
    public float carCenterReward = 0.09f;
    public float sideLinePenalty = -1.0f;
    public float centerLinePenalty = -0.3f;

    private HashSet<int> visitedCarCenters = new HashSet<int>();

    [Header("Checkpoint Order Reward")]
    public int maxCheckpointIndex = 18;
    public float checkpointReward = 1.0f;
    public float lapReward = 6.0f;
    public float wrongCheckpointPenalty = -5.0f;

    private int expectedCheckpointIndex = 1;

    [Header("Evaluation Stats")]
    private float lapStartTime = 0f;

    [Header("Cumulative Stat Settings")]
    public int finalWindowStartStep = 450000;
    public int finalTrainingStep = 500000;

    private static int totalCenterLineHits = 0;
    private static int totalSideLineHits = 0;
    private static int totalLapCompleted = 0;

    private static int finalWindowCenterLineHits = 0;
    private static int finalWindowSideLineHits = 0;

    // [FIX 2] 매 통계 호출마다 문자열을 새로 만들지 않도록 1회 계산 후 캐싱.
    private string _centerLineHitWindowStat;
    private string _sideLineHitWindowStat;

    [Header("Debug")]
    public bool enableDebugLog = false;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStaticStats()
    {
        totalCenterLineHits = 0;
        totalSideLineHits = 0;
        totalLapCompleted = 0;

        finalWindowCenterLineHits = 0;
        finalWindowSideLineHits = 0;
    }

    public override void Initialize()
    {
        if (car == null)
            car = GetComponent<PrometeoCarController>();

        if (car != null)
        {
            rb = car.GetComponent<Rigidbody>();
            // RL 제어 모드 활성화: Prometeo Update()의 키보드/터치 actuator 로직이
            // agent가 넣은 토크를 덮어쓰지 않도록 차단.
            car.agentControlEnabled = true;
        }

        // [FIX 2] suffix 및 stat 이름 문자열을 여기서 한 번만 생성.
        string finalWindowSuffix = $"{finalWindowStartStep / 1000}k_{finalTrainingStep / 1000}k";
        _centerLineHitWindowStat = $"Car/CenterLineHit_{finalWindowSuffix}";
        _sideLineHitWindowStat = $"Car/SideLineHit_{finalWindowSuffix}";

        RecordCumulativeStats();
    }

    public override void OnEpisodeBegin()
    {
        if (car == null || rb == null) return;

        if (spawnPoint != null)
        {
            car.transform.position = spawnPoint.position;
            car.transform.rotation = spawnPoint.rotation;
        }
        else
        {
            car.transform.localPosition = Vector3.zero;
            car.transform.localRotation = Quaternion.identity;
        }

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // motor/brake/steer/throttleAxis/LastApplied*/DecelerateCar Invoke까지
        // 한 번에 초기화. ThrottleOff+ResetSteeringAngle+RecoverTraction 조합은
        // brake torque·LastApplied* 잔류 위험이 있어 교체함.
        car.ResetAgentActuators();

        idleSteps = 0;

        lastSteerAction = 0f;
        lastThrottleAction = 0f;
        lastAppliedThrottleDebug = 0f;

        expectedCheckpointIndex = 1;
        visitedCarCenters.Clear();

        lapStartTime = Time.time;

        RecordCumulativeStats();
    }

    private int GetCurrentAcademyStep()
    {
        if (Academy.Instance == null)
            return 0;

        return Academy.Instance.StepCount;
    }

    private bool IsInFinalWindow()
    {
        int step = GetCurrentAcademyStep();
        return step >= finalWindowStartStep && step <= finalTrainingStep;
    }

    private void AddAverageStat(string statName, float value)
    {
        Academy.Instance.StatsRecorder.Add(statName, value, StatAggregationMethod.Average);
    }

    private void AddMostRecentStat(string statName, float value)
    {
        Academy.Instance.StatsRecorder.Add(statName, value, StatAggregationMethod.MostRecent);
    }

    private void RecordCumulativeStats()
    {
        // [FIX 2] 캐싱된 문자열 재사용 → 호출 시 문자열 할당(GC) 0.
        AddMostRecentStat("Car/CenterLineHitTotal", totalCenterLineHits);
        AddMostRecentStat(_centerLineHitWindowStat, finalWindowCenterLineHits);

        AddMostRecentStat("Car/SideLineHitTotal", totalSideLineHits);
        AddMostRecentStat(_sideLineHitWindowStat, finalWindowSideLineHits);

        AddMostRecentStat("Car/LapCompletedTotal", totalLapCompleted);
    }

    private void RegisterCenterLineHit()
    {
        totalCenterLineHits++;

        if (IsInFinalWindow())
            finalWindowCenterLineHits++;

        RecordCumulativeStats();
    }

    private void RegisterSideLineHit()
    {
        totalSideLineHits++;

        if (IsInFinalWindow())
            finalWindowSideLineHits++;

        RecordCumulativeStats();
    }

    private void RegisterLapCompleted(float lapTime)
    {
        totalLapCompleted++;

        AddAverageStat("Car/LapTime", lapTime);

        RecordCumulativeStats();
    }

    private void EndEpisodeWithLog(string reason)
    {
        RecordCumulativeStats();

        if (enableDebugLog)
            Debug.Log($"[CarAgent] EndEpisode: {reason}");

        EndEpisode();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (car == null || rb == null)
        {
            for (int i = 0; i < 5; i++)
                sensor.AddObservation(0f);

            return;
        }

        Vector3 localVelocity = car.transform.InverseTransformDirection(rb.velocity);

        float forwardVelocity = localVelocity.z;
        float lateralVelocity = localVelocity.x;
        float yawRate = Vector3.Dot(rb.angularVelocity, car.transform.up);

        sensor.AddObservation(Mathf.Clamp(forwardVelocity / VelocityScaleMps, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(lateralVelocity / VelocityScaleMps, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(yawRate / yawRateNorm, -1f, 1f));
        sensor.AddObservation(lastSteerAction);
        sensor.AddObservation(lastThrottleAction);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        if (car == null || rb == null) return;

        float steer = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float throttle = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        lastSteerAction = steer;

        car.SetSteeringNormalized(steer);

        // 연속 throttle: 데드존 없이 항상 비례 토크로 매핑.
        // deadzone을 두면 작은 throttle을 버려 Python 물리식 입력(raw)과
        // Unity 실제 입력이 어긋나므로 제거함. 정확히 0은 메서드 내부에서 처리.
        car.SetThrottleNormalized(throttle);

        // physics loss(command smoothness)는 raw-to-raw 비교를 쓰므로 관측에는
        // raw throttle command를 저장한다. (방식 B 정합성: Python의 throttle_now도
        // raw policy action이므로 obs의 직전 throttle도 raw여야 의미가 맞음)
        // 실제 적용값은 분석용으로 별도 보관(관측/loss에는 미사용).
        lastThrottleAction = throttle;
        lastAppliedThrottleDebug = car.LastAppliedThrottle;

        float forwardSpeed = Vector3.Dot(rb.velocity, car.transform.forward);
        float speedMag = rb.velocity.magnitude;

        AddReward(timePenalty);

        if (forwardSpeed > 0.1f)
        {
            float normForward = Mathf.Clamp01(forwardSpeed / VelocityScaleMps);
            AddReward(forwardRewardScale * normForward);
        }

        if (speedMag < 0.2f)
        {
            AddReward(idlePenalty);
            idleSteps++;
        }
        else
        {
            idleSteps = 0;
        }

        if (idleSteps > maxIdleSteps)
        {
            AddReward(idleEndPenalty);
            EndEpisodeWithLog("IdleStepsExceeded");
            return;
        }

        if (forwardSpeed < -0.1f)
        {
            float normBackward = Mathf.Clamp01(Mathf.Abs(forwardSpeed) / ReverseVelocityScaleMps);
            AddReward(backwardPenaltyScale * normBackward);
        }

        if (car.transform.position.y < -5f)
        {
            AddReward(fallPenalty);
            EndEpisodeWithLog("FellOffTrack");
            return;
        }

        // [FIX 1] 매 스텝마다 호출되던 RecordCumulativeStats() 제거.
        // 통계값은 충돌/랩 이벤트에서만 변하고, 해당 Register* 함수들이
        // 내부에서 이미 RecordCumulativeStats()를 호출하므로 결과 동일.
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("SideLine"))
        {
            RegisterSideLineHit();

            AddReward(sideLinePenalty);
            EndEpisodeWithLog("SideLineCollision");
            return;
        }

        if (collision.collider.CompareTag("CenterLine"))
        {
            RegisterCenterLineHit();

            AddReward(centerLinePenalty);
            return;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("CarCenter"))
        {
            RewardCarCenterOnce(other);
        }

        if (other.CompareTag("CenterLine"))
        {
            RegisterCenterLineHit();

            AddReward(centerLinePenalty);
        }

        if (other.CompareTag("SideLine"))
        {
            RegisterSideLineHit();

            AddReward(sideLinePenalty);
            EndEpisodeWithLog("SideLineTrigger");
            return;
        }

        int checkpointNumber = GetCheckpointNumberFromTag(other);

        if (checkpointNumber == 0)
            return;

        if (checkpointNumber == expectedCheckpointIndex)
        {
            AddReward(checkpointReward);
            idleSteps = 0;

            if (enableDebugLog)
            {
                Debug.Log(
                    $"[CarAgent] Checkpoint {checkpointNumber}/{maxCheckpointIndex} passed. " +
                    $"CumulativeReward={GetCumulativeReward():F2}"
                );
            }

            expectedCheckpointIndex++;

            if (expectedCheckpointIndex > maxCheckpointIndex)
            {
                expectedCheckpointIndex = 1;

                float lapTime = Time.time - lapStartTime;
                lapStartTime = Time.time;

                AddReward(lapReward);

                RegisterLapCompleted(lapTime);

                if (enableDebugLog)
                {
                    Debug.Log(
                        $"[CarAgent] Lap completed. " +
                        $"LapTime={lapTime:F2}, " +
                        $"TotalLapCompleted={totalLapCompleted}, " +
                        $"CumulativeReward={GetCumulativeReward():F2}"
                    );
                }
            }

            return;
        }

        AddReward(wrongCheckpointPenalty);

        if (enableDebugLog)
        {
            Debug.Log(
                $"[CarAgent] Wrong checkpoint. " +
                $"Expected={expectedCheckpointIndex}, Got={checkpointNumber}, " +
                $"CumulativeReward={GetCumulativeReward():F2}"
            );
        }

        // [FIX 3-b] 디버그 OFF일 때도 항상 문자열 보간이 일어나던 것을 분기 처리.
        if (enableDebugLog)
            EndEpisodeWithLog($"WrongCheckpoint Expected={expectedCheckpointIndex} Got={checkpointNumber}");
        else
            EndEpisodeWithLog("WrongCheckpoint");

        return;
    }

    private void RewardCarCenterOnce(Collider other)
    {
        int id = other.GetInstanceID();

        if (visitedCarCenters.Contains(id))
            return;

        visitedCarCenters.Add(id);
        AddReward(carCenterReward);
    }

    private int GetCheckpointNumberFromTag(Collider other)
    {
        // [FIX 3] Substring 할당 제거 + Ordinal 비교로 문화권 처리 오버헤드 제거.
        // 주의: other.tag getter 자체가 문자열 1개를 할당함(Unity 특성). 이 부분까지
        // 완전 무할당으로 만들려면 체크포인트 콜라이더에 int 인덱스를 들고 있는
        // 컴포넌트를 붙여 GetComponent로 읽는 방식이 정석(씬 수정 필요).
        string tagName = other.tag;
        const string prefix = "Checkpoint";

        if (!tagName.StartsWith(prefix, System.StringComparison.Ordinal))
            return 0;

        if (!int.TryParse(tagName.Substring(prefix.Length), out int checkpointNumber))
            return 0;

        if (checkpointNumber < 1 || checkpointNumber > maxCheckpointIndex)
            return 0;

        return checkpointNumber;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetAxis("Horizontal");
        ca[1] = Input.GetAxis("Vertical");
    }
}