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

    [SerializeField] private float lastAppliedThrottleDebug = 0f;

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
    public float carCenterReward = 0.5f;
    public float sideLinePenalty = -1.0f;
    public float centerLinePenalty = -1.0f;

    // 역주행 방지: 도로에 RVCarCenter 태그 콜라이더를 깔아 역방향 진행 시 밟히게 함.
    public float reverseDirectionPenalty = -1.0f;

    private HashSet<int> visitedCarCenters = new HashSet<int>();

    [Header("Checkpoint Order Reward")]
    public int maxCheckpointIndex = 24;
    public float checkpointReward = 5.0f;
    public float lapReward = 10.0f;
    public float wrongCheckpointPenalty = -5.0f;

    private int expectedCheckpointIndex = 1;

    [Header("Evaluation Stats")]
    private float lapStartTime = 0f;

    [Header("Cumulative Stat Settings")]
    // 5개 구간으로 나눠서 충돌 누적값을 기록한다.
    // 각 구간 그래프는 "전체 누적값"을 그 구간에서만 그린다.
    public int totalTrainingStep = 100000;
    public int numWindows = 1;   // 0-100k, 100k-200k, ..., 400k-500k

    // TensorBoard x축(trainer step) 환산용 나눗수.
    // Take Actions Between Decisions=true이면 OnActionReceived가 매 FixedUpdate
    // 호출되지만, 경험 수집(trainer step)은 Decision Period마다 1번이다.
    // 에이전트 수는 분자/분모에서 약분되므로 나눠야 하는 값은 Decision Period.
    // Inspector의 Decision Requester > Decision Period와 동일하게 설정할 것.
    public int decisionPeriod = 5;

    private static int totalCenterLineHits = 0;
    private static int totalSideLineHits = 0;
    private static int totalLapCompleted = 0;

    // 전체 에이전트의 OnActionReceived 누적 호출 수(모든 차 공유, static).
    private static int _totalActionCalls = 0;

    // 5개 구간 stat 이름 캐싱 (호출 시 GC 0).
    private string[] _centerLineWindowStats;
    private string[] _sideLineWindowStats;
    private int _windowSizeStep;

    [Header("Debug")]
    public bool enableDebugLog = false;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetStaticStats()
    {
        totalCenterLineHits = 0;
        totalSideLineHits = 0;
        totalLapCompleted = 0;
        _totalActionCalls = 0;
    }

    public override void Initialize()
    {
        if (car == null)
            car = GetComponent<PrometeoCarController>();

        if (car != null)
        {
            rb = car.GetComponent<Rigidbody>();
            car.agentControlEnabled = true;
        }

        int windows = Mathf.Max(numWindows, 1);
        _windowSizeStep = Mathf.Max(totalTrainingStep / windows, 1);

        _centerLineWindowStats = new string[windows];
        _sideLineWindowStats = new string[windows];

        for (int i = 0; i < windows; i++)
        {
            int startK = (i * _windowSizeStep) / 1000;
            int endK = ((i + 1) * _windowSizeStep) / 1000;
            _centerLineWindowStats[i] = $"Car/CenterLineHit_{startK}k_{endK}k";
            _sideLineWindowStats[i] = $"Car/SideLineHit_{startK}k_{endK}k";
        }

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

    private int GetCurrentTrainerStep()
    {
        // 전체 OnActionReceived 호출 수를 Decision Period로 나눠 trainer step에 맞춤.
        // Take Actions Between Decisions=true면 매 FixedUpdate마다 호출되지만
        // 경험 수집은 Decision Period마다 1번이라 그만큼 나눠야 함.
        int divisor = Mathf.Max(decisionPeriod, 1);
        return _totalActionCalls / divisor;
    }

    private int GetCurrentWindowIndex()
    {
        int step = GetCurrentTrainerStep();
        if (_windowSizeStep <= 0)
            return 0;

        int idx = step / _windowSizeStep;

        int lastIdx = (_centerLineWindowStats != null)
            ? _centerLineWindowStats.Length - 1
            : 0;

        return Mathf.Clamp(idx, 0, lastIdx);
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
        AddMostRecentStat("Car/CenterLineHitTotal", totalCenterLineHits);
        AddMostRecentStat("Car/SideLineHitTotal", totalSideLineHits);
        AddMostRecentStat("Car/LapCompletedTotal", totalLapCompleted);

        if (_centerLineWindowStats == null || _sideLineWindowStats == null)
            return;

        int w = GetCurrentWindowIndex();
        AddMostRecentStat(_centerLineWindowStats[w], totalCenterLineHits);
        AddMostRecentStat(_sideLineWindowStats[w], totalSideLineHits);
    }

    private void RegisterCenterLineHit()
    {
        totalCenterLineHits++;
        RecordCumulativeStats();
    }

    private void RegisterSideLineHit()
    {
        totalSideLineHits++;
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

        // 모든 에이전트 공유 카운터. trainer step 환산용(GetCurrentTrainerStep 참고).
        _totalActionCalls++;

        float steer = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
        float throttle = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

        lastSteerAction = steer;

        car.SetSteeringNormalized(steer);

        car.SetThrottleNormalized(throttle);

        lastThrottleAction = throttle;
        lastAppliedThrottleDebug = car.LastAppliedThrottle;

        float speedMag = rb.velocity.magnitude;

        float forwardSpeed = Vector3.Dot(rb.velocity, car.transform.forward);

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
    }

    private void OnCollisionEnter(Collision collision)
    {
        // SideLine은 Non-Trigger이므로 여기(OnCollisionEnter)에서만 처리.
        if (collision.collider.CompareTag("SideLine"))
        {
            RegisterSideLineHit();
            AddReward(sideLinePenalty);
            EndEpisodeWithLog("SideLineCollision");
            return;
        }
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("CarCenter"))
        {
            RewardCarCenterOnce(other);
        }

        // 역주행 방지: RVCarCenter 밟으면 감점(에피소드 종료 없음).
        if (other.CompareTag("RVCarCenter"))
        {
            AddReward(reverseDirectionPenalty);
        }

        if (other.CompareTag("CenterLine"))
        {
            RegisterCenterLineHit();
            AddReward(centerLinePenalty);
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