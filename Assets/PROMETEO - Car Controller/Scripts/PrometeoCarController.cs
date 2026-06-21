/*
MESSAGE FROM CREATOR: This script was coded by Mena. You can use it in your games either these are commercial or
personal projects. You can even add or remove functions as you wish. However, you cannot sell copies of this
script by itself, since it is originally distributed as a free product.
I wish you the best for your project. Good luck!

P.S: If you need more cars, you can check my other vehicle assets on the Unity Asset Store, perhaps you could find
something useful for your game. Best regards, Mena.
*/

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class PrometeoCarController : MonoBehaviour
{

    //CAR SETUP

      [Space(20)]
      //[Header("CAR SETUP")]
      [Space(10)]
      [Range(20, 190)]
      public int maxSpeed = 40; //The maximum speed that the car can reach in km/h.
      [Range(10, 120)]
      public int maxReverseSpeed = 30; //The maximum speed that the car can reach while going on reverse in km/h.
      [Range(1, 10)]
      public int accelerationMultiplier = 6; // How fast the car can accelerate. 1 is a slow acceleration and 10 is the fastest.
      [Space(10)]
      [Range(10, 45)]
      public int maxSteeringAngle = 45; // The maximum angle that the tires can reach while rotating the steering wheel.
      [Range(0.1f, 1f)]
      public float steeringSpeed = 0.6f; // How fast the steering wheel turns.
      [Space(10)]
      [Range(100, 600)]
      public int brakeForce = 450; // The strength of the wheel brakes.
      [Range(1, 10)]
      public int decelerationMultiplier = 2; // How fast the car decelerates when the user is not using the throttle.
      [Range(1, 10)]
      public int handbrakeDriftMultiplier = 5; // How much grip the car loses when the user hit the handbrake.
      [Space(10)]
      public Vector3 bodyMassCenter; // This is a vector that contains the center of mass of the car. I recommend to set this value
                                    // in the points x = 0 and z = 0 of your car. You can select the value that you want in the y axis,
                                    // however, you must notice that the higher this value is, the more unstable the car becomes.
                                    // Usually the y value goes from 0 to 1.5.

    //WHEELSs

      //[Header("WHEELS")]

      /*
      The following variables are used to store the wheels' data of the car. We need both the mesh-only game objects and wheel
      collider components of the wheels. The wheel collider components and 3D meshes of the wheels cannot come from the same
      game object; they must be separate game objects.
      */
      public GameObject frontLeftMesh;
      public WheelCollider frontLeftCollider;
      [Space(10)]
      public GameObject frontRightMesh;
      public WheelCollider frontRightCollider;
      [Space(10)]
      public GameObject rearLeftMesh;
      public WheelCollider rearLeftCollider;
      [Space(10)]
      public GameObject rearRightMesh;
      public WheelCollider rearRightCollider;

    //SPEED TEXT (UI)

      [Space(20)]
      //[Header("UI")]
      [Space(10)]
      //The following variable lets you to set up a UI text to display the speed of your car.
      public bool useUI = false;
      public Text carSpeedText; // Used to store the UI object that is going to show the speed of the car.

    //SOUNDS

      [Space(20)]
      //[Header("Sounds")]
      [Space(10)]
      //The following variable lets you to set up sounds for your car such as the car engine or tire screech sounds.
      public bool useSounds = false;
      public AudioSource carEngineSound; // This variable stores the sound of the car engine.
      public AudioSource tireScreechSound; // This variable stores the sound of the tire screech (when the car is drifting).
      float initialCarEngineSoundPitch; // Used to store the initial pitch of the car engine sound.

    //CONTROLS

      [Space(20)]
      //[Header("CONTROLS")]
      [Space(10)]
      //The following variables lets you to set up touch controls for mobile devices.
      public bool useTouchControls = false;
      public GameObject throttleButton;
      PrometeoTouchInput throttlePTI;
      public GameObject reverseButton;
      PrometeoTouchInput reversePTI;
      public GameObject turnRightButton;
      PrometeoTouchInput turnRightPTI;
      public GameObject turnLeftButton;
      PrometeoTouchInput turnLeftPTI;
      public GameObject handbrakeButton;
      PrometeoTouchInput handbrakePTI;

    //CAR DATA

      [HideInInspector]
      public float carSpeed; // Used to store the speed of the car.
      [HideInInspector]
      public bool isDrifting; // Used to know whether the car is drifting or not.
      [HideInInspector]
      public bool isTractionLocked; // Used to know whether the traction of the car is locked or not.

    //AGENT CONTROL

      // RL 에이전트가 actuator를 직접 제어하는 모드. true이면 Update()의
      // 키보드·터치 입력 처리(ThrottleOff/DecelerateCar 등)를 건너뛴다.
      // 학습/추론 시 CarAgent.Initialize()에서 true로 설정해야 한다.
      [Tooltip("RL 학습/추론 중에는 true. 키보드 수동 주행 시 false.")]
      public bool agentControlEnabled = true;

      // PINN temporal residual은 정책이 '명령한' raw action이 아니라 환경에
      // 실제 '적용된' 값을 필요로 한다. 속도 제한/브레이크/0 토크 등으로 raw와
      // 적용값이 달라질 수 있으므로 실제 적용값을 노출한다.
      public float LastAppliedThrottle { get; private set; }
      public float LastAppliedMotorTorque { get; private set; }
      public float LastAppliedBrakeTorque { get; private set; }

    //PRIVATE VARIABLES

      /*
      IMPORTANT: The following variables should not be modified manually since their values are automatically given via script.
      */
      Rigidbody carRigidbody; // Stores the car's rigidbody.
      float steeringAxis; // Used to know whether the steering wheel has reached the maximum value. It goes from -1 to 1.
      float throttleAxis; // Used to know whether the throttle has reached the maximum value. It goes from -1 to 1.
      float driftingAxis;
      float localVelocityZ;
      float localVelocityX;
      bool deceleratingCar;
      bool touchControlsSetup = false;
      /*
      The following variables are used to store information about sideways friction of the wheels (such as
      extremumSlip,extremumValue, asymptoteSlip, asymptoteValue and stiffness). We change this values to
      make the car to start drifting.
      */
      WheelFrictionCurve FLwheelFriction;
      float FLWextremumSlip;
      WheelFrictionCurve FRwheelFriction;
      float FRWextremumSlip;
      WheelFrictionCurve RLwheelFriction;
      float RLWextremumSlip;
      WheelFrictionCurve RRwheelFriction;
      float RRWextremumSlip;

    // Start is called before the first frame update
    void Start()
    {
      //In this part, we set the 'carRigidbody' value with the Rigidbody attached to this
      //gameObject. Also, we define the center of mass of the car with the Vector3 given
      //in the inspector.
      carRigidbody = gameObject.GetComponent<Rigidbody>();
      carRigidbody.centerOfMass = bodyMassCenter;

      //Initial setup to calculate the drift value of the car. This part could look a bit
      //complicated, but do not be afraid, the only thing we're doing here is to save the default
      //friction values of the car wheels so we can set an appropiate drifting value later.
      FLwheelFriction = new WheelFrictionCurve ();
        FLwheelFriction.extremumSlip = frontLeftCollider.sidewaysFriction.extremumSlip;
        FLWextremumSlip = frontLeftCollider.sidewaysFriction.extremumSlip;
        FLwheelFriction.extremumValue = frontLeftCollider.sidewaysFriction.extremumValue;
        FLwheelFriction.asymptoteSlip = frontLeftCollider.sidewaysFriction.asymptoteSlip;
        FLwheelFriction.asymptoteValue = frontLeftCollider.sidewaysFriction.asymptoteValue;
        FLwheelFriction.stiffness = frontLeftCollider.sidewaysFriction.stiffness;
      FRwheelFriction = new WheelFrictionCurve ();
        FRwheelFriction.extremumSlip = frontRightCollider.sidewaysFriction.extremumSlip;
        FRWextremumSlip = frontRightCollider.sidewaysFriction.extremumSlip;
        FRwheelFriction.extremumValue = frontRightCollider.sidewaysFriction.extremumValue;
        FRwheelFriction.asymptoteSlip = frontRightCollider.sidewaysFriction.asymptoteSlip;
        FRwheelFriction.asymptoteValue = frontRightCollider.sidewaysFriction.asymptoteValue;
        FRwheelFriction.stiffness = frontRightCollider.sidewaysFriction.stiffness;
      RLwheelFriction = new WheelFrictionCurve ();
        RLwheelFriction.extremumSlip = rearLeftCollider.sidewaysFriction.extremumSlip;
        RLWextremumSlip = rearLeftCollider.sidewaysFriction.extremumSlip;
        RLwheelFriction.extremumValue = rearLeftCollider.sidewaysFriction.extremumValue;
        RLwheelFriction.asymptoteSlip = rearLeftCollider.sidewaysFriction.asymptoteSlip;
        RLwheelFriction.asymptoteValue = rearLeftCollider.sidewaysFriction.asymptoteValue;
        RLwheelFriction.stiffness = rearLeftCollider.sidewaysFriction.stiffness;
      RRwheelFriction = new WheelFrictionCurve ();
        RRwheelFriction.extremumSlip = rearRightCollider.sidewaysFriction.extremumSlip;
        RRWextremumSlip = rearRightCollider.sidewaysFriction.extremumSlip;
        RRwheelFriction.extremumValue = rearRightCollider.sidewaysFriction.extremumValue;
        RRwheelFriction.asymptoteSlip = rearRightCollider.sidewaysFriction.asymptoteSlip;
        RRwheelFriction.asymptoteValue = rearRightCollider.sidewaysFriction.asymptoteValue;
        RRwheelFriction.stiffness = rearRightCollider.sidewaysFriction.stiffness;

        // We save the initial pitch of the car engine sound.
        if(carEngineSound != null){
          initialCarEngineSoundPitch = carEngineSound.pitch;
        }

        // We invoke 2 methods inside this script. CarSpeedUI() changes the text of the UI object that stores
        // the speed of the car and CarSounds() controls the engine and drifting sounds. Both methods are invoked
        // in 0 seconds, and repeatedly called every 0.1 seconds.
        if(useUI){
          InvokeRepeating("CarSpeedUI", 0f, 0.1f);
        }else if(!useUI){
          if(carSpeedText != null){
            carSpeedText.text = "0";
          }
        }

        if(useSounds){
          InvokeRepeating("CarSounds", 0f, 0.1f);
        }else if(!useSounds){
          if(carEngineSound != null){
            carEngineSound.Stop();
          }
          if(tireScreechSound != null){
            tireScreechSound.Stop();
          }
        }

        if(useTouchControls){
          if(throttleButton != null && reverseButton != null &&
          turnRightButton != null && turnLeftButton != null
          && handbrakeButton != null){

            throttlePTI = throttleButton.GetComponent<PrometeoTouchInput>();
            reversePTI = reverseButton.GetComponent<PrometeoTouchInput>();
            turnLeftPTI = turnLeftButton.GetComponent<PrometeoTouchInput>();
            turnRightPTI = turnRightButton.GetComponent<PrometeoTouchInput>();
            handbrakePTI = handbrakeButton.GetComponent<PrometeoTouchInput>();
            touchControlsSetup = true;

          }else{
            String ex = "Touch controls are not completely set up. You must drag and drop your scene buttons in the" +
            " PrometeoCarController component.";
            Debug.LogWarning(ex);
          }
        }

    
        LastAppliedThrottle = 0f;
        LastAppliedMotorTorque = 0f;
        LastAppliedBrakeTorque = 0f;

    }

    // Update is called once per frame
    void Update()
    {
      // CAR DATA

      // We determine the speed of the car.
      carSpeed = (2 * Mathf.PI * frontLeftCollider.radius * frontLeftCollider.rpm * 60) / 1000;

      // Save the local velocity of the car in the x axis. Used to know if the car is drifting.
      localVelocityX = transform.InverseTransformDirection(carRigidbody.velocity).x;

      // Save the local velocity of the car in the z axis. Used to know if the car is going forward or backwards.
      localVelocityZ = transform.InverseTransformDirection(carRigidbody.velocity).z;

      // ===== AGENT CONTROL GUARD =====
      // agentControlEnabled가 true이면(=RL 학습/추론 중) 아래의 키보드·터치
      // actuator 로직을 전부 건너뛴다. 그렇지 않으면 키를 누르지 않는 학습 중에
      // ThrottleOff()/DecelerateCar()가 매 프레임 호출되어 CarAgent가
      // SetThrottleNormalized()로 넣은 motorTorque를 즉시 0으로 덮어쓴다.
      // 표시용 carSpeed/localVelocity 계산과 AnimateWheelMeshes()는 계속 수행.
      if (agentControlEnabled)
      {
        AnimateWheelMeshes();
        return;
      }

      // CAR PHYSICS
      if (useTouchControls && touchControlsSetup)
      {
        if (throttlePTI.buttonPressed)
        {
          CancelInvoke("DecelerateCar");
          deceleratingCar = false;
          GoForward();
        }

        if (reversePTI.buttonPressed)
        {
          CancelInvoke("DecelerateCar");
          deceleratingCar = false;
          GoReverse();
        }

        if (turnLeftPTI.buttonPressed)
        {
          TurnLeft();
        }

        if (turnRightPTI.buttonPressed)
        {
          TurnRight();
        }

        if (handbrakePTI.buttonPressed)
        {
          CancelInvoke("DecelerateCar");
          deceleratingCar = false;
          Handbrake();
        }

        if (!handbrakePTI.buttonPressed)
        {
          RecoverTraction();
        }

        if ((!throttlePTI.buttonPressed && !reversePTI.buttonPressed))
        {
          ThrottleOff();
        }

        if ((!reversePTI.buttonPressed && !throttlePTI.buttonPressed) && !handbrakePTI.buttonPressed && !deceleratingCar)
        {
          InvokeRepeating("DecelerateCar", 0f, 0.1f);
          deceleratingCar = true;
        }

        // 삭제/비활성화:
        // Agent가 SetSteeringNormalized()로 넣은 조향값을 여기서 0으로 되돌리면 안 됨.
        /*
        if (!turnLeftPTI.buttonPressed && !turnRightPTI.buttonPressed && steeringAxis != 0f)
        {
          ResetSteeringAngle();
        }
        */
      }
      else
      {
        if (Input.GetKey(KeyCode.W))
        {
          CancelInvoke("DecelerateCar");
          deceleratingCar = false;
          GoForward();
        }

        if (Input.GetKey(KeyCode.S))
        {
          CancelInvoke("DecelerateCar");
          deceleratingCar = false;
          GoReverse();
        }

        if (Input.GetKey(KeyCode.A))
        {
          TurnLeft();
        }

        if (Input.GetKey(KeyCode.D))
        {
          TurnRight();
        }

        if (Input.GetKey(KeyCode.Space))
        {
          CancelInvoke("DecelerateCar");
          deceleratingCar = false;
          Handbrake();
        }

        if (Input.GetKeyUp(KeyCode.Space))
        {
          RecoverTraction();
        }

        if ((!Input.GetKey(KeyCode.S) && !Input.GetKey(KeyCode.W)))
        {
          ThrottleOff();
        }

        if ((!Input.GetKey(KeyCode.S) && !Input.GetKey(KeyCode.W)) && !Input.GetKey(KeyCode.Space) && !deceleratingCar)
        {
          InvokeRepeating("DecelerateCar", 0f, 0.1f);
          deceleratingCar = true;
        }

        // 삭제/비활성화:
        // Agent가 SetSteeringNormalized()로 넣은 조향값을 여기서 0으로 되돌리면 안 됨.
        /*
        if (!Input.GetKey(KeyCode.A) && !Input.GetKey(KeyCode.D) && steeringAxis != 0f)
        {
          ResetSteeringAngle();
        }
        */
      }

      // We call the method AnimateWheelMeshes() in order to match the wheel collider movements with the 3D meshes of the wheels.
      AnimateWheelMeshes();
    }
        // This method converts the car speed data from float to string, and then set the text of the UI carSpeedText with this value.
    public void CarSpeedUI(){

      if(useUI){
          try{
            float absoluteCarSpeed = Mathf.Abs(carSpeed);
            carSpeedText.text = Mathf.RoundToInt(absoluteCarSpeed).ToString();
          }catch(Exception ex){
            Debug.LogWarning(ex);
          }
      }

    }

    // This method controls the car sounds. For example, the car engine will sound slow when the car speed is low because the
    // pitch of the sound will be at its lowest point. On the other hand, it will sound fast when the car speed is high because
    // the pitch of the sound will be the sum of the initial pitch + the car speed divided by 100f.
    // Apart from that, the tireScreechSound will play whenever the car starts drifting or losing traction.
    public void CarSounds(){

      if(useSounds){
        try{
          if(carEngineSound != null){
            float engineSoundPitch = initialCarEngineSoundPitch + (Mathf.Abs(carRigidbody.velocity.magnitude) / 25f);
            carEngineSound.pitch = engineSoundPitch;
          }
          if((isDrifting) || (isTractionLocked && Mathf.Abs(carSpeed) > 12f)){
            if(!tireScreechSound.isPlaying){
              tireScreechSound.Play();
            }
          }else if((!isDrifting) && (!isTractionLocked || Mathf.Abs(carSpeed) < 12f)){
            tireScreechSound.Stop();
          }
        }catch(Exception ex){
          Debug.LogWarning(ex);
        }
      }else if(!useSounds){
        if(carEngineSound != null && carEngineSound.isPlaying){
          carEngineSound.Stop();
        }
        if(tireScreechSound != null && tireScreechSound.isPlaying){
          tireScreechSound.Stop();
        }
      }

    }

    //
    //STEERING METHODS
    //

    //The following method turns the front car wheels to the left. The speed of this movement will depend on the steeringSpeed variable.
    public void TurnLeft(){
      steeringAxis = steeringAxis - (Time.deltaTime * 10f * steeringSpeed);
      if(steeringAxis < -1f){
        steeringAxis = -1f;
      }
      var steeringAngle = steeringAxis * maxSteeringAngle;
      frontLeftCollider.steerAngle = Mathf.Lerp(frontLeftCollider.steerAngle, steeringAngle, steeringSpeed);
      frontRightCollider.steerAngle = Mathf.Lerp(frontRightCollider.steerAngle, steeringAngle, steeringSpeed);
    }

    //The following method turns the front car wheels to the right. The speed of this movement will depend on the steeringSpeed variable.
    public void TurnRight(){
      steeringAxis = steeringAxis + (Time.deltaTime * 10f * steeringSpeed);
      if(steeringAxis > 1f){
        steeringAxis = 1f;
      }
      var steeringAngle = steeringAxis * maxSteeringAngle;
      frontLeftCollider.steerAngle = Mathf.Lerp(frontLeftCollider.steerAngle, steeringAngle, steeringSpeed);
      frontRightCollider.steerAngle = Mathf.Lerp(frontRightCollider.steerAngle, steeringAngle, steeringSpeed);
    }

    //The following method takes the front car wheels to their default position (rotation = 0). The speed of this movement will depend
    // on the steeringSpeed variable.
    public void ResetSteeringAngle(){
      if(steeringAxis < 0f){
        steeringAxis = steeringAxis + (Time.deltaTime * 10f * steeringSpeed);
      }else if(steeringAxis > 0f){
        steeringAxis = steeringAxis - (Time.deltaTime * 10f * steeringSpeed);
      }
      if(Mathf.Abs(frontLeftCollider.steerAngle) < 1f){
        steeringAxis = 0f;
      }
      var steeringAngle = steeringAxis * maxSteeringAngle;
      frontLeftCollider.steerAngle = Mathf.Lerp(frontLeftCollider.steerAngle, steeringAngle, steeringSpeed);
      frontRightCollider.steerAngle = Mathf.Lerp(frontRightCollider.steerAngle, steeringAngle, steeringSpeed);
    }

    // ===== RL / PINN STEERING (proportional, non-accumulative) =====
    // RL 연속 액션(steer ∈ [-1, 1])을 받아 조향각을 "직접" 설정한다.
    // Prometeo 기본 TurnLeft()/TurnRight()는 매 프레임 steeringAxis를 누적시키는
    // 키보드용 함수라서, RL이 같은 부호를 연속으로 내면 풀 lock까지 쌓여 한쪽으로
    // 박는다. 또 if(steer>0.1)식 분기는 0.2든 0.9든 똑같이 처리해 액션 크기를 버린다.
    // 이 메서드는 value를 그대로 steerAngle = value * maxSteeringAngle 로 매핑하므로
    // 연속 액션의 크기가 보존되고, policy의 조향각 δ 와 실제 조향각이 1:1로 일치한다.
    // (PINN의 yaw_rate = v/L * tan(δ) 제약이 거짓말을 하지 않게 됨.)
    public void SetSteeringNormalized(float value){
      value = Mathf.Clamp(value, -1f, 1f);
      steeringAxis = value; // 누적 함수들과 상태 일관성 유지
      float angle = steeringAxis * maxSteeringAngle;
      frontLeftCollider.steerAngle = angle;
      frontRightCollider.steerAngle = angle;
    }

    // ===== RL / PINN THROTTLE (proportional, magnitude-preserving) =====
    // RL 연속 액션(throttle ∈ [-1, 1])의 크기를 그대로 motorTorque에 반영한다.
    // 기존 GoForward()/GoReverse()는 throttleAxis를 시간에 따라 ±1로 키우는
    // 키보드용이라 policy가 0.2를 내든 0.9를 내든 결국 같은 풀스로틀로 수렴해
    // 액션 크기를 버린다. 이 메서드는 throttle 값을 직접 토크로 매핑하므로
    // PINN의 종방향 가속도(ax) 제약이 실제 입력과 일치한다.
    //
    // 설계 노트:
    // - deadzone 없음: 작은 throttle도 그대로 반영해야 Python 물리식 입력과
    //   Unity 실제 입력이 일치한다(PINN 정합성). 정확히 0이면 토크 0.
    // - 매 호출 motor/brake를 완전 초기화하여 이전 step의 brake/motor 잔류를 제거.
    // - 속도 제한은 wheel RPM 기반 carSpeed 대신 Rigidbody 종방향 속도를 사용
    //   (바퀴 슬립/공중 뜸에 영향받지 않고 PINN 물리식과 같은 상태 변수 사용).
    // - 토크를 1차 지연 없이 즉시 매핑(의도된 설계: action-토크 시간지연 제거).
        public void SetThrottleNormalized(float value)
    {
      EnsureCarRigidbody();

      value = Mathf.Clamp(value, -1f, 1f);
      throttleAxis = value; // 상태 일관성 유지

      // 매 호출 actuator 상태 완전 초기화 (이전 step의 motor/brake 잔류 제거)
      CancelInvoke(nameof(DecelerateCar));
      deceleratingCar = false;
      SetAllMotorTorque(0f);
      SetAllBrakeTorque(0f);

      LastAppliedThrottle = 0f;
      LastAppliedMotorTorque = 0f;
      LastAppliedBrakeTorque = 0f;

      // 제어/물리용 속도: Rigidbody 종방향 속도(m/s) → km/h.
      // WheelCollider rpm 기반 carSpeed는 slip/공중뜸에 민감하므로 제어 한계에는 쓰지 않는다.
      float signedSpeedMps = Vector3.Dot(carRigidbody.velocity, transform.forward);
      float absoluteSpeedKph = Mathf.Abs(signedSpeedMps) * 3.6f;

      // 정확히 0 입력이면 코스팅(토크 0, 브레이크 0)
      if (Mathf.Approximately(value, 0f))
      {
        return;
      }

      // 전진 중 후진 입력, 또는 후진 중 전진 입력 시 우선 제동.
      // 이때 실제 적용 throttle은 0으로 기록한다.
      bool changingDirection =
        (value > 0f && signedSpeedMps < -1f) ||
        (value < 0f && signedSpeedMps > 1f);

      if (changingDirection)
      {
        SetAllBrakeTorque(brakeForce);
        LastAppliedBrakeTorque = brakeForce;
        return;
      }

      // 속도 한계 체크 (Rigidbody 속도 기반, RoundToInt 제거)
      bool overSpeedForward = (value > 0f) && (absoluteSpeedKph >= maxSpeed);
      bool overSpeedReverse = (value < 0f) && (absoluteSpeedKph >= maxReverseSpeed);

      if (overSpeedForward || overSpeedReverse)
      {
        // 한계 도달: 토크 0, 브레이크 0 상태 유지
        return;
      }

      // 비례 토크: |throttle|에 비례, 부호가 방향.
      // accelerationMultiplier * 50은 기존 GoForward/GoReverse와 같은 토크 스케일.
      float torque = (accelerationMultiplier * 50f) * value;
      SetAllMotorTorque(torque);

      LastAppliedThrottle = value;
      LastAppliedMotorTorque = torque;
      LastAppliedBrakeTorque = 0f;
    }

    // 네 바퀴 motor/brake를 한 번에 설정하는 helper (상태 잔류 방지용)
    private void SetAllMotorTorque(float torque)
    {
      frontLeftCollider.motorTorque = torque;
      frontRightCollider.motorTorque = torque;
      rearLeftCollider.motorTorque = torque;
      rearRightCollider.motorTorque = torque;
    }

    private void SetAllBrakeTorque(float torque)
    {
      frontLeftCollider.brakeTorque = torque;
      frontRightCollider.brakeTorque = torque;
      rearLeftCollider.brakeTorque = torque;
      rearRightCollider.brakeTorque = torque;
    }

    private void EnsureCarRigidbody()
    {
      if (carRigidbody == null)
      {
        carRigidbody = gameObject.GetComponent<Rigidbody>();
      }
    }

    // Agent episode 시작/환경 reset 전용.
    // 기존 ThrottleOff() + ResetSteeringAngle() 조합은 brakeTorque, Invoke, LastApplied* 값을
    // 완전히 초기화하지 못하므로 학습 episode 사이에 actuator 상태가 남을 수 있다.
    public void ResetAgentActuators()
    {
      EnsureCarRigidbody();

      CancelInvoke(nameof(DecelerateCar));
      CancelInvoke(nameof(RecoverTraction));
      deceleratingCar = false;

      throttleAxis = 0f;
      steeringAxis = 0f;

      SetAllMotorTorque(0f);
      SetAllBrakeTorque(0f);

      frontLeftCollider.steerAngle = 0f;
      frontRightCollider.steerAngle = 0f;

      LastAppliedThrottle = 0f;
      LastAppliedMotorTorque = 0f;
      LastAppliedBrakeTorque = 0f;

      isDrifting = false;
      isTractionLocked = false;
      driftingAxis = 0f;
    }


    // This method matches both the position and rotation of the WheelColliders with the WheelMeshes.
    void AnimateWheelMeshes(){
      try{
        Quaternion FLWRotation;
        Vector3 FLWPosition;
        frontLeftCollider.GetWorldPose(out FLWPosition, out FLWRotation);
        frontLeftMesh.transform.position = FLWPosition;
        frontLeftMesh.transform.rotation = FLWRotation;

        Quaternion FRWRotation;
        Vector3 FRWPosition;
        frontRightCollider.GetWorldPose(out FRWPosition, out FRWRotation);
        frontRightMesh.transform.position = FRWPosition;
        frontRightMesh.transform.rotation = FRWRotation;

        Quaternion RLWRotation;
        Vector3 RLWPosition;
        rearLeftCollider.GetWorldPose(out RLWPosition, out RLWRotation);
        rearLeftMesh.transform.position = RLWPosition;
        rearLeftMesh.transform.rotation = RLWRotation;

        Quaternion RRWRotation;
        Vector3 RRWPosition;
        rearRightCollider.GetWorldPose(out RRWPosition, out RRWRotation);
        rearRightMesh.transform.position = RRWPosition;
        rearRightMesh.transform.rotation = RRWRotation;
      }catch(Exception ex){
        Debug.LogWarning(ex);
      }
    }

    //
    //ENGINE AND BRAKING METHODS
    //

    // This method apply positive torque to the wheels in order to go forward.
    public void GoForward(){
      //If the forces aplied to the rigidbody in the 'x' axis are greater than
      //2.5f, it means that the car is losing traction.
      isDrifting = Mathf.Abs(localVelocityX) > 2.5f;
      // The following part sets the throttle power to 1 smoothly.
      throttleAxis = throttleAxis + (Time.deltaTime * 3f);
      if(throttleAxis > 1f){
        throttleAxis = 1f;
      }
      //If the car is going backwards, then apply brakes in order to avoid strange
      //behaviours. If the local velocity in the 'z' axis is less than -1f, then it
      //is safe to apply positive torque to go forward.
      if(localVelocityZ < -1f){
        Brakes();
      }else{
        if(Mathf.RoundToInt(carSpeed) < maxSpeed){
          //Apply positive torque in all wheels to go forward if maxSpeed has not been reached.
          frontLeftCollider.brakeTorque = 0;
          frontLeftCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
          frontRightCollider.brakeTorque = 0;
          frontRightCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
          rearLeftCollider.brakeTorque = 0;
          rearLeftCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
          rearRightCollider.brakeTorque = 0;
          rearRightCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
        }else {
          // If the maxSpeed has been reached, then stop applying torque to the wheels.
          // IMPORTANT: The maxSpeed variable should be considered as an approximation; the speed of the car
          // could be a bit higher than expected.
    			frontLeftCollider.motorTorque = 0;
    			frontRightCollider.motorTorque = 0;
          rearLeftCollider.motorTorque = 0;
    			rearRightCollider.motorTorque = 0;
    		}
      }
    }

    // This method apply negative torque to the wheels in order to go backwards.
    public void GoReverse(){
      //If the forces aplied to the rigidbody in the 'x' axis are greater than
      //2.5f, it means that the car is losing traction.
      isDrifting = Mathf.Abs(localVelocityX) > 2.5f;
      // The following part sets the throttle power to -1 smoothly.
      throttleAxis = throttleAxis - (Time.deltaTime * 3f);
      if(throttleAxis < -1f){
        throttleAxis = -1f;
      }
      //If the car is still going forward, then apply brakes in order to avoid strange
      //behaviours. If the local velocity in the 'z' axis is greater than 1f, then it
      //is safe to apply negative torque to go reverse.
      if(localVelocityZ > 1f){
        Brakes();
      }else{
        if(Mathf.Abs(Mathf.RoundToInt(carSpeed)) < maxReverseSpeed){
          //Apply negative torque in all wheels to go in reverse if maxReverseSpeed has not been reached.
          frontLeftCollider.brakeTorque = 0;
          frontLeftCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
          frontRightCollider.brakeTorque = 0;
          frontRightCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
          rearLeftCollider.brakeTorque = 0;
          rearLeftCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
          rearRightCollider.brakeTorque = 0;
          rearRightCollider.motorTorque = (accelerationMultiplier * 50f) * throttleAxis;
        }else {
          //If the maxReverseSpeed has been reached, then stop applying torque to the wheels.
          // IMPORTANT: The maxReverseSpeed variable should be considered as an approximation; the speed of the car
          // could be a bit higher than expected.
    			frontLeftCollider.motorTorque = 0;
    			frontRightCollider.motorTorque = 0;
          rearLeftCollider.motorTorque = 0;
    			rearRightCollider.motorTorque = 0;
    		}
      }
    }

    //The following function set the motor torque to 0 (in case the user is not pressing either W or S).
        public void ThrottleOff(){
      SetAllMotorTorque(0f);
      throttleAxis = 0f;

      LastAppliedThrottle = 0f;
      LastAppliedMotorTorque = 0f;
      // brakeTorque는 여기서 건드리지 않는다.
      // 수동 handbrake/Brakes 상태를 ThrottleOff가 풀어버리면 기존 Prometeo 동작이 깨진다.
    }

    // The following method decelerates the speed of the car according to the decelerationMultiplier variable, where
    // 1 is the slowest and 10 is the fastest deceleration. This method is called by the function InvokeRepeating,
    // usually every 0.1f when the user is not pressing W (throttle), S (reverse) or Space bar (handbrake).
        public void DecelerateCar(){
      EnsureCarRigidbody();

      isDrifting = Mathf.Abs(localVelocityX) > 2.5f;

      // The following part resets the throttle power to 0 smoothly.
      if(throttleAxis != 0f){
        if(throttleAxis > 0f){
          throttleAxis = throttleAxis - (Time.deltaTime * 10f);
        }else if(throttleAxis < 0f){
            throttleAxis = throttleAxis + (Time.deltaTime * 10f);
        }
        if(Mathf.Abs(throttleAxis) < 0.15f){
          throttleAxis = 0f;
        }
      }

      carRigidbody.velocity = carRigidbody.velocity * (1f / (1f + (0.025f * decelerationMultiplier)));

      // Since we want to decelerate the car, remove motor torque from all wheels.
      SetAllMotorTorque(0f);
      LastAppliedThrottle = 0f;
      LastAppliedMotorTorque = 0f;

      // If the magnitude of the car's velocity is less than 0.25f, stop completely.
      if(carRigidbody.velocity.magnitude < 0.25f){
        carRigidbody.velocity = Vector3.zero;
        CancelInvoke(nameof(DecelerateCar));
        deceleratingCar = false;
      }
    }

    // This function applies brake torque to the wheels according to the brake force given by the user.
        public void Brakes(){
      SetAllMotorTorque(0f);
      SetAllBrakeTorque(brakeForce);

      LastAppliedThrottle = 0f;
      LastAppliedMotorTorque = 0f;
      LastAppliedBrakeTorque = brakeForce;
    }

    // This function is used to make the car lose traction. By using this, the car will start drifting. The amount of traction lost
    // will depend on the handbrakeDriftMultiplier variable. If this value is small, then the car will not drift too much, but if
    // it is high, then you could make the car to feel like going on ice.
    public void Handbrake(){
      CancelInvoke("RecoverTraction");
      // We are going to start losing traction smoothly, there is were our 'driftingAxis' variable takes
      // place. This variable will start from 0 and will reach a top value of 1, which means that the maximum
      // drifting value has been reached. It will increase smoothly by using the variable Time.deltaTime.
      driftingAxis = driftingAxis + (Time.deltaTime);
      float secureStartingPoint = driftingAxis * FLWextremumSlip * handbrakeDriftMultiplier;

      if(secureStartingPoint < FLWextremumSlip){
        driftingAxis = FLWextremumSlip / (FLWextremumSlip * handbrakeDriftMultiplier);
      }
      if(driftingAxis > 1f){
        driftingAxis = 1f;
      }
      //If the forces aplied to the rigidbody in the 'x' axis are greater than
      //2.5f, it means that the car lost its traction.
      isDrifting = Mathf.Abs(localVelocityX) > 2.5f;
      //If the 'driftingAxis' value is not 1f, it means that the wheels have not reach their maximum drifting
      //value, so, we are going to continue increasing the sideways friction of the wheels until driftingAxis
      // = 1f.
      if(driftingAxis < 1f){
        FLwheelFriction.extremumSlip = FLWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        frontLeftCollider.sidewaysFriction = FLwheelFriction;

        FRwheelFriction.extremumSlip = FRWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        frontRightCollider.sidewaysFriction = FRwheelFriction;

        RLwheelFriction.extremumSlip = RLWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        rearLeftCollider.sidewaysFriction = RLwheelFriction;

        RRwheelFriction.extremumSlip = RRWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        rearRightCollider.sidewaysFriction = RRwheelFriction;
      }

      // Whenever the player uses the handbrake, it means that the wheels are locked.
      isTractionLocked = true;

    }

    // This function is used to recover the traction of the car when the user has stopped using the car's handbrake.
    public void RecoverTraction(){
      isTractionLocked = false;
      driftingAxis = driftingAxis - (Time.deltaTime / 1.5f);
      if(driftingAxis < 0f){
        driftingAxis = 0f;
      }

      //If the 'driftingAxis' value is not 0f, it means that the wheels have not recovered their traction.
      //We are going to continue decreasing the sideways friction of the wheels until we reach the initial
      // car's grip.
      if(FLwheelFriction.extremumSlip > FLWextremumSlip){
        FLwheelFriction.extremumSlip = FLWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        frontLeftCollider.sidewaysFriction = FLwheelFriction;

        FRwheelFriction.extremumSlip = FRWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        frontRightCollider.sidewaysFriction = FRwheelFriction;

        RLwheelFriction.extremumSlip = RLWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        rearLeftCollider.sidewaysFriction = RLwheelFriction;

        RRwheelFriction.extremumSlip = RRWextremumSlip * handbrakeDriftMultiplier * driftingAxis;
        rearRightCollider.sidewaysFriction = RRwheelFriction;

        Invoke("RecoverTraction", Time.deltaTime);

      }else if (FLwheelFriction.extremumSlip < FLWextremumSlip){
        FLwheelFriction.extremumSlip = FLWextremumSlip;
        frontLeftCollider.sidewaysFriction = FLwheelFriction;

        FRwheelFriction.extremumSlip = FRWextremumSlip;
        frontRightCollider.sidewaysFriction = FRwheelFriction;

        RLwheelFriction.extremumSlip = RLWextremumSlip;
        rearLeftCollider.sidewaysFriction = RLwheelFriction;

        RRwheelFriction.extremumSlip = RRWextremumSlip;
        rearRightCollider.sidewaysFriction = RRwheelFriction;

        driftingAxis = 0f;
      }
    }

}