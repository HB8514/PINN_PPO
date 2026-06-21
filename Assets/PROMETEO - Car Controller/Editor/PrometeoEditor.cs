using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(PrometeoCarController))]
[System.Serializable]
public class PrometeoEditor : Editor
{
    private SerializedObject SO;

    //
    // CAR SETUP
    //
    private SerializedProperty maxSpeed;
    private SerializedProperty maxReverseSpeed;
    private SerializedProperty accelerationMultiplier;
    private SerializedProperty maxSteeringAngle;
    private SerializedProperty steeringSpeed;
    private SerializedProperty brakeForce;
    private SerializedProperty decelerationMultiplier;
    private SerializedProperty handbrakeDriftMultiplier;
    private SerializedProperty bodyMassCenter;

    //
    // WHEELS VARIABLES
    //
    private SerializedProperty frontLeftMesh;
    private SerializedProperty frontLeftCollider;
    private SerializedProperty frontRightMesh;
    private SerializedProperty frontRightCollider;
    private SerializedProperty rearLeftMesh;
    private SerializedProperty rearLeftCollider;
    private SerializedProperty rearRightMesh;
    private SerializedProperty rearRightCollider;

    //
    // UI VARIABLES
    //
    private SerializedProperty useUI;
    private SerializedProperty carSpeedText;

    //
    // SOUND VARIABLES
    //
    private SerializedProperty useSounds;
    private SerializedProperty carEngineSound;
    private SerializedProperty tireScreechSound;

    //
    // TOUCH CONTROLS VARIABLES
    //
    private SerializedProperty useTouchControls;
    private SerializedProperty throttleButton;
    private SerializedProperty reverseButton;
    private SerializedProperty turnRightButton;
    private SerializedProperty turnLeftButton;
    private SerializedProperty handbrakeButton;

    private void OnEnable()
    {
        SO = new SerializedObject(target);

        maxSpeed = SO.FindProperty("maxSpeed");
        maxReverseSpeed = SO.FindProperty("maxReverseSpeed");
        accelerationMultiplier = SO.FindProperty("accelerationMultiplier");
        maxSteeringAngle = SO.FindProperty("maxSteeringAngle");
        steeringSpeed = SO.FindProperty("steeringSpeed");
        brakeForce = SO.FindProperty("brakeForce");
        decelerationMultiplier = SO.FindProperty("decelerationMultiplier");
        handbrakeDriftMultiplier = SO.FindProperty("handbrakeDriftMultiplier");
        bodyMassCenter = SO.FindProperty("bodyMassCenter");

        frontLeftMesh = SO.FindProperty("frontLeftMesh");
        frontLeftCollider = SO.FindProperty("frontLeftCollider");
        frontRightMesh = SO.FindProperty("frontRightMesh");
        frontRightCollider = SO.FindProperty("frontRightCollider");
        rearLeftMesh = SO.FindProperty("rearLeftMesh");
        rearLeftCollider = SO.FindProperty("rearLeftCollider");
        rearRightMesh = SO.FindProperty("rearRightMesh");
        rearRightCollider = SO.FindProperty("rearRightCollider");

        useUI = SO.FindProperty("useUI");
        carSpeedText = SO.FindProperty("carSpeedText");

        useSounds = SO.FindProperty("useSounds");
        carEngineSound = SO.FindProperty("carEngineSound");
        tireScreechSound = SO.FindProperty("tireScreechSound");

        useTouchControls = SO.FindProperty("useTouchControls");
        throttleButton = SO.FindProperty("throttleButton");
        reverseButton = SO.FindProperty("reverseButton");
        turnRightButton = SO.FindProperty("turnRightButton");
        turnLeftButton = SO.FindProperty("turnLeftButton");
        handbrakeButton = SO.FindProperty("handbrakeButton");
    }

    public override void OnInspectorGUI()
    {
        SO.Update();

        //
        // CAR SETUP
        //
        GUILayout.Space(25);
        GUILayout.Label("CAR SETUP", EditorStyles.boldLabel);
        GUILayout.Space(10);

        if (maxSpeed != null)
            maxSpeed.intValue = EditorGUILayout.IntSlider("Max Speed:", maxSpeed.intValue, 20, 190);

        if (maxReverseSpeed != null)
            maxReverseSpeed.intValue = EditorGUILayout.IntSlider("Max Reverse Speed:", maxReverseSpeed.intValue, 10, 120);

        if (accelerationMultiplier != null)
            accelerationMultiplier.intValue = EditorGUILayout.IntSlider("Acceleration Multiplier:", accelerationMultiplier.intValue, 1, 10);

        if (maxSteeringAngle != null)
            maxSteeringAngle.intValue = EditorGUILayout.IntSlider("Max Steering Angle:", maxSteeringAngle.intValue, 10, 45);

        if (steeringSpeed != null)
            steeringSpeed.floatValue = EditorGUILayout.Slider("Steering Speed:", steeringSpeed.floatValue, 0.1f, 1f);

        if (brakeForce != null)
            brakeForce.intValue = EditorGUILayout.IntSlider("Brake Force:", brakeForce.intValue, 100, 600);

        if (decelerationMultiplier != null)
            decelerationMultiplier.intValue = EditorGUILayout.IntSlider("Deceleration Multiplier:", decelerationMultiplier.intValue, 1, 10);

        if (handbrakeDriftMultiplier != null)
            handbrakeDriftMultiplier.intValue = EditorGUILayout.IntSlider("Drift Multiplier:", handbrakeDriftMultiplier.intValue, 1, 10);

        if (bodyMassCenter != null)
            EditorGUILayout.PropertyField(bodyMassCenter, new GUIContent("Mass Center of Car: "));

        //
        // WHEELS
        //
        GUILayout.Space(25);
        GUILayout.Label("WHEELS", EditorStyles.boldLabel);
        GUILayout.Space(10);

        if (frontLeftMesh != null)
            EditorGUILayout.PropertyField(frontLeftMesh, new GUIContent("Front Left Mesh: "));

        if (frontLeftCollider != null)
            EditorGUILayout.PropertyField(frontLeftCollider, new GUIContent("Front Left Collider: "));

        if (frontRightMesh != null)
            EditorGUILayout.PropertyField(frontRightMesh, new GUIContent("Front Right Mesh: "));

        if (frontRightCollider != null)
            EditorGUILayout.PropertyField(frontRightCollider, new GUIContent("Front Right Collider: "));

        if (rearLeftMesh != null)
            EditorGUILayout.PropertyField(rearLeftMesh, new GUIContent("Rear Left Mesh: "));

        if (rearLeftCollider != null)
            EditorGUILayout.PropertyField(rearLeftCollider, new GUIContent("Rear Left Collider: "));

        if (rearRightMesh != null)
            EditorGUILayout.PropertyField(rearRightMesh, new GUIContent("Rear Right Mesh: "));

        if (rearRightCollider != null)
            EditorGUILayout.PropertyField(rearRightCollider, new GUIContent("Rear Right Collider: "));

        //
        // UI
        //
        GUILayout.Space(25);
        GUILayout.Label("UI", EditorStyles.boldLabel);
        GUILayout.Space(10);

        if (useUI != null)
        {
            useUI.boolValue = EditorGUILayout.BeginToggleGroup("Use UI (Speed text)?", useUI.boolValue);
            GUILayout.Space(10);

            if (carSpeedText != null)
                EditorGUILayout.PropertyField(carSpeedText, new GUIContent("Speed Text (UI): "));

            EditorGUILayout.EndToggleGroup();
        }

        //
        // SOUNDS
        //
        GUILayout.Space(25);
        GUILayout.Label("SOUNDS", EditorStyles.boldLabel);
        GUILayout.Space(10);

        if (useSounds != null)
        {
            useSounds.boolValue = EditorGUILayout.BeginToggleGroup("Use sounds (car sounds)?", useSounds.boolValue);
            GUILayout.Space(10);

            if (carEngineSound != null)
                EditorGUILayout.PropertyField(carEngineSound, new GUIContent("Car Engine Sound: "));

            if (tireScreechSound != null)
                EditorGUILayout.PropertyField(tireScreechSound, new GUIContent("Tire Screech Sound: "));

            EditorGUILayout.EndToggleGroup();
        }

        //
        // TOUCH CONTROLS
        //
        GUILayout.Space(25);
        GUILayout.Label("TOUCH CONTROLS", EditorStyles.boldLabel);
        GUILayout.Space(10);

        if (useTouchControls != null)
        {
            useTouchControls.boolValue = EditorGUILayout.BeginToggleGroup("Use touch controls (mobile devices)?", useTouchControls.boolValue);
            GUILayout.Space(10);

            if (throttleButton != null)
                EditorGUILayout.PropertyField(throttleButton, new GUIContent("Throttle Button: "));

            if (reverseButton != null)
                EditorGUILayout.PropertyField(reverseButton, new GUIContent("Brakes/Reverse Button: "));

            if (turnLeftButton != null)
                EditorGUILayout.PropertyField(turnLeftButton, new GUIContent("Turn Left Button: "));

            if (turnRightButton != null)
                EditorGUILayout.PropertyField(turnRightButton, new GUIContent("Turn Right Button: "));

            if (handbrakeButton != null)
                EditorGUILayout.PropertyField(handbrakeButton, new GUIContent("Handbrake Button: "));

            EditorGUILayout.EndToggleGroup();
        }

        GUILayout.Space(10);

        SO.ApplyModifiedProperties();
    }
}