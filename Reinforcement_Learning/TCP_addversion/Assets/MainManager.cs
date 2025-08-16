using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System;
using NUnit.Framework;
using Random = UnityEngine.Random;
// Add libraries for connection
using System.Net; 
using System.Net.Sockets;
using System.Threading;
using System.Collections;


public enum GameState
{
    WaitingForAction,
    Simulation,
    End
}

public enum EventType 
{ 
    Cushion, 
    BallRed, 
    BallYellow 
}

public class MainManager : basicSingleton<MainManager>
{
    public int episode;

    // UI
    public Slider timeSlider;
    public TMP_Text timeSliderText;
    public TMP_Text episodeText;
    public TMP_Text detailText;
    public Toggle autostartToggle;

    public List<Ball> balls;

    public List<BallType> touchedBallType;
    // Order of the events
    public List<EventType> eventLog = new List<EventType>();
    public int cushionCount = 0;

    public GameState state;

    public List<Vector2> currentStartPos;
    // Backup if it failed
    private List<Vector2> lastTriedStartPos;
    public List<List<Vector2>> predefinedPositions = new List<List<Vector2>>();
    // Current position index
    public int positionIndex = 0;

    public float currentAngle;
    public float currentForce;
    public bool lastSuccess = false;

    private string folderPath;
    private string filePath;
    float baseFixedDeltaTime;

    // For TCP
    private TcpListener tcpListener;
    private Thread listenerThread;
    private TcpClient connectedClient;
    private NetworkStream stream;
    private bool clientConnected = false; 

    void Awake()
    {
        StartTCPServer();
        folderPath = Application.dataPath + "/SimulationData";

        if (!Directory.Exists(folderPath))
            Directory.CreateDirectory(folderPath);

        filePath = Path.Combine(folderPath, "EpisodeLog20.csv");

        if (balls == null || balls.Count == 0)
        {
            balls = new List<Ball>(FindObjectsOfType<Ball>());
            Debug.Log($"[Awake] {balls.Count} balls are loaded automatically");
        }

        if (!File.Exists(filePath))
        {
            using (StreamWriter writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                writer.WriteLine("Episode,WhiteBallStartX,WhiteBallStartY,RedBallStartX,RedBallStartY,YellowBallStartX,YellowBallStartY,HitAngle,HitForce,RedTouched,YellowTouched,CushionCount,Score,Reward,TimeStamp");
            }

            InitializePositions();
        }
        // Calculate the last episode number
        else
        {
            string lastLine = null;
            using (StreamReader reader = new StreamReader(filePath))
            {
                while (!reader.EndOfStream)
                {
                    lastLine = reader.ReadLine();
                }
            }
            if (!string.IsNullOrWhiteSpace(lastLine) && !lastLine.StartsWith("Episode"))
            {
                string[] parts = lastLine.Split(',');
                if (parts.Length >= 8)
                {
                    int lastEp = int.Parse(parts[0]);
                    float whiteX = float.Parse(parts[1]);
                    float whiteY = float.Parse(parts[2]);
                    float redX = float.Parse(parts[3]);
                    float redY = float.Parse(parts[4]);
                    float yellowX = float.Parse(parts[5]);
                    float yellowY = float.Parse(parts[6]);

                    episode = lastEp + 1;
                    
                    currentStartPos = new List<Vector2>()
                    {
                        new Vector2(whiteX, whiteY),
                        new Vector2(redX, redY),
                        new Vector2(yellowX, yellowY)
                    };

                    for (int i = 0; i < balls.Count; i++)
                    {
                        balls[i].transform.position = currentStartPos[i];
                    }
                }
            }
        }
    }

    void Start()
    {
        // Initialize time scale
        baseFixedDeltaTime = Time.fixedDeltaTime;
        if (timeSlider != null)
        {
            timeSlider.onValueChanged.AddListener(OnSpeedChanged);
            OnSpeedChanged(1);
        }

        // Set the balls' position to the succeded position
        predefinedPositions = new List<List<Vector2>>
        {
            new List<Vector2> {new Vector2(3.74f, 0.95f), new Vector2(-0.93f, 0.72f), new Vector2(5.44f, 1.75f)},
            new List<Vector2> {new Vector2(0.55f, -2.06f), new Vector2(1.30f, -1.0f), new Vector2(4.40f, -1.64f)},
            new List<Vector2> {new Vector2(-1.22f, 0.99f), new Vector2(5.04f, 0.14f), new Vector2(3.84f, 1.40f)},
            new List<Vector2> {new Vector2(3.16f, -0.46f), new Vector2(4.46f, -0.27f), new Vector2(-4.39f, 0.30f)},
            new List<Vector2> {new Vector2(-0.28f, 1.31f), new Vector2(5.15f, -2.45f), new Vector2(4.80f, -1.33f)},
            new List<Vector2> {new Vector2(-3.25f, 1.19f), new Vector2(4.55f, -0.71f), new Vector2(0.05f, -0.92f)},
            new List<Vector2> {new Vector2(-3.31f, 0.76f), new Vector2(-0.43f, 0.16f), new Vector2(4.82f, 0.51f)},
            new List<Vector2> {new Vector2(-2.91f, -1.23f), new Vector2(-4.70f, -1.07f), new Vector2(0.09f, -0.24f)},
            new List<Vector2> {new Vector2(-0.26f, 1.47f), new Vector2(-3.34f, -2.44f), new Vector2(-4.75f, -1.69f)},
            new List<Vector2> {new Vector2(-2.53f, -2.48f), new Vector2(2.44f, 1.04f), new Vector2(4.20f, 1.71f)},
            new List<Vector2> {new Vector2(-0.22f, -1.02f), new Vector2(0.00f, 2.31f), new Vector2(-1.15f, 0.59f)},
            new List<Vector2> {new Vector2(3.41f, 2.42f), new Vector2(0.93f, -2.19f), new Vector2(-1.99f, -2.40f)},
            new List<Vector2> {new Vector2(-2.38f, 0.03f), new Vector2(2.07f, -0.67f), new Vector2(5.23f, -1.02f)},
            new List<Vector2> {new Vector2(4.95f, -0.23f), new Vector2(2.20f, -2.40f), new Vector2(-4.96f, -0.59f)},
            new List<Vector2> {new Vector2(1.37f, -0.49f), new Vector2(-0.49f, -0.37f), new Vector2(-1.07f, -0.74f)},
            new List<Vector2> {new Vector2(-0.15f, 2.43f), new Vector2(-2.82f, -2.22f), new Vector2(4.86f, 0.88f)},
            new List<Vector2> {new Vector2(2.67f, 0.96f), new Vector2(0.92f, -2.20f), new Vector2(-3.30f, 2.24f)},
            new List<Vector2> {new Vector2(-3.36f, 1.75f), new Vector2(1.13f, -0.72f), new Vector2(-2.78f, -0.98f)},
            new List<Vector2> {new Vector2(-3.45f, 0.24f), new Vector2(0.06f, 0.86f), new Vector2(3.15f, 1.69f)},
            new List<Vector2> {new Vector2(1.10f, -1.96f), new Vector2(-4.41f, -1.84f), new Vector2(4.01f, -0.85f)},
            new List<Vector2> {new Vector2(-3.47f, -2.39f), new Vector2(0.38f, -1.96f), new Vector2(-3.22f, -1.49f)},
            new List<Vector2> {new Vector2(0.07f, -2.42f), new Vector2(-1.67f, 1.42f), new Vector2(4.19f, 2.09f)},
            new List<Vector2> {new Vector2(-4.21f, 1.26f), new Vector2(-5.06f, -1.75f), new Vector2(-4.62f, -1.82f)},
            new List<Vector2> {new Vector2(-4.62f, 0.06f), new Vector2(4.24f, 0.27f), new Vector2(0.44f, 1.73f)},
            new List<Vector2> {new Vector2(-3.07f, 2.30f), new Vector2(0.75f, -1.42f), new Vector2(3.59f, 0.26f)},
            new List<Vector2> {new Vector2(-2.73f, 1.69f), new Vector2(-4.72f, 1.16f), new Vector2(5.34f, 1.18f)},
            new List<Vector2> {new Vector2(4.23f, -0.59f), new Vector2(5.08f, 2.16f), new Vector2(-2.82f, 0.02f)},
            new List<Vector2> {new Vector2(3.82f, 0.19f), new Vector2(4.81f, 0.01f), new Vector2(1.21f, -0.55f)},
            new List<Vector2> {new Vector2(-3.26f, 1.33f), new Vector2(-2.27f, 0.98f), new Vector2(-4.94f, 2.38f)},
            new List<Vector2> {new Vector2(2.81f, 1.68f), new Vector2(4.34f, 1.01f), new Vector2(-1.44f, -0.62f)},
            new List<Vector2> {new Vector2(-4.07f, 0.16f), new Vector2(1.29f, -1.62f), new Vector2(-0.79f, -2.43f)},
            new List<Vector2> {new Vector2(2.89f, 0.06f), new Vector2(3.87f, -1.16f), new Vector2(-0.52f, 2.16f)},
            new List<Vector2> {new Vector2(4.11f, 1.39f), new Vector2(0.00f, 1.86f), new Vector2(-3.30f, -0.03f)},
            new List<Vector2> {new Vector2(-0.72f, -0.84f), new Vector2(-2.30f, -1.78f), new Vector2(-3.50f, 0.62f)},
            new List<Vector2> {new Vector2(1.04f, -0.50f), new Vector2(0.78f, -1.18f), new Vector2(-3.53f, 2.39f)},
            new List<Vector2> {new Vector2(-1.63f, 0.51f), new Vector2(-0.53f, 1.35f), new Vector2(3.36f, 0.67f)},
            new List<Vector2> {new Vector2(-1.09f, -0.26f), new Vector2(-1.73f, -0.02f), new Vector2(0.63f, 0.70f)},
            new List<Vector2> {new Vector2(3.37f, -2.28f), new Vector2(-3.07f, -0.32f), new Vector2(1.96f, 1.32f)},
            new List<Vector2> {new Vector2(5.00f, -1.40f), new Vector2(-2.08f, 1.43f), new Vector2(3.25f, -2.09f)},
            new List<Vector2> {new Vector2(1.98f, 1.26f), new Vector2(-5.11f, 1.75f), new Vector2(-1.58f, 0.11f)},
            new List<Vector2> {new Vector2(-4.34f, 0.72f), new Vector2(-1.73f, -0.15f), new Vector2(-0.90f, -1.95f)},
            new List<Vector2> {new Vector2(-1.23f, -0.88f), new Vector2(-3.75f, 1.17f), new Vector2(3.70f, 1.91f)},
            new List<Vector2> {new Vector2(3.04f, -1.34f), new Vector2(-3.49f, -1.71f), new Vector2(-4.76f, 1.94f)},
            new List<Vector2> {new Vector2(2.54f, 1.03f), new Vector2(-0.12f, -0.26f), new Vector2(-5.24f, -1.48f)},
            new List<Vector2> {new Vector2(2.07f, 2.26f), new Vector2(-0.01f, 0.47f), new Vector2(4.98f, 0.46f)},
            new List<Vector2> {new Vector2(1.13f, -0.38f), new Vector2(-4.53f, -1.40f), new Vector2(-1.05f, 0.05f)},
            new List<Vector2> {new Vector2(-5.36f, 0.10f), new Vector2(5.37f, 2.36f), new Vector2(3.99f, 1.47f)},
            new List<Vector2> {new Vector2(5.40f, -0.13f), new Vector2(-2.32f, -0.75f), new Vector2(-2.19f, 2.44f)},
            new List<Vector2> {new Vector2(-4.65f, 0.12f), new Vector2(1.20f, 0.47f), new Vector2(-0.77f, 0.58f)},
            new List<Vector2> {new Vector2(0.80f, 0.81f), new Vector2(-1.79f, 2.45f), new Vector2(1.81f, -0.48f)},
            new List<Vector2> {new Vector2(-1.50f, 1.75f), new Vector2(-1.98f, -0.24f), new Vector2(0.37f, 0.67f)},
            new List<Vector2> {new Vector2(1.66f, 0.91f), new Vector2(1.30f, -2.50f), new Vector2(-1.41f, -1.13f)},
            new List<Vector2> {new Vector2(0.11f, 0.70f), new Vector2(-3.01f, 0.81f), new Vector2(-5.00f, 2.46f)},
            new List<Vector2> {new Vector2(1.05f, -0.66f), new Vector2(1.94f, 1.39f), new Vector2(-4.41f, -0.97f)},
            new List<Vector2> {new Vector2(3.05f, -2.44f), new Vector2(-4.86f, -0.95f), new Vector2(-1.26f, -2.37f)},
            new List<Vector2> {new Vector2(-5.19f, 0.53f), new Vector2(-0.80f, 2.48f), new Vector2(1.40f, -0.13f)},
            new List<Vector2> {new Vector2(5.02f, -0.42f), new Vector2(-3.89f, -0.10f), new Vector2(1.37f, -0.26f)},
            new List<Vector2> {new Vector2(-1.91f, -1.87f), new Vector2(2.25f, -2.36f), new Vector2(-4.49f, -0.03f)},
            new List<Vector2> {new Vector2(1.53f, -2.34f), new Vector2(-2.66f, 2.49f), new Vector2(-1.49f, -1.41f)},
            new List<Vector2> {new Vector2(0.86f, -0.81f), new Vector2(0.55f, 0.11f), new Vector2(5.13f, 0.97f)},
            new List<Vector2> {new Vector2(-0.15f, -2.30f), new Vector2(4.95f, -2.48f), new Vector2(-0.41f, -1.91f)},
            new List<Vector2> {new Vector2(-3.36f, -0.81f), new Vector2(-5.50f, -0.32f), new Vector2(2.11f, 2.38f)},
            new List<Vector2> {new Vector2(-0.36f, 0.63f), new Vector2(-3.30f, -2.02f), new Vector2(4.62f, -2.15f)},
            new List<Vector2> {new Vector2(1.63f, 2.19f), new Vector2(0.56f, 1.62f), new Vector2(-4.47f, -0.69f)},
            new List<Vector2> {new Vector2(-3.75f, -2.35f), new Vector2(-2.45f, -1.61f), new Vector2(-5.03f, 1.37f)},
            new List<Vector2> {new Vector2(-3.28f, 2.26f), new Vector2(3.51f, 1.63f), new Vector2(2.81f, 2.41f)},
            new List<Vector2> {new Vector2(-2.09f, 0.56f), new Vector2(0.55f, -2.42f), new Vector2(-4.90f, -1.55f)},
            new List<Vector2> {new Vector2(-5.39f, -1.00f), new Vector2(-3.67f, -2.27f), new Vector2(2.91f, 1.68f)},
            new List<Vector2> {new Vector2(-1.02f, -1.81f), new Vector2(-0.32f, 0.83f), new Vector2(2.11f, 1.25f)},
            new List<Vector2> {new Vector2(-2.94f, 1.49f), new Vector2(0.34f, -0.90f), new Vector2(-4.49f, -1.15f)},
            new List<Vector2> {new Vector2(5.42f, 1.57f), new Vector2(4.57f, 1.68f), new Vector2(-3.49f, 1.50f)},
            new List<Vector2> {new Vector2(-1.50f, -1.58f), new Vector2(2.61f, 1.95f), new Vector2(-1.15f, -0.01f)},
            new List<Vector2> {new Vector2(0.73f, 2.22f), new Vector2(2.82f, 0.25f), new Vector2(-4.94f, 1.17f)},
            new List<Vector2> {new Vector2(2.09f, 0.27f), new Vector2(-2.93f, -0.49f), new Vector2(-1.33f, 0.63f)},
            new List<Vector2> {new Vector2(-4.98f, 0.46f), new Vector2(-1.79f, 1.71f), new Vector2(-0.26f, -1.57f)},
            new List<Vector2> {new Vector2(-2.71f, 0.85f), new Vector2(-3.85f, -1.34f), new Vector2(1.44f, -2.20f)},
            new List<Vector2> {new Vector2(5.22f, 0.47f), new Vector2(0.56f, -0.49f), new Vector2(1.26f, 0.69f)},
            new List<Vector2> {new Vector2(-3.91f, 1.67f), new Vector2(-1.62f, -1.40f), new Vector2(3.57f, 2.17f)},
            new List<Vector2> {new Vector2(-4.52f, 1.46f), new Vector2(-3.67f, -0.24f), new Vector2(3.34f, -2.26f)},
            new List<Vector2> {new Vector2(0.39f, 2.13f), new Vector2(-1.28f, 1.20f), new Vector2(-2.21f, -1.85f)},
            new List<Vector2> {new Vector2(-1.96f, 2.31f), new Vector2(-1.92f, 0.38f), new Vector2(1.91f, 0.39f)},
            new List<Vector2> {new Vector2(4.34f, -1.13f), new Vector2(1.34f, 2.14f), new Vector2(-2.40f, 1.99f)},
            new List<Vector2> {new Vector2(-1.57f, 0.10f), new Vector2(0.97f, 0.86f), new Vector2(1.65f, -1.80f)},
            new List<Vector2> {new Vector2(-3.22f, -1.19f), new Vector2(4.06f, 1.27f), new Vector2(0.48f, 0.52f)},
            new List<Vector2> {new Vector2(2.19f, -0.77f), new Vector2(3.43f, -0.68f), new Vector2(3.97f, 2.29f)},
            new List<Vector2> {new Vector2(-4.73f, 1.51f), new Vector2(-2.76f, -1.11f), new Vector2(2.59f, 0.20f)},
            new List<Vector2> {new Vector2(-0.01f, 1.00f), new Vector2(-4.96f, 1.91f), new Vector2(-4.58f, 1.16f)},
            new List<Vector2> {new Vector2(0.95f, 2.32f), new Vector2(-0.98f, 2.12f), new Vector2(-5.49f, 2.33f)},
            new List<Vector2> {new Vector2(3.46f, -1.82f), new Vector2(1.66f, 0.22f), new Vector2(-4.79f, 2.29f)},
            new List<Vector2> {new Vector2(-3.38f, -2.48f), new Vector2(-3.46f, -2.36f), new Vector2(-1.47f, -1.85f)},
            new List<Vector2> {new Vector2(-2.66f, -1.22f), new Vector2(-4.22f, -1.67f), new Vector2(1.23f, -1.10f)},
            new List<Vector2> {new Vector2(4.64f, -0.29f), new Vector2(-4.22f, 0.27f), new Vector2(5.16f, -0.68f)},
            new List<Vector2> {new Vector2(-3.29f, 0.46f), new Vector2(-5.12f, -1.59f), new Vector2(-2.03f, 0.39f)},
            new List<Vector2> {new Vector2(3.30f, -0.49f), new Vector2(-0.85f, -0.19f), new Vector2(-4.94f, -0.57f)},
            new List<Vector2> {new Vector2(5.05f, 0.05f), new Vector2(3.84f, -1.65f), new Vector2(-1.08f, 1.20f)},
            new List<Vector2> {new Vector2(-2.96f, -0.92f), new Vector2(4.09f, 1.69f), new Vector2(3.06f, -2.35f)},
            new List<Vector2> {new Vector2(1.90f, -0.40f), new Vector2(-2.92f, -2.30f), new Vector2(0.49f, 1.72f)},
            new List<Vector2> {new Vector2(4.74f, -2.33f), new Vector2(3.67f, 1.09f), new Vector2(2.81f, 2.40f)},
            new List<Vector2> {new Vector2(-5.27f, 0.94f), new Vector2(2.97f, 0.09f), new Vector2(4.64f, -1.34f)},
            new List<Vector2> {new Vector2(-3.21f, -0.33f), new Vector2(-1.71f, 2.36f), new Vector2(1.05f, 1.37f)},
            new List<Vector2> {new Vector2(-4.15f, -0.10f), new Vector2(4.92f, -2.49f), new Vector2(-4.63f, -1.94f)},
            new List<Vector2> {new Vector2(-5.24f, 1.28f), new Vector2(1.28f, -0.19f), new Vector2(3.85f, -0.36f)},
            new List<Vector2> {new Vector2(5.14f, -1.61f), new Vector2(-2.88f, 0.59f), new Vector2(4.74f, 1.85f)},
            new List<Vector2> {new Vector2(-2.74f, 0.79f), new Vector2(4.21f, 0.28f), new Vector2(-5.09f, -1.56f)},
            new List<Vector2> {new Vector2(-4.15f, 0.29f), new Vector2(-0.15f, 1.68f), new Vector2(-4.92f, -1.46f)},
            new List<Vector2> {new Vector2(-3.66f, 1.88f), new Vector2(-4.47f, -1.74f), new Vector2(-2.51f, 1.99f)},
            new List<Vector2> {new Vector2(-4.85f, -0.17f), new Vector2(0.11f, -1.11f), new Vector2(2.65f, -0.60f)},
            new List<Vector2> {new Vector2(0.21f, 2.01f), new Vector2(2.49f, -1.97f), new Vector2(-1.65f, 0.77f)},
            new List<Vector2> {new Vector2(-1.66f, -0.56f), new Vector2(-4.26f, -1.84f), new Vector2(1.00f, -1.52f)},
            new List<Vector2> {new Vector2(5.15f, -1.96f), new Vector2(2.51f, 1.97f), new Vector2(-4.50f, -0.58f)},
            new List<Vector2> {new Vector2(-1.07f, 2.48f), new Vector2(2.96f, 1.15f), new Vector2(-3.42f, 0.86f)},
            new List<Vector2> {new Vector2(-0.63f, -0.94f), new Vector2(-1.46f, 2.38f), new Vector2(-3.55f, -0.50f)},
            new List<Vector2> {new Vector2(1.77f, 0.26f), new Vector2(2.71f, -1.88f), new Vector2(4.47f, 1.37f)},
            new List<Vector2> {new Vector2(-5.40f, 1.49f), new Vector2(-1.49f, -0.27f), new Vector2(4.32f, -1.17f)},
            new List<Vector2> {new Vector2(-2.10f, -2.00f), new Vector2(1.50f, -1.97f), new Vector2(4.97f, 1.83f)},
            new List<Vector2> {new Vector2(3.06f, 0.27f), new Vector2(-2.92f, 1.19f), new Vector2(-5.26f, -1.95f)},
            new List<Vector2> {new Vector2(-2.30f, 0.50f), new Vector2(-1.18f, -0.24f), new Vector2(-2.19f, 1.74f)},
            new List<Vector2> {new Vector2(1.77f, -1.77f), new Vector2(-4.95f, -1.07f), new Vector2(2.78f, -0.67f)},
            new List<Vector2> {new Vector2(0.08f, 0.09f), new Vector2(-2.68f, 0.81f), new Vector2(2.59f, 0.00f)},
            new List<Vector2> {new Vector2(3.85f, 1.74f), new Vector2(-5.03f, 2.09f), new Vector2(-1.26f, -0.18f)},
            new List<Vector2> {new Vector2(-3.56f, 0.07f), new Vector2(0.97f, -0.33f), new Vector2(-2.72f, -1.80f)},
            new List<Vector2> {new Vector2(-5.38f, 1.79f), new Vector2(2.77f, 1.32f), new Vector2(-1.21f, -1.45f)},
            new List<Vector2> {new Vector2(4.39f, 0.92f), new Vector2(-2.59f, 0.90f), new Vector2(2.26f, 2.28f)},
            new List<Vector2> {new Vector2(4.56f, -0.68f), new Vector2(-0.02f, -2.18f), new Vector2(-2.09f, 0.04f)},
            new List<Vector2> {new Vector2(-5.40f, 1.48f), new Vector2(-1.50f, 0.64f), new Vector2(-0.98f, 2.34f)},
            new List<Vector2> {new Vector2(-4.05f, -0.13f), new Vector2(3.13f, -1.04f), new Vector2(2.03f, -0.11f)},
            new List<Vector2> {new Vector2(1.92f, 0.84f), new Vector2(-3.60f, -2.29f), new Vector2(-4.52f, 1.59f)},
            new List<Vector2> {new Vector2(-0.87f, 1.54f), new Vector2(-5.13f, -1.10f), new Vector2(3.62f, -1.47f)},
            new List<Vector2> {new Vector2(4.60f, 0.17f), new Vector2(4.68f, -1.62f), new Vector2(-3.04f, -0.64f)},
            new List<Vector2> {new Vector2(-4.56f, 0.21f), new Vector2(-2.64f, 1.74f), new Vector2(0.99f, 1.93f)},
            new List<Vector2> {new Vector2(3.19f, 0.59f), new Vector2(2.15f, 2.22f), new Vector2(-4.07f, 0.34f)},
            new List<Vector2> {new Vector2(1.39f, 1.36f), new Vector2(-2.35f, 0.63f), new Vector2(-1.39f, -1.96f)},
            new List<Vector2> {new Vector2(-2.62f, 0.48f), new Vector2(-1.43f, 0.64f), new Vector2(0.97f, -1.98f)},
            new List<Vector2> {new Vector2(-3.55f, 0.07f), new Vector2(3.37f, 0.28f), new Vector2(-2.62f, -0.56f)},
            new List<Vector2> {new Vector2(-4.99f, 2.35f), new Vector2(-3.41f, -2.04f), new Vector2(2.49f, -2.08f)},
            new List<Vector2> {new Vector2(3.94f, 2.42f), new Vector2(2.02f, -1.48f), new Vector2(-4.39f, 1.87f)},
            new List<Vector2> {new Vector2(-2.17f, 2.47f), new Vector2(-5.08f, 0.57f), new Vector2(-4.44f, 1.20f)},
            new List<Vector2> {new Vector2(5.10f, -0.19f), new Vector2(-4.27f, -1.19f), new Vector2(4.53f, 1.17f)},
            new List<Vector2> {new Vector2(-5.36f, -1.33f), new Vector2(-0.94f, -0.83f), new Vector2(-1.11f, -0.54f)},
            new List<Vector2> {new Vector2(-4.78f, -0.98f), new Vector2(2.39f, -2.48f), new Vector2(-2.07f, 0.07f)},
            new List<Vector2> {new Vector2(4.82f, -1.83f), new Vector2(-1.03f, -1.42f), new Vector2(-1.80f, -0.11f)},
            new List<Vector2> {new Vector2(2.72f, 2.35f), new Vector2(-4.50f, -0.81f), new Vector2(-2.79f, 0.18f)},
            new List<Vector2> {new Vector2(1.83f, 2.48f), new Vector2(0.05f, 1.14f), new Vector2(4.34f, -0.55f)},
            new List<Vector2> {new Vector2(-2.13f, 2.42f), new Vector2(-4.88f, -0.17f), new Vector2(1.93f, 1.21f)},
            new List<Vector2> {new Vector2(-1.15f, -1.96f), new Vector2(3.19f, -1.61f), new Vector2(-0.47f, -1.13f)},
            new List<Vector2> {new Vector2(3.25f, 0.50f), new Vector2(-0.10f, -1.87f), new Vector2(5.39f, -1.02f)},
            new List<Vector2> {new Vector2(2.99f, -0.17f), new Vector2(-3.07f, 0.29f), new Vector2(5.30f, 1.92f)},
            new List<Vector2> {new Vector2(0.68f, -0.36f), new Vector2(-1.35f, -0.28f), new Vector2(-1.40f, -0.65f)},
            new List<Vector2> {new Vector2(-2.81f, 0.35f), new Vector2(3.72f, 1.57f), new Vector2(-3.29f, -1.99f)},
            new List<Vector2> {new Vector2(5.29f, 1.04f), new Vector2(-3.90f, -0.53f), new Vector2(-3.62f, -1.99f)},
            new List<Vector2> {new Vector2(-1.62f, -0.37f), new Vector2(5.43f, -0.61f), new Vector2(2.64f, 0.61f)},
            new List<Vector2> {new Vector2(-0.56f, -1.53f), new Vector2(4.30f, -2.30f), new Vector2(-1.22f, -2.36f)},
            new List<Vector2> {new Vector2(-2.11f, 1.72f), new Vector2(3.28f, 2.27f), new Vector2(2.47f, 0.09f)},
            new List<Vector2> {new Vector2(-0.83f, -1.40f), new Vector2(-0.65f, -1.74f), new Vector2(1.76f, -0.07f)},
            new List<Vector2> {new Vector2(5.44f, 0.94f), new Vector2(-3.94f, 1.89f), new Vector2(0.73f, 1.32f)},
            new List<Vector2> {new Vector2(1.39f, 1.63f), new Vector2(1.58f, 0.75f), new Vector2(-3.64f, -2.31f)},
            new List<Vector2> {new Vector2(-2.65f, 0.17f), new Vector2(0.54f, -2.35f), new Vector2(3.65f, 2.32f)},
            new List<Vector2> {new Vector2(-1.10f, 1.18f), new Vector2(-0.22f, 0.08f), new Vector2(2.31f, -0.11f)},
            new List<Vector2> {new Vector2(-3.70f, -1.52f), new Vector2(-4.12f, -1.73f), new Vector2(0.25f, 0.89f)},
            new List<Vector2> {new Vector2(2.95f, 0.99f), new Vector2(-4.93f, -0.27f), new Vector2(-2.12f, -2.18f)},
            new List<Vector2> {new Vector2(-4.24f, 0.92f), new Vector2(4.18f, 2.11f), new Vector2(-2.23f, -1.04f)},
            new List<Vector2> {new Vector2(2.12f, 0.14f), new Vector2(-3.71f, 0.88f), new Vector2(-4.48f, -0.22f)},
            new List<Vector2> {new Vector2(-1.84f, 1.88f), new Vector2(3.49f, 0.07f), new Vector2(-2.99f, 1.59f)},
            new List<Vector2> {new Vector2(0.76f, -2.36f), new Vector2(0.63f, -1.44f), new Vector2(2.41f, 1.41f)},
            new List<Vector2> {new Vector2(-1.69f, -1.44f), new Vector2(2.60f, 2.22f), new Vector2(0.50f, 0.75f)},
            new List<Vector2> {new Vector2(1.66f, -2.42f), new Vector2(3.73f, 1.78f), new Vector2(2.71f, 1.46f)},
            new List<Vector2> {new Vector2(-2.54f, -2.23f), new Vector2(4.17f, 0.43f), new Vector2(5.34f, -0.01f)},
            new List<Vector2> {new Vector2(2.17f, 2.06f), new Vector2(5.10f, 1.73f), new Vector2(-4.43f, 1.31f)},
            new List<Vector2> {new Vector2(0.96f, 1.30f), new Vector2(-5.46f, -2.01f), new Vector2(-4.96f, -2.02f)},
            new List<Vector2> {new Vector2(-1.81f, 1.03f), new Vector2(-1.09f, 0.33f), new Vector2(4.07f, 1.18f)},
            new List<Vector2> {new Vector2(4.67f, -0.09f), new Vector2(-1.65f, 2.24f), new Vector2(4.17f, 0.94f)},
            new List<Vector2> {new Vector2(-4.36f, 0.12f), new Vector2(-3.56f, 1.46f), new Vector2(-5.07f, -0.11f)},
            new List<Vector2> {new Vector2(-4.04f, 0.72f), new Vector2(0.46f, -0.68f), new Vector2(-1.35f, 2.23f)},
            new List<Vector2> {new Vector2(4.42f, -0.76f), new Vector2(1.13f, -2.01f), new Vector2(-5.15f, 0.48f)},
            new List<Vector2> {new Vector2(0.35f, -1.52f), new Vector2(1.83f, -0.48f), new Vector2(-3.59f, -2.27f)},
            new List<Vector2> {new Vector2(-5.09f, 0.05f), new Vector2(0.16f, -1.63f), new Vector2(4.00f, 1.11f)},
            new List<Vector2> {new Vector2(4.58f, -0.08f), new Vector2(4.49f, 1.72f), new Vector2(-0.55f, -1.47f)},
            new List<Vector2> {new Vector2(-5.16f, -1.51f), new Vector2(-4.44f, -2.29f), new Vector2(-3.14f, -2.42f)},
            new List<Vector2> {new Vector2(4.03f, 1.74f), new Vector2(1.81f, -0.61f), new Vector2(-5.41f, 1.88f)},
            new List<Vector2> {new Vector2(-2.38f, -0.18f), new Vector2(-2.93f, 1.10f), new Vector2(-3.51f, -2.27f)},
            new List<Vector2> {new Vector2(-4.29f, 2.04f), new Vector2(4.27f, 1.63f), new Vector2(-4.25f, 1.89f)},
            new List<Vector2> {new Vector2(3.22f, 1.71f), new Vector2(-5.24f, -1.03f), new Vector2(-5.27f, 0.52f)},
            new List<Vector2> {new Vector2(1.03f, 0.05f), new Vector2(2.28f, 0.85f), new Vector2(1.58f, -0.79f)},
            new List<Vector2> {new Vector2(4.36f, -1.18f), new Vector2(4.10f, -0.25f), new Vector2(-4.27f, -1.77f)},
            new List<Vector2> {new Vector2(5.09f, -0.95f), new Vector2(0.73f, -0.18f), new Vector2(3.01f, 0.30f)},
            new List<Vector2> {new Vector2(3.09f, 1.18f), new Vector2(0.14f, 1.54f), new Vector2(0.16f, -1.21f)},
            new List<Vector2> {new Vector2(-5.15f, 1.88f), new Vector2(-3.32f, -0.93f), new Vector2(3.81f, -0.81f)},
            new List<Vector2> {new Vector2(4.17f, 1.22f), new Vector2(3.19f, -1.12f), new Vector2(4.02f, -0.79f)},
            new List<Vector2> {new Vector2(-0.64f, -2.25f), new Vector2(5.34f, -1.28f), new Vector2(1.22f, -2.37f)},
            new List<Vector2> {new Vector2(0.07f, 1.50f), new Vector2(5.36f, -0.91f), new Vector2(3.17f, 0.85f)},
            new List<Vector2> {new Vector2(4.37f, 0.50f), new Vector2(-5.34f, 1.63f), new Vector2(4.11f, -0.94f)},
            new List<Vector2> {new Vector2(1.83f, -0.89f), new Vector2(-1.09f, -1.39f), new Vector2(2.33f, 0.18f)},
            new List<Vector2> {new Vector2(4.47f, 2.28f), new Vector2(-3.59f, 1.53f), new Vector2(5.18f, -0.60f)},
            new List<Vector2> {new Vector2(-2.10f, -1.37f), new Vector2(3.33f, 1.11f), new Vector2(2.94f, 0.35f)},
            new List<Vector2> {new Vector2(5.44f, 1.04f), new Vector2(0.97f, -1.25f), new Vector2(1.49f, 2.02f)},
            new List<Vector2> {new Vector2(-3.76f, -1.48f), new Vector2(4.68f, -0.77f), new Vector2(2.44f, 2.13f)},
            new List<Vector2> {new Vector2(-4.33f, -0.46f), new Vector2(3.55f, 1.37f), new Vector2(-1.90f, -2.16f)},
            new List<Vector2> {new Vector2(5.43f, -0.58f), new Vector2(0.12f, -0.88f), new Vector2(3.59f, -2.42f)},
            new List<Vector2> {new Vector2(-4.64f, 0.91f), new Vector2(2.41f, -1.55f), new Vector2(-2.32f, 0.61f)},
            new List<Vector2> {new Vector2(-4.40f, -2.46f), new Vector2(3.42f, 1.84f), new Vector2(3.79f, 0.95f)},
            new List<Vector2> {new Vector2(1.12f, 0.16f), new Vector2(2.22f, 1.05f), new Vector2(5.41f, 0.92f)},
            new List<Vector2> {new Vector2(-0.46f, -0.65f), new Vector2(3.15f, 2.16f), new Vector2(1.99f, -1.50f)},
            new List<Vector2> {new Vector2(4.14f, -1.89f), new Vector2(-0.83f, 0.87f), new Vector2(4.82f, 1.13f)},
            new List<Vector2> {new Vector2(5.34f, -0.75f), new Vector2(-1.02f, -0.90f), new Vector2(0.32f, 1.48f)},
            new List<Vector2> {new Vector2(-4.53f, 0.95f), new Vector2(-4.71f, -1.09f), new Vector2(-5.40f, 1.37f)},
            new List<Vector2> {new Vector2(-1.37f, 0.10f), new Vector2(-4.01f, 0.24f), new Vector2(1.59f, -0.32f)},
            new List<Vector2> {new Vector2(0.40f, -1.99f), new Vector2(2.48f, 2.17f), new Vector2(-2.73f, -1.09f)},
            new List<Vector2> {new Vector2(3.63f, 1.93f), new Vector2(-2.30f, 2.11f), new Vector2(0.79f, -1.66f)},
            new List<Vector2> {new Vector2(3.58f, -0.72f), new Vector2(-3.91f, 1.76f), new Vector2(3.29f, -0.06f)},
            new List<Vector2> {new Vector2(-1.97f, -1.39f), new Vector2(3.96f, 0.17f), new Vector2(-2.00f, 1.62f)},
            new List<Vector2> {new Vector2(-3.34f, -0.49f), new Vector2(5.35f, 1.82f), new Vector2(2.23f, 0.44f)},
            new List<Vector2> {new Vector2(-1.09f, -1.57f), new Vector2(-4.84f, 2.15f), new Vector2(-1.43f, -2.07f)},
            new List<Vector2> {new Vector2(2.45f, 0.54f), new Vector2(0.33f, -2.09f), new Vector2(-3.13f, -1.97f)},
            new List<Vector2> {new Vector2(-2.03f, 1.54f), new Vector2(-2.05f, -1.31f), new Vector2(3.13f, -1.33f)},
            new List<Vector2> {new Vector2(-2.93f, -1.66f), new Vector2(1.51f, -1.53f), new Vector2(-3.11f, -0.41f)},
            new List<Vector2> {new Vector2(0.35f, 0.91f), new Vector2(-0.52f, 0.58f), new Vector2(1.38f, -1.98f)},
            new List<Vector2> {new Vector2(4.38f, 0.64f), new Vector2(-1.56f, -0.70f), new Vector2(-4.07f, -1.89f)},
            new List<Vector2> {new Vector2(2.40f, 2.21f), new Vector2(-1.40f, -0.38f), new Vector2(5.04f, -0.74f)},
            new List<Vector2> {new Vector2(-0.20f, 0.11f), new Vector2(-3.75f, 0.75f), new Vector2(2.37f, 0.49f)},
            new List<Vector2> {new Vector2(-1.09f, -0.87f), new Vector2(-0.50f, -2.42f), new Vector2(2.04f, -0.02f)},
            new List<Vector2> {new Vector2(-3.36f, -0.27f), new Vector2(2.54f, -2.17f), new Vector2(-2.69f, 2.20f)},
            new List<Vector2> {new Vector2(2.45f, 1.76f), new Vector2(-0.34f, -1.13f), new Vector2(0.87f, 2.44f)},
            new List<Vector2> {new Vector2(2.80f, 2.09f), new Vector2(-5.00f, 0.19f), new Vector2(4.58f, -1.17f)},
            new List<Vector2> {new Vector2(1.55f, 2.47f), new Vector2(-4.62f, -0.19f), new Vector2(3.55f, -1.82f)},
            new List<Vector2> {new Vector2(-3.84f, 0.83f), new Vector2(3.42f, -1.21f), new Vector2(0.15f, -0.50f)},
            new List<Vector2> {new Vector2(1.93f, -1.79f), new Vector2(-2.76f, -0.19f), new Vector2(4.93f, -1.14f)},
            new List<Vector2> {new Vector2(-2.23f, -0.41f), new Vector2(-1.89f, 1.95f), new Vector2(-0.35f, 0.72f)},
            new List<Vector2> {new Vector2(3.50f, 0.60f), new Vector2(5.39f, 0.26f), new Vector2(3.07f, 1.26f)},
            new List<Vector2> {new Vector2(-5.09f, 0.26f), new Vector2(3.83f, 1.70f), new Vector2(-5.38f, 0.03f)},
            new List<Vector2> {new Vector2(5.29f, -0.88f), new Vector2(0.86f, -0.87f), new Vector2(-0.01f, 1.99f)},
            new List<Vector2> {new Vector2(4.08f, -0.79f), new Vector2(5.12f, -1.83f), new Vector2(-4.23f, 2.48f)},
            new List<Vector2> {new Vector2(1.30f, -1.04f), new Vector2(4.73f, -2.12f), new Vector2(3.70f, 0.43f)},
            new List<Vector2> {new Vector2(3.16f, 0.47f), new Vector2(1.19f, 0.35f), new Vector2(4.483f, -2.30f)},
            new List<Vector2> {new Vector2(-3.51f, 1.61f), new Vector2(-1.49f, -0.10f), new Vector2(-1.37f, 2.15f)},
            new List<Vector2> {new Vector2(-0.55f, -0.58f), new Vector2(5.29f, 1.08f), new Vector2(-5.49f, 1.52f)},
            new List<Vector2> {new Vector2(4.55f, -0.94f), new Vector2(3.39f, 2.22f), new Vector2(-4.99f, -0.33f)},
            new List<Vector2> {new Vector2(-3.93f, 2.32f), new Vector2(0.21f, 2.08f), new Vector2(-2.98f, 0.26f)},
            new List<Vector2> {new Vector2(1.81f, 1.17f), new Vector2(4.72f, 0.40f), new Vector2(4.54f, -1.44f)},
            new List<Vector2> {new Vector2(-1.93f, 1.27f), new Vector2(1.09f, 1.50f), new Vector2(2.26f, -0.09f)},
            new List<Vector2> {new Vector2(-1.05f, -1.68f), new Vector2(5.41f, -1.59f), new Vector2(3.60f, 1.59f)},
            new List<Vector2> {new Vector2(2.70f, 0.45f), new Vector2(3.52f, 1.28f), new Vector2(-3.02f, -2.43f)},
            new List<Vector2> {new Vector2(-0.33f, 1.28f), new Vector2(-1.73f, 1.54f), new Vector2(0.46f, 2.48f)},
            new List<Vector2> {new Vector2(4.91f, 1.30f), new Vector2(-0.93f, -0.21f), new Vector2(-2.63f, -1.71f)},
            new List<Vector2> {new Vector2(1.42f, 2.20f), new Vector2(-3.06f, -2.08f), new Vector2(-4.57f, 0.35f)},
            new List<Vector2> {new Vector2(-5.12f, -1.85f), new Vector2(5.29f, 1.66f), new Vector2(-3.09f, -1.87f)},
            new List<Vector2> {new Vector2(-2.63f, -1.34f), new Vector2(-4.24f, -1.79f), new Vector2(4.43f, 1.84f)},
            new List<Vector2> {new Vector2(-4.32f, 2.08f), new Vector2(-4.16f, 1.38f), new Vector2(-4.03f, -1.22f)},
            new List<Vector2> {new Vector2(1.78f, -0.10f), new Vector2(1.67f, 0.41f), new Vector2(-4.96f, -2.43f)},
            new List<Vector2> {new Vector2(3.56f, 1.65f), new Vector2(-0.35f, 0.83f), new Vector2(5.50f, -1.96f)},
            new List<Vector2> {new Vector2(1.86f, 2.02f), new Vector2(-4.17f, 1.05f), new Vector2(-4.32f, -1.47f)},
            new List<Vector2> {new Vector2(-0.92f, -1.58f), new Vector2(2.06f, -1.52f), new Vector2(-0.42f, 1.56f)},
            new List<Vector2> {new Vector2(-3.94f, 1.49f), new Vector2(-3.98f, -1.47f), new Vector2(-1.37f, -1.94f)},
            new List<Vector2> {new Vector2(0.37f, -1.65f), new Vector2(1.74f, -2.40f), new Vector2(3.89f, -1.18f)},
            new List<Vector2> {new Vector2(-2.55f, 1.73f), new Vector2(3.61f, -0.32f), new Vector2(-5.05f, 0.27f)},
            new List<Vector2> {new Vector2(-1.19f, 2.50f), new Vector2(-0.55f, 1.90f), new Vector2(0.52f, 1.69f)},
            new List<Vector2> {new Vector2(0.25f, 2.43f), new Vector2(1.82f, -0.17f), new Vector2(3.43f, 0.57f)},
            new List<Vector2> {new Vector2(0.59f, -1.78f), new Vector2(-2.59f, -1.12f), new Vector2(-1.98f, -2.15f)},
            new List<Vector2> {new Vector2(-2.41f, -0.05f), new Vector2(-2.89f, 2.17f), new Vector2(-3.79f, 1.06f)},
            new List<Vector2> {new Vector2(-5.38f, 0.43f), new Vector2(-5.40f, 0.56f), new Vector2(0.79f, 0.10f)},
            new List<Vector2> {new Vector2(2.65f, -1.44f), new Vector2(-2.55f, 1.92f), new Vector2(-1.34f, 0.99f)},
            new List<Vector2> {new Vector2(1.37f, 2.01f), new Vector2(-2.73f, 0.52f), new Vector2(3.84f, -2.04f)},
            new List<Vector2> {new Vector2(2.43f, -1.00f), new Vector2(1.67f, 0.22f), new Vector2(1.41f, -1.39f)},
            new List<Vector2> {new Vector2(4.97f, -1.44f), new Vector2(4.08f, 1.27f), new Vector2(-1.55f, 2.10f)},
            new List<Vector2> {new Vector2(-3.08f, -1.29f), new Vector2(5.42f, -0.98f), new Vector2(-2.29f, -1.33f)},
            new List<Vector2> {new Vector2(-5.38f, -1.76f), new Vector2(2.88f, -1.93f), new Vector2(3.16f, 1.60f)},
            new List<Vector2> {new Vector2(-1.18f, -2.47f), new Vector2(4.33f, 0.31f), new Vector2(-2.65f, 1.28f)},
            new List<Vector2> {new Vector2(5.35f, 0.77f), new Vector2(-1.32f, -0.96f), new Vector2(4.68f, -2.13f)},
            new List<Vector2> {new Vector2(-2.22f, 1.59f), new Vector2(-1.64f, -0.75f), new Vector2(5.38f, 0.17f)},
            new List<Vector2> {new Vector2(-1.64f, -1.52f), new Vector2(-4.80f, -2.36f), new Vector2(0.93f, 1.60f)},
            new List<Vector2> {new Vector2(0.12f, -0.85f), new Vector2(-3.76f, -0.97f), new Vector2(1.29f, -2.06f)},
            // total 269
        };
        positionIndex = 0;
        currentStartPos = new List<Vector2>(predefinedPositions[positionIndex]);
        PrepareNewEpisode();
    }

    private void FixedUpdate()
    {
        // Finish the episode if all the balls stopped
        if (state == GameState.Simulation && balls[0].isStopped() && balls[1].isStopped() && balls[2].isStopped())
        {
            state = GameState.End;
            EndGame();
        }

        if (clientConnected && stream != null && stream.DataAvailable)
        {
            byte[] buffer = new byte[1024];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            string message = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            HandleTCPMessage(message);
        }

        // Result UI
        if (state != GameState.End)
        {
            string touchedBall = "";
            if (touchedBallType.Contains(BallType.Red))
            {
                touchedBall = "Red";
            }
            if (touchedBallType.Contains(BallType.Yellow))
            {
                if (string.IsNullOrEmpty(touchedBall))
                    touchedBall = "Yellow";
                else
                    touchedBall = "Red, Yellow";
            }
            // Check failure or not
            int score = IsFailure() ? 0 : 1;
            float reward = 0f;

            bool red = touchedBallType.Contains(BallType.Red);
            bool yellow = touchedBallType.Contains(BallType.Yellow);

            if (red && yellow) {
                if (cushionCount >= 3) {
                    reward += 10f;
                }
                else {
                    reward += 4f;
                    reward += Math.Min(cushionCount, 2);
                }
            }
            else if (red || yellow) {
                if (cushionCount >= 3) {
                    reward += 5f;
                }
                else reward += 2f;
                reward += Math.Min(cushionCount, 3);
            }
            else {
                if (cushionCount < 3) reward -= 10f;
                else reward -= 1f;
            }

            string detail = $"White: ({currentStartPos[0].x:F2},{currentStartPos[0].y:F2})\n" +
                $"Red: ({currentStartPos[1].x:F2},{currentStartPos[1].y:F2})\n" +
                $"Yellow: ({currentStartPos[2].x:F2},{currentStartPos[2].y:F2})\n" +
                $"Current Angle: {currentAngle:F2}\n" +
                $"Current Force: {currentForce:F2}\n" +
                $"Touched Balls: ({touchedBall})\n" +
                $"Cushion Count: {cushionCount}\n" +
                $"Score: {score}\n" +
                $"Reward: {reward}";
            detailText.text = detail;
        }
    }

    public void OnSpeedChanged(float value)
    {
        float realval = (value + 1) / 2f;
        Time.timeScale = realval;
        Time.fixedDeltaTime = baseFixedDeltaTime / realval;
        timeSliderText.text = $"Time Scale: {realval:F1}";
    }

    public void BallTouchesWall()
    {
        if (state == GameState.Simulation) {
            cushionCount += 1;
            // The order of cushions
            eventLog.Add(EventType.Cushion);
        }
    }

    public void BallTouchesBall(BallType type)
    {
        if (type == BallType.White || state != GameState.Simulation) return;

        if (!touchedBallType.Contains(type))
            touchedBallType.Add(type);

        // The order of collisions
        if (type == BallType.Red) eventLog.Add(EventType.BallRed);
        if (type == BallType.Yellow) eventLog.Add(EventType.BallYellow);

        // Condition of early finishing(If main ball hit the two different balls before hit the wall 3 times)
        int cushions = 0;
        bool redHit = false, yellowHit = false;

        foreach (var e in eventLog)
        {
            if (e == EventType.Cushion)
            {
                cushions++;
            }
            else if (e == EventType.BallRed)
            {
                if (cushions < 3) redHit = true;
            }
            else if (e == EventType.BallYellow)
            {
                if (cushions < 3) yellowHit = true;
            }
        }

        if ((redHit && yellowHit) && state != GameState.End)
        {
            state = GameState.End;
            // Early finishing
            EndGame();
        }
    }

    void EndGame()
    {
        // Check failure or not
        int score = IsFailure() ? 0 : 1;
        float reward = 0f;

        bool red = touchedBallType.Contains(BallType.Red);
        bool yellow = touchedBallType.Contains(BallType.Yellow);

        if (red && yellow) {
            if (cushionCount >= 3) {
                reward += 10f;
            }
            else {
                reward += 4f;
                reward += Math.Min(cushionCount, 2);
            }
        }
        else if (red || yellow) {
            if (cushionCount >= 3) {
                reward += 5f;
            }
            else reward += 2f;
            reward += Math.Min(cushionCount, 3);
        }
        else {
            if (cushionCount < 3) reward -= 10f;
            else reward -= 1f;
        }
        
        LogData(score, reward);
        if (clientConnected && stream != null)
        {
            bool redTouched = touchedBallType.Contains(BallType.Red);
            bool yellowTouched = touchedBallType.Contains(BallType.Yellow);
            string result = $"RES:{score},{reward:F2},{redTouched},{yellowTouched},{cushionCount}\n";

            byte[] resultBytes = Encoding.UTF8.GetBytes(result);
            stream.Write(resultBytes, 0, resultBytes.Length);
            stream.Flush();

            Debug.Log($"[TCP → Python] Sent RES after client connect: {result.Trim()}");
        }
        episode++;

        bool success = (score == 1);
        if (success) positionIndex++;
        PrepareNewEpisode();
    }

    bool IsFailure()
    {

        // Failure condition 1: hit with two other balls before hit the wall for 3 times
        // Failure condition 2: If it didn't hit two other balls before it stoppeds

        int cushions = 0;
        int ballsHitBefore3Cushion = 0;
        bool redHit = false, yellowHit = false;

        foreach (var e in eventLog)
        {
            if (e == EventType.Cushion)
            {
                cushions++;
            }
            else if (e == EventType.BallRed)
            {
                if (cushions < 3) ballsHitBefore3Cushion++;
                redHit = true;
            }
            else if (e == EventType.BallYellow)
            {
                if (cushions < 3) ballsHitBefore3Cushion++;
                yellowHit = true;
            }
            // Hit two other balls before hit the ball for 3 times → Failure
            if (ballsHitBefore3Cushion >= 2) return true;
        }
        // If it didn't hit two other balls → Failure
        if (!(redHit && yellowHit)) return true;

        return false;
    }

    void PrepareNewEpisode()
    {
        touchedBallType.Clear();
        cushionCount = 0;
        eventLog.Clear();
        
        if (positionIndex < predefinedPositions.Count)
        {
            currentStartPos = new List<Vector2>(predefinedPositions[positionIndex]);
        }
        else if (positionIndex >= predefinedPositions.Count) {
            positionIndex = 0;
            List<Vector2> basePos = predefinedPositions[positionIndex];
            currentStartPos = new List<Vector2>(basePos);
            float offset = 0.2f;
            int randomBallIndex = UnityEngine.Random.Range(0, 3);

            Vector2 original = currentStartPos[randomBallIndex];
            float newX = original.x + offset;
            float newY = original.y + offset;
            if (newX < -5.5f || newX > 5.5f) newX = original.x - 2*offset;
            if (newY < -2.5f || newY > 2.5f) newY = original.y - 2*offset;

            Vector2 newPos = new Vector2(newX, newY);
            currentStartPos[randomBallIndex] = newPos;

            predefinedPositions[positionIndex] = new List<Vector2>(currentStartPos);
        }
        
        SendCurrentPositions();
        state = GameState.WaitingForAction;
        
        for (int i = 0; i < balls.Count; i++)
        {
            balls[i].transform.position = currentStartPos[i];
            var rb = balls[i].GetComponent<Rigidbody2D>();
            rb.linearVelocity = Vector2.zero;
            rb.angularVelocity = 0f;
            rb.Sleep();

            rb.simulated=true;
        }

    }

    void ExecuteEpisode(float angle, float force)
    {
        if (state != GameState.WaitingForAction)
        {
            Debug.LogWarning("[ExecuteEpisode] Not ready for action.");
            return;
        }

        episodeText.text = $"Episode {episode}";

        if (currentForce >= 8f && currentForce <= 30f && currentAngle >= 0f && currentAngle <= 360f)
        {
            balls[0].Shoot(currentForce, currentAngle);
            Debug.Log($"[Shoot] angle={currentAngle}, force={currentForce}");
            state = GameState.Simulation;
        }
        else
        {
            Debug.LogWarning($"[Shoot BLOCKED] invalid force/angle: angle={currentAngle}, force={currentForce}");
            // Waiting after sending the location
            PrepareNewEpisode();
        }
    }

    void InitializePositions()
    {
        // Set initial position so that it can prevent the balls from overlapping
        currentStartPos = new List<Vector2>();
        for (int i = 0; i < 3; i++)
        {
            int trycount = 0;
            Vector2 p = Vector2.zero;
            bool failed = true;

            while (failed && trycount < 10000)
            {
                p = new Vector2(UnityEngine.Random.Range(-5.5f, 5.5f), UnityEngine.Random.Range(-2.5f, 2.5f));
                failed = false;
                foreach (Vector2 v in currentStartPos)
                {
                    if ((v - p).magnitude < 0.1f)
                    {
                        failed = true;
                        break;
                    }
                }
                trycount ++;
            }

            if (failed)
            {
                detailText.text = "Couldn't find new ball position.";
                return;
            }

            currentStartPos.Add(p);
            balls[i].transform.position = p;
        }
        // Save the initial position
        lastTriedStartPos = new List<Vector2>(currentStartPos);

        for (int i = 0; i < balls.Count; i++)
        {
            var rb = balls[i].GetComponent<Rigidbody2D>();
            rb.linearVelocity = Vector2.zero;
            rb.angularVelocity = 0f;
            rb.Sleep(); 
            rb.simulated = false;
        }
    }

    void SendCurrentPositions()
    {
        if (clientConnected && stream != null && currentStartPos != null && currentStartPos.Count == 3)
        {
            Vector2 w = currentStartPos[0];
            Vector2 r = currentStartPos[1];
            Vector2 y = currentStartPos[2];
            string msg = $"POS:{w.x},{w.y},{r.x},{r.y},{y.x},{y.y}\n";
            byte[] msgBytes = Encoding.UTF8.GetBytes(msg);
            stream.Write(msgBytes, 0, msgBytes.Length);
            stream.Flush();
            Debug.Log($"[TCP → Python] Sent POS: {msg.Trim()}");
        }
    }


    public void LogData(int score, float reward)
    {
        string timeStamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        bool redTouched = touchedBallType.Contains(BallType.Red);
        bool yellowTouched = touchedBallType.Contains(BallType.Yellow);

        string newLine = $"{episode},{currentStartPos[0].x},{currentStartPos[0].y},{currentStartPos[1].x},{currentStartPos[1].y},{currentStartPos[2].x},{currentStartPos[2].y},{currentAngle},{currentForce},{redTouched},{yellowTouched},{cushionCount},{score},{reward},{timeStamp}";
        using (StreamWriter writer = new StreamWriter(filePath, true, Encoding.UTF8))
        {
            writer.WriteLine(newLine);
        }
    }

    public void ResetEpisode(Vector2[] positions)
    {
        // reposition the balls when they initialize the episode
        touchedBallType.Clear();
        cushionCount = 0;
        currentStartPos = new List<Vector2>(positions);

        for (int i = 0; i < balls.Count; i++)
        {
            balls[i].transform.position = currentStartPos[i];
        }
    }

    public void SetAction(float angle, float force)
    {
        currentAngle = angle;
        currentForce = force;
        Debug.Log($"[SetAction] angle={angle}, force={force}");
    }

    void StartTCPServer()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, 9999);
            tcpListener.Start();
            Debug.Log("[TCP] Server started on port 9999");

            listenerThread = new Thread(new ThreadStart(ListenForClient));
            listenerThread.IsBackground = true;
            listenerThread.Start();
        }
        catch (Exception ex)
        {
            Debug.LogError($"[TCP] Failed to start server: {ex.Message}");
        }
    }

    void ListenForClient()
    {
        try
        {
            connectedClient = tcpListener.AcceptTcpClient();
            stream = connectedClient.GetStream();
            clientConnected = true;
            Debug.Log("[TCP] Client connected.");

            SendCurrentPositions();
        }
        catch (Exception e)
        {
            Debug.LogError($"[TCP] Listen Error: {e.Message}");
        }
    }

    void HandleTCPMessage(string msg)
    {
        try
        {
            string[] parts = msg.Split(',');
            if (parts.Length == 2)
            {
                float angle = float.Parse(parts[0]);
                float force = float.Parse(parts[1]);

                SetAction(angle, force);
                // WaitingForAction when the state is GameState
                if (state == GameState.WaitingForAction)
                {
                    ExecuteEpisode(angle, force);
                }
                else
                {
                    Debug.LogWarning($"[TCP] Ignored action: GameState={state}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[TCP] Message parse error: {e.Message}");
        }
    }

    void OnApplicationQuit()
    {
        if (tcpListener != null)
        {
            tcpListener.Stop();
            Debug.Log("[TCP] Listener stopped on quit.");
        }

        if (listenerThread != null && listenerThread.IsAlive)
        {
            listenerThread.Abort();
        }
    }
}