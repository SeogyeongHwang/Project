using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System;
using NUnit.Framework;
using Random = UnityEngine.Random;

// Game State
public enum GameState
{
    Simulation,
    End
}

// Check Event : Cushion, Hit red or yellow balls
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
    // Check hitting event order
    public List<EventType> eventLog = new List<EventType>();
    public int cushionCount = 0;

    public GameState state;

    public List<Vector2> currentStartPos;
    private List<Vector2> lastTriedStartPos; // Backup for failure
    public float currentAngle;
    public float currentForce;
    public bool lastSuccess = false;

    private string folderPath;
    private string filePath;
    float baseFixedDeltaTime;

    void Awake()
    {
        folderPath = Application.dataPath + "/SimulationData";

        if (!Directory.Exists(folderPath))
            Directory.CreateDirectory(folderPath);

        filePath = Path.Combine(folderPath, "EpisodeLog6.csv");

        if (balls == null || balls.Count == 0)
        {
            balls = new List<Ball>(FindObjectsOfType<Ball>());
            Debug.Log($"[Awake] Load {balls.Count} balls automatically");
        }

        if (!File.Exists(filePath))
        {
            using (StreamWriter writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                writer.WriteLine("Episode,WhiteBallStartX,WhiteBallStartY,RedBallStartX,RedBallStartY,YellowBallStartX,YellowBallStartY,HitAngle,HitForce,RedTouched,YellowTouched,CushionCount,Score,Reward,TimeStamp");
            }

            InitializePositions();
        }
        // Check the latest episode number
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
        if (currentStartPos == null) InitializePositions(); // Set the balls' initial locations
        // Start new episode
        NewEpisode(true);
    }

    private void FixedUpdate()
    {
        // Finish if all the balls stopped
        if (state != GameState.End && balls[0].isStopped() && balls[1].isStopped() && balls[2].isStopped())
        {
            state = GameState.End;
            EndGame();
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
            // Check cushion order
            eventLog.Add(EventType.Cushion);
        }
    }

    public void BallTouchesBall(BallType type)
    {
        if (type == BallType.White || state != GameState.Simulation) return;

        if (!touchedBallType.Contains(type))
            touchedBallType.Add(type);

        // Check hitting event order
        if (type == BallType.Red) eventLog.Add(EventType.BallRed);
        if (type == BallType.Yellow) eventLog.Add(EventType.BallYellow);

        // Early failure conditions: Hit 2 other balls before succeeding 3 cushion
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
            EndGame();  // Finish episode
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
        if (balls.Count > 0 && balls[0] != null)
        {
            var agent = balls[0].GetComponent<BilliardAgent>();
            if (agent != null)
            {
                agent.SetReward(reward);
                agent.EndEpisode();
            }
        }
        
        if (autostartToggle.isOn || autostartToggle == null)
        {
            if (score == 1)
            {
                InitializePositions();
            }
            episode++;
            NewEpisode(score == 1);
        }
    }

    bool IsFailure()
    {

         // Failure condition 1: Hit two other balls before finish 3 cushion
        // Failure condition 2: Couldn't hit two other balls before it stopped moving

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

            if (ballsHitBefore3Cushion >= 2) return true; // Hit two balls before finish 3 cushion → failure
        }

        if (!(redHit && yellowHit)) return true; // failed to hit two other balls → failure

        return false;
    }

    void NewEpisode(bool success)
    {
        episodeText.text = $"Episode {episode}";

        touchedBallType.Clear();
        cushionCount = 0;
        eventLog.Clear();

        if (!success)
        {
            // When it failed → initialize to the failed location
            for (int i = 0; i < 3; i++)
            {               
                balls[i].transform.position = lastTriedStartPos[i];
            }
            // Update failure location
            currentStartPos = new List<Vector2>(lastTriedStartPos);
        }

        lastTriedStartPos = new List<Vector2>();
        foreach (Ball ball in balls)
        {
            lastTriedStartPos.Add(ball.transform.position);
        }

        // Set random angle/force
        GetNextAction();
        balls[0].Shoot(currentForce, currentAngle);
        state = GameState.Simulation;
    }

    void InitializePositions()
    {
        // Initialize balls' locations so that it doesn't have same locations
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
    }

    public void GetNextAction()
    {
        currentAngle = Mathf.Round(UnityEngine.Random.Range(0f, 360f) * 100f) / 100f;
        currentForce = Mathf.Round(UnityEngine.Random.Range(8f,30f) * 100f) / 100f;
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
        // Relocation if episode starts from outside
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
    }
}