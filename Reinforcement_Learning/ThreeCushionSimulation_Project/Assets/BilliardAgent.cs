using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class BilliardAgent : Agent
{
    public override void OnEpisodeBegin()
    {
        // 실패 → 이전 위치에서 다시 시작
        MainManager.Instance.ResetEpisode(MainManager.Instance.currentStartPos.ToArray());
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // 공 3개의 위치를 관측값으로 사용
        foreach (Vector2 pos in MainManager.Instance.currentStartPos)
        {
            sensor.AddObservation(pos.x);
            sensor.AddObservation(pos.y);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float angle = Mathf.Clamp(actions.ContinuousActions[0], 0f, 360f);
        float force = Mathf.Clamp(actions.ContinuousActions[1], 3f, 25f);

        if (MainManager.Instance.state != GameState.Simulation)
        {
            MainManager.Instance.SetAction(angle, force);
            MainManager.Instance.balls[0].Shoot(force, angle);
            MainManager.Instance.state = GameState.Simulation;
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Random.Range(0f, 360f); // 각도 (0 ~ 1 → 0 ~ 360도)
        continuousActions[1] = Random.Range(3f, 25f); // 힘 (0 ~ 1 → 0 ~ 25)
    }
}