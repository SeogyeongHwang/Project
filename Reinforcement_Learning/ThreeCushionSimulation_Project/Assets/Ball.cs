using UnityEngine;

public enum BallType
{
    White,
    Red,
    Yellow
}

public class Ball : MonoBehaviour
{
    public BallType balltype;
    Rigidbody2D rb;

    private void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
    }


    public bool isStopped()
    {
        return rb.linearVelocity.magnitude < 0.03f;
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {        
        if (balltype != BallType.White) return;
        
        if (collision.gameObject.GetComponent<Ball>() != null)
        {
            MainManager.Instance.BallTouchesBall(collision.gameObject.GetComponent<Ball>().balltype);
        }
        else if (collision.gameObject.CompareTag("Wall"))
        {
            MainManager.Instance.BallTouchesWall();
        }
    }


    public void Shoot(float currentForce, float currentAngle)
    {
        float angleRad = currentAngle * Mathf.Deg2Rad;
        Vector2 direction = new Vector2(Mathf.Cos(angleRad), Mathf.Sin(angleRad)).normalized;

        rb.AddForce(direction * currentForce, ForceMode2D.Impulse);
    }
}
