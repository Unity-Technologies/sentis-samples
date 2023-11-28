using UnityEngine;

// controls a simple animation on the piece as you put it down, or when it is captured. 
public class Piece : MonoBehaviour
{
    float m_StartTime;
    Vector3 m_StartPos;
    float m_Duration = 1f;
    float m_MaxHeight = 0.25f;
    float m_Speed = 4f;

    void Start()
    {
        m_StartTime = Time.time;
        m_StartPos = transform.localPosition;
    }

    public void BeginAnimation()
    {
        m_StartTime = Time.time;
        m_MaxHeight = 0.3f;
    }

    public void BeginFlipAnimation()
    {
        m_StartTime = Time.time;
        m_MaxHeight = 0.1f;
    }

    void Update()
    {
        float T = (Time.time - m_StartTime) / m_Duration;
        float y = T >= 1 ? 0 : Mathf.Abs(Mathf.Sin((Time.time - m_StartTime) * m_Speed / (1 - T))) * (1 - T) * m_MaxHeight;
        Vector3 pos = transform.localPosition;
        transform.localPosition = new Vector3(pos.x, m_StartPos.y + y, pos.z);
    }
}
