using UnityEngine;

// Creates a behavior of the opponent spirit masks bobbing up and down.
public class SpiritMove : MonoBehaviour
{
    Vector3 m_StartPos;
    public float m_Movement = 0.1f;
    float m_Speed = 1.5f;

    void Start()
    {
        m_StartPos = transform.localPosition;
    }

    void Update()
    {
        transform.localPosition = m_StartPos + Mathf.Cos(Time.time * m_Speed) * m_Movement * Vector3.up;
    }
}
