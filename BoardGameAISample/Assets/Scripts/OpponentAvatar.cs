using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OpponentAvatar : MonoBehaviour
{
    public GameObject target;
    public enum Mode { Back, Forward};
    public Mode mode = Mode.Back;

    Vector3 m_StartingPosition;
    float m_Speed = 2f;

    void Start()
    {
        m_StartingPosition = transform.position;
    }

    void Update()
    {
        if (mode == Mode.Forward)
            transform.position = Vector3.MoveTowards(transform.position, target.transform.position, Time.deltaTime * m_Speed);
        else if (mode == Mode.Back)
            transform.position = Vector3.MoveTowards(transform.position, m_StartingPosition, Time.deltaTime * m_Speed);
    }
}
