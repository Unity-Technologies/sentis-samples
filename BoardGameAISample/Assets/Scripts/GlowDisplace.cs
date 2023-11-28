using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GlowDisplace : MonoBehaviour
{
    public float displacement = 1f;
    public float glowSize = 0.02f;
    
    void LateUpdate()
    {
        Vector3 dist = Camera.main.transform.position - transform.parent.position;
        Vector3 dir = dist.normalized * displacement;
        transform.position = transform.parent.position - dir;
        transform.localScale = Vector3.one * (displacement + dist.magnitude) / dist.magnitude * (1 + glowSize);
    }
}
