using System;
using UnityEngine;

public class BoundingBox : MonoBehaviour
{
    public LineRenderer lineRenderer;
    public Color color;
    public float width;

    void Start()
    {
        lineRenderer.startColor = color;
        lineRenderer.endColor = color;
        lineRenderer.startWidth = width;
        lineRenderer.endWidth = width;
    }

    public void Set(bool active, Vector3 position, Vector2 size)
    {
        gameObject.SetActive(active);
        lineRenderer.positionCount = 4;
        lineRenderer.SetPosition(0, position + new Vector3(-0.5f * size.x, -0.5f * size.y, 0));
        lineRenderer.SetPosition(1, position + new Vector3(-0.5f * size.x, +0.5f * size.y, 0));
        lineRenderer.SetPosition(2, position + new Vector3(+0.5f * size.x, +0.5f * size.y, 0));
        lineRenderer.SetPosition(3, position + new Vector3(+0.5f * size.x, -0.5f * size.y, 0));
        lineRenderer.loop = true;
    }
}
