using System;
using UnityEngine;

public class BoundingCircle : MonoBehaviour
{
    public LineRenderer lineRenderer;
    public Color color;
    public float width;
    public int numSegments;

    void Start()
    {
        lineRenderer.startColor = color;
        lineRenderer.endColor = color;
        lineRenderer.startWidth = width;
        lineRenderer.endWidth = width;
    }

    public void Set(bool active, Vector3 position, float radius)
    {
        gameObject.SetActive(active);
        lineRenderer.positionCount = numSegments;
        for (var i = 0; i < numSegments; i++)
        {
            var theta = 2 * Mathf.PI * i / (float)numSegments;
            lineRenderer.SetPosition(i, position + radius * new Vector3(Mathf.Cos(theta), Mathf.Sin(theta), 0));
        }
        lineRenderer.loop = true;
    }
}
