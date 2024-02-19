using UnityEngine;

public class VectorHelper {  
    public static Vector3 Bezier(Vector3 A, Vector3 B, Vector3 C1, Vector3 C2, float a)
    {
        float b = 1 - a;
        return (b * b * b) * A + (3 * a * b * b) * C1 + (3 * a * a * b) * C2 + (a * a * a) * B;
    }

    public static Vector3 BezierTangent(Vector3 A, Vector3 B, Vector3 C1, Vector3 C2, float a)
    {
        float b = 1 - a;
        return (-b * b) * A + (b * b - 2 * a * b) * C1 + (2 * a * b - a * a) * C2 + (a * a) * B;
    }
}
