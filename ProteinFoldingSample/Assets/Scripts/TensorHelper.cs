using UnityEngine;
using Unity.Sentis;

public static class TensorHelper
{
    // Helper function to replace a tensor with a new one. Disposes old one to avoid memory leakage
    public static void Replace(ref TensorFloat A, TensorFloat B)
    {
        A?.Dispose();
        A = B;
    }

    // Helper function to turn a tensor into an array of Vector3
    public static Vector3[] GetVectorArray(TensorFloat F)
    {
        F.MakeReadable();
        float[] f = F.ToReadOnlyArray();
        int N = f.Length / 3;
        Vector3[] pos = new Vector3[N];
        for (int i = 0; i < N; i++)
        {
            pos[i] = new Vector3(f[i * 3], f[i * 3 + 1], f[i * 3 + 2]);
        }
        return pos;
    }
}
