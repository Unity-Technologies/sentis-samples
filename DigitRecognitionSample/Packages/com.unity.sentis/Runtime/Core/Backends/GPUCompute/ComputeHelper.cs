using System;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Profiling;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis {

static class ComputeHelper
{
    public static int IDivC(int v, int div)
    {
        return (v + div - 1) / div;
    }

    /// <summary>
    /// Policy to decide the (log2) width of a texture given the number of pixels required
    /// </summary>
    public static int CalculateWidthShift(int numPixels)
    {
        // aim for square, this seems to be faster than using very thin textures (width/height 1 or 2)
        // variant of base 2 bit counting from https://stackoverflow.com/questions/8970101/whats-the-quickest-way-to-compute-log2-of-an-integer-in-c
        var n = numPixels;
        var shift = 0;

        if (n > 0x7FFF)
        {
            n >>= 16;
            shift = 0x8;
        }

        if (n > 0x7F)
        {
            n >>= 8;
            shift |= 0x4;
        }

        if (n > 0x7)
        {
            n >>= 4;
            shift |= 0x2;
        }

        if (n > 0x1)
        {
            shift |= 0x1;
        }

        return shift;
    }

    public static void SetTexture(this ComputeFunc fn, int nameID, Texture tex)
    {
        fn.shader.SetTexture(fn.kernelIndex, nameID, tex);
    }

    static readonly int[] s_unrolledDispatchArgs = new int[2];
    public static void UnrolledDispatch(this ComputeFunc fn, int numThread)
    {
        if (numThread == 0)
            return;

        int threadPerTG = (int)(fn.threadGroupSizeX * fn.threadGroupSizeY * fn.threadGroupSizeZ);
        int neededTG = ComputeHelper.IDivC(numThread, threadPerTG);
        int threadGroupZ = 1;
        int threadGroupY = ComputeHelper.IDivC(neededTG, (int)ComputeFunc.SafeDispatchLimit);
        int threadGroupX = ComputeHelper.IDivC(neededTG, threadGroupY);
        s_unrolledDispatchArgs[0] = threadGroupX * threadPerTG;
        s_unrolledDispatchArgs[1] = numThread;
        fn.shader.SetInts(k_ID_unrolledDispatchArgs, s_unrolledDispatchArgs);

        int workItemsZ = (int)(threadGroupZ * fn.threadGroupSizeZ);
        int workItemsY = (int)(threadGroupY * fn.threadGroupSizeY);
        int workItemsX = (int)(threadGroupX * fn.threadGroupSizeX);
        fn.Dispatch(workItemsX, workItemsY, workItemsZ);
    }

    public static void SetFloat(this ComputeFunc fn, int nameID, float data)
    {
        fn.shader.SetFloat(nameID, data);
    }

    public static void SetInt(this ComputeFunc fn, int nameID, int data)
    {
        fn.shader.SetInt(nameID, data);
    }

    public static void SetInts(this ComputeFunc fn, int nameID, int[] data)
    {
        fn.shader.SetInts(nameID, data);
    }

    public static void SetVector(this ComputeFunc fn, int nameID, Vector4 data)
    {
        fn.shader.SetVector(nameID, data);
    }

    public static void SetBool(this ComputeFunc fn, int nameID, bool data)
    {
        fn.shader.SetBool(nameID, data);
    }

    public static void EnableKeyword(this ComputeFunc fn, string keyword)
    {
        fn.shader.EnableKeyword(keyword);
    }

    public static void DisableKeyword(this ComputeFunc fn, string keyword)
    {
        fn.shader.DisableKeyword(keyword);
    }

    public static unsafe void SetTensorShapeStrides(this ComputeFunc fn, int shapeNameID, int strideNameID, TensorShape shape)
    {
        int* pShape = stackalloc int[TensorShape.maxRank];
        int* pStrides = stackalloc int[TensorShape.maxRank];
        OpsUtils.PinTensorShapeStrides(shape, pShape, pStrides);

        fn.SetInt8(shapeNameID, pShape);
        fn.SetInt8(strideNameID, pStrides);
    }

    // for setting uint4 and int4 values, no padding required
    static readonly int[] s_scratchPadInt4 = new int[4];

    public static void SetInt4(this ComputeFunc fn, int nameID, Span<int> ptr)
    {
        for (int i = 0; i < ptr.Length && i < 4; i++)
            s_scratchPadInt4[i] = ptr[i];

        fn.shader.SetInts(nameID, s_scratchPadInt4);
    }

    //See https://docs.unity3d.com/2020.2/Documentation/ScriptReference/ComputeShader.SetInts.html
    //SetInts API need CPU side to be padded
    static readonly int[] s_scratchPadInt16 = new int[16*4];
    static readonly int[] s_scratchPadInt8 = new int[8*4];
    static readonly int[] s_scratchPadInt6 = new int[6*4];

    public static void SetInt16(this ComputeFunc fn, int nameID, ReadOnlySpan<int> ptr)
    {
        Logger.AssertIsTrue(ptr.Length <= 16, "cannot pin array > 16, got {0}", ptr.Length);
        for (int i = 0; i < ptr.Length; i++)
            s_scratchPadInt16[4 * i] = ptr[i];

        fn.shader.SetInts(nameID, s_scratchPadInt16);
    }

    public static unsafe void SetInt8(this ComputeFunc fn, int nameID, int* ptr)
    {
        fixed (int* dst = &s_scratchPadInt8[0])
        {
            UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), 8);
        }
        fn.shader.SetInts(nameID, s_scratchPadInt8);
    }

    public static unsafe void SetInt8(this ComputeFunc fn, int nameID, int[] ptr)
    {
        Logger.AssertIsTrue(ptr.Length <= 8, "cannot pin array > 8, got {0}", ptr.Length);
        for (int i = 0; i < ptr.Length; i++)
            s_scratchPadInt8[4 * i] = ptr[i];

        fn.shader.SetInts(nameID, s_scratchPadInt8);
    }

    public static unsafe void SetInt6(this ComputeFunc fn, int nameID, int* ptr)
    {
        fixed (int* dst = &s_scratchPadInt6[0])
        {
            UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), 6);
        }
        fn.shader.SetInts(nameID, s_scratchPadInt6);
    }

    public static void SetTensorAsBuffer(this ComputeFunc fn, int bufferID, ComputeTensorData tensorData)
    {
        fn.shader.SetBuffer(fn.kernelIndex, bufferID, tensorData.buffer);
    }

    public static void Dispatch(this ComputeFunc fn, int workItemsX, int workItemsY, int workItemsZ)
    {
        Profiler.BeginSample(fn.kernelName);
        var x = IDivC(workItemsX, (int)fn.threadGroupSizeX);
        var y = IDivC(workItemsY, (int)fn.threadGroupSizeY);
        var z = IDivC(workItemsZ, (int)fn.threadGroupSizeZ);

        // some GFX APIs / GPU hw/drivers have limitation of 65535 per dimension
        if (x > ComputeFunc.SafeDispatchLimit || y > ComputeFunc.SafeDispatchLimit || z > ComputeFunc.SafeDispatchLimit)
            D.LogWarning($"Exceeded safe compute dispatch group count limit per dimension [{x}, {y}, {z}] for {fn.kernelName}");

        fn.shader.Dispatch(fn.kernelIndex, x, y, z);

        Profiler.EndSample();
    }

    public static void ScheduleXSBWO(this ComputeFunc fn, ComputeTensorData X, ComputeTensorData S, ComputeTensorData B, ComputeTensorData W, ComputeTensorData O, int numThread)
    {
        fn.SetTensorAsBuffer(k_ID_Xptr, X);
        fn.SetTensorAsBuffer(k_ID_Sptr, S);
        fn.SetTensorAsBuffer(k_ID_Bptr, B);
        fn.SetTensorAsBuffer(k_ID_Wptr, W);
        fn.SetTensorAsBuffer(k_ID_Optr, O);
        fn.UnrolledDispatch(numThread);
    }

    public static void ScheduleXSBO(this ComputeFunc fn, ComputeTensorData X, ComputeTensorData S, ComputeTensorData B, ComputeTensorData O, int numThread)
    {
        fn.SetTensorAsBuffer(k_ID_Xptr, X);
        fn.SetTensorAsBuffer(k_ID_Sptr, S);
        fn.SetTensorAsBuffer(k_ID_Bptr, B);
        fn.SetTensorAsBuffer(k_ID_Optr, O);
        fn.UnrolledDispatch(numThread);
    }

    public static void ScheduleXBO(this ComputeFunc fn, ComputeTensorData X, ComputeTensorData B, ComputeTensorData O, int numThread)
    {
        fn.SetTensorAsBuffer(k_ID_Xptr, X);
        fn.SetTensorAsBuffer(k_ID_Bptr, B);
        fn.SetTensorAsBuffer(k_ID_Optr, O);
        fn.UnrolledDispatch(numThread);
    }

    public static void ScheduleXO(this ComputeFunc fn, ComputeTensorData X, ComputeTensorData O, int numThread)
    {
        fn.SetTensorAsBuffer(k_ID_Xptr, X);
        fn.SetTensorAsBuffer(k_ID_Optr, O);
        fn.UnrolledDispatch(numThread);
    }

    public static void ScheduleO(this ComputeFunc fn, ComputeTensorData O, int numThread)
    {
        fn.SetTensorAsBuffer(k_ID_Optr, O);
        fn.UnrolledDispatch(numThread);
    }
}
} // namespace Unity.Sentis
