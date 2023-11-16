using UnityEngine;
using UnityEngine.Profiling;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using static Unity.Sentis.ShaderPropertyID;
using UnityEngine.Rendering;
using System;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis {

static class CommandBufferHelper
{
    public static void SetTexture(this CommandBuffer cb, ComputeFunc fn, int nameID, Texture tex)
    {
        cb.SetComputeTextureParam(fn.shader, fn.kernelIndex, nameID, tex);
    }

    public static void SetTexture(this CommandBuffer cb, ComputeFunc fn, int nameID, RenderTargetIdentifier tex)
    {
        cb.SetComputeTextureParam(fn.shader, fn.kernelIndex, nameID, tex);
    }

    static readonly int[] s_unrolledDispatchArgs = new int[2];
    public static void UnrolledDispatch(this CommandBuffer cb, ComputeFunc fn, int numThread)
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
        cb.SetComputeIntParams(fn.shader, k_ID_unrolledDispatchArgs, s_unrolledDispatchArgs);

        int workItemsZ = (int)(threadGroupZ * fn.threadGroupSizeZ);
        int workItemsY = (int)(threadGroupY * fn.threadGroupSizeY);
        int workItemsX = (int)(threadGroupX * fn.threadGroupSizeX);
        cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
    }

    public static void SetFloat(this CommandBuffer cb, ComputeFunc fn, int nameID, float data)
    {
        cb.SetComputeFloatParam(fn.shader, nameID, data);
    }

    public static void SetInt(this CommandBuffer cb, ComputeFunc fn, int nameID, int data)
    {
        cb.SetComputeIntParam(fn.shader, nameID, data);
    }

    public static void SetInts(this CommandBuffer cb, ComputeFunc fn, int nameID, int[] data)
    {
        cb.SetComputeIntParams(fn.shader, nameID, data);
    }

    public static void SetVector(this CommandBuffer cb, ComputeFunc fn, int nameID, Vector4 data)
    {
        cb.SetComputeVectorParam(fn.shader, nameID, data);
    }

    public static void SetBool(this CommandBuffer cb, ComputeFunc fn, int nameID, bool data)
    {
        cb.SetComputeIntParam(fn.shader, nameID, data ? 1 : 0);
    }

    public static void EnableKeyword(this CommandBuffer cb, ComputeFunc fn, string keyword)
    {
        cb.EnableKeyword(fn.shader, new LocalKeyword(fn.shader, keyword));
    }

    public static void DisableKeyword(this CommandBuffer cb, ComputeFunc fn, string keyword)
    {
        cb.DisableKeyword(fn.shader, new LocalKeyword(fn.shader, keyword));
    }

    // for setting uint4 and int4 values, no padding required
    static readonly int[] s_scratchPadInt4 = new int[4];

    public static void SetInt4(this CommandBuffer cb, ComputeFunc fn, int nameID, Span<int> ptr)
    {
        for (int i = 0; i < ptr.Length && i < 4; i++)
            s_scratchPadInt4[i] = ptr[i];

        cb.SetComputeIntParams(fn.shader, nameID, s_scratchPadInt4);
    }

    public static unsafe void SetTensorShapeStrides(this CommandBuffer cb, ComputeFunc fn, int shapeNameID, int strideNameID, TensorShape shape)
    {
        int* pShape = stackalloc int[TensorShape.maxRank];
        int* pStrides = stackalloc int[TensorShape.maxRank];
        OpsUtils.PinTensorShapeStrides(shape, pShape, pStrides);

        cb.SetInt8(fn, shapeNameID, pShape);
        cb.SetInt8(fn, strideNameID, pStrides);
    }

    //See https://docs.unity3d.com/2020.2/Documentation/ScriptReference/ComputeShader.SetInts.html
    //SetInts API need CPU side to be padded
    static readonly int[] s_scratchPadInt16 = new int[16*4];
    static readonly int[] s_scratchPadInt8 = new int[8*4];
    static readonly int[] s_scratchPadInt6 = new int[6*4];

    public static void SetInt16(this CommandBuffer cb, ComputeFunc fn, int nameID, ReadOnlySpan<int> ptr)
    {
        Logger.AssertIsTrue(ptr.Length <= 16, "cannot pin array > 16, got {0}", ptr.Length);
        for (int i = 0; i < ptr.Length; i++)
            s_scratchPadInt16[4 * i] = ptr[i];

        cb.SetComputeIntParams(fn.shader, nameID, s_scratchPadInt16);
    }

    public static void SetInt16(this CommandBuffer cb, ComputeFunc fn, int nameID, Span<int> ptr)
    {
        Logger.AssertIsTrue(ptr.Length <= 16, "cannot pin array > 16, got {0}", ptr.Length);
        for (int i = 0; i < ptr.Length; i++)
            s_scratchPadInt16[4 * i] = ptr[i];

        cb.SetComputeIntParams(fn.shader, nameID, s_scratchPadInt16);
    }

    public static void SetInt8(this CommandBuffer cb, ComputeFunc fn, int nameID, Span<int> ptr)
    {
        Logger.AssertIsTrue(ptr.Length <= 8, "cannot pin array > 8, got {0}", ptr.Length);
        for (int i = 0; i < ptr.Length; i++)
            s_scratchPadInt8[4 * i] = ptr[i];

        cb.SetComputeIntParams(fn.shader, nameID, s_scratchPadInt8);
    }

    public static unsafe void SetInt8(this CommandBuffer cb, ComputeFunc fn, int nameID, int* ptr)
    {
        fixed (int* dst = &s_scratchPadInt8[0])
        {
            UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), 8);
        }
        cb.SetComputeIntParams(fn.shader, nameID, s_scratchPadInt8);
    }

    public static unsafe void SetInt8(this CommandBuffer cb, ComputeFunc fn, int nameID, int[] ptr)
    {
        Logger.AssertIsTrue(ptr.Length <= 8, "cannot pin array > 8, got {0}", ptr.Length);
        for (int i = 0; i < ptr.Length; i++)
            s_scratchPadInt8[4 * i] = ptr[i];

        cb.SetComputeIntParams(fn.shader, nameID, s_scratchPadInt8);
    }

    public static unsafe void SetInt6(this CommandBuffer cb, ComputeFunc fn, int nameID, int* ptr)
    {
        fixed (int* dst = &s_scratchPadInt6[0])
        {
            UnsafeUtility.MemCpyStride(dst, 4 * sizeof(int), ptr, 1 * sizeof(int), sizeof(int), 6);
        }
        cb.SetComputeIntParams(fn.shader, nameID, s_scratchPadInt6);
    }

    public static void SetTensorAsBuffer(this CommandBuffer cb, ComputeFunc fn, int bufferID, ComputeTensorData tensorData)
    {
        cb.SetComputeBufferParam(fn.shader, fn.kernelIndex, bufferID, tensorData.buffer);
    }

    public static void Dispatch(this CommandBuffer cb, ComputeFunc fn, int workItemsX, int workItemsY, int workItemsZ)
    {
        Profiler.BeginSample(fn.kernelName);
        var x = ComputeHelper.IDivC(workItemsX, (int)fn.threadGroupSizeX);
        var y = ComputeHelper.IDivC(workItemsY, (int)fn.threadGroupSizeY);
        var z = ComputeHelper.IDivC(workItemsZ, (int)fn.threadGroupSizeZ);

        // some GFX APIs / GPU hw/drivers have limitation of 65535 per dimension
        if (x > ComputeFunc.SafeDispatchLimit || y > ComputeFunc.SafeDispatchLimit || z > ComputeFunc.SafeDispatchLimit)
            D.LogWarning($"Exceeded safe compute dispatch group count limit per dimension [{x}, {y}, {z}] for {fn.kernelName}");

        cb.DispatchCompute(fn.shader, fn.kernelIndex, x, y, z);

        Profiler.EndSample();
    }

    public static void ScheduleXSBWO(this CommandBuffer cb, ComputeFunc fn, ComputeTensorData X, ComputeTensorData S, ComputeTensorData B, ComputeTensorData W, ComputeTensorData O, int numThread)
    {
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, X);
        cb.SetTensorAsBuffer(fn, k_ID_Sptr, S);
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, B);
        cb.SetTensorAsBuffer(fn, k_ID_Wptr, W);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, O);
        cb.UnrolledDispatch(fn, numThread);
    }

    public static void ScheduleXSBO(this CommandBuffer cb, ComputeFunc fn, ComputeTensorData X, ComputeTensorData S, ComputeTensorData B, ComputeTensorData O, int numThread)
    {
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, X);
        cb.SetTensorAsBuffer(fn, k_ID_Sptr, S);
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, B);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, O);
        cb.UnrolledDispatch(fn, numThread);
    }

    public static void ScheduleXBO(this CommandBuffer cb, ComputeFunc fn, ComputeTensorData X, ComputeTensorData B, ComputeTensorData O, int numThread)
    {
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, X);
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, B);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, O);
        cb.UnrolledDispatch(fn, numThread);
    }

    public static void ScheduleXO(this CommandBuffer cb, ComputeFunc fn, ComputeTensorData X, ComputeTensorData O, int numThread)
    {
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, X);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, O);
        cb.UnrolledDispatch(fn, numThread);
    }

    public static void ScheduleO(this CommandBuffer cb, ComputeFunc fn, ComputeTensorData O, int numThread)
    {
        cb.SetTensorAsBuffer(fn, k_ID_Optr, O);
        cb.UnrolledDispatch(fn, numThread);
    }
}
} // namespace Unity.Sentis
