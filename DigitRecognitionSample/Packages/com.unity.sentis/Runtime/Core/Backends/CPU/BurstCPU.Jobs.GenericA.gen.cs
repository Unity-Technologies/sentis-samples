// This is auto-generated -- do not modify directly

using UnityEngine;
using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace Unity.Sentis
{
public partial class CPUBackend
{


[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct InstanceNormalizationTailJob : IJobParallelFor, IJobResourceDeclarationXSBWO
{
    
    public float epsilon;
    public int channels;
    public int spatialDims;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource S { get; set; } float* Sptr => (float*)S.ptr;
    public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
    public ReadOnlyMemResource W { get; set; } float* Wptr => (float*)W.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int bc = threadIdx / (spatialDims);
        int c = bc % channels;

        float mean = Wptr[bc * 2 + 0];
        float variance = Wptr[bc * 2 + 1];

        float scale = Sptr[c];
        float bias = Bptr[c];

        // normalization factor
        float invNormFactor = 1 / sqrt(variance + epsilon);

        float v = Xptr[threadIdx];
        v = v * invNormFactor - mean * invNormFactor;
        v = v * scale + bias;

        Optr[threadIdx] = v;
    }
}

}
}
