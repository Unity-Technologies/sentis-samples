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
internal unsafe struct EinsumOneJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public int sumRank;
    public int sumSize;
    public int outRank;
    public fixed int sumLengths[8];
    public fixed int sumStrides[8];
    public fixed int sumStridesA[8];
    public fixed int outLengths[8];
    public fixed int outStrides[8];
    public fixed int outStridesA[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int outIndex = threadIdx;

        int outOffsetA = 0;
        for (int i = 8 - outRank; i < 8; i++)
        {
            int outSubIndex = (outIndex / outStrides[i]) % outLengths[i];
            outOffsetA += outStridesA[i] * outSubIndex;
        }

        float sum = 0;

        for (int sumIndex = 0; sumIndex < sumSize; sumIndex++)
        {
            int sumOffsetA = 0;

            for (int i = 8 - sumRank; i < 8; i++)
            {
                int sumSubIndex = (sumIndex / sumStrides[i]) % sumLengths[i];
                sumOffsetA += sumStridesA[i] * sumSubIndex;
            }

            sum += Xptr[outOffsetA + sumOffsetA];
        }

        Optr[outIndex] = sum;
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct EinsumTwoJob : IJobParallelFor, IJobResourceDeclarationXBO
{
    
    public int sumRank;
    public int sumSize;
    public int outRank;
    public fixed int sumLengths[8];
    public fixed int sumStrides[8];
    public fixed int sumStridesA[8];
    public fixed int sumStridesB[8];
    public fixed int outLengths[8];
    public fixed int outStrides[8];
    public fixed int outStridesA[8];
    public fixed int outStridesB[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int outIndex = threadIdx;

        int outOffsetA = 0;
        int outOffsetB = 0;
        for (int i = 8 - outRank; i < 8; i++)
        {
            int outSubIndex = (outIndex / outStrides[i]) % outLengths[i];
            outOffsetA += outStridesA[i] * outSubIndex;
            outOffsetB += outStridesB[i] * outSubIndex;
        }

        float sum = 0;

        for (int sumIndex = 0; sumIndex < sumSize; sumIndex++)
        {
            int sumOffsetA = 0;
            int sumOffsetB = 0;

            for (int i = 8 - sumRank; i < 8; i++)
            {
                int sumSubIndex = (sumIndex / sumStrides[i]) % sumLengths[i];
                sumOffsetA += sumStridesA[i] * sumSubIndex;
                sumOffsetB += sumStridesB[i] * sumSubIndex;
            }

            sum += Xptr[outOffsetA + sumOffsetA] * Bptr[outOffsetB + sumOffsetB];
        }

        Optr[outIndex] = sum;
    }
}

}
}
