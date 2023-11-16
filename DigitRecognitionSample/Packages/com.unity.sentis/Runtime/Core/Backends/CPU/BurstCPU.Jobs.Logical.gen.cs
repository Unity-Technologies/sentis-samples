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
internal unsafe struct WhereJob : IJobParallelFor, IJobResourceDeclarationXSBO
{
    
    public int rank;
    public fixed int shapeO[8];
    public fixed int stridesO[8];
    public fixed int shapeC[8];
    public fixed int stridesC[8];
    public fixed int shapeA[8];
    public fixed int stridesA[8];
    public fixed int shapeB[8];
    public fixed int stridesB[8];
    public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
    public ReadOnlyMemResource S { get; set; } float* Sptr => (float*)S.ptr;
    public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int indexC = 0; int indexA = 0; int indexB = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            indexC += (((threadIdx / stridesO[(TensorShape.maxRank - 1) - axis]) % shapeO[(TensorShape.maxRank - 1) - axis]) % shapeC[(TensorShape.maxRank - 1) - axis]) * stridesC[(TensorShape.maxRank - 1) - axis];
            indexA += (((threadIdx / stridesO[(TensorShape.maxRank - 1) - axis]) % shapeO[(TensorShape.maxRank - 1) - axis]) % shapeA[(TensorShape.maxRank - 1) - axis]) * stridesA[(TensorShape.maxRank - 1) - axis];
            indexB += (((threadIdx / stridesO[(TensorShape.maxRank - 1) - axis]) % shapeO[(TensorShape.maxRank - 1) - axis]) % shapeB[(TensorShape.maxRank - 1) - axis]) * stridesB[(TensorShape.maxRank - 1) - axis];
        }

        bool cond = (Xptr[indexC] != 0);

        Optr[threadIdx] = cond ? Sptr[indexA] : Bptr[indexB];
    }
}

}
}
