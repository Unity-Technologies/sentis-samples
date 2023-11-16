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
internal unsafe struct TileJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public int rank;
    public fixed int shapeO[8];
    public fixed int stridesO[8];
    public fixed int shapeX[8];
    public fixed int stridesX[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int indexX = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            indexX += (((threadIdx / stridesO[(TensorShape.maxRank-1) - axis]) % shapeO[(TensorShape.maxRank-1) - axis]) % shapeX[(TensorShape.maxRank-1) - axis]) * stridesX[(TensorShape.maxRank-1) - axis];
        }

        Optr[threadIdx] = Xptr[indexX];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct GatherElementsJob : IJobParallelFor, IJobResourceDeclarationXBO
{
    
    public int axisDim;
    public int axisDimX;
    public int endLength;
    public int endLengthX;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int end = threadIdx % endLength;
        int start = threadIdx / (axisDim * endLength);

        int index = (int)Bptr[threadIdx];
        index = index < 0 ? axisDimX + index : index;

        Optr[threadIdx] = Xptr[start * endLengthX * axisDimX + index * endLengthX + end];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ScatterElementsJob : IJobParallelFor, IJobResourceDeclarationXBO
{
    
    public int axisDim;
    public int axisDimIndices;
    public int endLength;
    public int reduction;
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int end = threadIdx % endLength;
        int start = threadIdx / (endLength * axisDimIndices);

        int index = (int)Bptr[threadIdx];
        index = index < 0 ? axisDim + index : index;

        if (reduction == 0)
            Optr[start * axisDim * endLength + index * endLength + end] = Xptr[threadIdx];
        else if (reduction == 1)
            Optr[start * axisDim * endLength + index * endLength + end] += Xptr[threadIdx];
        else if (reduction == 2)
            Optr[start * axisDim * endLength + index * endLength + end] *= Xptr[threadIdx];
    }
}



[BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
internal unsafe struct ExpandJob : IJobParallelFor, IJobResourceDeclarationXO
{
    
    public int rank;
    public fixed int shapeO[8];
    public fixed int stridesO[8];
    public fixed int shapeX[8];
    public fixed int stridesX[8];
    public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
    public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

    public void Execute(int threadIdx)
    {
        int indexX = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            indexX += (((threadIdx / stridesO[(TensorShape.maxRank - 1) - axis]) % shapeO[(TensorShape.maxRank - 1) - axis]) % shapeX[(TensorShape.maxRank - 1) - axis]) * stridesX[(TensorShape.maxRank - 1) - axis];
        }

        Optr[threadIdx] = Xptr[indexX];
    }
}

}
}
