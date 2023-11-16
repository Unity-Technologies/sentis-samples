// This is auto-generated -- do not modify directly
using UnityEngine;
using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Avx2;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Sentis {
public partial class CPUBackend
{


    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct AbsIntJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public void Execute(int threadIdx)
        {
            int v = Xptr[threadIdx];
            Optr[threadIdx] = (math.abs(v));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct AbsFloatJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public void Execute(int threadIdx)
        {
            float v = Xptr[threadIdx];
            Optr[threadIdx] = (math.abs(v));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct NegIntJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public void Execute(int threadIdx)
        {
            int v = Xptr[threadIdx];
            Optr[threadIdx] = (-v);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct NegFloatJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public void Execute(int threadIdx)
        {
            float v = Xptr[threadIdx];
            Optr[threadIdx] = (-v);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct IsNaNJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public void Execute(int threadIdx)
        {
            float v = Xptr[threadIdx];
            Optr[threadIdx] = (math.isnan(v) ? 1 : 0);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct CastToFloatJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public void Execute(int threadIdx)
        {
            int v = Xptr[threadIdx];
            Optr[threadIdx] = ((float)v);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct CastToIntJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public void Execute(int threadIdx)
        {
            float v = Xptr[threadIdx];
            Optr[threadIdx] = ((int)v);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct SignFloatJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public void Execute(int threadIdx)
        {
            float v = Xptr[threadIdx];
            Optr[threadIdx] = (math.sign(v));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct SignIntJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public void Execute(int threadIdx)
        {
            int v = Xptr[threadIdx];
            Optr[threadIdx] = (v == 0 ? 0 : (v > 0 ? 1 : -1));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct NotJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public void Execute(int threadIdx)
        {
            int v = Xptr[threadIdx];
            Optr[threadIdx] = ((v == 0) ? 1 : 0);
        }
    }


    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RangeFloatJob : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public float start;
        public float delta;

        public void Execute(int threadIdx)
        {
            Optr[threadIdx] = start + (threadIdx * delta);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RangeIntJob : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public int start;
        public int delta;

        public void Execute(int threadIdx)
        {
            Optr[threadIdx] = start + (threadIdx * delta);
        }
    }
}
}
