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
    internal unsafe struct AddFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x + y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct SubFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x - y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MulFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x * y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct DivFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x / y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MinFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (math.min(x, y));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MaxFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (math.max(x, y));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct PowFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (math.pow(x, y));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct PowFloatIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (math.pow(x, y));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MeanFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public float alpha;
        public float beta;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x * alpha + y * beta);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct GreaterFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x > y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct GreaterOrEqualFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x >= y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct LessFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x < y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct LessOrEqualFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x <= y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct EqualFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (x == y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct OrJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x | y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct AndJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x & y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct XorJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x ^ y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct PReluJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (math.select(x * y, x, x >= 0.0f));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct AddIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x + y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct SubIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x - y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MulIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x * y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct DivIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x / y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MinIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (math.min(x, y));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MaxIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (math.max(x, y));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct GreaterIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x > y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct GreaterOrEqualIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x >= y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct LessIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x < y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct LessOrEqualIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x <= y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct EqualIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x == y ? 1 : 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ModIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = ((x % y + y) % y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct FModIntJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                int* Xp = Xptr + offsetX;
                int* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] int* Xp, uint stepX, [NoAlias] int* Bp, uint stepB, [NoAlias] int* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                int x = Xp[0]; Xp += stepX;
                int y = Bp[0]; Bp += stepB;
                Op[i] = (x % y);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct FModFloatJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public BroadcastHelperXBO broadcast;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            float* Op = Optr + i;

            int offsetX = broadcast.iteratorX.InitialOffset(i, countersX);
            int offsetB = broadcast.iteratorB.InitialOffset(i, countersB);

            while (count > 0)
            {
                int spanRemainingX = broadcast.iteratorX.SpanSize() - countersX[0];
                int spanRemainingB = broadcast.iteratorB.SpanSize() - countersB[0];
                int spanRemaining = math.min(spanRemainingX, spanRemainingB);
                int spanCount = math.min(count, spanRemaining);

                float* Xp = Xptr + offsetX;
                float* Bp = Bptr + offsetB;

                if (broadcast.iteratorX.IsScalarBroadcast())
                    ProcessSpan(Xp, 0, Bp, 1, Op, (uint)spanCount);
                else if (broadcast.iteratorB.IsScalarBroadcast())
                    ProcessSpan(Xp, 1, Bp, 0, Op, (uint)spanCount);
                else
                    ProcessSpan(Xp, 1, Bp, 1, Op, (uint)spanCount);

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                {
                    offsetX = broadcast.iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
                    offsetB = broadcast.iteratorB.AdvanceOffset(offsetB, spanCount, countersB);
                }
            }
        }

        private void ProcessSpan([NoAlias] float* Xp, uint stepX, [NoAlias] float* Bp, uint stepB, [NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
            {
                float x = Xp[0]; Xp += stepX;
                float y = Bp[0]; Bp += stepB;
                Op[i] = (math.fmod(x, y));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ScalarMadJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public float s;
        public float b;

        public void Execute(int i, int count)
        {
            float* Xp = Xptr + i;
            float* Op = Optr + i;

            for (; count > 0; count--)
            {
                float x = Xp[0];
                Op[0] = s * x + b;
                Xp++;
                Op++;
            }
        }
    }
}
}
