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
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceMaxFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = float.MinValue;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = float.MinValue;
                        float accVal1 = float.MinValue;
                        float accVal2 = float.MinValue;
                        float accVal3 = float.MinValue;
                        float accVal4 = float.MinValue;
                        float accVal5 = float.MinValue;
                        float accVal6 = float.MinValue;
                        float accVal7 = float.MinValue;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = float.MinValue;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            return math.max(accVal, v);
        }

        float Finalize(float accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceMaxIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                int* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    int accVal = int.MinValue;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                int* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    int* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        int accVal0 = int.MinValue;
                        int accVal1 = int.MinValue;
                        int accVal2 = int.MinValue;
                        int accVal3 = int.MinValue;
                        int accVal4 = int.MinValue;
                        int accVal5 = int.MinValue;
                        int accVal6 = int.MinValue;
                        int accVal7 = int.MinValue;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        int accVal0 = int.MinValue;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        int Reduce(int accVal, int v)
        {
            return math.max(accVal, v);
        }

        int Finalize(int accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceMinFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = float.MaxValue;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = float.MaxValue;
                        float accVal1 = float.MaxValue;
                        float accVal2 = float.MaxValue;
                        float accVal3 = float.MaxValue;
                        float accVal4 = float.MaxValue;
                        float accVal5 = float.MaxValue;
                        float accVal6 = float.MaxValue;
                        float accVal7 = float.MaxValue;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = float.MaxValue;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            return math.min(accVal, v);
        }

        float Finalize(float accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceMinIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                int* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    int accVal = int.MaxValue;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                int* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    int* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        int accVal0 = int.MaxValue;
                        int accVal1 = int.MaxValue;
                        int accVal2 = int.MaxValue;
                        int accVal3 = int.MaxValue;
                        int accVal4 = int.MaxValue;
                        int accVal5 = int.MaxValue;
                        int accVal6 = int.MaxValue;
                        int accVal7 = int.MaxValue;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        int accVal0 = int.MaxValue;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        int Reduce(int accVal, int v)
        {
            return math.min(accVal, v);
        }

        int Finalize(int accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceSumFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            return accVal + v;
        }

        float Finalize(float accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceSumIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                int* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    int accVal = 0;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                int* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    int* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        int accVal0 = 0;
                        int accVal1 = 0;
                        int accVal2 = 0;
                        int accVal3 = 0;
                        int accVal4 = 0;
                        int accVal5 = 0;
                        int accVal6 = 0;
                        int accVal7 = 0;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        int accVal0 = 0;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        int Reduce(int accVal, int v)
        {
            return accVal + v;
        }

        int Finalize(int accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceSumSquareFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            v = v * v;
            return accVal + v;
        }

        float Finalize(float accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceSumSquareIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                int* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    int accVal = 0;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                int* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    int* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        int accVal0 = 0;
                        int accVal1 = 0;
                        int accVal2 = 0;
                        int accVal3 = 0;
                        int accVal4 = 0;
                        int accVal5 = 0;
                        int accVal6 = 0;
                        int accVal7 = 0;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        int accVal0 = 0;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        int Reduce(int accVal, int v)
        {
            v = v * v;
            return accVal + v;
        }

        int Finalize(int accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceMeanFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;
        public float normalization;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            normalization = 1.0f / reduceLength;

            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            return accVal + v;
        }

        float Finalize(float accVal)
        {
            return accVal * normalization;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceProdFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 1.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 1.0f;
                        float accVal1 = 1.0f;
                        float accVal2 = 1.0f;
                        float accVal3 = 1.0f;
                        float accVal4 = 1.0f;
                        float accVal5 = 1.0f;
                        float accVal6 = 1.0f;
                        float accVal7 = 1.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 1.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            return accVal * v;
        }

        float Finalize(float accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceProdIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                int* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    int accVal = 1;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                int* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    int* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        int accVal0 = 1;
                        int accVal1 = 1;
                        int accVal2 = 1;
                        int accVal3 = 1;
                        int accVal4 = 1;
                        int accVal5 = 1;
                        int accVal6 = 1;
                        int accVal7 = 1;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        int accVal0 = 1;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        int Reduce(int accVal, int v)
        {
            return accVal * v;
        }

        int Finalize(int accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceL1FloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            v = math.abs(v);
            return accVal + v;
        }

        float Finalize(float accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceL1IntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                int* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    int accVal = 0;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                int* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    int* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        int accVal0 = 0;
                        int accVal1 = 0;
                        int accVal2 = 0;
                        int accVal3 = 0;
                        int accVal4 = 0;
                        int accVal5 = 0;
                        int accVal6 = 0;
                        int accVal7 = 0;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        int accVal0 = 0;

                        int* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        int Reduce(int accVal, int v)
        {
            v = math.abs(v);
            return accVal + v;
        }

        int Finalize(int accVal)
        {
            return accVal;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceL2FloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            v = v * v;
            return accVal + v;
        }

        float Finalize(float accVal)
        {
            return math.sqrt(accVal);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceSqrtFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            return accVal + v;
        }

        float Finalize(float accVal)
        {
            return math.sqrt(accVal);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceLogSumFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);


            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                // Horizontal reduction: use auto-vectorization to reduce multiple parts of the input in
                // parallel and then do a horizontal reduce of these parts to produce the output.
                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    for (int j = 0; j < reduceLength; j++)
                        accVal = Reduce(accVal, Xp[j]);

                    Xp += reduceLength;
                    *Op++ = Finalize(accVal);
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);
                            accVal1 = Reduce(accVal1, Xpr[1]);
                            accVal2 = Reduce(accVal2, Xpr[2]);
                            accVal3 = Reduce(accVal3, Xpr[3]);
                            accVal4 = Reduce(accVal4, Xpr[4]);
                            accVal5 = Reduce(accVal5, Xpr[5]);
                            accVal6 = Reduce(accVal6, Xpr[6]);
                            accVal7 = Reduce(accVal7, Xpr[7]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);
                        Op[1] = Finalize(accVal1);
                        Op[2] = Finalize(accVal2);
                        Op[3] = Finalize(accVal3);
                        Op[4] = Finalize(accVal4);
                        Op[5] = Finalize(accVal5);
                        Op[6] = Finalize(accVal6);
                        Op[7] = Finalize(accVal7);

                        Xpi += 8;
                        Op += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi;
                        for (int j = 0; j < reduceLength; j++)
                        {
                            accVal0 = Reduce(accVal0, Xpr[0]);

                            Xpr += innerLength;
                        }

                        Op[0] = Finalize(accVal0);

                        Xpi += 1;
                        Op += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        float Reduce(float accVal, float v)
        {
            return accVal + v;
        }

        float Finalize(float accVal)
        {
            return math.log(accVal);
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct CumSumFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;
        public bool reverse;
        public bool exclusive;

        public void Execute(int i, int count)
        {
            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;
                float* Op = Optr + i * reduceLength;

                int startOffset = reverse ? (reduceLength - 1) : 0;
                int accInnerLength = reverse ? -1 : 1;

                for (; count > 0; count--)
                {
                    float accVal = 0.0f;

                    float* Xpr = Xp + startOffset;
                    float* Opr = Op + startOffset;

                    int j = 0;
                    if (exclusive)
                    {
                        Opr[0] = 0;
                        Opr += accInnerLength;
                        j = 1;
                    }

                    for (; j < reduceLength; j++)
                    {
                        accVal += Xpr[0];
                        Opr[0] = accVal;

                        Xpr += accInnerLength;
                        Opr += accInnerLength;
                    }

                    Xp += reduceLength;
                    Op += reduceLength;
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;
                float* Op = Optr + outerIndex * outerStride;

                int startOffset = reverse ? innerLength * (reduceLength - 1) : 0;
                int accInnerLength = reverse ? -innerLength : innerLength;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;
                    float* Opi = Op + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        float accVal0 = 0.0f;
                        float accVal1 = 0.0f;
                        float accVal2 = 0.0f;
                        float accVal3 = 0.0f;
                        float accVal4 = 0.0f;
                        float accVal5 = 0.0f;
                        float accVal6 = 0.0f;
                        float accVal7 = 0.0f;

                        float* Xpr = Xpi + startOffset;
                        float* Opr = Opi + startOffset;

                        int j = 0;
                        if (exclusive)
                        {
                            Opr[0] = 0;
                            Opr[1] = 0;
                            Opr[2] = 0;
                            Opr[3] = 0;
                            Opr[4] = 0;
                            Opr[5] = 0;
                            Opr[6] = 0;
                            Opr[7] = 0;
                            Opr += accInnerLength;
                            j = 1;
                        }

                        for (; j < reduceLength; j++)
                        {
                            accVal0 += Xpr[0];
                            accVal1 += Xpr[1];
                            accVal2 += Xpr[2];
                            accVal3 += Xpr[3];
                            accVal4 += Xpr[4];
                            accVal5 += Xpr[5];
                            accVal6 += Xpr[6];
                            accVal7 += Xpr[7];
                            Opr[0] = accVal0;
                            Opr[1] = accVal1;
                            Opr[2] = accVal2;
                            Opr[3] = accVal3;
                            Opr[4] = accVal4;
                            Opr[5] = accVal5;
                            Opr[6] = accVal6;
                            Opr[7] = accVal7;

                            Xpr += accInnerLength;
                            Opr += accInnerLength;
                        }

                        Xpi += 8;
                        Opi += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        float accVal0 = 0.0f;

                        float* Xpr = Xpi + startOffset;
                        float* Opr = Opi + startOffset;

                        int j = 0;
                        if (exclusive)
                        {
                            Opr[0] = 0;
                            Opr += accInnerLength;
                            j = 1;
                        }

                        for (; j < reduceLength; j++)
                        {
                            accVal0 += Xpr[0];
                            Opr[0] = accVal0;

                            Xpr += accInnerLength;
                            Opr += accInnerLength;
                        }

                        Xpi += 1;
                        Opi += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                    Op += outerStride;
                }
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct CumSumIntJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;
        public bool reverse;
        public bool exclusive;

        public void Execute(int i, int count)
        {
            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            if (innerLength == 1)
            {
                int* Xp = Xptr + i * reduceLength;
                int* Op = Optr + i * reduceLength;

                int startOffset = reverse ? (reduceLength - 1) : 0;
                int accInnerLength = reverse ? -1 : 1;

                for (; count > 0; count--)
                {
                    int accVal = 0;

                    int* Xpr = Xp + startOffset;
                    int* Opr = Op + startOffset;

                    int j = 0;
                    if (exclusive)
                    {
                        Opr[0] = 0;
                        Opr += accInnerLength;
                        j = 1;
                    }

                    for (; j < reduceLength; j++)
                    {
                        accVal += Xpr[0];
                        Opr[0] = accVal;

                        Xpr += accInnerLength;
                        Opr += accInnerLength;
                    }

                    Xp += reduceLength;
                    Op += reduceLength;
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                int* Xp = Xptr + outerIndex * outerStride;
                int* Op = Optr + outerIndex * outerStride;

                int startOffset = reverse ? innerLength * (reduceLength - 1) : 0;
                int accInnerLength = reverse ? -innerLength : innerLength;

                // Vertical reduction: use auto-vectorization to reduce multiple output elements in parallel.
                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    int* Xpi = Xp + innerIndex;
                    int* Opi = Op + innerIndex;

                    for (; spanCount >= 8; spanCount -= 8)
                    {
                        int accVal0 = 0;
                        int accVal1 = 0;
                        int accVal2 = 0;
                        int accVal3 = 0;
                        int accVal4 = 0;
                        int accVal5 = 0;
                        int accVal6 = 0;
                        int accVal7 = 0;

                        int* Xpr = Xpi + startOffset;
                        int* Opr = Opi + startOffset;

                        int j = 0;
                        if (exclusive)
                        {
                            Opr[0] = 0;
                            Opr[1] = 0;
                            Opr[2] = 0;
                            Opr[3] = 0;
                            Opr[4] = 0;
                            Opr[5] = 0;
                            Opr[6] = 0;
                            Opr[7] = 0;
                            Opr += accInnerLength;
                            j = 1;
                        }

                        for (; j < reduceLength; j++)
                        {
                            accVal0 += Xpr[0];
                            accVal1 += Xpr[1];
                            accVal2 += Xpr[2];
                            accVal3 += Xpr[3];
                            accVal4 += Xpr[4];
                            accVal5 += Xpr[5];
                            accVal6 += Xpr[6];
                            accVal7 += Xpr[7];
                            Opr[0] = accVal0;
                            Opr[1] = accVal1;
                            Opr[2] = accVal2;
                            Opr[3] = accVal3;
                            Opr[4] = accVal4;
                            Opr[5] = accVal5;
                            Opr[6] = accVal6;
                            Opr[7] = accVal7;

                            Xpr += accInnerLength;
                            Opr += accInnerLength;
                        }

                        Xpi += 8;
                        Opi += 8;
                    }

                    for (; spanCount > 0; spanCount--)
                    {
                        int accVal0 = 0;

                        int* Xpr = Xpi + startOffset;
                        int* Opr = Opi + startOffset;

                        int j = 0;
                        if (exclusive)
                        {
                            Opr[0] = 0;
                            Opr += accInnerLength;
                            j = 1;
                        }

                        for (; j < reduceLength; j++)
                        {
                            accVal0 += Xpr[0];
                            Opr[0] = accVal0;

                            Xpr += accInnerLength;
                            Opr += accInnerLength;
                        }

                        Xpi += 1;
                        Opi += 1;
                    }

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                    Op += outerStride;
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMaxFloatFirstJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            float* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                float* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    float accVal0 = float.MinValue;
                    float accVal1 = float.MinValue;
                    float accVal2 = float.MinValue;
                    float accVal3 = float.MinValue;
                    float accVal4 = float.MinValue;
                    float accVal5 = float.MinValue;
                    float accVal6 = float.MinValue;
                    float accVal7 = float.MinValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    float accVal0 = float.MinValue;
                    int accValIndex0 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref float accVal, ref int accValIndex, float v, int index)
        {
                        // Compare if the reduced value has changed using an integer compare to avoid another
            // floating point compare that may order differently with respect to NaNs or other
            // classes of floating point numbers. Integer compares are also typically less expensive
            // compare to floating point compares.
            float reduceVal = math.max(accVal, v);
            if (math.asint(accVal) != math.asint(reduceVal))
                accValIndex = index;
            accVal = reduceVal;
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMinFloatFirstJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            float* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                float* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    float accVal0 = float.MaxValue;
                    float accVal1 = float.MaxValue;
                    float accVal2 = float.MaxValue;
                    float accVal3 = float.MaxValue;
                    float accVal4 = float.MaxValue;
                    float accVal5 = float.MaxValue;
                    float accVal6 = float.MaxValue;
                    float accVal7 = float.MaxValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    float accVal0 = float.MaxValue;
                    int accValIndex0 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref float accVal, ref int accValIndex, float v, int index)
        {
                        // Compare if the reduced value has changed using an integer compare to avoid another
            // floating point compare that may order differently with respect to NaNs or other
            // classes of floating point numbers. Integer compares are also typically less expensive
            // compare to floating point compares.
            float reduceVal = math.min(accVal, v);
            if (math.asint(accVal) != math.asint(reduceVal))
                accValIndex = index;
            accVal = reduceVal;
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMaxIntFirstJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            int* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                int* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    int accVal0 = int.MinValue;
                    int accVal1 = int.MinValue;
                    int accVal2 = int.MinValue;
                    int accVal3 = int.MinValue;
                    int accVal4 = int.MinValue;
                    int accVal5 = int.MinValue;
                    int accVal6 = int.MinValue;
                    int accVal7 = int.MinValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    int accVal0 = int.MinValue;
                    int accValIndex0 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref int accVal, ref int accValIndex, int v, int index)
        {
                        // Compare if the reduced value has changed using an integer compare to avoid another
            // floating point compare that may order differently with respect to NaNs or other
            // classes of floating point numbers. Integer compares are also typically less expensive
            // compare to floating point compares.
            int reduceVal = math.max(accVal, v);
            if (math.asint(accVal) != math.asint(reduceVal))
                accValIndex = index;
            accVal = reduceVal;
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMinIntFirstJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            int* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                int* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    int accVal0 = int.MaxValue;
                    int accVal1 = int.MaxValue;
                    int accVal2 = int.MaxValue;
                    int accVal3 = int.MaxValue;
                    int accVal4 = int.MaxValue;
                    int accVal5 = int.MaxValue;
                    int accVal6 = int.MaxValue;
                    int accVal7 = int.MaxValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    int accVal0 = int.MaxValue;
                    int accValIndex0 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref int accVal, ref int accValIndex, int v, int index)
        {
                        // Compare if the reduced value has changed using an integer compare to avoid another
            // floating point compare that may order differently with respect to NaNs or other
            // classes of floating point numbers. Integer compares are also typically less expensive
            // compare to floating point compares.
            int reduceVal = math.min(accVal, v);
            if (math.asint(accVal) != math.asint(reduceVal))
                accValIndex = index;
            accVal = reduceVal;
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMaxFloatLastJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            float* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                float* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    float accVal0 = float.MinValue;
                    float accVal1 = float.MinValue;
                    float accVal2 = float.MinValue;
                    float accVal3 = float.MinValue;
                    float accVal4 = float.MinValue;
                    float accVal5 = float.MinValue;
                    float accVal6 = float.MinValue;
                    float accVal7 = float.MinValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    float accVal0 = float.MinValue;
                    int accValIndex0 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref float accVal, ref int accValIndex, float v, int index)
        {
                        if (v >= accVal)
            {
                accVal = v;
                accValIndex = index;
            }
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMinFloatLastJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            float* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                float* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    float accVal0 = float.MaxValue;
                    float accVal1 = float.MaxValue;
                    float accVal2 = float.MaxValue;
                    float accVal3 = float.MaxValue;
                    float accVal4 = float.MaxValue;
                    float accVal5 = float.MaxValue;
                    float accVal6 = float.MaxValue;
                    float accVal7 = float.MaxValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    float accVal0 = float.MaxValue;
                    int accValIndex0 = 0;

                    float* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref float accVal, ref int accValIndex, float v, int index)
        {
                        if (v <= accVal)
            {
                accVal = v;
                accValIndex = index;
            }
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMaxIntLastJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            int* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                int* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    int accVal0 = int.MinValue;
                    int accVal1 = int.MinValue;
                    int accVal2 = int.MinValue;
                    int accVal3 = int.MinValue;
                    int accVal4 = int.MinValue;
                    int accVal5 = int.MinValue;
                    int accVal6 = int.MinValue;
                    int accVal7 = int.MinValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    int accVal0 = int.MinValue;
                    int accValIndex0 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref int accVal, ref int accValIndex, int v, int index)
        {
                        if (v >= accVal)
            {
                accVal = v;
                accValIndex = index;
            }
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ArgMinIntLastJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            int* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = i % innerLength;
            int outerIndex = i / innerLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = reduceLength * innerLength;

            int* Xp = Xptr + outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                int* Xpi = Xp + innerIndex;

                for (; spanCount >= 8; spanCount -= 8)
                {
                    int accVal0 = int.MaxValue;
                    int accVal1 = int.MaxValue;
                    int accVal2 = int.MaxValue;
                    int accVal3 = int.MaxValue;
                    int accVal4 = int.MaxValue;
                    int accVal5 = int.MaxValue;
                    int accVal6 = int.MaxValue;
                    int accVal7 = int.MaxValue;
                    int accValIndex0 = 0;
                    int accValIndex1 = 0;
                    int accValIndex2 = 0;
                    int accValIndex3 = 0;
                    int accValIndex4 = 0;
                    int accValIndex5 = 0;
                    int accValIndex6 = 0;
                    int accValIndex7 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);
                        Reduce(ref accVal1, ref accValIndex1, Xpr[1], j);
                        Reduce(ref accVal2, ref accValIndex2, Xpr[2], j);
                        Reduce(ref accVal3, ref accValIndex3, Xpr[3], j);
                        Reduce(ref accVal4, ref accValIndex4, Xpr[4], j);
                        Reduce(ref accVal5, ref accValIndex5, Xpr[5], j);
                        Reduce(ref accVal6, ref accValIndex6, Xpr[6], j);
                        Reduce(ref accVal7, ref accValIndex7, Xpr[7], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;
                    Op[1] = accValIndex1;
                    Op[2] = accValIndex2;
                    Op[3] = accValIndex3;
                    Op[4] = accValIndex4;
                    Op[5] = accValIndex5;
                    Op[6] = accValIndex6;
                    Op[7] = accValIndex7;

                    Xpi += 8;
                    Op += 8;
                }

                for (; spanCount > 0; spanCount--)
                {
                    int accVal0 = int.MaxValue;
                    int accValIndex0 = 0;

                    int* Xpr = Xpi;
                    for (int j = 0; j < reduceLength; j++)
                    {
                        Reduce(ref accVal0, ref accValIndex0, Xpr[0], j);

                        Xpr += innerLength;
                    }

                    Op[0] = accValIndex0;

                    Xpi += 1;
                    Op += 1;
                }

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
            }
        }

        static void Reduce(ref int accVal, ref int accValIndex, int v, int index)
        {
                        if (v <= accVal)
            {
                accVal = v;
                accValIndex = index;
            }
            
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct SoftmaxJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;
                float* Op = Optr + i * reduceLength;

                for (; count > 0; count--)
                {
                    Softmax(Xp, Op, reduceLength, innerLength: 1);

                    Xp += reduceLength;
                    Op += reduceLength;
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;
                float* Op = Optr + outerIndex * outerStride;

                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;
                    float* Opi = Op + innerIndex;

                    for (; spanCount > 0; spanCount--)
                        Softmax(Xpi++, Opi++, reduceLength, innerLength);

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                    Op += outerStride;
                }
            }
        }

        static void Softmax(float* Xp, float* Op, int reduceLength, int innerLength)
        {
            float maxVal = float.MinValue;
            {
                float* Xpr = Xp;
                for (int j = 0; j < reduceLength; j++)
                {
                    maxVal = math.max(maxVal, Xpr[0]);
                    Xpr += innerLength;
                }
            }

            float expSumVal = 0.0f;
            {
                float *Xpr = Xp;
                float *Opr = Op;
                for (int j = 0; j < reduceLength; j++)
                {
                    float expVal = math.exp(Xpr[0] - maxVal);
                    expSumVal += expVal;
                    Opr[0] = expVal;
                    Xpr += innerLength;
                    Opr += innerLength;
                }
            }

            float rcpExpSumVal = math.rcp(expSumVal);
            {
                float *Opr = Op;
                for (int j = 0; j < reduceLength; j++)
                {
                    Opr[0] *= rcpExpSumVal;
                    Opr += innerLength;
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct LogSoftmaxJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;
                float* Op = Optr + i * reduceLength;

                for (; count > 0; count--)
                {
                    LogSoftmax(Xp, Op, reduceLength, innerLength: 1);

                    Xp += reduceLength;
                    Op += reduceLength;
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;
                float* Op = Optr + outerIndex * outerStride;

                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;
                    float* Opi = Op + innerIndex;

                    for (; spanCount > 0; spanCount--)
                        LogSoftmax(Xpi++, Opi++, reduceLength, innerLength);

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                    Op += outerStride;
                }
            }
        }

        static void LogSoftmax(float* Xp, float* Op, int reduceLength, int innerLength)
        {
            float maxVal = float.MinValue;
            {
                float* Xpr = Xp;
                for (int j = 0; j < reduceLength; j++)
                {
                    maxVal = math.max(maxVal, Xpr[0]);
                    Xpr += innerLength;
                }
            }

            float expSumVal = 0.0f;
            {
                float *Xpr = Xp;
                float *Opr = Op;
                for (int j = 0; j < reduceLength; j++)
                {
                    float expVal = math.exp(Xpr[0] - maxVal);
                    expSumVal += expVal;
                    Xpr += innerLength;
                    Opr += innerLength;
                }
            }

            float logVal = math.log(expSumVal) + maxVal;
            {
                float *Xpr = Xp;
                float *Opr = Op;
                for (int j = 0; j < reduceLength; j++)
                {
                    Opr[0] = Xpr[0] - logVal;
                    Xpr += innerLength;
                    Opr += innerLength;
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ReduceLogSumExpFloatJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            if (innerLength == 1)
            {
                float* Xp = Xptr + i * reduceLength;

                for (; count > 0; count--)
                {
                    *Op++ = ReduceLogSumExp(Xp, reduceLength, innerLength: 1);

                    Xp += reduceLength;
                }
            }
            else
            {
                int innerIndex = i % innerLength;
                int outerIndex = i / innerLength;

                int innerRemaining = innerLength - innerIndex;
                int outerStride = reduceLength * innerLength;

                float* Xp = Xptr + outerIndex * outerStride;

                while (count > 0)
                {
                    int spanCount = math.min(count, innerRemaining);
                    count -= spanCount;

                    float* Xpi = Xp + innerIndex;

                    for (; spanCount > 0; spanCount--)
                        *Op++ = ReduceLogSumExp(Xpi++, reduceLength, innerLength);

                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    Xp += outerStride;
                }
            }
        }

        static float ReduceLogSumExp(float* Xp, int reduceLength, int innerLength)
        {
            float maxVal = float.MinValue;
            {
                float* Xpr = Xp;
                for (int j = 0; j < reduceLength; j++)
                {
                    maxVal = math.max(maxVal, Xpr[0]);
                    Xpr += innerLength;
                }
            }

            float expSumVal = 0.0f;
            {
                float *Xpr = Xp;
                for (int j = 0; j < reduceLength; j++)
                {
                    expSumVal += math.exp(Xpr[0] - maxVal);
                    Xpr += innerLength;
                }
            }

            return math.log(expSumVal) + maxVal;
        }
    }
}
}
