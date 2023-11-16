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
    internal unsafe struct PadJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public Layers.PadMode padMode;
        public float constant;
        public PadParameters padParams;

        public void Execute(int i, int count)
        {
            switch (padMode)
            {
                case Layers.PadMode.Constant:
                    PadConstant(i, count);
                    break;
                case Layers.PadMode.Reflect:
                    PadReflect(i, count);
                    break;
                case Layers.PadMode.Symmetric:
                    PadSymmetric(i, count);
                    break;
                case Layers.PadMode.Edge:
                    PadEdge(i, count);
                    break;
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void PadReflect(int i, int count)
        {
            float* Op = Optr + i;

            while (count > 0)
            {
                // Compute the innermost offset and count of elements that can be handled sequentially.
                int innerOffsetO = i % padParams.shapeO[0];
                int remaining = i / padParams.shapeO[0];
                int spanCount = math.min(count, padParams.shapeO[0] - innerOffsetO);

                Hint.Assume(spanCount > 0);

                i += spanCount;
                count -= spanCount;

                float* Xp = Xptr;

                // Compute the pointer to the innermost dimension by unraveling the remaining output index.
                for (int j = 1; j < padParams.lastIndex; j++)
                {
                    int offsetX = (remaining % padParams.shapeO[j]) - padParams.pad[j];
                    remaining = remaining / padParams.shapeO[j];

                    Xp += padParams.strideX[j] * ReflectOffset(offsetX, padParams.shapeX[j] - 1);
                }

                int innerOffsetX = innerOffsetO - padParams.pad[0];
                int lastInnerOffsetX = padParams.shapeX[0] - 1;

                // Pad the span of elements from the innermost dimension.
                while (spanCount > 0)
                {
                    // Check if the input offset is in the left/right padding region.
                    if ((uint)innerOffsetX <= (uint)lastInnerOffsetX)
                    {
                        int copyCount = math.min(spanCount, lastInnerOffsetX - innerOffsetX + 1);
                        UnsafeUtility.MemCpy(Op, &Xp[innerOffsetX], copyCount * sizeof(float));

                        Op += copyCount;
                        innerOffsetX += copyCount;
                        spanCount -= copyCount;
                    }
                    else
                    {
                        Op[0] = Xp[ReflectOffset(innerOffsetX, lastInnerOffsetX)];

                        Op += 1;
                        innerOffsetX += 1;
                        spanCount -= 1;
                    }
                }
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void PadSymmetric(int i, int count)
        {
            float* Op = Optr + i;

            while (count > 0)
            {
                // Compute the innermost offset and count of elements that can be handled sequentially.
                int innerOffsetO = i % padParams.shapeO[0];
                int remaining = i / padParams.shapeO[0];
                int spanCount = math.min(count, padParams.shapeO[0] - innerOffsetO);

                Hint.Assume(spanCount > 0);

                i += spanCount;
                count -= spanCount;

                float* Xp = Xptr;

                // Compute the pointer to the innermost dimension by unraveling the remaining output index.
                for (int j = 1; j < padParams.lastIndex; j++)
                {
                    int offsetX = (remaining % padParams.shapeO[j]) - padParams.pad[j];
                    remaining = remaining / padParams.shapeO[j];

                    Xp += padParams.strideX[j] * SymmetricOffset(offsetX, padParams.shapeX[j] - 1);
                }

                int innerOffsetX = innerOffsetO - padParams.pad[0];
                int lastInnerOffsetX = padParams.shapeX[0] - 1;

                // Pad the span of elements from the innermost dimension.
                while (spanCount > 0)
                {
                    // Check if the input offset is in the left/right padding region.
                    if ((uint)innerOffsetX <= (uint)lastInnerOffsetX)
                    {
                        int copyCount = math.min(spanCount, lastInnerOffsetX - innerOffsetX + 1);
                        UnsafeUtility.MemCpy(Op, &Xp[innerOffsetX], copyCount * sizeof(float));

                        Op += copyCount;
                        innerOffsetX += copyCount;
                        spanCount -= copyCount;
                    }
                    else
                    {
                        Op[0] = Xp[SymmetricOffset(innerOffsetX, lastInnerOffsetX)];

                        Op += 1;
                        innerOffsetX += 1;
                        spanCount -= 1;
                    }
                }
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void PadEdge(int i, int count)
        {
            float* Op = Optr + i;

            while (count > 0)
            {
                // Compute the innermost offset and count of elements that can be handled sequentially.
                int innerOffsetO = i % padParams.shapeO[0];
                int remaining = i / padParams.shapeO[0];
                int spanCount = math.min(count, padParams.shapeO[0] - innerOffsetO);

                Hint.Assume(spanCount > 0);

                i += spanCount;
                count -= spanCount;

                float* Xp = Xptr;

                // Compute the pointer to the innermost dimension by unraveling the remaining output index.
                for (int j = 1; j < padParams.lastIndex; j++)
                {
                    int offsetX = (remaining % padParams.shapeO[j]) - padParams.pad[j];
                    remaining = remaining / padParams.shapeO[j];

                    Xp += padParams.strideX[j] * EdgeOffset(offsetX, padParams.shapeX[j] - 1);
                }

                int innerOffsetX = innerOffsetO - padParams.pad[0];
                int lastInnerOffsetX = padParams.shapeX[0] - 1;

                // Pad the span of elements from the innermost dimension.
                while (spanCount > 0)
                {
                    // Check if the input offset is in the left/right padding region.
                    if ((uint)innerOffsetX <= (uint)lastInnerOffsetX)
                    {
                        int copyCount = math.min(spanCount, lastInnerOffsetX - innerOffsetX + 1);
                        UnsafeUtility.MemCpy(Op, &Xp[innerOffsetX], copyCount * sizeof(float));

                        Op += copyCount;
                        innerOffsetX += copyCount;
                        spanCount -= copyCount;
                    }
                    else
                    {
                        Op[0] = Xp[EdgeOffset(innerOffsetX, lastInnerOffsetX)];

                        Op += 1;
                        innerOffsetX += 1;
                        spanCount -= 1;
                    }
                }
            }
        }

        static int MirrorOffset(int offsetX, int lastOffsetX, int symmetric)
        {
            if (offsetX < 0)
                offsetX = math.min(-offsetX - symmetric, lastOffsetX);
            else if (offsetX > lastOffsetX)
                offsetX = math.max(0, lastOffsetX - (offsetX - lastOffsetX) + symmetric);
            return offsetX;
        }

        static int ReflectOffset(int offsetX, int lastOffsetX)
        {
            return MirrorOffset(offsetX, lastOffsetX, symmetric: 0);
        }

        static int SymmetricOffset(int offsetX, int lastOffsetX)
        {
            return MirrorOffset(offsetX, lastOffsetX, symmetric: 1);
        }

        static int EdgeOffset(int offsetX, int lastOffsetX)
        {
            return math.clamp(offsetX, 0, lastOffsetX);
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void PadConstant(int i, int count)
        {
            float* Op = Optr + i;

            while (count > 0)
            {
                // Compute the innermost offset and count of elements that can be handled sequentially.
                int innerOffsetO = i % padParams.shapeO[0];
                int remaining = i / padParams.shapeO[0];
                int spanCount = math.min(count, padParams.shapeO[0] - innerOffsetO);

                Hint.Assume(spanCount > 0);

                i += spanCount;
                count -= spanCount;

                float* Xp = Xptr;
                bool isPaddingRegion = false;

                // Compute the pointer to the innermost dimension by unraveling the remaining output index.
                for (int j = 1; j < padParams.lastIndex; j++)
                {
                    int offsetX = (remaining % padParams.shapeO[j]) - padParams.pad[j];
                    remaining = remaining / padParams.shapeO[j];

                    if ((uint)offsetX >= (uint)padParams.shapeX[j])
                    {
                        PadOutput(Op, (uint)spanCount);
                        Op += spanCount;
                        isPaddingRegion = true;
                        break;
                    }

                    Xp += padParams.strideX[j] * offsetX;
                }

                if (!isPaddingRegion)
                {
                    int innerOffsetX = innerOffsetO - padParams.pad[0];

                    // Pad the span of elements from the innermost dimension.
                    while (spanCount > 0)
                    {
                        int elementCount;

                        // Check if the input offset is in the left/right padding region.
                        if ((uint)innerOffsetX < (uint)padParams.shapeX[0])
                        {
                            elementCount = math.min(spanCount, padParams.shapeX[0] - innerOffsetX);
                            UnsafeUtility.MemCpy(Op, &Xp[innerOffsetX], elementCount * sizeof(float));
                        }
                        else
                        {
                            elementCount = (innerOffsetX < 0) ? math.min(spanCount, -innerOffsetX) : spanCount;
                            PadOutput(Op, (uint)elementCount);
                        }

                        Op += elementCount;
                        innerOffsetX += elementCount;
                        spanCount -= elementCount;
                    }
                }
            }
        }

        void PadOutput([NoAlias] float* Op, uint count)
        {
            for (uint i = 0; i < count; i++)
                Op[i] = constant;
        }
    }
}
}
