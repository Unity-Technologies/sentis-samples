using UnityEngine;
using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Burst.Intrinsics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Avx2;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Sentis {
public partial class CPUBackend
{
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CopyJob : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } uint* Xptr => (uint*)X.ptr;
        public ReadWriteMemResource O { get; set; } uint* Optr => (uint*)O.ptr;
        public int length;
        public int offsetX;
        public int offsetO;

        public void Execute()
        {
            UnsafeUtility.MemCpy(destination: Optr + offsetO, source: Xptr + offsetX, size: length * sizeof(float));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CopyStrideJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int strideX;
        public int strideO;
        public int offsetX;
        public int offsetO;
        public int length;

        public void Execute(int i)
        {
            UnsafeUtility.MemCpy(destination: Optr + offsetO + i * strideO, source: Xptr + offsetX + i * strideX, size: length * sizeof(int));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ClearJob : IJob, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public int length;

        public void Execute()
        {
            UnsafeUtility.MemClear(destination: Optr, size: length * sizeof(int));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SetJob : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public int memValue;

        public void Execute(int i)
        {
            Optr[i] = memValue;
        }
    }

    /// <summary>
    /// Creates a Mathematics.Random struct to be used inside the inner loop of a burst job.
    /// The seed is given per job and the threadIndex is per output tensor entry, subsequent calls
    /// to e.g. NextFloat will be randomly distributed across adjacent threads.
    /// </summary>
    static Mathematics.Random JobMathematicsRandom(uint seed, int threadIndex)
    {
        var index = seed + (uint)threadIndex;
        // index may not be uint.MaxValue, in this case move to distant value
        return Mathematics.Random.CreateFromIndex(index != uint.MaxValue ? index : 2147483647u);
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RandomNormalJob : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public uint seed;
        public float mean;
        public float scale;

        public void Execute(int i)
        {
            var random = JobMathematicsRandom(seed, i);
            float u, v, s;
            do {
                u = random.NextFloat() * 2 - 1;
                v = random.NextFloat() * 2 - 1;
                s = u * u + v * v;
            } while (s >= 1 || s == 0);
            float mul = Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
            Optr[i] = mean + scale * u * mul;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RandomUniformJob : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public uint seed;
        public float low;
        public float high;

        public void Execute(int i)
        {
            var random = JobMathematicsRandom(seed, i);
            Optr[i] = low + (high - low) * random.NextFloat();
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BernoulliJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; }
        public uint seed;
        public DataType dataType;

        public void Execute(int i)
        {
            var random = JobMathematicsRandom(seed, i);
            if (dataType == DataType.Float)
                ((float*)O.ptr)[i] = Xptr[i] > random.NextFloat() ? 1f : 0f;
            else if (dataType == DataType.Int)
                ((int*)O.ptr)[i] = Xptr[i] > random.NextFloat() ? 1 : 0;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct IsInfJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public bool detectNegative;
        public bool detectPositive;

        public void Execute(int threadIdx)
        {
            float v = Xptr[threadIdx];
            Optr[threadIdx] = (float.IsNegativeInfinity(v) && detectNegative || float.IsPositiveInfinity(v) && detectPositive) ? 1 : 0;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GatherJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int axisDim;
        public int innerLength;
        public int indicesLength;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            int innerIndex = i % innerLength;
            int indicesIndex = (i / innerLength) % indicesLength;
            int outerIndex = (i / innerLength) / indicesLength;

            int innerRemaining = innerLength - innerIndex;
            int outerStride = innerLength * axisDim;
            int outerOffset = outerIndex * outerStride;

            while (count > 0)
            {
                int spanCount = math.min(count, innerRemaining);
                count -= spanCount;

                int index = Bptr[indicesIndex];
                index = index < 0 ? axisDim + index : index;

                float* Xp = &Xptr[outerOffset + index * innerLength + innerIndex];
                UnsafeUtility.MemCpy(Op, Xp, spanCount * sizeof(float));
                Op += spanCount;

                if (count > 0)
                {
                    // Output is now always aligned to the start of the inner dimension.
                    innerIndex = 0;
                    innerRemaining = innerLength;

                    if (++indicesIndex == indicesLength)
                    {
                        indicesIndex = 0;
                        outerOffset += outerStride;
                    }
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GatherNDJob :  IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public int batchDims;
        public int indexSize;

        public int rankO;
        public fixed int shapeO[8];
        public fixed int stridesO[8];
        public int rankX;
        public fixed int shapeX[8];
        public fixed int stridesX[8];
        public int rankIndices;
        public fixed int shapeIndices[8];
        public fixed int stridesIndices[8];

        public void Execute(int threadIdx)
        {
            var itIndices = 0;
            for (var i = 0; i < rankIndices - 1; i++)
                itIndices += ((threadIdx / stridesO[(TensorShape.maxRank - rankO) + i]) % shapeO[(TensorShape.maxRank - rankO) + i]) * stridesIndices[(TensorShape.maxRank - rankIndices) + i];

            var itX = 0;
            for (var i = 0; i < batchDims; i++)
                itX += ((threadIdx / stridesO[(TensorShape.maxRank - rankO) + i]) % shapeO[(TensorShape.maxRank - rankO) + i]) * stridesX[(TensorShape.maxRank - rankX) + i];

            for (var i = 0; i < indexSize; i++)
            {
                int index = Bptr[itIndices + i];
                index = index < 0 ? shapeX[(TensorShape.maxRank - rankX) + (batchDims + i)] + index : index;

                itX += index * stridesX[(TensorShape.maxRank - rankX) + (batchDims + i)];
            }

            for (var i = batchDims + indexSize; i < rankX; i++)
            {
                var outIdx = rankO - (rankX - i);
                itX += ((threadIdx / stridesO[(TensorShape.maxRank - rankO) + outIdx]) % shapeO[(TensorShape.maxRank - rankO) + outIdx]) * stridesX[(TensorShape.maxRank - rankX) + i];
            }

            Optr[threadIdx] = Xptr[itX];
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SliceJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public SliceParameters sliceParams;

        public void Execute(int i, int count)
        {
            float* Op = Optr + i;

            while (count > 0)
            {
                // Compute the innermost offset and count of elements that can be handled sequentially.
                int innerOffsetO = i % sliceParams.shapeO[0];
                int remaining = i / sliceParams.shapeO[0];
                int spanCount = math.min(count, sliceParams.shapeO[0] - innerOffsetO);

                Unity.Burst.CompilerServices.Hint.Assume(spanCount > 0);

                int innerOffsetX = sliceParams.stridedStarts[0] + innerOffsetO * sliceParams.stridedSteps[0];
                float* Xp = Xptr + innerOffsetX;

                // Compute the pointer to the innermost dimension by unraveling the remaining output index.
                for (int j = 1; j < sliceParams.lastIndex; j++)
                {
                    int offsetO = remaining % sliceParams.shapeO[j];
                    remaining = remaining / sliceParams.shapeO[j];
                    Xp += sliceParams.stridedStarts[j] + offsetO * sliceParams.stridedSteps[j];
                }

                if (sliceParams.stridedSteps[0] == 1)
                    UnsafeUtility.MemCpy(Op, &Xp[0], spanCount * sizeof(float));
                else
                {
                    for (int j = 0; j < spanCount; j++)
                    {
                        Op[j] = Xp[0];
                        Xp += sliceParams.stridedSteps[0];
                    }
                }

                i += spanCount;
                count -= spanCount;
                Op += spanCount;
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct HardmaxJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int reduceLength;
        public int innerLength;

        public void Execute(int i, int count)
        {
            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

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
                    Hardmax(Xpi++, Opi++, reduceLength, innerLength);

                // Output is now always aligned to the start of the inner dimension.
                innerIndex = 0;
                innerRemaining = innerLength;

                Xp += outerStride;
                Op += outerStride;
            }
        }

        static void Hardmax(float* Xp, float* Op, int reduceLength, int innerLength)
        {
            float* Xpr = Xp;
            float* Opr = Op;

            var maxVal = float.MinValue;
            float* maxOpr = Opr;

            for (var j = 0; j < reduceLength; j++)
            {
                Opr[0] = 0.0f;
                if (Xpr[0] > maxVal)
                {
                    maxVal = Xpr[0];
                    maxOpr = Opr;
                }
                Xpr += innerLength;
                Opr += innerLength;
            }

            *maxOpr = 1.0f;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TransposeJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public TransposeIterator iteratorX;

        public void Execute(int i, int count)
        {
            int* countersX = stackalloc int[TensorShape.maxRank];

            int offsetX = iteratorX.InitialOffset(i, countersX);
            int innerStride = iteratorX.InnerStride();

            float* Op = Optr + i;

            while (count > 0)
            {
                int spanRemainingX = iteratorX.InnerSpanSize() - countersX[0];
                int spanCount = math.min(count, spanRemainingX);

                float* Xp = Xptr + offsetX;

                for (int j = 0; j < spanCount; j++)
                {
                    Op[j] = Xp[0];
                    Xp += innerStride;
                }

                Op += spanCount;
                count -= spanCount;

                if (count > 0)
                    offsetX = iteratorX.AdvanceOffset(offsetX, spanCount, countersX);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TrilJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int widthX;
        public int heightX;
        public int diagonalK;

        public void Execute(int i)
        {
            float* rowPtrO = Optr + i * widthX;
            float* rowPtrX = Xptr + i * widthX;

            int index = math.clamp((i % heightX) + diagonalK + 1, 0, widthX);

            UnsafeUtility.MemCpy(rowPtrO, rowPtrX, index * sizeof(float));
            UnsafeUtility.MemClear(rowPtrO + index, (widthX - index) * sizeof(float));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TriuJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int widthX;
        public int heightX;
        public int diagonalK;

        public void Execute(int i)
        {
            float* rowPtrO = Optr + i * widthX;
            float* rowPtrX = Xptr + i * widthX;

            int index = math.clamp((i % heightX) + diagonalK, 0, widthX);

            UnsafeUtility.MemClear(rowPtrO, index * sizeof(float));
            UnsafeUtility.MemCpy(rowPtrO + index, rowPtrX + index, (widthX - index) * sizeof(float));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct OneHotJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;
        public int onValue;
        public int offValue;
        public int rankO;
        public int inputRank;
        public int depth;
        public int axis;
        public fixed int shapeO[8];
        public fixed int stridesO[8];
        public fixed int shapeX[8];
        public fixed int stridesX[8];

        public void Execute(int threadIdx)
        {
            int stridesToAxis = stridesO[(TensorShape.maxRank - rankO) + axis];
            int axisDim = shapeO[(TensorShape.maxRank - rankO) + axis];

            int j = (threadIdx / stridesToAxis) % axisDim;
            int index = Xptr[(threadIdx - stridesToAxis*j) / axisDim + (threadIdx % stridesToAxis)];
            index = index < 0 ? depth + index : index;

            int v = (j == index) ? onValue : offValue;

            Optr[threadIdx] = v;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ResizeLinearInitTablesJob : IJob, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; }
        public int spatialDims;
        public fixed int inputShape[ResizeLinearJob.maxSpatialDims];
        public fixed int outputShape[ResizeLinearJob.maxSpatialDims];
        public fixed float scales[ResizeLinearJob.maxSpatialDims];
        public fixed float biases[ResizeLinearJob.maxSpatialDims];

        public void Execute()
        {
            int* buffer = (int*)O.ptr;

            for (int i = 0; i < spatialDims; i++)
            {
                int inputDim = inputShape[i];
                int outputDim = outputShape[i];
                float scale = scales[i];
                float bias = biases[i];

                int* indexTable0 = buffer;
                buffer += outputDim;
                int* indexTable1 = buffer;
                buffer += outputDim;
                float* fracTable = (float*)buffer;
                buffer += outputDim;

                // Scale the indices with the input stride.
                int inputStride = 1;
                for (int j = i + 1; j < spatialDims; j++)
                    inputStride *= inputShape[j];

                for (int j = 0; j < outputDim; j++)
                {
                    float inputCoord = math.max(0.0f, (float)j * scale + bias);

                    int indexValue = (int)inputCoord;

                    indexTable0[j] = inputStride * indexValue;
                    indexTable1[j] = inputStride * math.min(indexValue + 1, inputDim - 1);
                    fracTable[j] = inputCoord - math.floor(inputCoord);
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ResizeLinearJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        internal const int maxSpatialDims = 3;

        public int spatialDims;
        public int inputSize;
        public fixed int outputShape[ResizeLinearJob.maxSpatialDims];

        public void Execute(int i, int count)
        {
            if (spatialDims == 2)
                ResizeLinear2D(i, count);
            else
                ResizeLinear3D(i, count);
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void ResizeLinear2D(int i, int count)
        {
            int inputSize = this.inputSize;
            int outputHeight = outputShape[0];
            int outputWidth = outputShape[1];

            // Compute the addresses of the tables contained in the table blob.
            void* nextTableData = (int*)B.ptr;
            int* indexTableH0 = TableDataAddr<int>(ref nextTableData, outputHeight);
            int* indexTableH1 = TableDataAddr<int>(ref nextTableData, outputHeight);
            float* fracTableH = TableDataAddr<float>(ref nextTableData, outputHeight);
            int* indexTableW0 = TableDataAddr<int>(ref nextTableData, outputWidth);
            int* indexTableW1 = TableDataAddr<int>(ref nextTableData, outputWidth);
            float* fracTableW = TableDataAddr<float>(ref nextTableData, outputWidth);

            float* Op = Optr + i;

            // Extract the starting output position from the index.
            int ow = i % outputWidth;
            i = i / outputWidth;
            int oh = i % outputHeight;
            i = i / outputHeight;

            // Advance to the starting input channel.
            float* Xp = Xptr + i * inputSize;

            int outputWidthRemaining = outputWidth - ow;

            while (count > 0)
            {
                int outputCountW = math.min(count, outputWidthRemaining);
                count -= outputCountW;

                float* Xph0 = Xp + indexTableH0[oh];
                float* Xph1 = Xp + indexTableH1[oh];

                float fracH0 = fracTableH[oh];
                float fracH1 = 1.0f - fracH0;

                if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
                {
                    v256 fracH0V = new v256(fracH0);
                    v256 fracH1V = new v256(fracH1);
                    v256 onesV = new v256(1.0f);

                    // Use mm256_i32gather_ps to gather the data from several indices at once, which in turn allows
                    // the interpolation to be computed using the entire vector.
                    for (; outputCountW >= 8; outputCountW -= 8)
                    {
                        v256 indexW0V = mm256_loadu_si256(&indexTableW0[ow]);
                        v256 indexW1V = mm256_loadu_si256(&indexTableW1[ow]);
                        v256 fracW0V = mm256_loadu_ps(&fracTableW[ow]);
                        v256 fracW1V = mm256_sub_ps(onesV, fracW0V);

                        v256 input00V = mm256_i32gather_ps(Xph0, indexW0V, sizeof(float));
                        v256 input01V = mm256_i32gather_ps(Xph0, indexW1V, sizeof(float));
                        v256 input10V = mm256_i32gather_ps(Xph1, indexW0V, sizeof(float));
                        v256 input11V = mm256_i32gather_ps(Xph1, indexW1V, sizeof(float));

                        // Bilinear interpolation.
                        v256 temp0V = mm256_fmadd_ps(input01V, fracW0V, mm256_mul_ps(input00V, fracW1V));
                        v256 temp1V = mm256_fmadd_ps(input11V, fracW0V, mm256_mul_ps(input10V, fracW1V));
                        v256 outputV = mm256_fmadd_ps(temp1V, fracH0V, mm256_mul_ps(temp0V, fracH1V));

                        mm256_storeu_ps(Op, outputV);
                        Op += 8;
                        ow += 8;
                    }
                }

                for (; outputCountW > 0; outputCountW -= 1)
                {
                    int indexW0 = indexTableW0[ow];
                    int indexW1 = indexTableW1[ow];

                    float fracW0 = fracTableW[ow];
                    float fracW1 = 1.0f - fracW0;

                    float input00 = Xph0[indexW0];
                    float input01 = Xph0[indexW1];
                    float input10 = Xph1[indexW0];
                    float input11 = Xph1[indexW1];

                    // Bilinear interpolation.
                    float temp0 = math.mad(input01, fracW0, input00 * fracW1);
                    float temp1 = math.mad(input11, fracW0, input10 * fracW1);
                    float output = math.mad(temp1, fracH0, temp0 * fracH1);

                    *Op++ = (float)output;
                    ow++;
                }

                if (count > 0)
                {
                    // Output is now always aligned to the start of a row.
                    ow = 0;
                    outputWidthRemaining = outputWidth;

                    if (++oh == outputHeight)
                    {
                        // Advance to the next input channel.
                        oh = 0;
                        Xp += inputSize;
                    }
                }
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void ResizeLinear3D(int i, int count)
        {
            int inputSize = this.inputSize;
            int outputDepth = outputShape[0];
            int outputHeight = outputShape[1];
            int outputWidth = outputShape[2];

            // Compute the addresses of the tables contained in the table blob.
            void* nextTableData = (int*)B.ptr;
            int* indexTableD0 = TableDataAddr<int>(ref nextTableData, outputDepth);
            int* indexTableD1 = TableDataAddr<int>(ref nextTableData, outputDepth);
            float* fracTableD = TableDataAddr<float>(ref nextTableData, outputDepth);
            int* indexTableH0 = TableDataAddr<int>(ref nextTableData, outputHeight);
            int* indexTableH1 = TableDataAddr<int>(ref nextTableData, outputHeight);
            float* fracTableH = TableDataAddr<float>(ref nextTableData, outputHeight);
            int* indexTableW0 = TableDataAddr<int>(ref nextTableData, outputWidth);
            int* indexTableW1 = TableDataAddr<int>(ref nextTableData, outputWidth);
            float* fracTableW = TableDataAddr<float>(ref nextTableData, outputWidth);

            float* Op = Optr + i;

            // Extract the starting output position from the index.
            int ow = i % outputWidth;
            i = i / outputWidth;
            int oh = i % outputHeight;
            i = i / outputHeight;
            int od = i % outputDepth;
            i = i / outputDepth;

            // Advance to the starting input channel.
            float* Xp = Xptr + i * inputSize;

            int outputWidthRemaining = outputWidth - ow;

            while (count > 0)
            {
                int outputCountW = math.min(count, outputWidthRemaining);
                count -= outputCountW;

                float* Xpd0 = Xp + indexTableD0[od];
                float* Xpd1 = Xp + indexTableD1[od];

                float fracD0 = fracTableD[od];
                float fracD1 = 1.0f - fracD0;

                float* Xph00 = Xpd0 + indexTableH0[oh];
                float* Xph01 = Xpd0 + indexTableH1[oh];
                float* Xph10 = Xpd1 + indexTableH0[oh];
                float* Xph11 = Xpd1 + indexTableH1[oh];

                float fracH0 = fracTableH[oh];
                float fracH1 = 1.0f - fracH0;

                if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
                {
                    v256 fracD0V = new v256(fracD0);
                    v256 fracD1V = new v256(fracD1);
                    v256 fracH0V = new v256(fracH0);
                    v256 fracH1V = new v256(fracH1);
                    v256 onesV = new v256(1.0f);

                    // Use mm256_i32gather_ps to gather the data from several indices at once, which in turn allows
                    // the interpolation to be computed using the entire vector.
                    for (; outputCountW >= 8; outputCountW -= 8)
                    {
                        v256 indexW0V = mm256_loadu_si256(&indexTableW0[ow]);
                        v256 indexW1V = mm256_loadu_si256(&indexTableW1[ow]);
                        v256 fracW0V = mm256_loadu_ps(&fracTableW[ow]);
                        v256 fracW1V = mm256_sub_ps(onesV, fracW0V);

                        v256 input000V = mm256_i32gather_ps(Xph00, indexW0V, sizeof(float));
                        v256 input001V = mm256_i32gather_ps(Xph00, indexW1V, sizeof(float));
                        v256 input010V = mm256_i32gather_ps(Xph01, indexW0V, sizeof(float));
                        v256 input011V = mm256_i32gather_ps(Xph01, indexW1V, sizeof(float));

                        // Bilinear interpolation, plane 0.
                        v256 temp00V = mm256_fmadd_ps(input001V, fracW0V, mm256_mul_ps(input000V, fracW1V));
                        v256 temp01V = mm256_fmadd_ps(input011V, fracW0V, mm256_mul_ps(input010V, fracW1V));
                        v256 plane0V = mm256_fmadd_ps(temp01V, fracH0V, mm256_mul_ps(temp00V, fracH1V));

                        v256 input100V = mm256_i32gather_ps(Xph10, indexW0V, sizeof(float));
                        v256 input101V = mm256_i32gather_ps(Xph10, indexW1V, sizeof(float));
                        v256 input110V = mm256_i32gather_ps(Xph11, indexW0V, sizeof(float));
                        v256 input111V = mm256_i32gather_ps(Xph11, indexW1V, sizeof(float));

                        // Bilinear interpolation, plane 1.
                        v256 temp10V = mm256_fmadd_ps(input101V, fracW0V, mm256_mul_ps(input100V, fracW1V));
                        v256 temp11V = mm256_fmadd_ps(input111V, fracW0V, mm256_mul_ps(input110V, fracW1V));
                        v256 plane1V = mm256_fmadd_ps(temp11V, fracH0V, mm256_mul_ps(temp10V, fracH1V));

                        // Trilinear interpolation.
                        v256 outputV = mm256_fmadd_ps(plane1V, fracD0V, mm256_mul_ps(plane0V, fracD1V));

                        mm256_storeu_ps(Op, outputV);
                        Op += 8;
                        ow += 8;
                    }
                }

                for (; outputCountW > 0; outputCountW -= 1)
                {
                    int indexW0 = indexTableW0[ow];
                    int indexW1 = indexTableW1[ow];

                    float fracW0 = fracTableW[ow];
                    float fracW1 = 1.0f - fracW0;

                    float input000 = Xph00[indexW0];
                    float input001 = Xph00[indexW1];
                    float input010 = Xph01[indexW0];
                    float input011 = Xph01[indexW1];

                    // Bilinear interpolation, plane 0.
                    float temp00 = math.mad(input001, fracW0, input000 * fracW1);
                    float temp01 = math.mad(input011, fracW0, input010 * fracW1);
                    float plane0 = math.mad(temp01, fracH0, temp00 * fracH1);

                    float input100 = Xph10[indexW0];
                    float input101 = Xph10[indexW1];
                    float input110 = Xph11[indexW0];
                    float input111 = Xph11[indexW1];

                    // Bilinear interpolation, plane 1.
                    float temp10 = math.mad(input101, fracW0, input100 * fracW1);
                    float temp11 = math.mad(input111, fracW0, input110 * fracW1);
                    float plane1 = math.mad(temp11, fracH0, temp10 * fracH1);

                    // Trilinear interpolation.
                    float output = math.mad(plane1, fracD0, plane0 * fracD1);

                    *Op++ = (float)output;
                    ow++;
                }

                if (count > 0)
                {
                    // Output is now always aligned to the start of a row.
                    ow = 0;
                    outputWidthRemaining = outputWidth;

                    if (++oh == outputHeight)
                    {
                        oh = 0;

                        if (++od == outputDepth)
                        {
                            // Advance to the next input channel.
                            od = 0;
                            Xp += inputSize;
                        }
                    }
                }
            }
        }

        static T* TableDataAddr<T>(ref void* nextTableData, int count) where T : unmanaged
        {
            T* address = (T*)nextTableData;
            nextTableData = address + count;
            return address;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ResizeNearestInitTablesJob : IJob, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; }
        public int spatialDims;
        public fixed int inputShape[ResizeNearestJob.maxSpatialDims];
        public fixed int outputShape[ResizeNearestJob.maxSpatialDims];
        public fixed float scales[ResizeNearestJob.maxSpatialDims];
        public fixed float biases[ResizeNearestJob.maxSpatialDims];
        public Layers.NearestMode nearestMode;

        public void Execute()
        {
            int* buffer = (int*)O.ptr;

            for (int i = 0; i < spatialDims; i++)
            {
                int inputDim = inputShape[i];
                int outputDim = outputShape[i];
                float scale = scales[i];
                float bias = biases[i];

                int* indexTable = buffer;
                buffer += outputDim;

                // Scale the indices with the input stride.
                int inputStride = 1;
                for (int j = i + 1; j < spatialDims; j++)
                    inputStride *= inputShape[j];

                for (int j = 0; j < outputDim; j++)
                {
                    float inputCoord = math.max(0.0f, (float)j * scale + bias);

                    int indexValue = 0;
                    switch (nearestMode)
                    {
                        case Layers.NearestMode.Floor:
                            indexValue = (int)math.floor(inputCoord);
                            break;
                        case Layers.NearestMode.Ceil:
                            indexValue = (int)math.ceil(inputCoord);
                            break;
                    }
                    indexValue = math.clamp(indexValue, 0, inputShape[i] - 1);
                    indexTable[j] = inputStride * indexValue;
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ResizeNearestJob : IParallelForBatch, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        internal const int maxSpatialDims = 3;

        public int spatialDims;
        public int inputSize;
        public fixed int outputShape[ResizeNearestJob.maxSpatialDims];

        public void Execute(int i, int count)
        {
            if (spatialDims == 2)
                ResizeNearest2D(i, count);
            else
                ResizeNearest3D(i, count);
        }

        static float* ResizeRow([NoAlias] float* Xph, [NoAlias] float* Op, [NoAlias] int* indexTableW, int ow, int outputCountW)
        {
            if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                for (; outputCountW >= 8; outputCountW -= 8)
                {
                    // Use mm256_i32gather_ps to gather the data from several indices at once.
                    mm256_storeu_ps(Op, mm256_i32gather_ps(Xph, mm256_loadu_si256(&indexTableW[ow]), sizeof(float)));
                    Op += 8;
                    ow += 8;
                }
            }

            for (; outputCountW > 0; outputCountW -= 1)
            {
                *Op++ = Xph[indexTableW[ow]];
                ow++;
            }

            return Op;
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void ResizeNearest2D(int i, int count)
        {
            int inputSize = this.inputSize;
            int outputHeight = outputShape[0];
            int outputWidth = outputShape[1];

            // Compute the addresses of the tables contained in the table blob.
            void* nextTableData = (int*)B.ptr;
            int* indexTableH = TableDataAddr<int>(ref nextTableData, outputHeight);
            int* indexTableW = TableDataAddr<int>(ref nextTableData, outputWidth);

            float* Op = Optr + i;

            // Extract the starting output position from the index.
            int ow = i % outputWidth;
            i = i / outputWidth;
            int oh = i % outputHeight;
            i = i / outputHeight;

            // Advance to the starting input channel.
            float* Xp = Xptr + i * inputSize;

            int outputWidthRemaining = outputWidth - ow;

            while (count > 0)
            {
                int outputCountW = math.min(count, outputWidthRemaining);
                count -= outputCountW;

                Op = ResizeRow(Xp + indexTableH[oh], Op, indexTableW, ow, outputCountW);

                if (count > 0)
                {
                    // Output is now always aligned to the start of a row.
                    ow = 0;
                    outputWidthRemaining = outputWidth;

                    if (++oh == outputHeight)
                    {
                        // Advance to the next input channel.
                        oh = 0;
                        Xp += inputSize;
                    }
                }
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void ResizeNearest3D(int i, int count)
        {
            int inputSize = this.inputSize;
            int outputDepth = outputShape[0];
            int outputHeight = outputShape[1];
            int outputWidth = outputShape[2];

            // Compute the addresses of the tables contained in the table blob.
            void* nextTableData = (int*)B.ptr;
            int* indexTableD = TableDataAddr<int>(ref nextTableData, outputDepth);
            int* indexTableH = TableDataAddr<int>(ref nextTableData, outputHeight);
            int* indexTableW = TableDataAddr<int>(ref nextTableData, outputWidth);

            float* Op = Optr + i;

            // Extract the starting output position from the index.
            int ow = i % outputWidth;
            i = i / outputWidth;
            int oh = i % outputHeight;
            i = i / outputHeight;
            int od = i % outputDepth;
            i = i / outputDepth;

            // Advance to the starting input channel.
            float* Xp = Xptr + i * inputSize;

            int outputWidthRemaining = outputWidth - ow;

            while (count > 0)
            {
                int outputCountW = math.min(count, outputWidthRemaining);
                count -= outputCountW;

                Op = ResizeRow(Xp + indexTableD[od] + indexTableH[oh], Op, indexTableW, ow, outputCountW);

                if (count > 0)
                {
                    // Output is now always aligned to the start of a row.
                    ow = 0;
                    outputWidthRemaining = outputWidth;

                    if (++oh == outputHeight)
                    {
                        oh = 0;

                        if (++od == outputDepth)
                        {
                            // Advance to the next input channel.
                            od = 0;
                            Xp += inputSize;
                        }
                    }
                }
            }
        }

        static T* TableDataAddr<T>(ref void* nextTableData, int count) where T : unmanaged
        {
            T* address = (T*)nextTableData;
            nextTableData = address + count;
            return address;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct UpsampleNearest2DJob : IParallelForBatch, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;
        public int inputHeight;
        public int inputWidth;
        public int scaleHeight;
        public int scaleWidth;

        public void Execute(int i, int count)
        {
            float* Xp = Xptr + i;

            // Extract the starting input position from the index.
            int iw = i % inputWidth;
            i = i / inputWidth;
            int ih = i % inputHeight;
            i = i / inputHeight;

            int outputHeight = inputHeight * scaleHeight;
            int outputWidth = inputWidth * scaleWidth;

            float* Op = Optr + (i * outputHeight + ih * scaleHeight) * outputWidth + iw * scaleWidth;
            var nstrideOutput = new System.IntPtr(outputWidth);
            int inputWidthRemaining = inputWidth - iw;

            while (count > 0)
            {
                int inputCountW = math.min(count, inputWidthRemaining);
                count -= inputCountW;

                if (scaleHeight == 2 && scaleWidth == 2)
                    UpsampleRow_Scale2x2(ref Xp, inputCountW, ref Op, nstrideOutput);
                else
                    UpsampleRow_ScaleNxN(ref Xp, inputCountW, ref Op, nstrideOutput);

                inputWidthRemaining = inputWidth;
            }
        }

        void UpsampleRow_Scale2x2(ref float* Xp, int inputCountW, ref float* Op, System.IntPtr nstrideOutput)
        {
            for (; inputCountW >= 4; inputCountW -= 4)
            {
                // Shuffle four input elements to eight output elements.
                float4 inputV = *((float4*)&Xp[0]);
                Xp += 4;

                *(float4*)StrideAddress(&Op[0], nstrideOutput, 0) = inputV.xxyy;
                *(float4*)StrideAddress(&Op[4], nstrideOutput, 0) = inputV.zzww;
                *(float4*)StrideAddress(&Op[0], nstrideOutput, 1) = inputV.xxyy;
                *(float4*)StrideAddress(&Op[4], nstrideOutput, 1) = inputV.zzww;
                Op += 8;
            }

            for (; inputCountW > 0; inputCountW -= 1)
            {
                // Shuffle one input element to two output elements.
                float2 inputV = new float2(Xp[0]);
                Xp += 1;

                *(float2*)StrideAddress(&Op[0], nstrideOutput, 0) = inputV;
                *(float2*)StrideAddress(&Op[0], nstrideOutput, 1) = inputV;
                Op += 2;
            }

            Op = StrideAddress(Op, nstrideOutput, 1);
        }

        void UpsampleRow_ScaleNxN(ref float* Xp, int inputCountW, ref float* Op, System.IntPtr nstrideOutput)
        {
            float* initialOp = Op;

            do
            {
                // Replicate one input element to scaleWidth output elements.
                int outputCountW = scaleWidth;
                float4 inputV = new float4(Xp[0]);
                Xp += 1;

                for (; outputCountW >= 4; outputCountW -= 4)
                {
                    *(float4*)Op = inputV;
                    Op += 4;
                }

                if ((outputCountW & 2) != 0)
                {
                    *(float2*)Op = inputV.xy;
                    Op += 2;
                }

                if ((outputCountW & 1) != 0)
                {
                    *Op++ = inputV.x;
                }
            }
            while (--inputCountW != 0);

            // Replicate the first row of output elements to the remaining scaleHeight rows.
            var copyOutputCountW = Op - initialOp;
            for (int h = 1; h < scaleHeight; h++)
            {
                Op = StrideAddress(Op, nstrideOutput, 1);
                UnsafeUtility.MemCpy(Op - copyOutputCountW, initialOp, copyOutputCountW * sizeof(float));
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ConvTransposeJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public bool useBias;
        const int k_MaxSpatialDims = 2;

        public int offsetO;
        public int inputSize;
        public int kernelSize;
        public int outputSize;
        fixed int inputShape[k_MaxSpatialDims];
        fixed int kernelShape[k_MaxSpatialDims];
        fixed int outputShape[k_MaxSpatialDims];
        fixed int stride[k_MaxSpatialDims];
        fixed int padLeft[k_MaxSpatialDims];
        fixed int dilation[k_MaxSpatialDims];

        public void Prepare(TensorShape shapeX, TensorShape shapeW, TensorShape shapeO, Span<int> stride, Span<int> pad)
        {
            var spatialDims = shapeX.rank - 2;

            // Implement ConvTranspose1D using the ConvTranspose2D path by unsqueezing the shapes.
            if (spatialDims == 1)
            {
                inputShape[0] = 1;
                inputShape[1] = shapeX[2];
                kernelShape[0] = 1;
                kernelShape[1] = shapeW[2];
                outputShape[0] = 1;
                outputShape[1] = shapeO[2];
                this.stride[0] = 1;
                this.stride[1] = stride[0];
                padLeft[0] = 0;
                padLeft[1] = pad[0];
            }
            else
            {
                inputShape[0] = shapeX[2];
                inputShape[1] = shapeX[3];
                kernelShape[0] = shapeW[2];
                kernelShape[1] = shapeW[3];
                outputShape[0] = shapeO[2];
                outputShape[1] = shapeO[3];
                this.stride[0] = stride[0];
                this.stride[1] = stride[1];
                padLeft[0] = pad[0];
                padLeft[1] = pad[1];
            }
            dilation[0] = 1;
            dilation[1] = 1;

            inputSize = inputShape[0] * inputShape[1];
            kernelSize = kernelShape[0] * kernelShape[1];
            outputSize = outputShape[0] * outputShape[1];
        }

        public void Execute(int i)
        {
            float* Xp = Xptr + i * inputSize * kernelSize;
            float* Bp = useBias ? Bptr + i : null;
            float* Op = Optr + offsetO + i * outputSize;

            if (useBias)
            {
                // Initialize the output image with the channel bias.
                for (int ohw = 0; ohw < outputSize; ohw++)
                    Op[ohw] = Bp[0];
            }
            else
            {
                UnsafeUtility.MemClear(destination: Op, size: outputSize * sizeof(float));
            }

            // Perform a col2im operation (see https://onnx.ai/onnx/operators/onnx__Col2Im.html).
            // The column buffer has the format: [kernelHeight][kernelWidth][inputHeight][inputWidth].
            // Walk over each column buffer element and accumulate into the output image.
            for (int kh = 0; kh < kernelShape[0]; kh++)
            {
                for (int kw = 0; kw < kernelShape[1]; kw++)
                {
                    for (int ih = 0; ih < inputShape[0]; ih++)
                    {
                        int oh = ih * stride[0] + kh * dilation[0] - padLeft[0];

                        if ((uint)oh < (uint)outputShape[0])
                        {
                            for (int iw = 0; iw < inputShape[1]; iw++)
                            {
                                int ow = iw * stride[1] + kw * dilation[1] - padLeft[1];

                                if ((uint)ow < (uint)outputShape[1])
                                {
                                    Op[oh * outputShape[1] + ow] += Xp[0];
                                }

                                Xp += 1;
                            }
                        }
                        else
                        {
                            Xp += inputShape[1];
                        }
                    }
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScatterNDFloatJob : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public Layers.ScatterReductionMode reduction;
        public int updatesLength;
        public int indexRemapDim;
        public fixed int trailing[8];
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource S { get; set; } int* Sptr => (int*)S.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public void Execute(int threadIdx)
        {
            int k = threadIdx % updatesLength;
            int i = threadIdx / updatesLength;

            int indexO = 0;
            for (int j = 0; j < indexRemapDim; j++)
            {
                indexO += trailing[j] * (int)Sptr[i * indexRemapDim + j];
            }
            float vw = Bptr[i * updatesLength + k];

            if (reduction == Layers.ScatterReductionMode.None)
                Optr[indexO * updatesLength + k] = vw;
            else if (reduction == Layers.ScatterReductionMode.Add)
                Optr[indexO * updatesLength + k] += vw;
            else if (reduction == Layers.ScatterReductionMode.Mul)
                Optr[indexO * updatesLength + k] *= vw;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScatterNDIntJob : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public Layers.ScatterReductionMode reduction;
        public int updatesLength;
        public int indexRemapDim;
        public fixed int trailing[8];
        public ReadOnlyMemResource X { get; set; } int* Xptr => (int*)X.ptr;
        public ReadOnlyMemResource S { get; set; } int* Sptr => (int*)S.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } int* Optr => (int*)O.ptr;

        public void Execute(int threadIdx)
        {
            int k = threadIdx % updatesLength;
            int i = threadIdx / updatesLength;

            int indexO = 0;
            for (int j = 0; j < indexRemapDim; j++)
            {
                indexO += trailing[j] * (int)Sptr[i * indexRemapDim + j];
            }
            int vw = Bptr[i * updatesLength + k];

            if (reduction == Layers.ScatterReductionMode.None)
                Optr[indexO * updatesLength + k] = vw;
            else if (reduction == Layers.ScatterReductionMode.Add)
                Optr[indexO * updatesLength + k] += vw;
            else if (reduction == Layers.ScatterReductionMode.Mul)
                Optr[indexO * updatesLength + k] *= vw;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TopKJob : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][Collections.ReadOnly] public float* Xptr;
        [NoAlias][NativeDisableUnsafePtrRestriction] public float* Valuesptr;
        [NoAlias][NativeDisableUnsafePtrRestriction] public int* Indicesptr;

        public int reduceLength;
        public int innerLength;
        public int maxK;
        public int direction;

        public void Execute(int index)
        {
            Hint.Assume(reduceLength > 0);
            Hint.Assume(innerLength > 0);

            int innerIndex = index % innerLength;
            int outerIndex = index / innerLength;

            float* Xp = Xptr + outerIndex * reduceLength * innerLength + innerIndex;
            float* Vp = Valuesptr + outerIndex * maxK * innerLength + innerIndex;
            int* Ip = Indicesptr + outerIndex * maxK * innerLength + innerIndex;

            // insertion sort + binary search index
            // https://en.wikipedia.org/wiki/Insertion_sort#Variants
            // O(k*n) complexity, fine for small k but should have other algorithm for k ~ n
            // output is always sorted, for each values of v, find index to insert
            // keep track of min-k to early out
            // shift all top-k starting from the selected index, insert v
            float value = Xp[0];
            Vp[0] = value;
            Ip[0] = 0;
            float minOValue = direction * value;
            int k = 1;
            for (int j = 1; j < reduceLength; j++)
            {
                Xp += innerLength;
                value = Xp[0];

                if (k == maxK && direction * value <= minOValue)
                    continue;

                k = math.min(k + 1, maxK);
                int idxLast = (k - 1) * innerLength;
                Vp[idxLast] = value;
                Ip[idxLast] = j;

                for (int i = k - 1; i > 0; i--)
                {
                    int idx = i * innerLength;
                    int idxPrev = (i - 1) * innerLength;

                    float swapf = Vp[idx];

                    if (direction * Vp[idxPrev] >= direction * swapf)
                        break;

                    int swapi = Ip[idx];

                    Vp[idx] = Vp[idxPrev];
                    Vp[idxPrev] = swapf;

                    Ip[idx] = Ip[idxPrev];
                    Ip[idxPrev] = swapi;
                }
                minOValue = math.min(direction * Vp[(k - 1) * innerLength], value);
            }
        }
    }
}
}
