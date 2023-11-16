using UnityEngine;
using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

namespace Unity.Sentis {

static class BurstSchedulingHelper
{
    static unsafe JobHandle ScheduleXSBWOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrS,
        void* ptrB,
        void* ptrW,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXSBWO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        Logger.AssertIsTrue(ptrO != ptrS, "BurstCPU.ScheduleJob.ValueError: output must be different from inputS");
        Logger.AssertIsTrue(ptrO != ptrB, "BurstCPU.ScheduleJob.ValueError: output must be different from inputB");
        Logger.AssertIsTrue(ptrO != ptrW, "BurstCPU.ScheduleJob.ValueError: output must be different from inputW");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.S = new CPUBackend.ReadOnlyMemResource() { ptr = ptrS };
        jobData.B = new CPUBackend.ReadOnlyMemResource() { ptr = ptrB };
        jobData.W = new CPUBackend.ReadOnlyMemResource() { ptr = ptrW };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleXSBOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrS,
        void* ptrB,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXSBO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        Logger.AssertIsTrue(ptrO != ptrS, "BurstCPU.ScheduleJob.ValueError: output must be different from inputS");
        Logger.AssertIsTrue(ptrO != ptrB, "BurstCPU.ScheduleJob.ValueError: output must be different from inputB");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.S = new CPUBackend.ReadOnlyMemResource() { ptr = ptrS };
        jobData.B = new CPUBackend.ReadOnlyMemResource() { ptr = ptrB };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleXBOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrB,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXBO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        Logger.AssertIsTrue(ptrO != ptrB, "BurstCPU.ScheduleJob.ValueError: output must be different from inputB");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.B = new CPUBackend.ReadOnlyMemResource() { ptr = ptrB };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleBatchXBOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrB,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IParallelForBatch, CPUBackend.IJobResourceDeclarationXBO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        Logger.AssertIsTrue(ptrO != ptrB, "BurstCPU.ScheduleJob.ValueError: output must be different from inputB");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.B = new CPUBackend.ReadOnlyMemResource() { ptr = ptrB };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.ScheduleBatch(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleBatchXSBOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrS,
        void* ptrB,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IParallelForBatch, CPUBackend.IJobResourceDeclarationXSBO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        Logger.AssertIsTrue(ptrO != ptrB, "BurstCPU.ScheduleJob.ValueError: output must be different from inputB");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.S = new CPUBackend.ReadOnlyMemResource() { ptr = ptrS };
        jobData.B = new CPUBackend.ReadOnlyMemResource() { ptr = ptrB };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.ScheduleBatch(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleXOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleBatchXOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IParallelForBatch, CPUBackend.IJobResourceDeclarationXO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.ScheduleBatch(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleXOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrO)
        where T : struct, IJob, CPUBackend.IJobResourceDeclarationXO
    {
        Logger.AssertIsTrue(ptrO != ptrX, "BurstCPU.ScheduleJob.ValueError: output must be different from inputX");
        jobData.X = new CPUBackend.ReadOnlyMemResource() { ptr = ptrX };
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.Schedule(fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrO)
        where T : struct, IJob, CPUBackend.IJobResourceDeclarationO
    {
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.Schedule(fenceBeforeJobStart);
    }

    static unsafe JobHandle ScheduleOInternal<T>(ref this T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationO
    {
        jobData.O = new CPUBackend.ReadWriteMemResource() { ptr = ptrO };
        return jobData.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    static unsafe JobHandle GetFenceBeforeJobStartXSBWO(
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinS,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinW,
        IDependableMemoryResource pinO)
    {
        JobHandle* jobs = stackalloc JobHandle[5] { pinX.fence, pinS.fence, pinB.fence, pinW.fence, pinO.reuse };
        return JobHandleUnsafeUtility.CombineDependencies(jobs, 5);
    }

    static unsafe JobHandle GetFenceBeforeJobStartXSBO(
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinS,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        JobHandle* jobs = stackalloc JobHandle[4] { pinX.fence, pinS.fence, pinB.fence, pinO.reuse };
        return JobHandleUnsafeUtility.CombineDependencies(jobs, 4);
    }

    static JobHandle GetFenceBeforeJobStartXBO(
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        return JobHandle.CombineDependencies(pinX.fence, pinB.fence, pinO.reuse);
    }

    static JobHandle GetFenceBeforeJobStartXBO(
        BurstTensorData pinX,
        BurstTensorData pinB,
        BurstTensorData pinO)
    {
        return JobHandle.CombineDependencies(pinX.fence, pinB.fence, pinO.reuse);
    }

    static JobHandle GetFenceBeforeJobStartXO(
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinO)
    {
        return JobHandle.CombineDependencies(pinX.fence, pinO.reuse);
    }

    static void SetXSBWOFences(this JobHandle jobFence,
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinS,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinW,
        IDependableMemoryResource pinO)
    {
        pinX.reuse = jobFence;
        pinS.reuse = jobFence;
        pinB.reuse = jobFence;
        pinW.reuse = jobFence;
        pinO.fence = jobFence;
    }

    static void SetXSBOFences(this JobHandle jobFence,
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinS,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        pinX.reuse = jobFence;
        pinS.reuse = jobFence;
        pinB.reuse = jobFence;
        pinO.fence = jobFence;
    }

    static void SetXBOFences(this JobHandle jobFence,
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        pinX.reuse = jobFence;
        pinB.reuse = jobFence;
        pinO.fence = jobFence;
    }

    static void SetXOFences(this JobHandle jobFence,
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinO)
    {
        pinX.reuse = jobFence;
        pinO.fence = jobFence;
    }

    public static unsafe JobHandle ScheduleXSBWO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource S,
        IDependableMemoryResource B,
        IDependableMemoryResource W,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXSBWO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXSBWO(X, S, B, W, O);
        var jobFence = jobData.ScheduleXSBWOInternal(fenceBeforeJobStart, X.rawPtr, S.rawPtr, B.rawPtr, W.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        jobFence.SetXSBWOFences(X, S, B, W, O);
        return jobFence;
    }

    public static unsafe JobHandle ScheduleXSBO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource S,
        IDependableMemoryResource B,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXSBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXSBO(X, S, B, O);
        var jobFence = jobData.ScheduleXSBOInternal(fenceBeforeJobStart, X.rawPtr, S.rawPtr, B.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        jobFence.SetXSBOFences(X, S, B, O);
        return jobFence;
    }

    public static unsafe JobHandle ScheduleXBO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource B,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXBO(X, B, O);
        var jobFence = jobData.ScheduleXBOInternal(fenceBeforeJobStart, X.rawPtr, B.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        jobFence.SetXBOFences(X, B, O);
        return jobFence;
    }

    public static unsafe JobHandle ScheduleBatchXBO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource B,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IParallelForBatch, CPUBackend.IJobResourceDeclarationXBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXBO(X, B, O);
        var jobFence = jobData.ScheduleBatchXBOInternal(fenceBeforeJobStart, X.rawPtr, B.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        jobFence.SetXBOFences(X, B, O);
        return jobFence;
    }

    public static unsafe JobHandle ScheduleBatchXSBO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource S,
        IDependableMemoryResource B,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IParallelForBatch, CPUBackend.IJobResourceDeclarationXSBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXSBO(X, S, B, O);
        var jobFence = jobData.ScheduleBatchXSBOInternal(fenceBeforeJobStart, X.rawPtr, S.rawPtr, B.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        jobFence.SetXSBOFences(X, S, B, O);
        return jobFence;
    }

    public static unsafe JobHandle ScheduleO<T>(ref this T jobData,
        IDependableMemoryResource O)
        where T : struct, IJob, CPUBackend.IJobResourceDeclarationO
    {
        var fenceBeforeJobStart = O.reuse;
        var jobFence = jobData.ScheduleOInternal(fenceBeforeJobStart, O.rawPtr);
        O.fence = jobFence;
        return jobFence;
    }

    public static unsafe JobHandle ScheduleXO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(X, O);
        var jobFence = jobData.ScheduleXOInternal(fenceBeforeJobStart, X.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        jobFence.SetXOFences(X, O);
        return jobFence;
    }

    public static unsafe JobHandle ScheduleBatchXO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IParallelForBatch, CPUBackend.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(X, O);
        var jobFence = jobData.ScheduleBatchXOInternal(fenceBeforeJobStart, X.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        jobFence.SetXOFences(X, O);
        return jobFence;
    }

    public static unsafe JobHandle ScheduleO<T>(ref this T jobData,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, CPUBackend.IJobResourceDeclarationO
    {
        var fenceBeforeJobStart = pinO.reuse;
        var jobFence = jobData.ScheduleOInternal(fenceBeforeJobStart, pinO.rawPtr, arrayLength, innerloopBatchCount);
        pinO.fence = jobFence;
        return jobFence;
    }

    public static unsafe JobHandle ScheduleXO<T>(ref this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource O)
        where T : struct, IJob, CPUBackend.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(X, O);
        var jobFence = jobData.ScheduleXOInternal(fenceBeforeJobStart, X.rawPtr, O.rawPtr);
        jobFence.SetXOFences(X, O);
        return jobFence;
    }
}

unsafe class FencedMemoryAlloc : IDependableMemoryResource
{
    JobHandle m_ReadFence;
    JobHandle m_WriteFence;
    void* data;
    public void* rawPtr => data;
    public int elementCount;
    public int elementSize;

    /// <inheritdoc/>
    public JobHandle fence { get { return m_ReadFence; }  set { m_ReadFence = value; m_WriteFence = value; } }

    /// <inheritdoc/>
    public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = value; } }

    public void Allocate(int numElement, int elementSize, Allocator allocator)
    {
        m_ReadFence = new JobHandle();
        m_WriteFence = new JobHandle();
        elementCount = numElement;
        this.elementSize = elementSize;
        Logger.AssertIsTrue(data == null, "FencedMemoryAlloc.Error: Please call ClearState() when freeing underlying memory.");
        data = UnsafeUtility.Malloc(elementCount * elementSize, elementSize, allocator);
        Logger.AssertIsTrue(data != null, "FencedMemoryAlloc.Error: failed to allocate data memory");
    }

    public FencedMemoryAlloc(int numElement, int elementSize, Allocator allocator)
    {
        m_ReadFence = new JobHandle();
        m_WriteFence = new JobHandle();
        elementCount = numElement;
        this.elementSize = elementSize;
        data = UnsafeUtility.Malloc(elementCount * elementSize, elementSize, allocator);
        Logger.AssertIsTrue(data != null, "FencedMemoryAlloc.Error: failed to allocate data memory");
    }

    public void ClearState()
    {
        m_ReadFence = new JobHandle();
        m_WriteFence = new JobHandle();
        elementCount = 0;
        elementSize = 0;
        data = null;
    }

    public FencedMemoryAlloc() { }
}

public partial class CPUBackend
{
    /// <summary>
    /// Utility structure to iterate the tensor offset for a broadcast operation.
    /// Based on the broadcast support from Microsoft's ONNX Runtime.
    /// </summary>
    internal unsafe struct BroadcastIterator
    {
        fixed int counts[TensorShape.maxRank];
        fixed int deltas[TensorShape.maxRank];
        uint lastIndex;
        int totalCount;

        /// <summary>
        /// Appends next innermost 'dim' to the iterator state.
        /// </summary>
        internal void AppendDim(int dim, int dimLargest)
        {
            if (lastIndex == 0)
            {
                deltas[0] = (dim > 1) ? 1 : 0;
                counts[0] = dimLargest;
                totalCount = dim;
                lastIndex = 1;
            }
            else
            {
                if (dim == 1)
                {
                    if (deltas[lastIndex - 1] > 0)
                    {
                        // Transition to stop broadcasting from this tensor.
                        deltas[lastIndex] = -totalCount;
                        counts[lastIndex] = 1;
                        lastIndex++;
                    }
                }
                else
                {
                    if (deltas[lastIndex - 1] <= 0)
                    {
                        // Transition to start broadcasting from this tensor.
                        deltas[lastIndex] = totalCount;
                        counts[lastIndex] = 1;
                        lastIndex++;
                    }
                }

                counts[lastIndex - 1] *= dimLargest;
                totalCount *= dim;
            }
        }

        /// <summary>
        /// Computes the input tensor offset given the starting output tensor index.
        /// </summary>
        internal int ComputeOffset(int i)
        {
            int offset = 0;

            for (uint j = 0; i > 0; j++)
            {
                offset += i * deltas[j];
                i = i / counts[j];
            }

            return offset;
        }

        /// <summary>
        /// Initializes the input tensor offset given the starting output tensor index.
        /// Assumes 'counters' is zero-initialized.
        /// </summary>
        internal int InitialOffset(int i, int *counters)
        {
            int offset = 0;

            for (uint j = 0; i > 0; j++)
            {
                offset += i * deltas[j];
                counters[j] = i % counts[j];
                i = i / counts[j];
            }

            return offset;
        }

        /// <summary>
        /// Advances the input tensor offset by 'count' elements. Assumes that 'count' does not
        /// overflow the innermost span size.
        /// </summary>
        internal int AdvanceOffset(int offset, int count, int* counters)
        {
            offset += count * deltas[0];
            counters[0] += count;

            if (counters[0] == counts[0])
            {
                // Propagate the carry through the counters.
                for (uint j = 1; j < lastIndex; j++)
                {
                    offset += deltas[j];
                    counters[j - 1] = 0;
                    if (++counters[j] != counts[j])
                        break;
                }
            }

            return offset;
        }

        /// <summary>
        /// Returns the number of elements that can be operated on sequentially at the innermost span.
        /// Ex: (4,64) -> 64
        /// </summary>
        internal int SpanSize()
        {
            return counts[0];
        }

        /// <summary>
        /// Returns true if the tensor is broadcasting a scalar input element to multiple output elements
        /// of a span, else false that the tensor is broadcasting a vector of input elements.
        /// </summary>
        internal bool IsScalarBroadcast()
        {
            return deltas[0] == 0;
        }
    }

    /// <summary>
    /// Utility structure to wrap broadcasting two inputs tensors to an output tensor.
    /// </summary>
    internal unsafe struct BroadcastHelperXBO
    {
        public BroadcastIterator iteratorX;
        public BroadcastIterator iteratorB;

        internal int Prepare(TensorShape shapeX, TensorShape shapeB)
        {
            bool appendScalarDim = true;
            int rank = Math.Max(shapeX.rank, shapeB.rank);
            int outputLength = 1;

            for (int i = 0; i < rank; i++)
            {
                int dimX = (i < shapeX.rank) ? shapeX.UnsafeGet(TensorShape.maxRank - i - 1) : 1;
                int dimB = (i < shapeB.rank) ? shapeB.UnsafeGet(TensorShape.maxRank - i - 1) : 1;

                int dimLargest = Math.Max(dimX, dimB);
                outputLength *= dimLargest;

                if (dimLargest > 1)
                {
                    iteratorX.AppendDim(dimX, dimLargest);
                    iteratorB.AppendDim(dimB, dimLargest);
                    appendScalarDim = false;
                }
            }

            if (appendScalarDim)
            {
                iteratorX.AppendDim(1, 1);
                iteratorB.AppendDim(1, 1);
            }

            return outputLength;
        }
    }

    /// <summary>
    /// Utility structure to wrap the batch matrix multiplication operation.
    /// </summary>
    internal unsafe struct BatchMatrixMultiplyHelper
    {
        public BroadcastIterator iteratorA;
        public BroadcastIterator iteratorB;
        public int batchCount;
        [NoAlias][NativeDisableUnsafePtrRestriction] public unsafe float* A, B, C;
        public int M, N, K;
        public int lda, ldb, ldc;
        public bool transposeA, transposeB;
        public bool accumulateC;

        internal void Prepare(TensorShape shapeA, TensorShape shapeB)
        {
            bool appendScalarDim = true;
            int rank = Math.Max(shapeA.rank, shapeB.rank);
            int batchCount = 1;

            for (int i = 2; i < rank; i++)
            {
                int dimA = (i < shapeA.rank) ? shapeA.UnsafeGet(TensorShape.maxRank - i - 1) : 1;
                int dimB = (i < shapeB.rank) ? shapeB.UnsafeGet(TensorShape.maxRank - i - 1) : 1;

                int dimLargest = Math.Max(dimA, dimB);
                batchCount *= dimLargest;

                if (dimLargest > 1)
                {
                    iteratorA.AppendDim(dimA, dimLargest);
                    iteratorB.AppendDim(dimB, dimLargest);
                    appendScalarDim = false;
                }
            }

            if (appendScalarDim)
            {
                iteratorA.AppendDim(1, 1);
                iteratorB.AppendDim(1, 1);
            }

            this.batchCount = batchCount;
        }
    }

    /// <summary>
    /// Utility structure to iterate the tensor offset for a transpose operation.
    /// </summary>
    internal unsafe struct TransposeIterator
    {
        fixed int counts[TensorShape.maxRank];
        fixed int strides[TensorShape.maxRank];
        uint lastIndex;

        /// <summary>
        /// Initializes the input tensor offset given the starting output tensor index.
        /// Assumes 'counters' is zero-initialized.
        /// </summary>
        internal int InitialOffset(int i, int *counters)
        {
            int offset = 0;

            for (uint j = 0; i > 0; j++)
            {
                counters[j] = i % counts[j];
                offset += counters[j] * strides[j];
                i = i / counts[j];
            }

            return offset;
        }

        /// <summary>
        /// Advances the input tensor offset by 'count' elements. Assumes that 'count' does not
        /// overflow the innermost span size.
        /// </summary>
        internal int AdvanceOffset(int offset, int count, int* counters)
        {
            offset += count * strides[0];
            counters[0] += count;

            if (counters[0] == counts[0])
            {
                // Propagate the carry through the counters.
                for (uint j = 1; j < lastIndex; j++)
                {
                    offset += strides[j] - (counts[j - 1] * strides[j - 1]);
                    counters[j - 1] = 0;
                    if (++counters[j] != counts[j])
                        break;
                }
            }

            return offset;
        }

        /// <summary>
        /// Returns the number of elements at the innermost span of the output tensor.
        /// Ex: (4,64) -> 64
        /// </summary>
        internal int InnerSpanSize()
        {
            return counts[0];
        }

        /// <summary>
        /// Returns the input tensor stride for the innermost span.
        /// Ex: (4,64) -> 64
        /// </summary>
        internal int InnerStride()
        {
            return strides[0];
        }

        internal void Prepare(TensorShape shapeX, int[] permutations = null)
        {
            // Compute the strides of the input tensor.
            int* stridesX = stackalloc int[TensorShape.maxRank];
            for (int i = shapeX.rank - 1, outputLength = 1; i >= 0; i--)
            {
                stridesX[i] = outputLength;
                outputLength *= shapeX[i];
            }

            for (int i = shapeX.rank - 1, lastPermutation = 0; i >= 0; i--)
            {
                int permutation = permutations?[i] ?? (shapeX.rank - i - 1);
                int dim = shapeX[permutation];

                // If the last permutation index is contiguous with this permutation index, then
                // flatten the transposed shape to reduce the number of index computations in the
                // kernel.
                if (lastIndex > 0 && lastPermutation == permutation + 1)
                {
                    counts[lastIndex - 1] *= dim;
                }
                else
                {
                    // Moving a dimension with size 1 has no impact on the index computations.
                    if (dim == 1)
                        continue;

                    strides[lastIndex] = stridesX[permutation];
                    counts[lastIndex] = dim;
                    lastIndex++;
                }

                lastPermutation = permutation;
            }

            // Special case for tensor with a single element as these are ignored above.
            if (lastIndex == 0)
            {
                strides[0] = 1;
                counts[0] = 1;
                lastIndex = 1;
            }
        }
    }

    /// <summary>
    /// Utility structure to combine parameters needed for a padding operation.
    /// </summary>
    internal unsafe struct PadParameters
    {
        public fixed int shapeX[TensorShape.maxRank];
        public fixed int shapeO[TensorShape.maxRank];
        public fixed int strideX[TensorShape.maxRank];
        public fixed int pad[TensorShape.maxRank];
        public int lastIndex;

        internal void Prepare(TensorShape shapeX, ReadOnlySpan<int> pad)
        {
            int rank = shapeX.rank;
            int lastIndex = 0;
            int strideX = 1;
            bool isLastZeroPadding = false;

            for (int i = rank - 1; i >= 0; i--)
            {
                int padLeft = pad[i];
                int padRight = pad[i + rank];
                int dimX = shapeX[i];
                int dimO = dimX + padLeft + padRight;
                bool isZeroPadding = (padLeft == 0) && (padRight == 0);

                // Flatten the shapes if the last dimension did not have any padding. This dimension
                // can have padding and the padding is scaled by the size of the last dimension.
                if (isLastZeroPadding)
                {
                    int lastDimX = this.shapeX[lastIndex - 1];
                    this.shapeX[lastIndex - 1] = dimX * lastDimX;
                    this.shapeO[lastIndex - 1] = dimO * lastDimX;
                    this.pad[lastIndex - 1] = padLeft * lastDimX;
                }
                else
                {
                    // A dimension of size 1 with no padding can be ignored to further flatten the shape.
                    if (dimX == 1 && isZeroPadding)
                        continue;

                    this.shapeX[lastIndex] = dimX;
                    this.shapeO[lastIndex] = dimO;
                    this.strideX[lastIndex] = strideX;
                    this.pad[lastIndex] = padLeft;
                    lastIndex++;
                }

                strideX *= dimX;
                isLastZeroPadding = isZeroPadding;
            }

            // Special case for tensor with a single element as these are ignored above.
            if (lastIndex == 0)
            {
                this.shapeX[0] = 1;
                this.shapeO[0] = 1;
                this.strideX[0] = 1;
                lastIndex = 1;
            }

            this.lastIndex = lastIndex;
        }
    }

    /// <summary>
    /// Utility structure to combine parameters needed for a slice operation.
    /// </summary>
    internal unsafe struct SliceParameters
    {
        public fixed int shapeO[TensorShape.maxRank];
        public fixed int stridedStarts[TensorShape.maxRank];
        public fixed int stridedSteps[TensorShape.maxRank];
        public int lastIndex;

        internal void Prepare(TensorShape shapeX, TensorShape shapeO, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
        {
            int* startsLocal = stackalloc int[TensorShape.maxRank];
            int* stepsLocal = stackalloc int[TensorShape.maxRank];

            // Expand and effectively sort the axes to a flat array to allow easier indexing in the
            // below loop.
            for (int i = 0; i < TensorShape.maxRank; i++)
            {
                startsLocal[i] = 0;
                stepsLocal[i] = 1;
            }

            for (int i = 0; i < starts.Length; i++)
            {
                int axis = axes != null ? shapeX.Axis(axes[i]) : i;
                int dimX = shapeX[axis];
                int start = Math.Min(starts[i], dimX - 1);
                startsLocal[axis] = start >= 0 ? start : start + dimX;
                stepsLocal[axis] = steps != null ? steps[i] : 1;
            }

            PrepareWithLocals(shapeX, shapeO, startsLocal, stepsLocal);
        }

        /// <summary>
        /// Prepare for slice on single axis with start value between 0 and the size of the slice axis.
        /// </summary>
        internal void PrepareSplit(TensorShape shapeX, TensorShape shapeO, int axis, int start)
        {
            int* startsLocal = stackalloc int[TensorShape.maxRank];
            int* stepsLocal = stackalloc int[TensorShape.maxRank];

            for (var i = 0; i < TensorShape.maxRank; i++)
            {
                startsLocal[i] = 0;
                stepsLocal[i] = 1;
            }

            axis = shapeX.Axis(axis);
            startsLocal[axis] = start;

            PrepareWithLocals(shapeX, shapeO, startsLocal, stepsLocal);
        }

        void PrepareWithLocals(TensorShape shapeX, TensorShape shapeO, int* startsLocal, int* stepsLocal)
        {
            int lastIndex = 0;
            int strideX = 1;
            bool isLastUnsliced = false;

            for (int i = shapeX.rank - 1; i >= 0; i--)
            {
                int dimX = shapeX[i];
                int dimO = shapeO[i];
                bool isUnsliced = (dimX == dimO) && (stepsLocal[i] == 1);

                // Flatten the shapes if this dimension and the last are not sliced.
                if (isUnsliced && isLastUnsliced)
                {
                    this.shapeO[lastIndex - 1] *= dimO;
                }
                else
                {
                    // A dimension of size 1 with no slicing can be ignored to further flatten the shape.
                    if (dimX == 1 && isUnsliced)
                        continue;

                    this.shapeO[lastIndex] = dimO;
                    this.stridedStarts[lastIndex] = startsLocal[i] * strideX;
                    this.stridedSteps[lastIndex] = stepsLocal[i] * strideX;
                    lastIndex++;
                }

                strideX *= dimX;
                isLastUnsliced = isUnsliced;
            }

            // Special case for tensor with a single element as these are ignored above.
            if (lastIndex == 0)
            {
                this.shapeO[0] = 1;
                this.stridedStarts[0] = 0;
                this.stridedSteps[0] = 1;
                lastIndex = 1;
            }

            this.lastIndex = lastIndex;
        }
    }
}

} // namespace Unity.Sentis
