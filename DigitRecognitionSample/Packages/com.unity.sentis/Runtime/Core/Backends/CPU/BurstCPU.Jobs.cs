using UnityEngine;
using System;
using System.Runtime.CompilerServices;
using System.Threading;
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
using Constant = Unity.Burst.CompilerServices.Constant;

[assembly: BurstCompile(OptimizeFor = OptimizeFor.FastCompilation)]

namespace Unity.Sentis {

// BurstCPU.Core.cs -- definition of class CPUBackend, Pin(), BurstTensorData
// BurstCPU.Ops.cs  -- impl. IOps, job schedulers
// BurstCPU.Jobs.cs -- impl. jobs

public partial class CPUBackend
{
    internal static readonly Thread MainThread = Thread.CurrentThread;

    internal unsafe struct ReadOnlyMemResource
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public void* ptr;
    }

    internal unsafe struct ReadWriteMemResource
    {
        [NoAlias][NativeDisableUnsafePtrRestriction] public void* ptr;
    }

    internal interface IJobResourceDeclarationO
    {
        ReadWriteMemResource O { get; set; }
    }

    internal interface IJobResourceDeclarationXO
    {
        ReadOnlyMemResource X { get; set; }
        ReadWriteMemResource O { get; set; }
    }

    internal interface IJobResourceDeclarationXBO
    {
        ReadOnlyMemResource X { get; set; }
        ReadOnlyMemResource B { get; set; }
        ReadWriteMemResource O { get; set; }
    }

    internal interface IJobResourceDeclarationXSBO
    {
        ReadOnlyMemResource X { get; set; }
        ReadOnlyMemResource S { get; set; }
        ReadOnlyMemResource B { get; set; }
        ReadWriteMemResource O { get; set; }
    }

    internal interface IJobResourceDeclarationXSBWO
    {
        ReadOnlyMemResource X { get; set; }
        ReadOnlyMemResource S { get; set; }
        ReadOnlyMemResource B { get; set; }
        ReadOnlyMemResource W { get; set; }
        ReadWriteMemResource O { get; set; }
    }

    internal static class VectorUtils
    {
        // Equivalent to unpacklo(a, b); vzip1q_f32(a, b); a(3,2,1,0),b(7,6,5,4)=>(0,4,1,5)
        internal static float4 UnpackLo(float4 a, float4 b)
        {
            return math.shuffle(a, b, math.ShuffleComponent.LeftX, math.ShuffleComponent.RightX, math.ShuffleComponent.LeftY, math.ShuffleComponent.RightY);
        }

        // Equivalent to unpackhi(a, b); vzip2q_f32(a, b); a(3,2,1,0),b(7,6,5,4)=>(2,6,3,7)
        internal static float4 UnpackHi(float4 a, float4 b)
        {
            return math.shuffle(a, b, math.ShuffleComponent.LeftZ, math.ShuffleComponent.RightZ, math.ShuffleComponent.LeftW, math.ShuffleComponent.RightW);
        }

        // Equivalent to vuzp1q_f32(a, b); a(3,2,1,0),b(7,6,5,4)=>(0,2,4,6)
        internal static float4 PackStride2(float4 a, float4 b)
        {
            return math.shuffle(a, b, math.ShuffleComponent.LeftX, math.ShuffleComponent.LeftZ, math.ShuffleComponent.RightX, math.ShuffleComponent.RightZ);
        }

        // Equivalent to vextq_f32(a, b, 1); a(3,2,1,0),b(7,6,5,4)=>(4,3,2,1)
        internal static float4 Extract1(float4 a, float4 b)
        {
            return math.shuffle(a, b, math.ShuffleComponent.LeftY, math.ShuffleComponent.LeftZ, math.ShuffleComponent.LeftW, math.ShuffleComponent.RightX);
        }

        // Equivalent to vextq_f32(a, b, 2); a(3,2,1,0),b(7,6,5,4)=>(5,4,3,2)
        internal static float4 Extract2(float4 a, float4 b)
        {
            return math.shuffle(a, b, math.ShuffleComponent.LeftZ, math.ShuffleComponent.LeftW, math.ShuffleComponent.RightX, math.ShuffleComponent.RightY);
        }

        // Equivalent to vextq_f32(a, b, 3); a(3,2,1,0),b(7,6,5,4)=>(6,5,4,3)
        internal static float4 Extract3(float4 a, float4 b)
        {
            return math.shuffle(a, b, math.ShuffleComponent.LeftW, math.ShuffleComponent.RightX, math.ShuffleComponent.RightY, math.ShuffleComponent.RightZ);
        }

        internal static v256 MulAdd(v256 a, v256 b, v256 c)
        {
            return new v256(math.mad(a.Float0, b.Float0, c.Float0), math.mad(a.Float1, b.Float1, c.Float1),
                            math.mad(a.Float2, b.Float2, c.Float2), math.mad(a.Float3, b.Float3, c.Float3),
                            math.mad(a.Float4, b.Float4, c.Float4), math.mad(a.Float5, b.Float5, c.Float5),
                            math.mad(a.Float6, b.Float6, c.Float6), math.mad(a.Float7, b.Float7, c.Float7));
        }
    }

    internal static unsafe T* StrideAddress<T>(T* ptr, IntPtr stride, uint scale) where T : unmanaged
    {
        return ptr + stride.ToInt64() * scale;
    }

    internal static unsafe float* AllocTempBlock(int blockSizeM, int blockSizeN)
    {
        int sz = blockSizeM * blockSizeN * sizeof(float);
        // Allocator.Temp is the fastest allocator, but can only be used within jobs; No explicit need to deallocate
        // Source: https://docs.unity3d.com/Packages/com.unity.collections@1.0/manual/allocation.html#allocatortemp
        return (float*)UnsafeUtility.Malloc(sz, JobsUtility.CacheLineSize, Allocator.Temp);
    }

    internal static unsafe void WritePaddingN([NoAlias] float* dstPadding)
    {
        // Allow the compiler to generate a constant number of vector store instructions given a fixed byte count.
        UnsafeUtility.MemClear(dstPadding, multiplyBlockWidthN * sizeof(float));
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct BatchMatrixMultiplyJob : IJobParallelFor
    {
        public BatchMatrixMultiplyHelper data;
        int innerCount;
        int blockSizeM;
        int blockSizeN;
        int blockSizeK;

        public JobHandle Schedule(JobHandle dependsOn)
        {
            if (blockSizeM == 0 || blockSizeN == 0 || blockSizeK == 0)
            {
                // Profiling across a range of matrices for best block size revealed:
                // (32, 384, 16) was the best common block size for matrices <= 576
                // (32, 768, 32) for matrices > 576 and <= 1152
                // (64, 96, 32) for matrices > 1200
                int maxM = 32;
                int maxN = 384;
                int maxK = 16;

                if (data.M > 1200)
                {
                    maxM = 64;
                    maxN = 96;
                    maxK = 32;
                }
                else if (data.M > 576)
                {
                    maxM = 32;
                    maxN = 768;
                    maxK = 32;
                }

                blockSizeM = Mathf.Min(data.M, maxM);

                var sizeN = Mathf.ClosestPowerOfTwo(data.N);
                sizeN = (sizeN / multiplyBlockWidthN) * multiplyBlockWidthN;
                sizeN = Mathf.Max(sizeN, multiplyBlockWidthN);
                blockSizeN = Mathf.Min(sizeN, maxN);

                // Adjust block size down to the actual count of rows, so no allocation takes place needlessly.
                blockSizeK = Mathf.Min(data.K, maxK);
            }

            // Distribute jobs over a single axis.
            int longerAxis = data.M;
            int blockSizeForLongerAxis = blockSizeM;
            if (data.N > data.M)
            {
                longerAxis = data.N; blockSizeForLongerAxis = blockSizeN;
            }

            innerCount = (longerAxis + blockSizeForLongerAxis - 1) / blockSizeForLongerAxis;
            return IJobParallelForExtensions.Schedule(this, data.batchCount * innerCount, 1, dependsOn);
        }

        // Keep the temporary blocks available across multiple calls to Execute(int). AllocTempBlock() uses
        // the Allocator.Temp allocator. These allocations are not reclaimed until the outermost thread level
        // Execute() returns. Repeated calls to Execute(int) from this thread may allocate temporary blocks
        // that cause the Allocator.Temp allocator to keep extending the heap. Avoid this by tracking the
        // pointers here.
        [NativeDisableUnsafePtrRestriction]
        unsafe float* blockTempA, blockTempB, blockTempC;

        public void Execute(int i)
        {
            float* batchA = data.A;
            float* batchB = data.B;
            float* batchC = data.C;

            // Advance the buffer pointers if this is a batched GEMM.
            if (i >= innerCount)
            {
                int batchIndex = i / innerCount;
                i = i % innerCount;

                batchA = batchA + data.M * data.K * data.iteratorA.ComputeOffset(batchIndex);
                batchB = batchB + data.K * data.N * data.iteratorB.ComputeOffset(batchIndex);
                batchC = batchC + data.M * data.N * batchIndex;
            }

            int shorterAxis = data.N;
            int blockSizeForShorterAxis = blockSizeN;
            if (data.N > data.M)
            {
                shorterAxis = data.M; blockSizeForShorterAxis = blockSizeM;
            }

            // this job is scheduled over the Max(N, M)
            // need to pick the remaining (shorter) axis
            for (int j = 0; j < shorterAxis; j += blockSizeForShorterAxis)
            {
                int m = (data.M >= data.N) ? i * blockSizeM : j;
                int n = (data.M >= data.N) ? j : i * blockSizeN;

                int blockCountM = Math.Min(data.M - m, blockSizeM);
                int blockCountN = Math.Min(data.N - n, blockSizeN);
                int alignedCountN = (blockCountN + multiplyBlockWidthN - 1) & ~(multiplyBlockWidthN - 1);
                bool doPaddingN = blockCountN != alignedCountN;
                bool accumulateC = data.accumulateC;

                // Double buffer C if the block width is not aligned to the kernel width.
                float* Cp;
                int strideC;
                if (doPaddingN)
                {
                    if (blockTempC == null)
                        blockTempC = AllocTempBlock(blockSizeM, blockSizeN);
                    if (accumulateC)
                        CopyBlock(blockCountM, blockCountN, batchC + m * data.ldc + n, data.ldc, blockTempC, alignedCountN, true);
                    Cp = blockTempC;
                    strideC = alignedCountN;
                }
                else
                {
                    Cp = batchC + m * data.ldc + n;
                    strideC = data.ldc;
                }

                for (int k = 0; k < data.K; k += blockSizeK)
                {
                    int blockCountK = Math.Min(data.K - k, blockSizeK);

                    // Double buffer A if the matrix is transposed.
                    float* Ap;
                    int strideA;
                    if (data.transposeA)
                    {
                        if (blockTempA == null)
                            blockTempA = AllocTempBlock(blockSizeM, blockSizeK);
                        TransposeBlock(blockCountM, blockCountK, batchA + k * data.lda + m, data.lda, blockTempA, blockCountK, false);
                        Ap = blockTempA;
                        strideA = blockCountK;
                    }
                    else
                    {
                        Ap = batchA + m * data.lda + k;
                        strideA = data.lda;
                    }

                    // Double buffer B if the matrix is transposed or the block width is not aligned to the kernel width.
                    float* Bp;
                    int strideB;
                    if (data.transposeB || doPaddingN)
                    {
                        if (blockTempB == null)
                            blockTempB = AllocTempBlock(blockSizeK, blockSizeN);
                        if (data.transposeB)
                            TransposeBlock(blockCountK, blockCountN, batchB + n * data.ldb + k, data.ldb, blockTempB, alignedCountN, doPaddingN);
                        else
                            CopyBlock(blockCountK, blockCountN, batchB + k * data.ldb + n, data.ldb, blockTempB, alignedCountN, true);
                        Bp = blockTempB;
                        strideB = alignedCountN;
                    }
                    else
                    {
                        Bp = batchB + k * data.ldb + n;
                        strideB = data.ldb;
                    }

                    MultiplyBlockUnroll(Ap, strideA, Bp, strideB, Cp, strideC, blockCountM, alignedCountN, blockCountK, accumulateC);
                    accumulateC = true;
                }

                if (Cp == blockTempC)
                    CopyBlock(blockCountM, blockCountN, blockTempC, alignedCountN, batchC + m * data.ldc + n, data.ldc, false);
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        static void CopyBlock(int M, int N, [NoAlias] float* src, int srcStride, [NoAlias] float* dst, int dstStride, bool doPaddingN)
        {
            // Avoid generating compiled code for the case where M or N are negative or zero.
            Hint.Assume(M > 0);
            Hint.Assume(N > 0);

            var alignedN = N & ~(multiplyBlockWidthN - 1);

            for (int m = 0; m < M; m++)
            {
                if (doPaddingN)
                    WritePaddingN(dst + alignedN);

                UnsafeUtility.MemCpy(dst, src, N * sizeof(float));
                src += srcStride;
                dst += dstStride;
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        static void TransposeBlock(int M, int N, [NoAlias] float* src, int srcStride, [NoAlias] float* dst, int dstStride, bool doPaddingN)
        {
            // Avoid generating compiled code for the case where M or N are negative or zero.
            Hint.Assume(M > 0);
            Hint.Assume(N > 0);

            var alignedN = N & ~(multiplyBlockWidthN - 1);

            var nsrcStride = new IntPtr(srcStride);
            var ndstStride = new IntPtr(dstStride);

            for (; M >= 4; M -= 4)
            {
                if (doPaddingN)
                {
                    float* dstPadding = dst + alignedN;
                    WritePaddingN(StrideAddress(dstPadding, ndstStride, 0));
                    WritePaddingN(StrideAddress(dstPadding, ndstStride, 1));
                    WritePaddingN(StrideAddress(dstPadding, ndstStride, 2));
                    WritePaddingN(StrideAddress(dstPadding, ndstStride, 3));
                }

                float* srcLoop = src;
                float* dstLoop = dst;
                var n = (uint)N;

                // Transpose 4x4 blocks to the output buffer.
                for (; n >= 4; n -= 4)
                {
                    float4 v0 = *((float4*)StrideAddress(srcLoop, nsrcStride, 0));
                    float4 v1 = *((float4*)StrideAddress(srcLoop, nsrcStride, 1));
                    float4 v2 = *((float4*)StrideAddress(srcLoop, nsrcStride, 2));
                    float4 v3 = *((float4*)StrideAddress(srcLoop, nsrcStride, 3));

                    float4 t0 = VectorUtils.UnpackLo(v0, v2);   // v0.1,v2.1,v0.0,v2.0
                    float4 t1 = VectorUtils.UnpackHi(v0, v2);   // v0.3,v2.3,v0.2,v2.2
                    float4 t2 = VectorUtils.UnpackLo(v1, v3);   // v1.1,v2.1,v1.0,v2.0
                    float4 t3 = VectorUtils.UnpackHi(v1, v3);   // v0.3,v3.3,v1.2,v3.2

                    float4 t4 = VectorUtils.UnpackLo(t0, t2);   // v0.0,v1.0,v2.0,v3.0
                    float4 t5 = VectorUtils.UnpackHi(t0, t2);   // v0.1,v1.1,v2.1,v3.1
                    float4 t6 = VectorUtils.UnpackLo(t1, t3);   // v0.2,v1.2,v2.2,v3.2
                    float4 t7 = VectorUtils.UnpackHi(t1, t3);   // v0.3,v1.3,v2.3,v3.3

                    *((float4*)StrideAddress(dstLoop, ndstStride, 0)) = t4;
                    *((float4*)StrideAddress(dstLoop, ndstStride, 1)) = t5;
                    *((float4*)StrideAddress(dstLoop, ndstStride, 2)) = t6;
                    *((float4*)StrideAddress(dstLoop, ndstStride, 3)) = t7;

                    srcLoop += nsrcStride.ToInt64() * 4;
                    dstLoop += 4;
                }

                // Transpose the remaining 4x1 blocks to the output buffer.
                for (; n > 0; n -= 1)
                {
                    *StrideAddress(dstLoop, ndstStride, 0) = srcLoop[0];
                    *StrideAddress(dstLoop, ndstStride, 1) = srcLoop[1];
                    *StrideAddress(dstLoop, ndstStride, 2) = srcLoop[2];
                    *StrideAddress(dstLoop, ndstStride, 3) = srcLoop[3];

                    srcLoop += nsrcStride.ToInt64();
                    dstLoop += 1;
                }

                src += 4;
                dst += ndstStride.ToInt64() * 4;
            }

            for (int m = 0; m < M; m++)
            {
                if (doPaddingN)
                    WritePaddingN(dst + alignedN);

                float* srcLoop = src;
                for (int n = 0; n < N; n++, srcLoop += nsrcStride.ToInt64())
                    dst[n] = srcLoop[0];

                src += 1;
                dst += ndstStride.ToInt64();
            }
        }
    }

    internal unsafe struct BatchMatrixMultiplyWithPluginJob : IJob
    {
        public BatchMatrixMultiplyHelper data;

        public void Execute()
        {
            int* countersA = stackalloc int[TensorShape.maxRank];
            int* countersB = stackalloc int[TensorShape.maxRank];

            int offsetA = 0;
            int offsetB = 0;

            float* Cp = data.C;
            float beta = data.accumulateC ? 1.0f : 0.0f;

            for (int i = 0; i < data.batchCount; i++)
            {
                float* Ap = data.A + offsetA * data.M * data.K;
                float* Bp = data.B + offsetB * data.K * data.N;

                s_BLAS.SGEMM(
                    data.M, data.N, data.K,
                    Ap, data.lda, Bp, data.ldb, Cp, data.ldc,
                    beta, data.transposeA, data.transposeB);

                offsetA = data.iteratorA.AdvanceOffset(offsetA, 1, countersA);
                offsetB = data.iteratorB.AdvanceOffset(offsetB, 1, countersB);
                Cp += data.M * data.N;
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct ConvJob : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource S { get; set; } float* Wptr => (float*)S.ptr;
        public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public bool useBias;

        const int maxSpatialDims = 3;

        const int blockSizeM = 64;
        const int blockSizeN = 120;
        const int blockSizeK = 64;

        int innerCount;
        int batchCount;
        int groupCount;
        int inputChannels;
        int outputChannels;
        int spatialDims;
        int inputSize;
        int kernelSize;
        int outputSize;
        fixed int inputShape[ConvJob.maxSpatialDims];
        fixed int kernelShape[ConvJob.maxSpatialDims];
        fixed int outputShape[ConvJob.maxSpatialDims];
        fixed int stride[ConvJob.maxSpatialDims];
        fixed int padLeft[ConvJob.maxSpatialDims];
        fixed int padRight[ConvJob.maxSpatialDims];
        fixed int dilation[ConvJob.maxSpatialDims];
        float minValue;
        bool useDepthwiseKernel;

        // Keep the temporary blocks available across multiple calls to Execute(int). AllocTempBlock() uses
        // the Allocator.Temp allocator. These allocations are not reclaimed until the outermost thread level
        // Execute() returns. Repeated calls to Execute(int) from this thread may allocate temporary blocks
        // that cause the Allocator.Temp allocator to keep extending the heap. Avoid this by tracking the
        // pointers here.
        [NativeDisableUnsafePtrRestriction]
        unsafe float* columnBuffer, blockTempC;

        public int Prepare(TensorShape X, TensorShape W, TensorShape O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            int spatialDims = X.rank - 2;

            this.batchCount = X[0];
            this.groupCount = groups;
            this.inputChannels = X[1] / groups;
            this.outputChannels = O[1] / groups;
            this.spatialDims = spatialDims;

            int baseOutputIndex = 0;

            // Implement Conv1D using the Conv2D path by unsqueezing the shapes.
            if (spatialDims == 1)
            {
                this.spatialDims = 2;
                this.padLeft[0] = 0;
                this.padRight[0] = 0;
                this.stride[0] = 1;
                this.dilation[0] = 1;
                this.inputShape[0] = 1;
                this.kernelShape[0] = 1;
                this.outputShape[0] = 1;
                baseOutputIndex = 1;
            }

            this.inputSize = 1;
            this.kernelSize = 1;
            this.outputSize = 1;

            bool allStridesEqualOne = true;
            bool allPaddingEqualsZero = true;

            for (int i = 0; i < spatialDims; i++)
            {
                inputSize *= inputShape[baseOutputIndex + i] = X[2 + i];
                kernelSize *= kernelShape[baseOutputIndex + i] = W[2 + i];
                outputSize *= outputShape[baseOutputIndex + i] = O[2 + i];
                stride[baseOutputIndex + i] = strides[i];
                padLeft[baseOutputIndex + i] = pads[i];
                padRight[baseOutputIndex + i] = pads[i + spatialDims];
                dilation[baseOutputIndex + i] = dilations[i];
                allStridesEqualOne &= strides[i] == 1;
                allPaddingEqualsZero &= pads[i] == 0 && pads[i + spatialDims] == 0;
            }

            // Detect a pointwise convolution and handle by flattening the shapes (a 3D convolution
            // is also converted to a 2D convolution). The current implementation uses Im2Col to
            // gather an input block for the GEMM and using a flattened shape allows CopyToColumnBuffer
            // to use a large loop size for the copy (for example a 14x14 input shape can be treated as
            // 1x196 instead).
            if (this.kernelSize == 1 && allStridesEqualOne && allPaddingEqualsZero)
            {
                this.spatialDims = 2;
                this.inputShape[0] = 1;
                this.inputShape[1] = this.inputSize;
                this.outputShape[0] = 1;
                this.outputShape[1] = this.outputSize;
            }

            this.minValue = (fusedActivation == Layers.FusableActivation.Relu) ? 0.0f : float.MinValue;

            if (groups > 1 && CanUseDepthwiseConvKernel(this))
                this.useDepthwiseKernel = true;

            // Distribute jobs over a single axis.
            int longerAxis = this.outputSize;
            int blockSizeForLongerAxis = ConvJob.blockSizeN;
            if (this.outputChannels >= this.outputSize || this.useDepthwiseKernel)
            {
                longerAxis = this.outputChannels; blockSizeForLongerAxis = ConvJob.blockSizeM;
            }
            this.innerCount = (longerAxis + blockSizeForLongerAxis - 1) / blockSizeForLongerAxis;
            return this.batchCount * this.groupCount * this.innerCount;
        }

        public void Execute(int i)
        {
            int batchGroupIndex = i / innerCount;
            i = i % innerCount;

            int groupIndex = batchGroupIndex % groupCount;
            int K = inputChannels * kernelSize;

            float *Xp = Xptr + batchGroupIndex * inputChannels * inputSize;
            float *Op = Optr + batchGroupIndex * outputChannels * outputSize;
            float *Wp = Wptr + groupIndex * outputChannels * K;
            float *Bp = useBias ? Bptr + groupIndex * outputChannels : null;

            if (useDepthwiseKernel)
            {
                if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
                {
                    float bias = useBias ? Bp[0] : 0;
                    DepthwiseConv2D_Avx2(Xp, Wp, bias, Op);
                    return;
                }
                else if (Unity.Burst.Intrinsics.Arm.Neon.IsNeonSupported)
                {
                    float bias = useBias ? Bp[0] : 0;
                    DepthwiseConv2D_Neon_Kernel3x3_Stride1x1(Xp, Wp, bias, Op);
                    return;
                }
            }

            if (columnBuffer == null)
            {
                columnBuffer = AllocTempBlock(ConvJob.blockSizeK + ConvJob.blockSizeM, ConvJob.blockSizeN);
                blockTempC = columnBuffer + ConvJob.blockSizeK * ConvJob.blockSizeN;
            }

            int shorterAxis = outputChannels;
            int blockSizeForShorterAxis = ConvJob.blockSizeM;
            if (outputChannels >= outputSize)
            {
                shorterAxis = outputSize; blockSizeForShorterAxis = ConvJob.blockSizeN;
            }

            // this job is scheduled over the Max(outputSize, outputChannels)
            // need to pick the remaining (shorter) axis
            for (int j = 0; j < shorterAxis; j += blockSizeForShorterAxis)
            {
                int m = (outputChannels >= outputSize) ? i * blockSizeM : j;
                int n = (outputChannels >= outputSize) ? j : i * blockSizeN;

                int blockCountM = Math.Min(outputChannels - m, blockSizeM);
                int blockCountN = Math.Min(outputSize - n, blockSizeN);
                int alignedCountN = (blockCountN + multiplyBlockWidthN - 1) & ~(multiplyBlockWidthN - 1);

                float* Wpj = Wp + m * K;

                for (int k = 0; k < K;)
                {
                    int blockCountK = Math.Min(K - k, blockSizeK);

                    if (Hint.Likely(spatialDims <= 2))
                        Im2Col(Xp, columnBuffer, k, blockCountK, n, blockCountN, alignedCountN);
                    else
                        Vol2Col(Xp, columnBuffer, k, blockCountK, n, blockCountN, alignedCountN);

                    MultiplyBlockUnroll(
                        Wpj + k, K,
                        columnBuffer, alignedCountN,
                        blockTempC, alignedCountN,
                        blockCountM, alignedCountN, blockCountK, accumulateC: (k > 0));

                    k += blockCountK;
                }

                CopyBlock(blockCountM, blockCountN, blockTempC, alignedCountN, Op + m * outputSize + n, outputSize, useBias, useBias ? Bp + m : null);
            }
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        static void CopyToColumnBuffer([NoAlias] float* Xptr, int inputWidth, int strideWidth, ref float* Optr, ref int inputW, ref int outputCountW)
        {
            Hint.Assume(inputWidth > 0);
            Hint.Assume(strideWidth > 0);
            Hint.Assume(inputW >= 0);

            int availableW = (inputWidth - inputW + strideWidth - 1) / strideWidth;
            int copyCountW = Math.Min(availableW, outputCountW);

            // This routine always copies at least one input value to the column buffer.
            Hint.Assume(copyCountW > 0);

            // The auto-vectorizer transforms the below stock loop into a loop that uses vector loads
            // and stores to copy 32 elements at time, then a loop that copies the remaining scalar
            // elements. For convolutions where `inputWidth` is small (such as the resnet50v1 blocks
            // with 7/14 elements), this results in too much time looping. Manually unroll the copy
            // loop here to reduce the total number of instructions for these smaller spatial sizes.
            if (Constant.IsConstantExpression(strideWidth) && strideWidth == 1)
            {
                uint i = 0;

                for (uint endingIndex = (uint)copyCountW & (~15u); i < endingIndex; i += 16)
                {
                    float4 v0 = ((float4*)&Xptr[i])[0];
                    float4 v1 = ((float4*)&Xptr[i])[1];
                    float4 v2 = ((float4*)&Xptr[i])[2];
                    float4 v3 = ((float4*)&Xptr[i])[3];
                    ((float4*)(&Optr[i]))[0] = v0;
                    ((float4*)(&Optr[i]))[1] = v1;
                    ((float4*)(&Optr[i]))[2] = v2;
                    ((float4*)(&Optr[i]))[3] = v3;
                }

                for (uint endingIndex = (uint)copyCountW & (~3u); i < endingIndex; i += 4)
                    ((float4*)&Optr[i])[0] = ((float4*)&Xptr[i])[0];

                if ((copyCountW & 2) != 0)
                {
                    ((float2*)&Optr[i])[0] = ((float2*)&Xptr[i])[0];
                    i += 2;
                }

                if ((copyCountW & 1) != 0)
                    Optr[i] = Xptr[i];
            }
            else
            {
                for (uint i = 0; i < (uint)copyCountW; i++)
                    Optr[i] = Xptr[i * strideWidth];
            }

            Optr += copyCountW;
            inputW += copyCountW * strideWidth;
            outputCountW -= copyCountW;
        }

        /// <summary>
        /// Extracts a portion of the im2col buffer from the starting offset (k,n) and of size (countK,countN).
        /// This column buffer is then used with a portion of the weights buffer to partially compute the
        /// output buffer using matrix multiplication. Iterating over all of K (inputChannels * kernelSize)
        /// produces the final result.
        ///
        /// Portions of the im2col buffer are used to reduce intermediate buffer requirements from (K,N) to
        /// (countK,countN). Each thread of the job can independently extract the im2col buffer needed for
        /// its local operation.
        ///
        /// The offset and size is optimized for MatrixBlockUnroll and these values are not related to the
        /// sizes of the input/weights/output shape. Part of the complexity of the below code is mapping the
        /// (k,n) coordinate to the ((inputChannel,kernelH,kernelW),(outputW,outputH)). The loops must handle
        /// the case of starting partially into a kernel or an output line.
        ///
        /// Reference implementation:
        ///
        /// float Xptr[inputChannels][inputSize=inputHeight,inputWidth];
        /// float Optr[K=inputChannels,kernelHeight,kernelWidth][N=outputHeight,outputWidth];
        ///
        /// for (int c = 0; c < inputChannels; c++)
        ///     for (int kh = 0; kh < kernelHeight; kh++)
        ///         for (int kw = 0; kw < kernelWidth; kw++)
        ///             for (int oh = 0; oh < outputHeight; oh++)
        ///                 for (int ow = 0; ow < outputWidth; ow++)
        ///                 {
        ///                     int ih = (oh * strideHeight) + kh - padHeight;
        ///                     int iw = (ow * strideWidth) + kw - padWidth;
        ///                     Optr[c,kh,kw][oh,ow] =
        ///                         (ih < 0 || ih >= inputHeight || iw < 0 || iw >= inputWidth) ? 0 : Xptr[c][ih,iw];
        ///                 }
        ///
        /// The signed bounds checks above can be reduced to unsigned bounds checks to detect lower/upper padding.
        ///
        /// The row size of `Optr` is aligned up to `multiplyBlockWidthN' to satisfy MultiplyBlockUnroll()
        /// requirements. The alignment values are zero filled.
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        void Im2ColImpl([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN, int strideWidth, int strideHeight)
        {
            Hint.Assume(countK > 0);
            Hint.Assume(countN > 0);

            // Compute the input and output origin position from n.
            int outputWidth = outputShape[1];

            int originOutputW = n % outputWidth;
            int originOutputH = n / outputWidth;

            int originInputW = originOutputW * strideWidth;
            int originInputH = originOutputH * strideHeight;

            // Compute the starting kernel counters and advance the input pointer to the starting channel
            // from k.
            int kernelWidth = kernelShape[1];
            int kernelHeight = kernelShape[0];

            int kw = k % kernelWidth;
            int kh = (k / kernelWidth) % kernelHeight;

            Xptr = Xptr + ((k / kernelSize) * inputSize);

            int inputWidth = inputShape[1];
            int inputHeight = inputShape[0];
            int padWidth = padLeft[1];
            int padHeight = padLeft[0];
            int dilationWidth = dilation[1];
            int dilationHeight = dilation[0];

            // Flatten one output image starting at (originInputH,originInputW) to the column buffer.
            //
            // Note: referring to the reference implementation, this uses a single loop to iterate over
            // the (c,kh,kw) space. These counters are updated independently to allow starting/stopping
            // at arbitrary positions easier.
            do
            {
                float* nextOptr = Optr + alignedCountN;

                if (countN != alignedCountN)
                    WritePaddingN(nextOptr - multiplyBlockWidthN);

                // The starting `n` is not aligned to outputWidth, so compute the starting offset and count
                // for the first iteration of the below loop. These are also shifted by the current kernel
                // position (kh,kw).
                int rowInitialInputW = dilationWidth * kw - padWidth;
                int outputCountW = outputWidth - originOutputW;
                int inputH = dilationHeight * kh + originInputH - padHeight;
                int initialInputW = rowInitialInputW + originInputW;

                int remainingN = countN;

                // Loop over `remainingN` to produce one or more output lines in the column buffer.
                //
                // Note: referring to the reference implementation, this uses a single loop to iterate over
                // the (oh,ow) space. As above, these counters may start at an arbitrary position. The
                // innermost dimension `ow` is optimized to copy spans of non-padding elements to the
                // column buffer.
                do
                {
                    outputCountW = Math.Min(outputCountW, remainingN);
                    remainingN -= outputCountW;

                    Hint.Assume(outputCountW > 0);

                    // Check if the input is in the top/bottom padding region.
                    if ((uint)inputH < (uint)inputHeight)
                    {
                        int inputW = initialInputW;

                        do
                        {
                            // Check if the input is in the left/right padding region.
                            if ((uint)inputW < (uint)inputWidth)
                            {
                                float* Xp = Xptr + inputH * inputWidth + inputW;
                                CopyToColumnBuffer(Xp, inputWidth, strideWidth, ref Optr, ref inputW, ref outputCountW);
                            }
                            else
                            {
                                *Optr++ = 0;
                                inputW += strideWidth;
                                outputCountW--;
                            }
                        }
                        while (outputCountW != 0);
                    }
                    else
                    {
                        UnsafeUtility.MemClear(Optr, outputCountW * sizeof(float));
                        Optr += outputCountW;
                    }

                    // Any remaining iterations of this loop has the current `n` aligned to the start
                    // of an output line.
                    outputCountW = outputWidth;
                    inputH += strideHeight;
                    initialInputW = rowInitialInputW;
                }
                while (remainingN != 0);

                // Advance the kernel counters and input pointer for the next iteration of K.
                if (++kw == kernelWidth)
                {
                    kw = 0;
                    if (++kh == kernelHeight)
                    {
                        kh = 0;

                        // Advance to the next input channel.
                        Xptr += inputSize;
                    }
                }

                Optr = nextOptr;
            }
            while (--countK != 0);
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void Im2Col_Stride1x1([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            Im2ColImpl(Xptr, Optr, k, countK, n, countN, alignedCountN, 1, 1);
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void Im2Col_Stride2x2([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            Im2ColImpl(Xptr, Optr, k, countK, n, countN, alignedCountN, 2, 2);
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void Im2Col_StrideNxN([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            Im2ColImpl(Xptr, Optr, k, countK, n, countN, alignedCountN, stride[1], stride[0]);
        }

        // Dispatch the common cases of a strided 1x1 or 2x2 convolution with a fallback to a generic
        // implementation. Breaking up the cases in this way avoids extra branching inside inner loops
        // to invoke optimized copy loops.
        void Im2Col([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            int strideWidth = stride[1];
            int strideHeight = stride[0];

            if (strideWidth == strideHeight)
            {
                if (strideWidth == 1)
                {
                    Im2Col_Stride1x1(Xptr, Optr, k, countK, n, countN, alignedCountN);
                    return;
                }
                else if (strideWidth == 2)
                {
                    Im2Col_Stride2x2(Xptr, Optr, k, countK, n, countN, alignedCountN);
                    return;
                }
            }

            Im2Col_StrideNxN(Xptr, Optr, k, countK, n, countN, alignedCountN);
        }

        /// <summary>
        /// Implements a 3D version of im2col. See Im2Col() for more information.
        /// </summary>
        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        void Vol2ColImpl([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN, int strideWidth, int strideHeight, int strideDepth)
        {
            Hint.Assume(countK > 0);
            Hint.Assume(countN > 0);

            // Compute the input and output origin position from n.
            int outputWidth = outputShape[2];
            int outputHeight = outputShape[1];

            int originOutputW = n % outputWidth;
            int originOutputH = (n / outputWidth) % outputHeight;
            int originOutputD = (n / outputWidth) / outputHeight;

            int originInputW = originOutputW * strideWidth;
            int originInputH = originOutputH * strideHeight;
            int originInputD = originOutputD * strideDepth;

            // Compute the starting kernel counters and advance the input pointer to the starting channel
            // from k.
            int kernelWidth = kernelShape[2];
            int kernelHeight = kernelShape[1];
            int kernelDepth = kernelShape[0];

            int kw = k % kernelWidth;
            int kh = (k / kernelWidth) % kernelHeight;
            int kd = ((k / kernelWidth) / kernelHeight) % kernelDepth;

            Xptr = Xptr + ((k / kernelSize) * inputSize);

            int inputWidth = inputShape[2];
            int inputHeight = inputShape[1];
            int inputDepth = inputShape[0];
            int padWidth = padLeft[2];
            int padHeight = padLeft[1];
            int padDepth = padLeft[0];
            int dilationWidth = dilation[2];
            int dilationHeight = dilation[1];
            int dilationDepth = dilation[0];

            // Flatten one output volume starting at (originInputD,originInputH,originInputW) to the column buffer.
            //
            // Note: referring to the reference implementation, this uses a single loop to iterate over
            // the (c,kd,kh,kw) space. These counters are updated independently to allow starting/stopping
            // at arbitrary positions easier.
            for (; countK > 0; countK--)
            {
                float* nextOptr = Optr + alignedCountN;

                if (countN != alignedCountN)
                    WritePaddingN(nextOptr - multiplyBlockWidthN);

                // The starting `n` is not aligned to outputWidth, so compute the starting offset and count
                // for the first iteration of the below loop. These are also shifted by the current kernel
                // position (kd,kh,kw).
                int rowInitialInputH = dilationHeight * kh - padHeight;
                int rowInitialInputW = dilationWidth * kw - padWidth;
                int outputCountW = outputWidth - originOutputW;
                int inputH = rowInitialInputH + originInputH;
                int initialInputW = rowInitialInputW + originInputW;
                int outputCountH = outputHeight - originOutputH;
                int inputD = dilationDepth * kd + originInputD - padDepth;

                int remainingN = countN;

                // Loop over `remainingN` to produce one or more output lines in the column buffer.
                //
                // Note: referring to the reference implementation, this uses a single loop to iterate over
                // the (od,oh,ow) space. As above, these counters may start at an arbitrary position. The
                // innermost dimension `ow` is optimized to copy spans of non-padding elements to the
                // column buffer.
                do {

                    outputCountW = Math.Min(outputCountW, remainingN);
                    remainingN -= outputCountW;

                    Hint.Assume(outputCountW > 0);

                    // Check if the input is in the top/bottom or front/back padding region.
                    if ((uint)inputH < (uint)inputHeight && (uint)inputD < (uint)inputDepth)
                    {
                        int inputW = initialInputW;

                        do {

                            // Check if the input is in the left/right padding region.
                            if ((uint)inputW < (uint)inputWidth)
                            {
                                float* Xp = Xptr + (inputD * inputHeight + inputH) * inputWidth + inputW;
                                CopyToColumnBuffer(Xp, inputWidth, strideWidth, ref Optr, ref inputW, ref outputCountW);
                            }
                            else
                            {
                                *Optr++ = 0;
                                inputW += strideWidth;
                                outputCountW--;
                            }

                        } while (outputCountW > 0);
                    }
                    else
                    {
                        UnsafeUtility.MemClear(Optr, outputCountW * sizeof(float));
                        Optr += outputCountW;
                    }

                    // Any remaining iterations of this loop has the current `n` aligned to the start
                    // of an output line.
                    outputCountW = outputWidth;
                    inputH += strideHeight;
                    initialInputW = rowInitialInputW;

                    // Advance to the next output plane if necessary.
                    if (--outputCountH == 0)
                    {
                        outputCountH = outputHeight;
                        inputH = rowInitialInputH;
                        inputD += strideDepth;
                    }

                } while (remainingN > 0);

                // Advance the kernel counters and input pointer for the next iteration of K.
                if (++kw == kernelWidth)
                {
                    kw = 0;
                    if (++kh == kernelHeight)
                    {
                        kh = 0;
                        if (++kd == kernelDepth)
                        {
                            kd = 0;

                            // Advance to the next input channel.
                            Xptr += inputSize;
                        }
                    }
                }

                Optr = nextOptr;
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void Vol2Col_Stride1x1x1([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            Vol2ColImpl(Xptr, Optr, k, countK, n, countN, alignedCountN, 1, 1, 1);
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void Vol2Col_StrideNxNx2([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            Vol2ColImpl(Xptr, Optr, k, countK, n, countN, alignedCountN, 2, stride[1], stride[0]);
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void Vol2Col_StrideNxNxN([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            Vol2ColImpl(Xptr, Optr, k, countK, n, countN, alignedCountN, stride[2], stride[1], stride[0]);
        }

        // Dispatch the common cases of a strided 1x1x1 or NxNx2 convolution with a fallback to a generic
        // implementation. Breaking up the cases in this way avoids extra branching inside inner loops
        // to invoke optimized copy loops.
        void Vol2Col([NoAlias] float* Xptr, [NoAlias] float* Optr, int k, int countK, int n, int countN, int alignedCountN)
        {
            int strideWidth = stride[2];
            int strideHeight = stride[1];
            int strideDepth = stride[0];

            if (strideWidth == 1 && strideHeight == 1 && strideDepth == 1)
            {
                Vol2Col_Stride1x1x1(Xptr, Optr, k, countK, n, countN, alignedCountN);
                return;
            }
            else if (strideWidth == 2)
            {
                Vol2Col_StrideNxNx2(Xptr, Optr, k, countK, n, countN, alignedCountN);
                return;
            }

            Vol2Col_StrideNxNxN(Xptr, Optr, k, countK, n, countN, alignedCountN);
        }

        void CopyBlock(int M, int N, [NoAlias] float* src, int srcStride, [NoAlias] float* dst, int dstStride, bool useBias, [NoAlias] float* bias)
        {
            // Avoid generating compiled code for the case where M or N are negative or zero.
            Hint.Assume(M > 0);
            Hint.Assume(N > 0);

            if (useBias)
            {
                for (int m = 0; m < M; m++)
                {
                    for (int n = 0; n < N; n++)
                        dst[n] = math.max(src[n] + bias[m], minValue);
                    src += srcStride;
                    dst += dstStride;
                }
            }
            else
            {
                for (int m = 0; m < M; m++)
                {
                    for (int n = 0; n < N; n++)
                        dst[n] = math.max(src[n], minValue);
                    src += srcStride;
                    dst += dstStride;
                }
            }
        }

        static bool CanUseDepthwiseConvKernel(in ConvJob job)
        {
            if (job.inputChannels != 1 || job.outputChannels != 1)
                return false;
            if (job.spatialDims != 2)
                return false;
            if (job.dilation[0] != 1 || job.dilation[1] != 1)
                return false;

            fixed (ConvJob* jobPtr = &job)
            {
                return CanUseDepthwiseConvKernel(jobPtr);
            }
        }

        // Platform dependent tests that require checking for intrinsic support. From non-Burst
        // compiled code, these intrinsic fields read as false, so these checks must be done from
        // a Burst compiled method.
        [BurstCompile]
        static bool CanUseDepthwiseConvKernel(ConvJob* job)
        {
            if (Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
            {
                if (job->stride[0] > 2 || job->stride[1] > 2)
                    return false;

                return true;
            }
            else if (Unity.Burst.Intrinsics.Arm.Neon.IsNeonSupported)
            {
                int inputWidth = job->inputShape[1];

                // Require a square kernel and stride.
                if (job->kernelShape[0] != job->kernelShape[1] || job->stride[0] != job->stride[1])
                    return false;

                if (job->kernelShape[0] == 3 && job->stride[0] == 1)
                {
                    // DepthwiseConv2D_Neon_Kernel3x3_Stride1x1 needs to be able to read at least four
                    // elements from an input line.
                    if (inputWidth < 4)
                        return false;

                    if (job->padLeft[0] > 1 || job->padRight[0] > 1 || job->padLeft[1] > 1 || job->padRight[1] > 1)
                        return false;

                    return true;
                }
            }

            return false;
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void DepthwiseConv2D_Avx2([NoAlias] float* Xptr, [NoAlias] float* Wptr, [NoAlias] float bias, [NoAlias] float* Optr)
        {
            if (!Unity.Burst.Intrinsics.X86.Avx2.IsAvx2Supported)
                return;

            int outputWidth = outputShape[1];
            int outputHeight = outputShape[0];
            int inputWidth = inputShape[1];
            int inputHeight = inputShape[0];
            int padWidth = padLeft[1];
            int padHeight = padLeft[0];
            int kernelWidth = kernelShape[1];
            int kernelHeight = kernelShape[0];
            int strideWidth = stride[1];
            int strideHeight = stride[0];

            v256 biasV = new v256(bias);
            v256 minValueV = new v256(minValue);
            v256 laneIndexV = new v256(0, 1, 2, 3, 4, 5, 6, 7);
            v256 inputWidthV = new v256(inputWidth);
            v256 negativeOneV = new v256(-1);
            int vectorWidth = 8;
            v256 vectorWidthV = new v256(vectorWidth);

            v256 outputPermuteV = laneIndexV;
            int outputLanes = vectorWidth;
            if (strideWidth == 2)
            {
                // Handle strideWidth=2 by using the same logic for strideWidth=1, but discard the odd
                // numbered elements. This is faster than the fallback path of Im2Col+MatMul while also
                // not increasing code size by using a dedicated kernel.
                outputPermuteV = new v256(0, 2, 4, 6, 0, 2, 4, 6);
                outputLanes = vectorWidth / 2;
            }
            v256 initialInputOffsetsV = mm256_sub_epi32(laneIndexV, new v256(padWidth));

            float *Xph = Xptr - padHeight * inputWidth - padWidth;

            int ihOuter = -padHeight;
            int ihOuterEnd = ihOuter + outputHeight * strideHeight;

            Hint.Assume(ihOuter < ihOuterEnd);

            for (; ihOuter < ihOuterEnd; ihOuter += strideHeight)
            {
                v256 rowInputOffsetsV = initialInputOffsetsV;
                int ihInnerEnd = ihOuter + kernelHeight;

                float* Xpw = Xph;
                Xph += (uint)(strideHeight * inputWidth);

                Hint.Assume(ihOuter < ihInnerEnd);

                for (int outputWidthRemaining = outputWidth; outputWidthRemaining > 0;)
                {
                    v256 accumV = biasV;
                    float* Xpk = Xpw;
                    float *Wpk = Wptr;

                    for (int ihInner = ihOuter; ihInner < ihInnerEnd; ihInner++)
                    {
                        Hint.Assume(kernelWidth > 0);

                        if (Hint.Likely((uint)ihInner < (uint)inputHeight))
                        {
                            v256 inputOffsetsV = rowInputOffsetsV;

                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                // Prepare a mask to load elements from the input tensor. mm256_maskload_ps uses the
                                // most significant bit (MSB) for each 32-bit element. First set the MSB for any
                                // input offset less than the input width (including negative offsets). Then
                                // the sign bit of the input offsets to clear the MSB for any negative input
                                // offsets.
                                v256 loadMaskV = mm256_xor_si256(mm256_cmpgt_epi32(inputWidthV, inputOffsetsV), inputOffsetsV);
                                v256 inputDataV = mm256_maskload_ps(&Xpk[kw], loadMaskV);
                                // Advance the input offsets by subtracting -1: generating a register with all -1's
                                // is a single instruction while an extra instruction or memory load is required
                                // to load all 1's.
                                inputOffsetsV = mm256_sub_epi32(inputOffsetsV, negativeOneV);
                                accumV = mm256_fmadd_ps(inputDataV, new v256(Wpk[kw]), accumV);
                            }
                        }

                        Xpk += (uint)inputWidth;
                        Wpk += (uint)kernelWidth;
                    }

                    // Apply the fused activation.
                    accumV = mm256_max_ps(accumV, minValueV);

                    // Prepare a mask to store elements to the output tensor. As above, mm256_maskstore_ps uses
                    // the MSB for each 32-bit element. Compare each lane index with the number of output elements
                    // to generate this mask.
                    // Also permute the accumulator to either pass through the vector (strideWidth=1) or pack the
                    // even numbered elements to the bottom of the output vector (strideWidth=2).
                    int outputCount = math.min(outputWidthRemaining, outputLanes);
                    v256 storeMaskV = mm256_cmpgt_epi32(new v256(outputCount), laneIndexV);
                    rowInputOffsetsV = mm256_add_epi32(rowInputOffsetsV, vectorWidthV);
                    mm256_maskstore_ps(Optr, storeMaskV, mm256_permutevar8x32_epi32(accumV, outputPermuteV));

                    Xpw += vectorWidth;
                    Optr += (uint)outputCount;
                    outputWidthRemaining -= outputCount;
                }
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void DepthwiseConv2D_Neon_Kernel3x3_Stride1x1([NoAlias] float* Xptr, [NoAlias] float* Wptr, float bias, [NoAlias] float* Optr)
        {
            int outputWidth = outputShape[1];
            int outputHeight = outputShape[0];
            int inputWidth = inputShape[1];
            int inputHeight = inputShape[0];
            int padWidth = padLeft[1];
            int padHeight = padLeft[0];

            float4 biasV = new float4(bias);
            float4 weight0V = *((float4*)(&Wptr[0]));
            float4 weight1V = *((float4*)(&Wptr[4]));
            float4 weight2V = new float4(Wptr[8], 0, 0, 0);
            float4 minValueV = new float4(minValue);

            float* zeroBuffer = stackalloc float[8];

            // Each iteration of the loop produces two output lines.
            for (int oh = 0; oh < outputHeight; oh += 2)
            {
                int ih = oh - padHeight;

                float* row0Op = &Optr[0];
                float* row1Op = &Optr[outputWidth];

                // If the number of remaining output lines is one, then use the output buffer as a temporary buffer
                // to store the unused line. The code below carefully writes to the unused odd line first, then
                // stores the used even line second.
                if (oh + 1 == outputHeight)
                    row1Op = row0Op;

                Optr += outputWidth * 2;

                // Each output line requires three input lines to be consumed. These output lines are contiguous, so
                // the middle lines are shared: in total, four inputs lines are used.
                float* row0Xp = Xptr + ih * inputWidth;
                float* row1Xp = row0Xp + inputWidth;
                float* row2Xp = row1Xp + inputWidth;
                float* row3Xp = row2Xp + inputWidth;

                int inputStride0 = 4;
                int inputStride1 = 4;
                int inputStride2 = 4;
                int inputStride3 = 4;

                int inputInitialRead = 4 - padWidth;

                // Handle the height padding by redirecting the input lines to a stack zero buffer and disable
                // pointer advancement for this pointer.
                if ((uint)(ih + 0) >= (uint)inputHeight)
                {
                    row0Xp = &zeroBuffer[4 - inputInitialRead];
                    inputStride0 = 0;
                }

                if ((uint)(ih + 1) >= (uint)inputHeight)
                {
                    row1Xp = &zeroBuffer[4 - inputInitialRead];
                    inputStride1 = 0;
                }

                if ((uint)(ih + 2) >= (uint)inputHeight)
                {
                    row2Xp = &zeroBuffer[4 - inputInitialRead];
                    inputStride2 = 0;
                }

                if ((uint)(ih + 3) >= (uint)inputHeight)
                {
                    row3Xp = &zeroBuffer[4 - inputInitialRead];
                    inputStride3 = 0;
                }

                float4 inputLo0V = *(float4*)(row0Xp);
                row0Xp += inputInitialRead;
                float4 inputLo1V = *(float4*)(row1Xp);
                row1Xp += inputInitialRead;
                float4 inputLo2V = *(float4*)(row2Xp);
                row2Xp += inputInitialRead;
                float4 inputLo3V = *(float4*)(row3Xp);
                row3Xp += inputInitialRead;

                // Shift in a zero padding element: (a,b,c,d)=>(0,a,b,c).
                if (inputInitialRead == 3)
                {
                    inputLo0V = VectorUtils.Extract3(float4.zero, inputLo0V);
                    inputLo1V = VectorUtils.Extract3(float4.zero, inputLo1V);
                    inputLo2V = VectorUtils.Extract3(float4.zero, inputLo2V);
                    inputLo3V = VectorUtils.Extract3(float4.zero, inputLo3V);
                }

                int inputCountW = inputWidth - inputInitialRead;
                int outputCountW = outputWidth;

                do
                {
                    float4 inputHi0V;
                    float4 inputHi1V;
                    float4 inputHi2V;
                    float4 inputHi3V;

                    if (Hint.Likely(inputCountW >= 4))
                    {
                        // Read the next interior elements, no padding required.
                        inputHi0V = *(float4*)(row0Xp);
                        row0Xp += inputStride0;
                        inputHi1V = *(float4*)(row1Xp);
                        row1Xp += inputStride1;
                        inputHi2V = *(float4*)(row2Xp);
                        row2Xp += inputStride2;
                        inputHi3V = *(float4*)(row3Xp);
                        row3Xp += inputStride3;

                        inputCountW -= 4;
                    }
                    else if (inputCountW > 0)
                    {
                        // Read the next interior elements, some padding required.
                        int padRightCount = 4 - inputCountW;

                        inputHi0V = *(float4*)(row0Xp - padRightCount);
                        inputHi1V = *(float4*)(row1Xp - padRightCount);
                        inputHi2V = *(float4*)(row2Xp - padRightCount);
                        inputHi3V = *(float4*)(row3Xp - padRightCount);

                        if ((padRightCount & 1) != 0)
                        {
                            inputHi0V = VectorUtils.Extract1(inputHi0V, float4.zero);
                            inputHi1V = VectorUtils.Extract1(inputHi1V, float4.zero);
                            inputHi2V = VectorUtils.Extract1(inputHi2V, float4.zero);
                            inputHi3V = VectorUtils.Extract1(inputHi3V, float4.zero);
                        }

                        if ((padRightCount & 2) != 0)
                        {
                            inputHi0V = VectorUtils.Extract2(inputHi0V, float4.zero);
                            inputHi1V = VectorUtils.Extract2(inputHi1V, float4.zero);
                            inputHi2V = VectorUtils.Extract2(inputHi2V, float4.zero);
                            inputHi3V = VectorUtils.Extract2(inputHi3V, float4.zero);
                        }

                        inputCountW = 0;
                    }
                    else
                    {
                        // Zero the next interior elements. This occurs when the remaining output count is
                        // one or two elements, so there is no remaining input elements to gather.
                        inputHi0V = float4.zero;
                        inputHi1V = float4.zero;
                        inputHi2V = float4.zero;
                        inputHi3V = float4.zero;
                    }

                    // At this point, the input vectors have gathered. Extract slices from these vectors to
                    // compute the final output.
                    //
                    // inputLo0V:inputHi0V:  x00 x01 x02 x03 x04 x05 x06 x07
                    // inputLo1V:inputHi1V:  x10 x11 x12 x13 x14 x15 x16 x17
                    // inputLo2V:inputHi2V:  x20 x21 x22 x23 x24 x25 x26 x27
                    // inputLo3V:inputHi3V:  x30 x31 x32 x33 x34 x35 x36 x37

                    float4 accum0V = biasV;
                    float4 accum1V = biasV;

                    accum0V = math.mad(inputLo0V, weight0V.xxxx, accum0V);
                    accum0V = math.mad(VectorUtils.Extract1(inputLo0V, inputHi0V), weight0V.yyyy, accum0V);
                    accum0V = math.mad(VectorUtils.Extract2(inputLo0V, inputHi0V), weight0V.zzzz, accum0V);

                    accum1V = math.mad(inputLo1V, weight0V.xxxx, accum1V);
                    accum1V = math.mad(VectorUtils.Extract1(inputLo1V, inputHi1V), weight0V.yyyy, accum1V);
                    accum1V = math.mad(VectorUtils.Extract2(inputLo1V, inputHi1V), weight0V.zzzz, accum1V);

                    accum0V = math.mad(inputLo1V, weight0V.wwww, accum0V);
                    accum0V = math.mad(VectorUtils.Extract1(inputLo1V, inputHi1V), weight1V.xxxx, accum0V);
                    accum0V = math.mad(VectorUtils.Extract2(inputLo1V, inputHi1V), weight1V.yyyy, accum0V);

                    accum1V = math.mad(inputLo2V, weight0V.wwww, accum1V);
                    accum1V = math.mad(VectorUtils.Extract1(inputLo2V, inputHi2V), weight1V.xxxx, accum1V);
                    accum1V = math.mad(VectorUtils.Extract2(inputLo2V, inputHi2V), weight1V.yyyy, accum1V);

                    accum0V = math.mad(inputLo2V, weight1V.zzzz, accum0V);
                    accum0V = math.mad(VectorUtils.Extract1(inputLo2V, inputHi2V), weight1V.wwww, accum0V);
                    accum0V = math.mad(VectorUtils.Extract2(inputLo2V, inputHi2V), weight2V.xxxx, accum0V);

                    accum1V = math.mad(inputLo3V, weight1V.zzzz, accum1V);
                    accum1V = math.mad(VectorUtils.Extract1(inputLo3V, inputHi3V), weight1V.wwww, accum1V);
                    accum1V = math.mad(VectorUtils.Extract2(inputLo3V, inputHi3V), weight2V.xxxx, accum1V);

                    // Apply the fused activation.
                    accum0V = math.max(accum0V, minValueV);
                    accum1V = math.max(accum1V, minValueV);

                    Hint.Assume(outputCountW > 0);

                    if ((outputCountW -= 4) >= 0)
                    {
                        *((float4*)row1Op) = accum1V;
                        row1Op += 4;
                        *((float4*)row0Op) = accum0V;
                        row0Op += 4;
                    }
                    else
                    {
                        // Handle the remaining non-vector aligned output elements.
                        // Note that the below tests are using bitwise tests to help the ARM64 backend generate a
                        // single tbz instruction instead of a cmp/blt instruction pair.

                        if ((outputCountW & 2) != 0)
                        {
                            *((float2*)row1Op) = accum1V.xy;
                            row1Op += 2;
                            *((float2*)row0Op) = accum0V.xy;
                            row0Op += 2;

                            // Move the third element of the accumulator to the first element for the ow=3 case.
                            accum1V = accum1V.zzzz;
                            accum0V = accum0V.zzzz;
                        }

                        if ((outputCountW & 1) != 0)
                        {
                            row1Op[0] = accum1V.x;
                            row0Op[0] = accum0V.x;
                        }

                        break;
                    }

                    // Shift the input vectors down for the next iteration of the loop.
                    inputLo0V = inputHi0V;
                    inputLo1V = inputHi1V;
                    inputLo2V = inputHi2V;
                    inputLo3V = inputHi3V;

                } while (outputCountW != 0);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct VectorBroadcast1DJob : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public int length;
        public int repeat;

        public void Execute()
        {
            UnsafeUtility.MemCpyReplicate(destination: O.ptr,
                                          source:      X.ptr,
                                          size:        length * sizeof(float),
                                          count:       repeat);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
    internal unsafe struct MemFreeJob : IJob
    {
        [NoAlias] [NativeDisableUnsafePtrRestriction]           public void* buffer0;
        [NoAlias] [NativeDisableUnsafePtrRestriction]           public void* buffer1;
                                                     [ReadOnly] public Allocator allocator;
        public void Execute()
        {
            if (buffer0 != null)
                UnsafeUtility.Free(buffer0, allocator);
            if (buffer1 != null)
                UnsafeUtility.Free(buffer1, allocator);
        }
    }
}

} // namespace Unity.Sentis
