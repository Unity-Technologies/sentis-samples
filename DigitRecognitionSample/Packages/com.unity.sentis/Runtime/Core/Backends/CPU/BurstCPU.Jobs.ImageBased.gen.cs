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
    internal unsafe struct RoiAlignJob : IParallelForBatch, IJobResourceDeclarationXSBO
    {
        public float spatialScale;
        public int numRois;
        public int inputChannels;
        public int inputHeight;
        public int inputWidth;
        public int inputSpatialSize;
        public int inputBatchOffset;
        public int outputHeight;
        public int outputWidth;
        public float normalizeOHeight;
        public float normalizeOWidth;
        public int samplingRatio;
        public Layers.RoiPoolingMode mode;
        public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
        public ReadOnlyMemResource S { get; set; } float* Sptr => (float*)S.ptr;
        public ReadOnlyMemResource B { get; set; } int* Bptr => (int*)B.ptr;
        public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

        public void Execute(int i, int count)
        {
            switch (mode)
            {
                case Layers.RoiPoolingMode.Avg:
                    RoiAlignAvg(i, count);
                    break;
                case Layers.RoiPoolingMode.Max:
                    RoiAlignMax(i, count);
                    break;
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void RoiAlignAvg(int i, int count)
        {
            float* Op = Optr + i;

            // Extract the starting output position from the index.
            int ow = i % outputWidth;
            i = i / outputWidth;
            int oh = i % outputHeight;
            i = i / outputHeight;
            int oc = i % inputChannels;
            i = i / inputChannels;
            int on = i;

            // Advance to the starting input channel.
            int* Bp = Bptr + on;
            float* Sp = Sptr + on * 4;

            int batchIdx = Bp[0];
            float* Xp = Xptr + batchIdx * inputBatchOffset + oc * inputSpatialSize;

            float roiStartW = Sp[0] * spatialScale;
            float roiStartH = Sp[1] * spatialScale;
            float roiEndW = Sp[2] * spatialScale;
            float roiEndH = Sp[3] * spatialScale;

            float roiWidth = roiEndW - roiStartW;
            float roiHeight = roiEndH - roiStartH;

            roiWidth = math.max(roiWidth, 1.0f);
            roiHeight = math.max(roiHeight, 1.0f);

            float binSizeH = roiHeight / ((float)outputHeight);
            float binSizeW = roiWidth / ((float)outputWidth);

            int roiBinGridH = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiHeight * normalizeOHeight);
            int roiBinGridW = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiWidth * normalizeOWidth);

            int countHW = math.max(roiBinGridH * roiBinGridW, 1);

            int outputWidthRemaining = outputWidth - ow;

            while (count > 0)
            {
                int outputCountW = math.min(count, outputWidthRemaining);
                count -= outputCountW;

                for (; outputCountW > 0; outputCountW -= 1)
                {
                    float startH = roiStartH + oh * binSizeH;
                    float startW = roiStartW + ow * binSizeW;

                    float v = 0.0f;
                    for (int iy = 0; iy < roiBinGridH; iy++)
                    {
                        float y = startH + (iy + 0.5f) * binSizeH / ((float)roiBinGridH);

                        for (int ix = 0; ix < roiBinGridW; ix++)
                        {
                            float x = startW + (ix + 0.5f) * binSizeW / ((float)roiBinGridW);

                            if (y >= (float)inputHeight || y < -1.0 || x >= (float)inputWidth || x < -1.0)
                                continue;

                            y = math.max(y, 0.0f);
                            x = math.max(x, 0.0f);

                            int yLow = (int)y;
                            int xLow = (int)x;
                            int yHigh;
                            int xHigh;

                            if (yLow >= inputHeight - 1)
                            {
                                yHigh = yLow = inputHeight - 1;
                                y = (float)yLow;
                            }
                            else
                            {
                                yHigh = yLow + 1;
                            }

                            if (xLow >= inputWidth - 1)
                            {
                                xHigh = xLow = inputWidth - 1;
                                x = (float)xLow;
                            }
                            else
                            {
                                xHigh = xLow + 1;
                            }

                            float ly = y - yLow;
                            float lx = x - xLow;
                            float hy = 1.0f - ly;
                            float hx = 1.0f - lx;
                            float w0 = hy * hx;
                            float w1 = hy * lx;
                            float w2 = ly * hx;
                            float w3 = ly * lx;

                            int pos0 = yLow * inputWidth + xLow;
                            int pos1 = yLow * inputWidth + xHigh;
                            int pos2 = yHigh * inputWidth + xLow;
                            int pos3 = yHigh * inputWidth + xHigh;
                            // TODO bake out pos*/w* as a separate kernel

                            float x0 = w0 * Xp[pos0];
                            float x1 = w1 * Xp[pos1];
                            float x2 = w2 * Xp[pos2];
                            float x3 = w3 * Xp[pos3];

                            v = v + x0 + x1 + x2 + x3;
                        }
                    }
                    v /= countHW;

                    *Op++ = (float)(v);
                    ow++;
                }

                if (count > 0)
                {
                    // Output is now always aligned to the start of a row.
                    ow = 0;
                    outputWidthRemaining = outputWidth;

                    if (++oh == outputHeight)
                    {
                        // Advance to the next output channel.
                        oh = 0;
                        Xp += inputHeight * inputWidth;
                        oc++;
                    }

                    if (oc == inputChannels)
                    {
                        // Advance to the next output batch.
                        oc = 0;
                        Bp += 1;
                        Sp += 4;

                        batchIdx = Bp[0];
                        Xp = Xptr + batchIdx * inputBatchOffset;

                        roiStartW = Sp[0] * spatialScale;
                        roiStartH = Sp[1] * spatialScale;
                        roiEndW = Sp[2] * spatialScale;
                        roiEndH = Sp[3] * spatialScale;

                        roiWidth = roiEndW - roiStartW;
                        roiHeight = roiEndH - roiStartH;

                        roiWidth = math.max(roiWidth, 1.0f);
                        roiHeight = math.max(roiHeight, 1.0f);

                        binSizeH = roiHeight / ((float)outputHeight);
                        binSizeW = roiWidth / ((float)outputWidth);

                        roiBinGridH = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiHeight * normalizeOHeight);
                        roiBinGridW = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiWidth * normalizeOWidth);

                        countHW = math.max(roiBinGridH * roiBinGridW, 1);
                    }
                }
            }
        }

        [MethodImplAttribute(MethodImplOptions.NoInlining)]
        void RoiAlignMax(int i, int count)
        {
            float* Op = Optr + i;

            // Extract the starting output position from the index.
            int ow = i % outputWidth;
            i = i / outputWidth;
            int oh = i % outputHeight;
            i = i / outputHeight;
            int oc = i % inputChannels;
            i = i / inputChannels;
            int on = i;

            // Advance to the starting input channel.
            int* Bp = Bptr + on;
            float* Sp = Sptr + on * 4;

            int batchIdx = Bp[0];
            float* Xp = Xptr + batchIdx * inputBatchOffset + oc * inputSpatialSize;

            float roiStartW = Sp[0] * spatialScale;
            float roiStartH = Sp[1] * spatialScale;
            float roiEndW = Sp[2] * spatialScale;
            float roiEndH = Sp[3] * spatialScale;

            float roiWidth = roiEndW - roiStartW;
            float roiHeight = roiEndH - roiStartH;

            roiWidth = math.max(roiWidth, 1.0f);
            roiHeight = math.max(roiHeight, 1.0f);

            float binSizeH = roiHeight / ((float)outputHeight);
            float binSizeW = roiWidth / ((float)outputWidth);

            int roiBinGridH = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiHeight * normalizeOHeight);
            int roiBinGridW = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiWidth * normalizeOWidth);

            int countHW = math.max(roiBinGridH * roiBinGridW, 1);

            int outputWidthRemaining = outputWidth - ow;

            while (count > 0)
            {
                int outputCountW = math.min(count, outputWidthRemaining);
                count -= outputCountW;

                for (; outputCountW > 0; outputCountW -= 1)
                {
                    float startH = roiStartH + oh * binSizeH;
                    float startW = roiStartW + ow * binSizeW;

                    float v = 0.0f;
                    for (int iy = 0; iy < roiBinGridH; iy++)
                    {
                        float y = startH + (iy + 0.5f) * binSizeH / ((float)roiBinGridH);

                        for (int ix = 0; ix < roiBinGridW; ix++)
                        {
                            float x = startW + (ix + 0.5f) * binSizeW / ((float)roiBinGridW);

                            if (y >= (float)inputHeight || y < -1.0 || x >= (float)inputWidth || x < -1.0)
                                continue;

                            y = math.max(y, 0.0f);
                            x = math.max(x, 0.0f);

                            int yLow = (int)y;
                            int xLow = (int)x;
                            int yHigh;
                            int xHigh;

                            if (yLow >= inputHeight - 1)
                            {
                                yHigh = yLow = inputHeight - 1;
                                y = (float)yLow;
                            }
                            else
                            {
                                yHigh = yLow + 1;
                            }

                            if (xLow >= inputWidth - 1)
                            {
                                xHigh = xLow = inputWidth - 1;
                                x = (float)xLow;
                            }
                            else
                            {
                                xHigh = xLow + 1;
                            }

                            float ly = y - yLow;
                            float lx = x - xLow;
                            float hy = 1.0f - ly;
                            float hx = 1.0f - lx;
                            float w0 = hy * hx;
                            float w1 = hy * lx;
                            float w2 = ly * hx;
                            float w3 = ly * lx;

                            int pos0 = yLow * inputWidth + xLow;
                            int pos1 = yLow * inputWidth + xHigh;
                            int pos2 = yHigh * inputWidth + xLow;
                            int pos3 = yHigh * inputWidth + xHigh;
                            // TODO bake out pos*/w* as a separate kernel

                            float x0 = w0 * Xp[pos0];
                            float x1 = w1 * Xp[pos1];
                            float x2 = w2 * Xp[pos2];
                            float x3 = w3 * Xp[pos3];

                            v = math.max(math.max(math.max(math.max(v, x0), x1), x2), x3);
                        }
                    }

                    *Op++ = (float)(v);
                    ow++;
                }

                if (count > 0)
                {
                    // Output is now always aligned to the start of a row.
                    ow = 0;
                    outputWidthRemaining = outputWidth;

                    if (++oh == outputHeight)
                    {
                        // Advance to the next output channel.
                        oh = 0;
                        Xp += inputHeight * inputWidth;
                        oc++;
                    }

                    if (oc == inputChannels)
                    {
                        // Advance to the next output batch.
                        oc = 0;
                        Bp += 1;
                        Sp += 4;

                        batchIdx = Bp[0];
                        Xp = Xptr + batchIdx * inputBatchOffset;

                        roiStartW = Sp[0] * spatialScale;
                        roiStartH = Sp[1] * spatialScale;
                        roiEndW = Sp[2] * spatialScale;
                        roiEndH = Sp[3] * spatialScale;

                        roiWidth = roiEndW - roiStartW;
                        roiHeight = roiEndH - roiStartH;

                        roiWidth = math.max(roiWidth, 1.0f);
                        roiHeight = math.max(roiHeight, 1.0f);

                        binSizeH = roiHeight / ((float)outputHeight);
                        binSizeW = roiWidth / ((float)outputWidth);

                        roiBinGridH = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiHeight * normalizeOHeight);
                        roiBinGridW = (samplingRatio > 0) ? samplingRatio : (int)math.ceil(roiWidth * normalizeOWidth);

                        countHW = math.max(roiBinGridH * roiBinGridW, 1);
                    }
                }
            }
        }
    }
}
}
