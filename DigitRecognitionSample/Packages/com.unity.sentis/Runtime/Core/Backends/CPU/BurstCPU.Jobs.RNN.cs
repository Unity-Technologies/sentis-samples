using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Sentis
{
    public partial class CPUBackend
    {
        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
        unsafe struct LSTMEndJob : IJobParallelFor
        {
            [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public float* Pptr;
            [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public float* Bptr;
            [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public float* XsixWTptr;
            [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public float* HtxRTptr;
            [NoAlias][NativeDisableUnsafePtrRestriction] public float* Yptr;
            [NoAlias][NativeDisableUnsafePtrRestriction] public float* YCptr;
            [NoAlias][NativeDisableUnsafePtrRestriction] public float* YHptr;

            [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public int* SequenceLensptr;
            public int seqIndex;

            public int xStride;
            public int yStride;

            public int hiddenSize;
            public bool inputForget;
            public float clip;
            public Layers.RnnActivation fActivation;
            public float fAlpha;
            public float fBeta;
            public Layers.RnnActivation gActivation;
            public float gAlpha;
            public float gBeta;
            public Layers.RnnActivation hActivation;
            public float hAlpha;
            public float hBeta;

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyRelu(float* v, int count)
            {
                for (var i = 0; i < count; i++) v[i] = 0.5f * (v[i] + math.abs(v[i]));
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyTanh(float* v, int count)
            {
                for (var i = 0; i < count; i++) v[i] = math.tanh(math.clamp(v[i], -16.0f, 16.0f));
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplySigmoid(float* v, int count)
            {
                for (var i = 0; i < count; i++) v[i] = 1.0f / (1.0f + math.exp(math.clamp(-v[i], -60.0f, 60.0f)));
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyAffine(float* v, int count, float alpha, float beta)
            {
                for (var i = 0; i < count; i++) v[i] = alpha * v[i] + beta;
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyLeakyRelu(float* v, int count, float alpha)
            {
                for (var i = 0; i < count; i++) v[i] = 0.5f * ((1.0f + alpha) * v[i] + (1.0f - alpha) * math.abs(v[i]));
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyThresholdedRelu(float* v, int count, float alpha)
            {
                for (var i = 0; i < count; i++) v[i] = v[i] >= alpha ? v[i] : 0.0f;
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyScaledTanh(float* v, int count, float alpha, float beta)
            {
                for (var i = 0; i < count; i++) v[i] = alpha * math.tanh(math.clamp(beta * v[i], -16.0f, 16.0f));
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyHardSigmoid(float* v, int count, float alpha, float beta)
            {
                for (var i = 0; i < count; i++) v[i] = math.clamp(alpha * v[i] + beta, 0.0f, 1.0f);
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplyElu(float* v, int count, float alpha)
            {
                for (var i = 0; i < count; i++) v[i] = v[i] >= 0.0f ? v[i] : alpha * (math.exp(math.clamp(v[i], -60.0f, 60.0f)) - 1.0f);
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplySoftsign(float* v, int count)
            {
                for (var i = 0; i < count; i++) v[i] = 1.0f / (1.0f + math.abs(v[i]));
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            static void ApplySoftplus(float* v, int count)
            {
                for (var i = 0; i < count; i++) v[i] = math.log(1.0f + math.exp(math.clamp(v[i], -60.0f, 60.0f)));
            }

            static void ApplyActivation(float* v, int count, Layers.RnnActivation activation, float alpha, float beta)
            {
                switch (activation)
                {
                    case Layers.RnnActivation.Relu:
                        ApplyRelu(v, count);
                        break;
                    case Layers.RnnActivation.Tanh:
                        ApplyTanh(v, count);
                        break;
                    case Layers.RnnActivation.Sigmoid:
                        ApplySigmoid(v, count);
                        break;
                    case Layers.RnnActivation.Affine:
                        ApplyAffine(v, count, alpha, beta);
                        break;
                    case Layers.RnnActivation.LeakyRelu:
                        ApplyLeakyRelu(v, count, alpha);
                        break;
                    case Layers.RnnActivation.ThresholdedRelu:
                        ApplyThresholdedRelu(v, count, alpha);
                        break;
                    case Layers.RnnActivation.ScaledTanh:
                        ApplyScaledTanh(v, count, alpha, beta);
                        break;
                    case Layers.RnnActivation.HardSigmoid:
                        ApplyHardSigmoid(v, count, alpha, beta);
                        break;
                    case Layers.RnnActivation.Elu:
                        ApplyElu(v, count, alpha);
                        break;
                    case Layers.RnnActivation.Softsign:
                        ApplySoftsign(v, count);
                        break;
                    case Layers.RnnActivation.Softplus:
                        ApplySoftplus(v, count);
                        break;
                    default:
                        break;
                }
            }

            const int k_InnerLoopLength = 32;

            [SkipLocalsInit]
            public void Execute(int batchIndex)
            {
                var Yp = Yptr + batchIndex * yStride;

                if (seqIndex >= SequenceLensptr[batchIndex])
                {
                    UnsafeUtility.MemClear(Yp, sizeof(float) * hiddenSize);
                    return;
                }

                var it = stackalloc float[k_InnerLoopLength];
                var ft = stackalloc float[k_InnerLoopLength];
                var ct = stackalloc float[k_InnerLoopLength];
                var ot = stackalloc float[k_InnerLoopLength];

                var YCp = YCptr + batchIndex * hiddenSize;
                var YHp = YHptr + batchIndex * hiddenSize;
                var XsixWTp = XsixWTptr + batchIndex * xStride;
                var HtxRTp = HtxRTptr + batchIndex * 4 * hiddenSize;
                var Bp = Bptr;
                var Pp = Pptr;

                for (var start = 0; start < hiddenSize; start += k_InnerLoopLength)
                {
                    var count = math.min(k_InnerLoopLength, hiddenSize - start);
                    int i;

                    for (i = 0; i < count; i++)
                    {
                        it[i] = math.clamp(XsixWTp[0 * hiddenSize + i] + HtxRTp[0 * hiddenSize + i] + Bp[0 * hiddenSize + i] + Bp[4 * hiddenSize + i] + Pp[0 * hiddenSize + i] * YCp[i], -clip, clip);
                    }
                    ApplyActivation(it, count, fActivation, fAlpha, fBeta);

                    if (inputForget)
                    {
                        for (i = 0; i < count; i++)
                        {
                            ft[i] = 1f - it[i];
                        }
                    }
                    else
                    {
                        for (i = 0; i < count; i++)
                        {
                            ft[i] = math.clamp(XsixWTp[2 * hiddenSize + i] + HtxRTp[2 * hiddenSize + i] + Bp[2 * hiddenSize + i] + Bp[6 * hiddenSize + i] + Pp[2 * hiddenSize + i] * YCp[i], -clip, clip);
                        }
                        ApplyActivation(ft, count, fActivation, fAlpha, fBeta);
                    }

                    for (i = 0; i < count; i++)
                    {
                        ct[i] = math.clamp(XsixWTp[3 * hiddenSize + i] + HtxRTp[3 * hiddenSize + i] + Bp[3 * hiddenSize + i] + Bp[7 * hiddenSize + i], -clip, clip);
                    }
                    ApplyActivation(ct, count, gActivation, gAlpha, gBeta);

                    for (i = 0; i < count; i++)
                    {
                        YCp[i] = ft[i] * YCp[i] + it[i] * ct[i];
                        ot[i] = math.clamp(XsixWTp[1 * hiddenSize + i] + HtxRTp[1 * hiddenSize + i] + Bp[1 * hiddenSize + i] + Bp[5 * hiddenSize + i] + Pp[1 * hiddenSize + i] * YCp[i], -clip, clip);
                    }
                    UnsafeUtility.MemCpy(YHp, YCp, sizeof(float) * count);

                    ApplyActivation(ot, count, fActivation, fAlpha, fBeta);
                    ApplyActivation(YHp, count, hActivation, hAlpha, hBeta);

                    for (i = 0; i < count; i++)
                    {
                        YHp[i] = ot[i] * YHp[i];
                    }
                    UnsafeUtility.MemCpy(Yp, YHp, sizeof(float) * count);

                    YCp += count;
                    YHp += count;
                    Yp += count;
                    XsixWTp += count;
                    HtxRTp += count;
                    Bp += count;
                    Pp += count;
                }
            }
        }
    }
}
