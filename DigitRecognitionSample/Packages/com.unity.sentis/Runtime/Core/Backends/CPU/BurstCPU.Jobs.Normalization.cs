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
        unsafe struct LayerNormalizationTailJob : IJobParallelFor, IJobResourceDeclarationXSBWO
        {
            public float epsilon;
            public int axisDim;
            public int outerLength;
            public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
            public ReadOnlyMemResource S { get; set; } float* Sptr => (float*)S.ptr;
            public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
            public ReadOnlyMemResource W { get; set; } float* Wptr => (float*)W.ptr;
            public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

            const int k_InnerLoopLength = 32;

            [SkipLocalsInit]
            public void Execute(int outerIndex)
            {
                var Xp = Xptr + outerIndex * axisDim;
                var Sp = Sptr;
                var Bp = Bptr;
                var Op = Optr + outerIndex * axisDim;

                float mean = Wptr[outerIndex * 2 + 0];
                float variance = Wptr[outerIndex * 2 + 1];

                var it = stackalloc float[k_InnerLoopLength];

                for (var start = 0; start < axisDim; start += k_InnerLoopLength)
                {
                    var count = math.min(k_InnerLoopLength, axisDim - start);
                    int i;

                    for (i = 0; i < count; i++)
                    {
                        float scale = Sp[i];
                        float bias = Bp[i];
                        float v = Xp[i];

                        v = (v - mean) / math.sqrt(variance + epsilon);
                        v = scale * v + bias;

                        it[i] = v;
                    }

                    UnsafeUtility.MemCpy(Op, it, sizeof(float) * count);

                    Xp += count;
                    Sp += count;
                    Bp += count;
                    Op += count;
                }
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
        unsafe struct BatchNormalizationJob : IParallelForBatch
        {
            public float epsilon;
            public int channels;
            public int spatialLength;

            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public float* Xptr;
            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public float* Sptr;
            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public float* Bptr;
            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public float* Mptr;
            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public float* Vptr;

            [NoAlias] [NativeDisableUnsafePtrRestriction] public float* Optr;

            public void Execute(int i, int count)
            {
                float* Op = Optr + i;
                float* Xp = Xptr + i;

                // Extract the starting output position from the index.
                int os = i % spatialLength;
                i = i / spatialLength;
                int oc = i % channels;
                i = i / channels;

                float* Sp = Sptr + oc;
                float* Bp = Bptr + oc;
                float* Mp = Mptr + oc;
                float* Vp = Vptr + oc;

                float scale = Sp[0];
                float bias = Bp[0];
                float mean = Mp[0];
                float variance = Vp[0];

                // Advance to the starting input channel.
                int spatialLengthRemaining = spatialLength - os;

                while (count > 0)
                {
                    int spatialCountW = math.min(count, spatialLengthRemaining);
                    count -= spatialCountW;

                    for (; spatialCountW > 0; spatialCountW -= 1)
                    {
                        float v = Xp[0];
                        v = (v - mean) / math.sqrt(variance + epsilon);
                        v = scale * v + bias;

                        Xp++;
                        *Op++ = v;
                        os++;
                    }

                    if (count > 0)
                    {
                        // Output is now always aligned to the start of a row.
                        os = 0;
                        spatialLengthRemaining = spatialLength;

                        oc++;
                        Sp++;
                        Bp++;
                        Mp++;
                        Vp++;

                        if (oc == channels)
                        {
                            // Advance to the next output batch.
                            oc = 0;

                            Sp = Sptr;
                            Bp = Bptr;
                            Mp = Mptr;
                            Vp = Vptr;
                        }

                        scale = Sp[0];
                        bias = Bp[0];
                        mean = Mp[0];
                        variance = Vp[0];
                    }
                }
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Standard)]
        unsafe struct ScaleBiasJob : IParallelForBatch, IJobResourceDeclarationXSBO
        {
            public int channels;
            public int spatialLength;

            public ReadOnlyMemResource X { get; set; } float* Xptr => (float*)X.ptr;
            public ReadOnlyMemResource S { get; set; } float* Sptr => (float*)S.ptr;
            public ReadOnlyMemResource B { get; set; } float* Bptr => (float*)B.ptr;
            public ReadWriteMemResource O { get; set; } float* Optr => (float*)O.ptr;

            public void Execute(int i, int count)
            {
                float* Op = Optr + i;
                float* Xp = Xptr + i;

                // Extract the starting output position from the index.
                int os = i % spatialLength;
                i = i / spatialLength;
                int oc = i % channels;
                i = i / channels;

                float* Sp = Sptr + oc;
                float* Bp = Bptr + oc;

                float scale = Sp[0];
                float bias = Bp[0];

                // Advance to the starting input channel.
                int spatialLengthRemaining = spatialLength - os;

                while (count > 0)
                {
                    int spatialCountW = math.min(count, spatialLengthRemaining);
                    count -= spatialCountW;

                    for (; spatialCountW > 0; spatialCountW -= 1)
                    {
                        float v = Xp[0];
                        v = scale * v + bias;

                        Xp++;
                        *Op++ = v;
                        os++;
                    }

                    if (count > 0)
                    {
                        // Output is now always aligned to the start of a row.
                        os = 0;
                        spatialLengthRemaining = spatialLength;

                        oc++;
                        Sp++;
                        Bp++;

                        if (oc == channels)
                        {
                            // Advance to the next output batch.
                            oc = 0;

                            Sp = Sptr;
                            Bp = Bptr;
                        }

                        scale = Sp[0];
                        bias = Bp[0];
                    }
                }
            }
        }
    }
}
