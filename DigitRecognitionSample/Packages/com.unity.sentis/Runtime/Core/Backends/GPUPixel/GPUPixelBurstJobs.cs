using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;

namespace Unity.Sentis
{
    static class GPUPixelBurstJobs
    {
        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct IntBytesAsFloatJob : IJobParallelFor
        {
            [NativeDisableUnsafePtrRestriction]
            public int* src;
            public NativeArray<float> dest;

            public void Execute(int index)
            {
                var n = math.asuint(math.clamp(src[index], -1073741824, 1073741823));
                // switch second bit to inverse of third bit to void denormal, NaNs and infinite values
                var v = math.asfloat((n & 0xbfffffffu) | (~(n << 1) & 0x40000000u));
                dest[index] = v;
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct FloatBytesAsIntJob : IJobParallelFor
        {
            public NativeArray<float> src;
            [NativeDisableUnsafePtrRestriction]
            public int* dest;

            public void Execute(int index)
            {
                var n = math.asuint(src[index]);
                // set second bit to same as first bit to inverse int to float conversion
                dest[index] = (int)((n & 0xbfffffffu) + ((n >> 1) & 0x40000000u));
            }
        }
    }
}
