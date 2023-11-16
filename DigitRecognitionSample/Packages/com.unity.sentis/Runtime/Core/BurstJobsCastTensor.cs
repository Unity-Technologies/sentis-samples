using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;
using UnityEngine;

namespace Unity.Sentis
{
    static class BurstJobsCastTensor
    {
        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct DoubleBytesAsFloatJob : IJobParallelFor
        {
            [NoAlias][NativeDisableUnsafePtrRestriction] [ReadOnly] public long* src;
            [NoAlias][NativeDisableUnsafePtrRestriction]            public float* dst;

            public void Execute(int index)
            {
                double v = math.asdouble(src[index]);
                dst[index] = v < int.MinValue ? (float)int.MinValue : v > int.MaxValue ? (float)int.MaxValue : (float)v;
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct Float16BytesAsFloatJob : IJobParallelFor
        {
            [NoAlias][NativeDisableUnsafePtrRestriction] [ReadOnly] public ushort* src;
            [NoAlias][NativeDisableUnsafePtrRestriction]            public float* dst;

            public void Execute(int index)
            {
                dst[index] = Mathf.HalfToFloat(src[index]);
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct LongBytesAsFloatJob : IJobParallelFor
        {
            [NoAlias][NativeDisableUnsafePtrRestriction] [ReadOnly] public long* src;
            [NoAlias][NativeDisableUnsafePtrRestriction]            public int* dst;

            public void Execute(int index)
            {
                long v = src[index];
                dst[index] = v < (long)int.MinValue ? int.MinValue : v > (long)int.MaxValue ? int.MaxValue : (int)v;
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct BoolBytesAsFloatJob : IJobParallelFor
        {
            [NoAlias][NativeDisableUnsafePtrRestriction] [ReadOnly] public bool* src;
            [NoAlias][NativeDisableUnsafePtrRestriction]            public int* dst;

            public void Execute(int index)
            {
                bool v = src[index];
                dst[index] = v ? 1 : 0;
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct Uint8BytesAsFloatJob : IJobParallelFor
        {
            [NoAlias][NativeDisableUnsafePtrRestriction] [ReadOnly] public byte* src;
            [NoAlias][NativeDisableUnsafePtrRestriction]            public int* dst;

            public void Execute(int index)
            {
                byte v = src[index];
                dst[index] = (int)v;
            }
        }

        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        public unsafe struct Int8BytesAsFloatJob : IJobParallelFor
        {
            [NoAlias] [NativeDisableUnsafePtrRestriction] [ReadOnly] public sbyte* src;
            [NoAlias] [NativeDisableUnsafePtrRestriction]            public int* dst;

            public void Execute(int index)
            {
                sbyte v = src[index];
                dst[index] = (int)v;
            }
        }
    }
}
