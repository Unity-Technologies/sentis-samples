using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class RoundDenormalWeightsPass : IModelPass
    {
        [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
        internal unsafe struct RoundDenormalJob : IJobParallelFor
        {
            [NoAlias] [NativeDisableUnsafePtrRestriction] public uint* ptr;

            public void Execute(int index)
            {
                // Perform the equivalent of Single.IsSubnormal which is not available in the
                // .NET profile used by the Unity 2020.3 editor. Treat the float buffer as
                // unsigned integers and use bit checks to detect denormal numbers. Denormals
                // have a zero exponent field and a non-zero fraction field. The sign bit is
                // ignored. Replace denormals with a zero (same bit pattern for integer versus
                // float).
                //
                // IEEE-754 float: SEEE'EEEE'EFFF'FFFF'FFFF'FFFF'FFFF'FFFF (sign/exponent/fraction)
                if (((ptr[index] & 0x7f800000) == 0) && ((ptr[index] & 0x007fffff) != 0))
                    ptr[index] = 0;
            }
        }

        public void Run(ref Model model)
        {
            foreach (var constant in model.constants)
            {
                if (constant.weights == null || constant.dataType != DataType.Float)
                    continue;

                unsafe
                {
                    var job = new RoundDenormalJob
                    {
                        ptr = (uint*)constant.weights.RawPtr
                    };
                    var jobHandle = job.Schedule(constant.weights.Length, 1024);
                    jobHandle.Complete();
                }
            }
        }
    }
}
