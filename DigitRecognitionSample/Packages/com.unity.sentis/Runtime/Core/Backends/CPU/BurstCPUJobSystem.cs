using System;
using System.Diagnostics;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

namespace Unity.Sentis {

[JobProducerType(typeof(IParallelForBatchExtensions.ParallelForBatchJobStruct<>))]
interface IParallelForBatch
{
    void Execute(int i, int count);
}

static class IParallelForBatchExtensions
{
    internal struct ParallelForBatchJobStruct<T> where T : struct, IParallelForBatch
    {
        internal static readonly SharedStatic<IntPtr> jobReflectionData = SharedStatic<IntPtr>.GetOrCreate<ParallelForBatchJobStruct<T>>();

        [BurstDiscard]
        internal static unsafe void Initialize()
        {
            if (jobReflectionData.Data == IntPtr.Zero)
                jobReflectionData.Data = JobsUtility.CreateJobReflectionData(typeof(T), (ExecuteJobFunction)Execute);
        }

        delegate void ExecuteJobFunction(ref T data, IntPtr additionalPtr, IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex);

        static unsafe void Execute(ref T jobData, IntPtr additionalPtr, IntPtr bufferRangePatchData, ref JobRanges ranges, int jobIndex)
        {
            while (true)
            {
                int begin;
                int end;
                if (!JobsUtility.GetWorkStealingRange(ref ranges, jobIndex, out begin, out end))
                    break;

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                JobsUtility.PatchBufferMinMaxRanges(bufferRangePatchData, UnsafeUtility.AddressOf(ref jobData), begin, end - begin);
#endif

                jobData.Execute(begin, end - begin);
            }
        }
    }

    public static void EarlyJobInit<T>()
        where T : struct, IParallelForBatch
    {
        ParallelForBatchJobStruct<T>.Initialize();
    }

    static IntPtr GetReflectionData<T>()
        where T : struct, IParallelForBatch
    {
        ParallelForBatchJobStruct<T>.Initialize();
        var reflectionData = ParallelForBatchJobStruct<T>.jobReflectionData.Data;
        CheckReflectionDataCorrect(reflectionData);
        return reflectionData;
    }

    [Conditional("ENABLE_UNITY_COLLECTIONS_CHECKS")]
    static void CheckReflectionDataCorrect(IntPtr reflectionData)
    {
        if (reflectionData == IntPtr.Zero)
            throw new InvalidOperationException("Support for burst compiled calls to Schedule depends on the Jobs package.\n\nFor generic job types, please include [assembly: RegisterGenericJobType(typeof(MyJob<MyJobSpecialization>))] in your source file.");
    }

    public static unsafe JobHandle ScheduleBatch<T>(ref this T jobData, int arrayLength, int innerloopBatchCount, JobHandle dependsOn = new JobHandle()) where T : struct, IParallelForBatch
    {
        var scheduleParams = new JobsUtility.JobScheduleParameters(UnsafeUtility.AddressOf(ref jobData), GetReflectionData<T>(), dependsOn, ScheduleMode.Parallel);
        return JobsUtility.ScheduleParallelFor(ref scheduleParams, arrayLength, innerloopBatchCount);
    }
}

} // namespace Unity.Sentis
