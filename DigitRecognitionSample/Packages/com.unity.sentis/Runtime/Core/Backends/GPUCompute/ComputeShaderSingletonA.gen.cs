// This is auto-generated -- do not modify directly

namespace Unity.Sentis
{
    public partial class ComputeShaderSingleton
    {
        void RegisterGeneratedKernelsA()
        {
            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.Broadcast.gen", new[]
            {
                "ScalarBroadcastPowFloat", "BroadcastPowFloat", "ElementwisePowFloat",
                "ScalarBroadcastPowInt", "BroadcastPowInt", "ElementwisePowInt",
                "ScalarBroadcastAddFloat", "BroadcastAddFloat", "ElementwiseAddFloat",
                "ScalarBroadcastSubFloat", "BroadcastSubFloat", "ElementwiseSubFloat",
                "ScalarBroadcastMulFloat", "BroadcastMulFloat", "ElementwiseMulFloat",
                "ScalarBroadcastDivFloat", "BroadcastDivFloat", "ElementwiseDivFloat",
                "ScalarBroadcastMinFloat", "BroadcastMinFloat", "ElementwiseMinFloat",
                "ScalarBroadcastMaxFloat", "BroadcastMaxFloat", "ElementwiseMaxFloat",
                "ScalarBroadcastMeanFloat", "BroadcastMeanFloat", "ElementwiseMeanFloat",
                "ScalarBroadcastFModFloat", "BroadcastFModFloat", "ElementwiseFModFloat",
                "ScalarBroadcastAddInt", "BroadcastAddInt", "ElementwiseAddInt",
                "ScalarBroadcastSubInt", "BroadcastSubInt", "ElementwiseSubInt",
                "ScalarBroadcastMulInt", "BroadcastMulInt", "ElementwiseMulInt",
                "ScalarBroadcastDivInt", "BroadcastDivInt", "ElementwiseDivInt",
                "ScalarBroadcastMinInt", "BroadcastMinInt", "ElementwiseMinInt",
                "ScalarBroadcastMaxInt", "BroadcastMaxInt", "ElementwiseMaxInt",
                "ScalarBroadcastModInt", "BroadcastModInt", "ElementwiseModInt",
                "ScalarBroadcastFModInt", "BroadcastFModInt", "ElementwiseFModInt",
"ScalarMad",
            });

            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.Conv.gen", new[]
            {
              "Conv2D_KxK",
              "Conv2D_1x1",
              "Conv1D_KxK",
              "Conv1D_1x1",
            });

            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.ConvTranspose.gen", new[]
            {
              "ConvTranspose2D_KxK",
              "ConvTranspose1D_KxK",
            });

            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.Reduction.gen", new[]
            {
                "ReduceMaxFloat",
                "GlobalReduceMaxFloat",
                "ReduceMinFloat",
                "GlobalReduceMinFloat",
                "ReduceSumFloat",
                "GlobalReduceSumFloat",
                "ReduceSumSquareFloat",
                "GlobalReduceSumSquareFloat",
                "ReduceMeanFloat",
                "GlobalReduceMeanFloat",
                "ReduceProdFloat",
                "GlobalReduceProdFloat",
                "ReduceL1Float",
                "GlobalReduceL1Float",
                "ReduceL2Float",
                "GlobalReduceL2Float",
                "ReduceSqrtFloat",
                "GlobalReduceSqrtFloat",
                "ReduceLogSumFloat",
                "GlobalReduceLogSumFloat",
                "ReduceLogSumExpFloat",
                "GlobalReduceLogSumExpFloat",
                "ReduceSumExpFloat",
                "GlobalReduceSumExpFloat",
                "ReduceMaxInt",
                "GlobalReduceMaxInt",
                "ReduceMinInt",
                "GlobalReduceMinInt",
                "ReduceSumInt",
                "GlobalReduceSumInt",
                "ReduceSumSquareInt",
                "GlobalReduceSumSquareInt",
                "ReduceProdInt",
                "GlobalReduceProdInt",
                "ReduceL1Int",
                "GlobalReduceL1Int",
            });
            RegisterKernels("Sentis/ComputeShaders/Compute.Shaders.ReductionUnrolled.gen", new[]
            {
                "UnrolledReduceMaxFloat",
                "UnrolledReduceMinFloat",
                "UnrolledReduceSumFloat",
                "UnrolledReduceSumSquareFloat",
                "UnrolledReduceMeanFloat",
                "UnrolledReduceProdFloat",
                "UnrolledReduceL1Float",
                "UnrolledReduceL2Float",
                "UnrolledReduceSqrtFloat",
                "UnrolledReduceLogSumFloat",
                "UnrolledReduceLogSumExpFloat",
                "UnrolledReduceSumExpFloat",
                "UnrolledReduceMaxInt",
                "UnrolledReduceMinInt",
                "UnrolledReduceSumInt",
                "UnrolledReduceSumSquareInt",
                "UnrolledReduceProdInt",
                "UnrolledReduceL1Int",
            });
        }
    }
}
