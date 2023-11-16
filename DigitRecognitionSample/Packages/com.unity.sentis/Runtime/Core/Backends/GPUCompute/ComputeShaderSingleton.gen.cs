// This is auto-generated -- do not modify directly

namespace Unity.Sentis
{
    public partial class ComputeShaderSingleton
    {
        private void RegisterGeneratedKernels()
        {
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.GenericA.gen", new[] {
                "Transpose",
                "InstanceNormalizationTail",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.ActivationA.gen", new[] {
                "LeakyRelu",
                "PRelu",
                "Swish",
                "Clip",
                "Relu",
                "Relu6",
                "Tanh",
                "Sigmoid",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.ActivationB.gen", new[] {
                "HardSigmoid",
                "Square",
                "Gelu",
                "Erf",
                "Celu",
                "Shrink",
                "ThresholdedRelu",
                "Elu",
                "Selu",
                "Softplus",
                "Ceil",
                "Floor",
                "Round",
                "Reciprocal",
                "Exp",
                "Log",
                "Sqrt",
                "Acos",
                "Acosh",
                "Asin",
                "Asinh",
                "Atan",
                "Atanh",
                "Cos",
                "Cosh",
                "Sin",
                "Sinh",
                "Tan",
                "Softsign",
                "HardSwish",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.PadA.gen", new[] {
                "PadBorderND",
                "PadReflectND",
                "PadSymmetricND",
                "PadEdgeND",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.PoolA.gen", new[] {
                "MaxPool2D",
                "AveragePool2D",
                "MaxPool1D",
                "AveragePool1D",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.Einsum.gen", new[] {
                "EinsumOne",
                "EinsumTwo",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.IndexingOpsA.gen", new[] {
                "Tile",
                "Gather",
                "GatherElements",
                "ScatterElements",
                "Expand",
                "Slice",
            });
            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl.Logical.gen", new[] {
                "Where",
            });
        }
    }
}