using System;

namespace Unity.Sentis
{
    /// <summary>
    /// Layer output is non deterministic, i.e. may be different each time the layer is run
    /// e.g. Layers which use random number generation such as RandomUniform, Bernoulli, Multinomial
    /// These layers will not have outputs precalculated at import time for optimization passes
    /// </summary>
    public class NonDeterministicOutput : Attribute { }

    namespace Optimization.CPUFallback
    {
        /// <summary>
        /// Flags that specified layer inputs needs to be read on the CPU
        /// allows to schedule upstream layers to be scheduled on the CPU, avoiding CPU/GPU sync
        /// </summary>
        class CPUReadInputs : Attribute
        {
            public int[] InputsOnCPU { get; set; }
            public CPUReadInputs(params int[] inputsOnCPU)
            {
                this.InputsOnCPU = inputsOnCPU;
            }
        }

        /// <summary>
        /// Layer doesn't need input data
        /// allows to stop propagating that upstream data should run on the CPU
        /// layer might have additional inputs that needs to be run on the CPU (e.g. CastLike)
        /// </summary>
        class NoDataDependencyInputs : Attribute
        {
            public int[] InputsNoDataDependency { get; set; }
            public NoDataDependencyInputs(params int[] inputsNoDataDependency)
            {
                this.InputsNoDataDependency = inputsNoDataDependency;
            }
        }
    }
}
