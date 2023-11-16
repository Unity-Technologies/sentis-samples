using System.Collections.Generic;
using Unity.Sentis.Compiler.Analyser;
using Unity.Sentis;

namespace Unity.Sentis.Compiler.Validation
{
    struct ValidateShapeInference : IValidationPass
    {
        public void Run(Model model)
        {
            MemoryFootprintAnalysis.FindLayersThatRequireStorage(model);
        }
    }
}
