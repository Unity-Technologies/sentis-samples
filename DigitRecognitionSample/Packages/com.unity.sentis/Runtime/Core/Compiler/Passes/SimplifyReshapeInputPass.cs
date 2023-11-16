using System;
using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using Unity.Sentis.Layers;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class SimplifyReshapeInputPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var reshapeLayers = model.layers.Where(l => l is Reshape).ToList();
            if (reshapeLayers.Count == 0)
                return;

            var allocator = new TensorCachingAllocator();
            var ctx = PartialInferenceAnalysis.InferModelPartialTensors(model, true);

            foreach (var layer in reshapeLayers)
            {
                var reshapeLayer = (Reshape)layer;
                var shapePartialTensor = ctx.GetPartialTensor(reshapeLayer.inputs[1]);
                if (!shapePartialTensor.isPartiallyKnown)
                    continue;
                var newShape = new PartialTensor(DataType.Int, shapePartialTensor.shape);
                for (var i = 0; i < shapePartialTensor.length; i++)
                    newShape[i] = shapePartialTensor[i];

                var input = ctx.GetPartialTensor(reshapeLayer.inputs[0]);
                var output = ctx.GetPartialTensor(reshapeLayer.name);

                // try and replace params and unknowns with values
                for (var i = 0; i < output.shape.rank; i++)
                {
                    if (!output.shape[i].isValue)
                        continue;
                    newShape[i] = (PartialTensorElement)output.shape[i];
                }

                // try and replace params with 0
                if (input.shape.hasRank && !reshapeLayer.allowZero)
                {
                    for (var i = 0; i < Mathf.Min(input.shape.rank, shapePartialTensor.length); i++)
                    {
                        if (input.shape[i].EqualsParam(output.shape[i]))
                            newShape[i] = PartialTensorElement.Zero;
                    }
                }

                // try and replace single param or unknown with -1
                var numZero = 0;
                var numMinusOne = 0;
                var numUnknown = 0;
                var unknownIndex = 0;
                for (var i = 0; i < newShape.length; i++)
                {
                    if (!newShape[i].isIntValue)
                    {
                        numUnknown++;
                        unknownIndex = i;
                        continue;
                    }

                    if (newShape[i] == 0)
                        numZero++;
                    else if (newShape[i] == -1)
                        numMinusOne++;
                }

                if (numMinusOne == 0 && numUnknown == 1 && (!reshapeLayer.allowZero || numZero == 0))
                    newShape[unknownIndex] = new PartialTensorElement(-1);

                if (!newShape.IsFullyKnown())
                    continue;

                var shapeName = model.GetUniqueName(reshapeLayer.name + "_Shape");
                using var shapeTensor = newShape.ToTensor();
                var shapeConstant = new Constant(model.GetUniqueName(shapeName), shapeTensor);
                reshapeLayer.inputs[1] = shapeName;
                model.AddConstant(shapeConstant);
            }
            allocator.Dispose();
        }
    }
}
