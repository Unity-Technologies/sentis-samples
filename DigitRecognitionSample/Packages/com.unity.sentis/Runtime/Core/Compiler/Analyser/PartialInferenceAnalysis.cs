using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Sentis.Compiler.Analyser
{
    static class PartialInferenceAnalysis
    {
        public static PartialInferenceContext InferModelPartialTensors(Model model, bool useConstantWeights, IDictionary<string, TensorShape> inputShapes = null)
        {
            Profiler.BeginSample("Sentis.Compiler.Analyser.ShapeInferenceAnalysis.InferModelSymbolicTensors");

            var ctx = new PartialInferenceContext();

            foreach (var constant in model.constants)
            {
                if (useConstantWeights)
                    ctx.AddPartialTensor(constant.name, PartialTensor.FromTensor(constant.DataSetToTensor()));
                else
                    ctx.AddPartialTensor(constant.name, new PartialTensor(constant.dataType, new SymbolicTensorShape(constant.shape)));
            }

            // model inputs
            foreach (var input in model.inputs)
            {
                if (inputShapes != null && inputShapes.TryGetValue(input.name, out var inputShape))
                    ctx.AddPartialTensor(input.name, new PartialTensor(input.dataType, new SymbolicTensorShape(inputShape)));
                else
                    ctx.AddPartialTensor(input.name, new PartialTensor(input.dataType, input.shape));
            }

            foreach (var layer in model.layers)
            {
                // symbolic tensor inference
                var layerInputSymbolicTensors = ctx.GetPartialTensors(layer.inputs);
                var layerOutputSymbolicTensor = layer.InferPartialTensor(layerInputSymbolicTensors, ctx);
                ctx.AddPartialTensor(layer.name, layerOutputSymbolicTensor);
            }

            Profiler.EndSample();

            return ctx;
        }
    }
}

