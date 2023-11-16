using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Sentis.Layers;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseConstantsPass : IModelPass
    {
        public void Run(ref Model model)
        {
            FuseConstants(ref model);
        }

        static void FuseConstants(ref Model model)
        {
            var computedLayerTensors = ComputeKnownLayerTensors(model);

            // remove precalculated layers
            model.layers.RemoveAll(x => computedLayerTensors.ContainsKey(x.name));

            // add precalculated constants
            foreach (var kvp in computedLayerTensors)
            {
                model.constants.Add(new Constant(kvp.Key, kvp.Value));
                kvp.Value.Dispose();
            }

            // remove unused constants
            var removeUnusedPass = new Cleanup.RemoveUnusedPass();
            removeUnusedPass.Run(ref model);
        }

        static Dictionary<string, Tensor> ComputeKnownLayerTensors(Model model)
        {
            var ctx = new PartialInferenceContext();
            var backend = new CPUBackend();
            var vars = new DefaultVars();
            var executionContext = new ExecutionContext
            {
                backend = backend,
                vars = vars
            };

            var constantTensors = new Dictionary<string, Tensor>();
            var calculatedTensors = new Dictionary<string, Tensor>();

            // model constants
            foreach (var constant in model.constants)
            {
                var constantTensor = constant.DataSetToTensor();
                constantTensors.Add(constant.name, constantTensor);
                ctx.AddPartialTensor(constant.name, PartialTensor.FromTensor(constantTensor));
            }

            // model inputs
            foreach (var input in model.inputs)
            {
                ctx.AddPartialTensor(input.name, new PartialTensor(input.dataType, input.shape));
            }

            // iterate through layers executing if layer inputs are all known
            foreach (var layer in model.layers)
            {
                var isDeterministic = !layer.GetType().IsDefined(typeof(NonDeterministicOutput));
                for (var i = 0; i < layer.inputs.Length && isDeterministic; i++)
                {
                    isDeterministic &= string.IsNullOrEmpty(layer.inputs[i]) || calculatedTensors.ContainsKey(layer.inputs[i]) || constantTensors.ContainsKey(layer.inputs[i]);
                }

                if (!isDeterministic)
                {
                    // partial tensor inference
                    var layerInputPartialTensors = ctx.GetPartialTensors(layer.inputs);
                    var outputPartialTensor = layer.InferPartialTensor(layerInputPartialTensors, ctx);
                    ctx.AddPartialTensor(layer.name, outputPartialTensor);
                    if (outputPartialTensor.IsFullyKnown())
                        calculatedTensors.Add(layer.name, outputPartialTensor.ToTensor());
                    for (var i = 1; i < (layer.outputs?.Length ?? 0); i++)
                    {
                        outputPartialTensor = ctx.GetPartialTensor(layer.outputs[i]);
                        if (outputPartialTensor.IsFullyKnown())
                            calculatedTensors.Add(layer.outputs[i], outputPartialTensor.ToTensor());
                    }
                    continue;
                }

                var inputs = new Tensor[layer.inputs.Length];
                for (var i = 0; i < layer.inputs.Length; i++)
                {
                    if (string.IsNullOrEmpty(layer.inputs[i]))
                        continue;
                    if (calculatedTensors.TryGetValue(layer.inputs[i], out var calculatedInputTensor))
                        inputs[i] = calculatedInputTensor;
                    else
                        inputs[i] = constantTensors[layer.inputs[i]];
                }

                // full inference
                var outputTensor = layer.Execute(inputs, executionContext);
                calculatedTensors.Add(layer.name, outputTensor);
                ctx.AddPartialTensor(layer.name, PartialTensor.FromTensor(outputTensor));

                if (layer.outputs == null)
                    continue;

                for (var i = 1; i < layer.outputs.Length; i++)
                {
                    outputTensor = vars.PeekOutput(layer.outputs[i]);
                    calculatedTensors.Add(layer.outputs[i], outputTensor);
                    ctx.AddPartialTensor(layer.outputs[i], PartialTensor.FromTensor(outputTensor));
                }
            }

            foreach (var constantTensor in constantTensors.Values)
            {
                constantTensor.Dispose();
            }

            vars.Dispose();

            return calculatedTensors;
        }
    }
}
