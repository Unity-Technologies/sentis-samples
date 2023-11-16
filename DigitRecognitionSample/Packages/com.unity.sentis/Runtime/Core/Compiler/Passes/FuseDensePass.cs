using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using Unity.Sentis;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseDensePass : IModelPass
    {
        public void Run(ref Model model)
        {
            using var ops = new CPUOps();

            var preserve = new HashSet<string>(model.outputs);

            var inputs = new HashSet<string>();
            foreach (var input in model.inputs)
                inputs.Add(input.name);

            Dictionary<string, Tensor> constTensors = new Dictionary<string, Tensor>();
            foreach (var constant in model.constants)
                constTensors.Add(constant.name, constant.DataSetToTensor());

            var layerDownstream = new Dictionary<string, List<Layers.Layer>>();
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layers.Layer layer = model.layers[l];
                layerDownstream.Add(layer.name, new List<Layers.Layer>());

                foreach (var input in layer.inputs)
                {
                    if (string.IsNullOrEmpty(input) || inputs.Contains(input) || constTensors.ContainsKey(input))
                        continue;
                    layerDownstream[input].Add(layer);
                }

                if (layer.outputs == null)
                    continue;

                foreach (var output in layer.outputs)
                {
                    if (string.IsNullOrEmpty(output))
                        continue;
                    if (!layerDownstream.ContainsKey(output))
                        layerDownstream.Add(output, new List<Layers.Layer>());
                }
            }

            var removeLayers = new HashSet<string>();
            var remap = new Dictionary<string, string>();

            for (int l = 0; l < model.layers.Count - 1; ++l)
            {
                Layers.Layer layer = model.layers[l];
                if (!(layer is Layers.MatMul || (layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeA != true)))
                    continue;

                // const weights of rank 2
                string weightsName = layer.inputs[1];
                if (!(constTensors.ContainsKey(weightsName) && constTensors[weightsName].shape.rank == 2))
                    continue;

                // const bias of rank 1
                List<Layers.Layer> downStreamLayers = layerDownstream[layer.name];
                if (!downStreamLayers.Any(x => x is Layers.Add) && !downStreamLayers.Any(x => x is Layers.ScalarMad))
                    continue;

                Layers.Layer bias;
                string biasName;
                if (downStreamLayers.Any(x => x is Layers.ScalarMad))
                {
                    bias = downStreamLayers.Find(x => x is Layers.ScalarMad);
                    var biasMad = bias as Layers.ScalarMad;
                    if (biasMad.s != 1)
                        continue;
                    var biasS = biasMad.b;
                    using TensorFloat biasT = ops.ConstantOfShape(new TensorShape(constTensors[weightsName].shape[-1]), biasS);
                    biasName = bias.name + "_Bias";
                    constTensors.Add(biasName, biasT);
                    model.constants.Add(new Layers.Constant(biasName, biasT));
                }
                else
                {
                    bias = downStreamLayers.Find(x => x is Layers.Add);
                    var biasInputsConst = bias.inputs.Where(x =>
                        x != layer.name && constTensors.ContainsKey(x) && constTensors[x].shape.rank == 1).ToList();
                    if (biasInputsConst.Count != 1)
                        continue;
                    biasName = biasInputsConst[0];
                }

                if (preserve.Contains(bias.name))
                    continue;

                TensorFloat weightT = constTensors[weightsName] as TensorFloat;

                bool transposeWeights = (layer is Layers.MatMul2D && (layer as Layers.MatMul2D).transposeB == true);

                removeLayers.Add(weightsName);
                removeLayers.Add(bias.name);

                if (transposeWeights)
                {
                    weightsName = model.GetUniqueName(weightsName + "_t_for" + layer.name);
                    model.constants.Add(new Layers.Constant(weightsName, ops.Transpose(weightT)));
                }

                model.layers[l] = new Layers.Dense(layer.name, layer.inputs[0], weightsName, biasName);
                remap[bias.name] = layer.name;
            }

            Passes.PassesUtils.RemoveAndRemap(ref model, removeLayers, remap);

            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layers.Layer layer = model.layers[l];
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (remap.ContainsKey(input) && layer.name != remap[input])
                        model.layers[l].inputs[i] = remap[input];
                }
            }

            foreach (var t in constTensors.Values)
                t.Dispose();

            // remove unused constants
            var removeUnusedPass = new Cleanup.RemoveUnusedPass();
            removeUnusedPass.Run(ref model);
        }
    }
}
