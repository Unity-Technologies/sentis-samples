using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseLinearLayersPass : IModelPass
    {
        public void Run(ref Model model)
        {
            using LinearLayerFusing linearLayerFusing = new LinearLayerFusing();
            Dictionary<string, Layers.Constant> constTensors = new Dictionary<string, Layers.Constant>();
            Dictionary<string, bool> sharedConstants = new Dictionary<string, bool>();
            foreach (var constant in model.constants)
                constTensors.Add(constant.name, constant);

            var layerNameToIndex = new Dictionary<string, int>();
            int idx = 0;
            foreach (var l in model.layers)
            {
                foreach (var i in l.inputs)
                {
                    if (!constTensors.ContainsKey(i))
                        continue;

                    if (sharedConstants.ContainsKey(i))
                        sharedConstants[i] = true;
                    else
                        sharedConstants.Add(i, false);
                }

                layerNameToIndex.Add(l.name, idx);
                idx++;
            }

            var preserve = new HashSet<string>(model.outputs);
            var remap = new Dictionary<string, string>();
            var mergedLayers = new HashSet<string>();

            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                bool isLayerLinear = LinearLayerFusing.IsLayerLinear(layer, constTensors);
                bool isLayerPreserved = layer.flags.HasFlag(Layers.Flags.Preserve) || preserve.Contains(layer.name);
                bool layerHasActivation = IsLayerFusedActivation(layer);

                if (!isLayerLinear)
                    continue;

                // if layer has an activation, we fuse it, but treat it as non linear for future children
                if (!layerHasActivation)
                {
                    remap[layer.name] = layer.name;
                }

                // Multi input nodes can only fuse constants and same inputs
                // only merge constants. @TODO: fuse equal input nodes
                var nonLinearInputs = layer.inputs.Where(x => !remap.ContainsKey(x) && !constTensors.ContainsKey(x)).ToList();
                var linearInputs = layer.inputs.Where(x => remap.ContainsKey(x)).ToList();

                // merge layer with one linearInput and eventual constants
                if (nonLinearInputs.Count > 0 || linearInputs.Count > 1)
                    continue;

                var input = linearInputs[0];

                // input is a linear layer, fuse it
                int inputLayerIndex = model.layers.FindIndex(x => x.name == remap[input]);
                Layers.Layer inputLayer = model.layers[inputLayerIndex];

                if (!AreLayersFusable(inputLayer, layer, constTensors, sharedConstants, linearLayerFusing))
                    continue;

                //  if input has more than 1 child, we're going to duplicate layers, not worth merging
                if (GraphLogicAnalysis.GetDownStreamLayersCount(model, input) != 1)
                    continue;

                // convention: layer will be fused into inputLayer
                // => fused layer will have the same inputs as inputLayer
                Layers.Layer fusedLayer = linearLayerFusing.FuseLayers(inputLayer, layer, constTensors);

                if (layerHasActivation)
                {
                    (fusedLayer as Layers.FusedActivation).fusedActivation = (layer as Layers.FusedActivation).fusedActivation;
                }

                // preserve layer if output/memory
                if (isLayerPreserved)
                {
                    // cannot merge layer into input:
                    // remove input, no need to remap as inputs == input.inputs
                    fusedLayer.name = layer.name;
                    model.layers[l] = fusedLayer;

                    if (!preserve.Contains(inputLayer.name))
                        mergedLayers.Add(inputLayer.name);
                }
                else
                {
                    // merge layer into input
                    // remove current and remap input names
                    mergedLayers.Add(layer.name);
                    remap[layer.name] = fusedLayer.name;
                    model.layers[inputLayerIndex] = fusedLayer;
                }
            }

            // remove merged layers
            Passes.PassesUtils.RemoveAndRemap(ref model, mergedLayers, remap);

            model.constants = constTensors.Values.ToList();

            // unpack maths ops const inputs into new const layer
            // remove unused constants
            var removeUnusedPass = new Cleanup.RemoveUnusedPass();
            removeUnusedPass.Run(ref model);
        }

        static bool IsLayerFusedActivation(Layers.Layer layer)
        {
            if (layer is Layers.FusedActivation)
                return (layer as Layers.FusedActivation).fusedActivation != Layers.FusableActivation.None;
            return false;
        }

        static bool AreLayersFusable(Layers.Layer l0, Layers.Layer l1, Dictionary<string, Layers.Constant> constTensors, Dictionary<string, bool> sharedConstants, LinearLayerFusing linearLayerFuser)
        {
            if (l0.inputs.Any(i => sharedConstants.ContainsKey(i) && sharedConstants[i]))
                return false;
            if (l1.inputs.Any(i => sharedConstants.ContainsKey(i) && sharedConstants[i]))
                return false;
            // can't fuse if input has a fused activation or if fusing code not implemented
            return !IsLayerFusedActivation(l0) && linearLayerFuser.AreLayersFusable(l0, l1, constTensors);
        }
    }
}
