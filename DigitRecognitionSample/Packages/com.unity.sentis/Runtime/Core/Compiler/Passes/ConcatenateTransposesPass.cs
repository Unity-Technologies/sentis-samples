using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class ConcatenateTransposesPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var preserve =  new HashSet<string>(model.outputs);
            var removeLayers = new HashSet<string>();
            var transposeReferences = new Dictionary<string, int>();
            var layerDownstreamCounts = new Dictionary<string, int>();
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layers.Layer layer = model.layers[l];

                layerDownstreamCounts[layer.name] = 0;

                foreach (var input in layer.inputs)
                {
                    if (string.IsNullOrEmpty(input))
                        continue;
                    if (layerDownstreamCounts.ContainsKey(input))
                        layerDownstreamCounts[input] += 1;
                }

                if (!(layer is Layers.Transpose))
                    continue;

                transposeReferences[layer.name] = l;
            }

            for (int l = 0; l < model.layers.Count; ++l)
            {
                if (!(model.layers[l] is Layers.Transpose))
                    continue;
                Layers.Transpose layer = model.layers[l] as Layers.Transpose;

                string input = layer.inputs[0];

                if (!transposeReferences.ContainsKey(input))
                    continue;

                Layers.Transpose previousLayer = model.layers[transposeReferences[input]] as Layers.Transpose;

                if (previousLayer.flags.HasFlag(Layers.Flags.Preserve) && layer.flags.HasFlag(Layers.Flags.Preserve))
                    continue;

                // previous layer is a transpose and current layer is the only downstream layer
                var permutations = MergeTranspose(previousLayer.permutations, layer.permutations);

                model.layers[l] = new Layers.Transpose(layer.name, previousLayer.inputs[0], permutations);

                if (!preserve.Contains(input) && (layerDownstreamCounts[input] == 1))
                    removeLayers.Add(input);
            }

            Passes.PassesUtils.RemoveAndRemap(ref model, removeLayers, new Dictionary<string, string>());
        }

        int[] MergeTranspose(int[] transpose0, int[] transpose1)
        {
            return (new TensorShape(transpose0)).Transpose(transpose1).ToArray();
        }
    }
}
