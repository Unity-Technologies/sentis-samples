using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Sentis.Compiler.Passes.Cleanup
{
    // TODO remove useless patterns:
    // Reduce keepdim 0 -> * -> Reshape
    class RemoveNoOpsPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var noopLayers = new HashSet<string>();
            var remap = new Dictionary<string, string>();
            var preserve = new HashSet<string>(model.outputs);

            // algorithm:
            // - if input is pointing to a noop, we need to remap it to upstream layer
            // - if layer is a noop, store its link to upstream layer
            // layers are in order of appearance, so if layer_N has layer_M as input, we'd have treated layer_M before
            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                // replace removed layers with their upstream inputs
                for (int i = 0; i < layer.inputs.Length; ++i)
                {
                    var input = layer.inputs[i];

                    if (remap.ContainsKey(input))
                    {
                        Assert.IsTrue(noopLayers.Contains(input));
                        model.layers[l].inputs[i] = remap[input];
                    }
                    else
                    {
                        Assert.IsFalse(noopLayers.Contains(input));
                    }
                }

                if (layer.flags.HasFlag(Layers.Flags.Preserve))
                    continue;

                if (layer.inputs.Length == 0) // const
                    continue;

                // if layer is noop = nop, identity or flatten
                if (layer is Layers.Identity)
                {
                    remap[layer.name] = layer.inputs[0];
                    noopLayers.Add(layer.name);
                }
            }

            model.layers.RemoveAll(x => noopLayers.Contains(x.name) && !preserve.Contains(x.name));
        }

        static bool IsLayerNoop(Model model, Layers.Layer layer)
        {
            if (layer is Layers.Identity)
                return true;
            else
                return false;
        }
    }
}
