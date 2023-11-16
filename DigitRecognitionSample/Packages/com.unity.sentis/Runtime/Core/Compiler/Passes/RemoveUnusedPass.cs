using System.Collections.Generic;
using System.Linq;

namespace Unity.Sentis.Compiler.Passes.Cleanup
{
    class RemoveUnusedPass : IModelPass
    {
        bool IsOutputUsed(Layers.Layer layer, HashSet<string> outputsUsed)
        {
            if (outputsUsed.Contains(layer.name))
                return true;

            if (layer.outputs == null)
                return false;

            foreach (var lo in layer.outputs)
            {
                if (outputsUsed.Contains(lo))
                    return true;
            }

            return false;
        }

        public void Run(ref Model model)
        {
            // algorithm:
            // bottom up graph iteration
            //  layer l
            //  check if previous layers uses l or any of its outputs
            //  if keep and add l.inputs as used outputs
            //  else remove
            var layersToRemove = new HashSet<string>();
            var outputsUsed = new HashSet<string>(model.outputs);
            for (var i = model.layers.Count - 1; i >= 0; i--)
            {
                var layer = model.layers[i];

                bool isOutputUsed = IsOutputUsed(layer, outputsUsed);

                if (isOutputUsed || layer.flags.HasFlag(Layers.Flags.Preserve))
                {
                    foreach (var input in layer.inputs)
                    {
                        if (string.IsNullOrEmpty(input))
                            continue;
                        outputsUsed.Add(input);
                    }
                }
                else
                {
                    layersToRemove.Add(layer.name);
                }
            }

            model.layers = model.layers.Where(l => !layersToRemove.Contains(l.name)).ToList();
            model.constants = model.constants.Where(c => outputsUsed.Contains(c.name)).ToList();
        }
    }
}
