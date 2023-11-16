using System.Collections.Generic;

namespace Unity.Sentis.Compiler.Passes
{
static class PassesUtils
{
    /// <summary>
    /// removes specified layers and remap inputs accordingly
    /// </summary>
    public static void RemoveAndRemap(ref Model model, HashSet<string> removeLayers, Dictionary<string, string> remap)
    {
        model.layers.RemoveAll(l => removeLayers.Contains(l.name));
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
    }
}
} // namespace Unity.Sentis
