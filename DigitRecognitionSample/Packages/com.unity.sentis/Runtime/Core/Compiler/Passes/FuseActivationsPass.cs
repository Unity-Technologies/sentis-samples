using System;
using System.Linq;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class FuseActivationPass : IModelPass
    {
        public void Run(ref Model model)
        {
            //Fused activation
            var fusableActivations = model.layers.Where(l => IsActivationFusable(l)).ToList();
            // Fused activation
            foreach (var activationLayer in fusableActivations)
            {
                if (activationLayer.inputs.Length != 1)
                    continue;

                var mainLayer = model.layers.Find(l => l.name == activationLayer.inputs[0]);
                if (mainLayer == null)
                    continue;

                if (!(mainLayer is Layers.FusedActivation))
                    continue;
                if ((mainLayer as Layers.FusedActivation).fusedActivation != Layers.FusableActivation.None)
                    continue;

                if (model.outputs.Contains(mainLayer.name))
                    continue;

                //Need to check that no other layers uses mainLayer directly.
                //Activation in the graph below can not be fused because (concat) layer needs raw output of (conv) layer
                //conv -> relu -----.
                //    \             v
                //     `---------> concat
                if (model.layers.Exists(l => l != activationLayer && l.inputs.Contains(mainLayer.name)))
                    continue;

                if (activationLayer.flags.HasFlag(Layers.Flags.Preserve))
                    continue;

                FuseActivation(ref model, mainLayer, activationLayer);
            }
        }

        public static bool IsActivationFusable(Layers.Layer layer)
        {
            return (layer is Layers.Relu);
        }

        public static Layers.FusableActivation LayerToActivation(Layers.Layer layer)
        {
            if (layer is Layers.Relu)
                return Layers.FusableActivation.Relu;
            else
                return Layers.FusableActivation.None;
        }

        static private void FuseActivation(ref Model model, Layers.Layer mainLayer, Layers.Layer activationToFuse)
        {
            //patch `mainLayer`
            if (mainLayer is Layers.FusedActivation)
                (mainLayer as Layers.FusedActivation).fusedActivation = LayerToActivation(activationToFuse);

            //patch all layers depending on `activationToFuse`
            foreach (var l in model.layers)
            {
                for (int i = 0; i < l.inputs.Length; ++i)
                {
                    if (l.inputs[i] == activationToFuse.name)
                        l.inputs[i] = mainLayer.name;
                }
            }

            //remove `activationToFuse` if not an output, if an output make it an identity layer instead.
            if (model.outputs.Contains(activationToFuse.name))
            {
                int activationToFuseIndex = model.layers.FindIndex(x => x == activationToFuse);
                model.layers[activationToFuseIndex] = new Layers.Identity(activationToFuse.name, activationToFuse.inputs[0]);
            }
            else
                model.layers.Remove(activationToFuse);
        }
    }
}
