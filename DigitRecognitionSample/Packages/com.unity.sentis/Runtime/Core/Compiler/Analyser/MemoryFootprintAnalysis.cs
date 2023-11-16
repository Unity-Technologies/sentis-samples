using System;
using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;

namespace Unity.Sentis.Compiler.Analyser
{
    static class MemoryFootprintAnalysis
    {
        public static HashSet<Layers.Layer> FindLayersThatRequireStorage(Model model)
        {
            var allInputsExceptFromPreviousLayer = new HashSet<string>();
            Layers.Layer prevLayer = null;
            foreach (var layer in model.layers)
            {
                foreach (var input in layer.inputs)
                {
                    if (string.IsNullOrEmpty(input))
                        continue;
                    if (prevLayer != null && input != prevLayer.name)
                        allInputsExceptFromPreviousLayer.Add(input);
                }
                prevLayer = layer;
            }

            var allOutputs = new HashSet<string>();
            foreach (var output in model.outputs)
                allOutputs.Add(output);
            allOutputs.Add(GraphLogicAnalysis.GetDefaultOutputName(model));

            var requireStorage = new HashSet<Layers.Layer>();
            foreach (var layer in model.layers)
            {
                if (allInputsExceptFromPreviousLayer.Contains(layer.name) ||
                    allOutputs.Contains(layer.name))
                    requireStorage.Add(layer);
            }

            return requireStorage;
        }
    }
}

