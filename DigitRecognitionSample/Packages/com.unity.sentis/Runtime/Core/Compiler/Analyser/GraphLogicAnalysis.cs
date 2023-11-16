using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.Sentis.Compiler.Analyser
{
    static class GraphLogicAnalysis
    {
        public static string GetDefaultInputName(Model model)
        {
            return model.inputs.Count > 0 ? model.inputs[0].name : null;
        }

        public static string GetDefaultOutputName(Model model)
        {
            return model.outputs.Count > 0 ? model.outputs[0] : null;
        }

        public static int GetDownStreamLayersCount(Model model, string name)
        {
            return model.layers.Count(x => x.inputs.Contains(name));
        }
    }
}

