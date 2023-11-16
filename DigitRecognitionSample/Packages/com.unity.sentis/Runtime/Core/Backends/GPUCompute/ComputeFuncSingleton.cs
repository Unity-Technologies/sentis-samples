using System.Collections.Generic;

namespace Unity.Sentis
{
    class ComputeFuncSingleton
    {
        public static ComputeFuncSingleton Instance { get; } = new();

        Dictionary<string, ComputeFunc> s_Cache = new ();

        public ComputeFunc Get(string name)
        {
            var found = s_Cache.TryGetValue(name, out var cf);
            if (!found)
            {
                cf = new ComputeFunc(name);
                s_Cache[name] = cf;
            }

            return cf;
        }
    }
}
