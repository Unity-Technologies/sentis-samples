using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents the compute kernel cache for backends that use GPU pixel shaders.
    /// </summary>
    public sealed class PixelShaderSingleton
    {
        /// <summary>
        /// Whether kernel usage tracking is enabled.
        /// </summary>
        public bool EnableDebug = false;

        static readonly PixelShaderSingleton instance = new PixelShaderSingleton();

        // Maps shader name -> Shader
        Dictionary<string, Material> m_shaderNameToMaterial = new Dictionary<string, Material>();

        HashSet<string> m_usedShaders = new HashSet<string>();

        internal Material FindMaterial(string kernelName)
        {
            if (EnableDebug) m_usedShaders.Add(kernelName);

            if (!m_shaderNameToMaterial.TryGetValue(kernelName, out var material) || material == null)
            {
                Profiler.BeginSample(kernelName);
                material = new Material(Shader.Find(kernelName));
                m_shaderNameToMaterial[kernelName] = material;
                Profiler.EndSample();
                return material;
            }

            // Avoid state leaking by disabling all shader keyword before returning material
            var keywords = material.shaderKeywords;
            for (int i = 0; i < keywords.Length; ++i)
            {
                material.DisableKeyword(keywords[i]);
            }

            return material;
        }

        /// <summary>
        /// Loads and compiles given pixel shaders without running them.
        /// </summary>
        /// <param name="shaders">List of shader names to load and compile.</param>
        /// <returns>Enumerator to iterate.</returns>
        public IEnumerator WarmupKernels(List<string> shaders)
        {
            foreach (var shader in shaders)
            {
                if (!m_shaderNameToMaterial.ContainsKey(shader))
                {
                    FindMaterial(shader);
                    yield return null;
                }
            }
            yield break;
        }

        /// <summary>
        /// Returns used pixel shaders as a list.
        /// </summary>
        /// <returns>List of used pixel shaders.</returns>
        public List<string> GetUsedShaders()
        {
            if (!EnableDebug)
            {
                D.LogWarning("List of used pixel shaders was requested while PixelShaderSingleton.EnableDebug == false");
                return null;
            }

            return m_usedShaders.ToList();
        }

        /// <summary>
        /// Initializes or returns the instance of `PixelShaderSingleton`.
        /// </summary>
        public static PixelShaderSingleton Instance => instance;
    }
}
