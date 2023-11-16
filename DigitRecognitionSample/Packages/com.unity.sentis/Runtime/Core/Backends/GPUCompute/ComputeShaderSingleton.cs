using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents the compute kernel cache for backends that use GPU compute.
    /// </summary>
    public partial class ComputeShaderSingleton
    {
        /// <summary>
        /// Whether kernel usage tracking is enabled.
        /// </summary>
        public bool EnableDebug = false;

        static readonly ComputeShaderSingleton instance = new ComputeShaderSingleton();

        // Maps kernel name -> shader name
        Dictionary<string, string> m_KernelToShaderName = new Dictionary<string, string>();

        // Maps kernel name -> kernel index
        Dictionary<string, int> m_KernelNameToID = new Dictionary<string, int>();

        // Maps kernel name -> kernel threadgroup size
        Dictionary<string, ValueTuple<uint, uint, uint>> m_KernelNameToTGS = new Dictionary<string, ValueTuple<uint, uint, uint>>();

        // Maps shader name -> ComputeShader
        Dictionary<string, ComputeShader> m_ShaderNameToComputeShader = new Dictionary<string, ComputeShader>();

        HashSet<string> m_UsedKernels = new HashSet<string>();

        ComputeShaderSingleton()
        {
            RegisterGeneratedKernels();
            RegisterGeneratedKernelsA();

            RegisterKernels("Sentis/TextureConversion/TextureToTensor",
                new[] { "TextureToTensorExact", "TextureToTensorLinear" });

            RegisterKernels("Sentis/TextureConversion/TensorToTexture",
                new[] { "TensorToTextureExact", "TensorToTextureLinear" });

            RegisterKernels("Sentis/ComputeShaders/AxisActivations", new[]
            {
                "LogSoftmaxEnd",
                "SoftmaxEnd",
                "HardmaxEnd",
            });

            RegisterKernels("Sentis/ComputeShaders/CumSum", new[]
            {
                "CumSumFloatForwardInclusive",
                "CumSumFloatForwardExclusive",
                "CumSumFloatReverseInclusive",
                "CumSumFloatReverseExclusive",
                "CumSumIntForwardInclusive",
                "CumSumIntForwardExclusive",
                "CumSumIntReverseInclusive",
                "CumSumIntReverseExclusive",
            });

            RegisterKernels("Sentis/ComputeShaders/ReferenceImpl",
                new[] { "MatMul" });

            RegisterKernels("Sentis/ComputeShaders/RNN",
                new[] { "LSTMEnd" });

            RegisterKernels("Sentis/ComputeShaders/Activations",
                new[] { "AbsFloat", "NegFloat", "SignFloat", "AbsInt", "NegInt", "SignInt", "IsInf", "IsNaN" });

            RegisterKernels("Sentis/ComputeShaders/LogicalOps",
                new[] { "Or", "And", "Xor", "Not" });

            RegisterKernels("Sentis/ComputeShaders/CompareOps",
                new[]
                {
                    "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Equal",
                    "GreaterInt", "GreaterOrEqualInt", "LessInt", "LessOrEqualInt", "EqualInt"
                });

            RegisterKernels("Sentis/ComputeShaders/GroupConv",
                new[] { "GroupedConv3D", "GroupedConv2D", "GroupedConv1D",
                        "GroupedConv3D_GroupLower64", "GroupedConv2D_GroupLower64", "GroupedConv1D_GroupLower64" });

            RegisterKernels("Sentis/ComputeShaders/Conv",
                new[]
                {
                    "Conv3D_T16x16_R4x4", "Conv3D_1x1_T16x16_R4x4",
                    "Conv2D_T16x16_R4x4", "Conv2D_1x1_T16x16_R4x4",
                    "Conv1D_T16x16_R4x4", "Conv1D_1x1_T16x16_R4x4"
                });

            RegisterKernels("Sentis/ComputeShaders/DepthwiseConv",
                new[] { "DepthwiseConv2DDirect", "DepthwiseConv2DWinograd", "KernelWinoExpand" });

            RegisterKernels("Sentis/ComputeShaders/ConvTranspose",
                new[]
                {
                    "ConvTranspose3D_T16x16_R4x4",
                    "ConvTranspose2D_T16x16_R4x4",
                    "ConvTranspose1D_T16x16_R4x4"
                });

            RegisterKernels("Sentis/ComputeShaders/Dense",
                new[]
                {
                    "Dense_T8x8_R4x4", "Dense_T16x16_R4x4", "Dense_V_L1Cached64",
                    "Gemm_T8x8_R4x4", "Gemm_T16x16_R4x4", "Gemm_V_L1Cached64",
                    "GemmBatched_T8x8_R4x4", "GemmBatched_T16x16_R4x4",
                });

            RegisterKernels("Sentis/ComputeShaders/GemmT",
                new[]
                {
                    "GemmT_XT_T8x8_R4x4", "GemmT_WT_T8x8_R4x4", "GemmT_XT_WT_T8x8_R4x4",
                });

            RegisterKernels("Sentis/ComputeShaders/Pool",
                new[]
                {
                    "AveragePoolReduce", "MaxPoolReduce",
                    "GlobalAveragePool", "GlobalMaxPool",
                    "AverageVariancePoolReduce", "GlobalAverageVariancePool",
                    "ArgMaxReduce", "GlobalArgMaxReduce"
                });

            RegisterKernels("Sentis/ComputeShaders/Normalization",
                new[] { "LayerNormalizationTail", "BatchNormalization", "ScaleBias" });

            RegisterKernels("Sentis/ComputeShaders/ReduceIndices",
                new[]
                {
                    "ArgMaxFloatFirst", "ArgMinFloatFirst", "ArgMaxFloatLast", "ArgMinFloatLast",
                    "ArgMaxIntFirst", "ArgMinIntFirst", "ArgMaxIntLast", "ArgMinIntLast",
                });

            RegisterKernels("Sentis/ComputeShaders/CopyOps",
                new[] { "MemCopy", "MemCopyStride", "MemSet", "Split", "Tril", "Triu", "RangeFloat", "RangeInt", "CastToInt", "CastToFloat", "Transpose2D" });

            RegisterKernels("Sentis/ComputeShaders/Random",
                new[] { "RandomUniform", "RandomNormal", "BernoulliFloat", "BernoulliInt" });

            RegisterKernels("Sentis/ComputeShaders/IndexingOps",
                new[] { "OneHot", "GatherND" });

            RegisterKernels("Sentis/ComputeShaders/SortingOps",
                new[] { "TopKLargest", "TopKSmallest" });

            RegisterKernels("Sentis/ComputeShaders/ScatterOps",
                new[] { "ScatterNDFloat", "ScatterNDInt" });

            RegisterKernels("Sentis/ComputeShaders/Upsample",
                new[]
                {
                    "Upsample1D_Nearest_Floor", "Upsample1D_Nearest_Ceil", "Upsample1D_Linear_None",
                    "Upsample2D_Nearest_Floor", "Upsample2D_Nearest_Ceil", "Upsample2D_Linear_None",
                    "Upsample3D_Nearest_Floor", "Upsample3D_Nearest_Ceil", "Upsample3D_Linear_None",
                });

            RegisterKernels("Sentis/ComputeShaders/ImageBased",
                new[] { "DepthToSpaceDepthColumnRow", "DepthToSpaceColumnRowDepth", "SpaceToDepth", "RoiAlignAvg", "RoiAlignMax" });
        }

        void RegisterKernels(string shaderName, string[] kernelNames)
        {
            foreach (var kernelName in kernelNames)
            {
                m_KernelToShaderName[kernelName] = shaderName;

                var shader = FindComputeShader(kernelName);
                var kernelIndex = shader.FindKernel(kernelName);
                shader.GetKernelThreadGroupSizes(kernelIndex, out uint threadGroupSizeX, out uint threadGroupSizeY, out uint threadGroupSizeZ);

                m_KernelNameToID[kernelName] = kernelIndex;
                m_KernelNameToTGS[kernelName] = new ValueTuple<uint, uint, uint>(threadGroupSizeX, threadGroupSizeY, threadGroupSizeZ);
            }
        }

        internal int GetKernelIndex(string kernelName)
        {
            if (!m_KernelNameToID.TryGetValue(kernelName, out int kernelIndex))
                throw new ArgumentException($"Kernel {kernelName} is missing");

            return kernelIndex;
        }

        internal void GetKernelThreadGroupSizes(string kernelName, out uint x, out uint y, out uint z)
        {
            if (!m_KernelNameToTGS.TryGetValue(kernelName, out var kernelTGS))
                throw new ArgumentException($"Kernel {kernelName} is missing");

            x = kernelTGS.Item1;
            y = kernelTGS.Item2;
            z = kernelTGS.Item3;
        }

        internal ComputeShader FindComputeShader(string kernelName)
        {
            string shaderName = null;
            m_KernelToShaderName.TryGetValue(kernelName, out shaderName);

            // Kernel not found
            if (shaderName == null)
                return null;

            if (EnableDebug)
                m_UsedKernels.Add(kernelName);

            return FindComputeShaderFromShaderName(shaderName);
        }

        ComputeShader FindComputeShaderFromShaderName(string shaderName)
        {
            var found = m_ShaderNameToComputeShader.TryGetValue(shaderName, out var cs);
            if (!found)
            {
                Profiler.BeginSample(shaderName);
                cs = Resources.Load<ComputeShader>(shaderName);
                m_ShaderNameToComputeShader[shaderName] = cs;
                Profiler.EndSample();
            }
            return cs;
        }

        /// <summary>
        /// Loads and compiles given compute kernels without running them.
        /// </summary>
        /// <param name="kernels">List of kernel names to load and compile.</param>
        /// <returns>Enumerator to iterate.</returns>
        public IEnumerator WarmupKernels(List<string> kernels)
        {
            foreach (var kernel in kernels)
            {
                var shader = m_KernelToShaderName[kernel];
                if (!m_ShaderNameToComputeShader.ContainsKey(shader))
                {
                    FindComputeShaderFromShaderName(shader);
                    yield return null;
                }
            }
            yield break;
        }

        /// <summary>
        /// Returns used compute kernels as a list.
        /// </summary>
        /// <returns>List of used kernels.</returns>
        public List<string> GetUsedKernels()
        {
            if (!EnableDebug)
            {
                D.LogWarning("List of used kernels was requested while ComputeShaderSingleton.EnableDebug == false");
                return null;
            }

            return m_UsedKernels.ToList();
        }

        /// <summary>
        /// Initializes or returns the instance of `ComputeShaderSingleton`.
        /// </summary>
        public static ComputeShaderSingleton Instance {
            get { return instance; }
        }

        /// <summary>
        /// Whether the GPU supports compute.
        /// </summary>
        public bool supported { get { return SystemInfo.supportsComputeShaders; } }
    }
}
