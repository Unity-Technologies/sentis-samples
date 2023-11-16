using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;

[assembly: InternalsVisibleTo("Unity.Sentis.MacBLAS")]
[assembly: InternalsVisibleTo("Unity.Sentis.iOSBLAS")]

namespace Unity.Sentis
{
    /// <summary>
    /// BLAS plugin interface, allows to supply platform specific implementation of matrix multiplication
    /// </summary>
    interface BLASPlugin
    {
        /// <summary>
        /// Query if current platform is supported by the BLAS plugin
        /// </summary>
        /// <returns>`true` if plugin supports current platform</returns>
        bool IsCurrentPlatformSupported();

        /// <summary>
        /// Perform matrix multiplication C = A x B + C * beta
        /// </summary>
        /// <param name="dependsOn">input data dependency job handle</param>
        /// <param name="M">matrix A and C row count</param>
        /// <param name="N">matrix B and C column count</param>
        /// <param name="K">matrix A column count and matrix B row count</param>
        /// <param name="A">pointer to the matrix A</param>
        /// <param name="lda">matrix A leading dimension count</param>
        /// <param name="B">pointer to the matrix B</param>
        /// <param name="ldb">matrix B leading dimension count</param>
        /// <param name="C">pointer to the matrix C</param>
        /// <param name="ldc">matrix C leading dimension count</param>
        /// <param name="beta">matrix C scaling factor</param>
        /// <param name="transposeA">matrix A data is in transposed layout</param>
        /// <param name="transposeB">matrix B data is in transposed layout</param>
        /// <returns>job handle</returns>
        unsafe void SGEMM(int M, int N, int K, float* A, int lda, float* B, int ldb,
            float* C, int ldc, float beta, bool transposeA, bool transposeB);
    }

    internal class BLASPluginFactory
    {
        public static BLASPlugin CreateNativeBLASPlugin()
        {
            BLASPlugin blas = null;

            // TODO make plugins discoverable via custom attributes
            Stack<string> plugins = new Stack<string>();

            if (Application.platform == RuntimePlatform.IPhonePlayer)
                plugins.Push("Unity.Sentis.iOSBLAS");
            // TODO: MAC-BLAS using this fails on deeppose test
            //else if (Application.platform == RuntimePlatform.OSXPlayer || Application.platform == RuntimePlatform.OSXEditor)
            //    plugins.Push("Unity.Sentis.MacBLAS");

            while (plugins.Count > 0)
            {
                var candidate = plugins.Pop();
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
                {
                    var t = assembly.GetType(candidate);
                    if (t != null)
                    {
                        try
                        {
                            var inst = Activator.CreateInstance(t) as BLASPlugin;

                            if (inst != null && inst.IsCurrentPlatformSupported())
                                blas = inst;
                        }
                        catch (Exception e)
                        {
                            D.LogWarning($"Failed to load {t} with exception {e}");
                            break;
                        }
                    }
                }

                // Found working candidate
                if (blas != null)
                    break;
            }

            return blas;
        }
    }
}
