using UnityEngine;
using UnityEngine.Rendering;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents information about the compute capabilities of the GPU.
    /// </summary>
    public class ComputeInfo
    {
        /// <summary>
        /// Whether the GPU supports shared memory.
        /// </summary>
        public static bool supportsComputeSharedMemory = true;

        /// <summary>
        /// Whether the GPU supports dense 32 x 32 kernels.
        /// </summary>
        public static bool supportsDense32x32 = true;

        /// <summary>
        /// Whether the GPU supports dense 64 x 64 kernels.
        /// </summary>
        public static bool supportsDense64x64 = true;

        /// <summary>
        /// Whether the GPU supports compute.
        /// </summary>
        public static bool supportsCompute = true;

        /// <summary>
        /// The maximum compute work group size the GPU supports.
        /// </summary>
        public static uint maxComputeWorkGroupSize = 1024;

        /// <summary>
        /// The vendor of the GPU.
        /// </summary>
        public static string graphicsDeviceVendor = "";

        /// <summary>
        /// Determines whether the GPU is a mobile GPU, for example Android, iPhone or Intel.
        /// </summary>
        /// <returns>Whether the GPU is a mobile GPU.</returns>
        public static bool IsMobileGPU() { return
            (Application.platform == RuntimePlatform.Android) ||
            (Application.platform == RuntimePlatform.IPhonePlayer);
        }

        /// <summary>
        /// Determines whether the GPU is a Mac GPU.
        /// </summary>
        /// <returns>Whether the GPU is a Mac GPU.</returns>
        public static bool IsMacGPU() { return
            (Application.platform == RuntimePlatform.OSXEditor) ||
            (Application.platform == RuntimePlatform.OSXPlayer);
        }

        /// <summary>
        /// Determines whether the GPU is an iPhone GPU.
        /// </summary>
        /// <returns>Whether the GPU is an iPhone GPU.</returns>
        public static bool IsiPhoneGPU() { return
            (Application.platform == RuntimePlatform.IPhonePlayer);
        }

        /// <summary>
        /// Determines whether the GPU is an Android Qualcomm GPU.
        /// </summary>
        /// <returns>Whether the GPU is an Android Qualcomm GPU.</returns>
        public static bool IsQualcommGPU() { return
            (Application.platform == RuntimePlatform.Android) && graphicsDeviceVendor.Contains("Qualcomm");
        }

        /// <summary>
        /// Determines whether the GPU is an Android ARM GPU.
        /// </summary>
        /// <returns>Whether the GPU is an Android ARM GPU.</returns>
        public static bool IsARMGPU() { return
            (Application.platform == RuntimePlatform.Android) && graphicsDeviceVendor.Contains("ARM");
        }

        static ComputeInfo()
        {
            supportsCompute = SystemInfo.supportsComputeShaders;

            graphicsDeviceVendor = SystemInfo.graphicsDeviceVendor;

            // TODO switch to SystemInfo.maxComputeWorkGroupSize when we bump min spec to 2019.3
            if (Application.platform == RuntimePlatform.Android)
            {
                maxComputeWorkGroupSize = (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Vulkan) ? 256u : 128u;

                var gpuName = SystemInfo.graphicsDeviceName ?? "";
                var osName = SystemInfo.operatingSystem ?? "";

                // Known issue with Adreno Vulkan drivers on Android 8.x
                if (gpuName.Contains("Adreno") && osName.StartsWith("Android OS 8") &&
                    SystemInfo.graphicsDeviceType == GraphicsDeviceType.Vulkan)
                    maxComputeWorkGroupSize = 128u;
            }
            else if (Application.platform == RuntimePlatform.IPhonePlayer || Application.platform == RuntimePlatform.tvOS)
            {
                var gpuName = SystemInfo.graphicsDeviceName;
                if (gpuName != null && gpuName.StartsWith("Apple A"))
                {
                    int gpuNumber = 0, idx = "Apple A".Length;
                    while (idx < gpuName.Length && '0' <= gpuName[idx] && gpuName[idx] <= '9')
                    {
                        gpuNumber = gpuNumber * 10 + gpuName[idx++] - '0';
                    }

                    // TODO check on lower end iOS devices
                    maxComputeWorkGroupSize = (gpuNumber <= 10) ? 224u : 256u;
                }
                else
                {
                    maxComputeWorkGroupSize = 256u;
                }
            }
        }
    }
}
