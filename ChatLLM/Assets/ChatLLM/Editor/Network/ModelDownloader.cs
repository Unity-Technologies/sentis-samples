using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using HuggingfaceHub;
using UnityEditor;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    public class ModelDownloader
    {
        readonly string m_ModelsPath = Application.dataPath + "/ChatLLM/Resources/Models";
        readonly Dictionary<string, int> m_LastLoggedProgress = new();

        public async Task DownloadModels()
        {
            var downloadTasks = new List<Task<string>>
            {
                CreateDownloadTask(LlavaConfig.DecoderModelFile, $"onnx/{LlavaConfig.DecoderModelFile}"),
                CreateDownloadTask(LlavaConfig.VisionEncoderModelFile, $"onnx/{LlavaConfig.VisionEncoderModelFile}"),
                CreateDownloadTask(LlavaConfig.EmbeddingModelFile, $"onnx/{LlavaConfig.EmbeddingModelFile}"),
                CreateDownloadTask(LlavaConfig.TokenizerModelFile, LlavaConfig.TokenizerModelFile)
            };

            try
            {
                await Task.WhenAll(downloadTasks);
                Debug.Log("All model files downloaded successfully.");
            }
            catch (Exception e)
            {
                Debug.LogError($"One or more downloads failed: {e}");
            }
            finally
            {
                var cacheDir = HFGlobalConfig.DefaultCacheDir;
                if (System.IO.Directory.Exists(cacheDir))
                {
                    System.IO.Directory.Delete(cacheDir, true);
                    Debug.Log($"Deleted cache directory: {cacheDir}");
                }

                AssetDatabase.Refresh();
            }
        }

        Task<string> CreateDownloadTask(string fileName, string remotePath)
        {
            var progress = new Progress<int>((p) => LogProgress(fileName, p));

            return Task.Run(() =>
                HFDownloader.DownloadFileAsync(
                    LlavaConfig.ModelId,
                    remotePath,
                    progress: progress,
                    localDir: m_ModelsPath
                )
            );
        }

        void LogProgress(string file, int progress)
        {
            m_LastLoggedProgress.TryAdd(file, -5);

            var lastProgress = m_LastLoggedProgress[file];
            if (progress - lastProgress >= 5 || progress == 100)
            {
                m_LastLoggedProgress[file] = progress;
                Debug.Log($"Downloading {file}: {progress}% complete.");
            }
        }

        public static bool VerifyModelsExist()
        {
            var basePath = Application.dataPath + "/ChatLLM/Runtime/Resources/";
            return System.IO.File.Exists(basePath + LlavaConfig.DecoderModelPath + ".onnx") &&
                   System.IO.File.Exists(basePath + LlavaConfig.VisionEncoderModelPath + ".onnx") &&
                   System.IO.File.Exists(basePath + LlavaConfig.EmbeddingModelPath + ".onnx") &&
                   System.IO.File.Exists(basePath + LlavaConfig.TokenizerConfigPath + ".json");
        }
    }
}
