using System;
using System.Collections.Generic;
using System.IO;
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

        static readonly (string fileName, string remotePath, string localPath)[] k_ModelFiles = {
            (LlavaConfig.DecoderModelFile, $"onnx/{LlavaConfig.DecoderModelFile}", LlavaConfig.DecoderModelPath + ".onnx"),
            (LlavaConfig.VisionEncoderModelFile, $"onnx/{LlavaConfig.VisionEncoderModelFile}", LlavaConfig.VisionEncoderModelPath + ".onnx"),
            (LlavaConfig.EmbeddingModelFile, $"onnx/{LlavaConfig.EmbeddingModelFile}", LlavaConfig.EmbeddingModelPath + ".onnx"),
            (LlavaConfig.TokenizerModelFile, LlavaConfig.TokenizerModelFile, LlavaConfig.TokenizerConfigPath + ".json")
        };

        public async Task DownloadModels()
        {
            var downloadTasks = new List<Task<string>>();
            foreach (var (fileName, remotePath, localPath) in k_ModelFiles)
            {
                if (VerifyModelExist(localPath))
                {
                    Debug.Log($"{fileName} already exists at {localPath}, skipping download.");
                    continue;
                }
                downloadTasks.Add(CreateDownloadTask(fileName, remotePath));
            }

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
                if (Directory.Exists(cacheDir))
                {
                    Directory.Delete(cacheDir, true);
                    Debug.Log($"Deleted cache directory: {cacheDir}");
                }

                AssetDatabase.Refresh();
            }
        }

        Task<string> CreateDownloadTask(string fileName, string remotePath)
        {
            var progress = new Progress<int>(p => LogProgress(fileName, p));

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
            if (progress - lastProgress >= 5)
            {
                m_LastLoggedProgress[file] = progress;
                Debug.Log($"Downloading {file}: {progress}% complete.");
            }
        }

        public static bool VerifyModelsExist()
        {
            foreach (var (_, _, localPath) in k_ModelFiles)
            {
                if (!VerifyModelExist(localPath))
                    return false;
            }
            return true;
        }

        public static bool VerifyModelExist(string localPath)
        {
            var basePath = Application.dataPath + "/ChatLLM/Resources/";
            return File.Exists(basePath + localPath);
        }
    }
}
