using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    public class ModelDownloader
    {
        readonly string m_ModelsPath = Application.dataPath + "/ChatLLM/Resources/Models";
        readonly HfDownloader m_Downloader;

        static readonly (string fileName, string remotePath, string localPath)[] k_ModelFiles = {
            (LlavaConfig.DecoderModelFile, $"onnx/{LlavaConfig.DecoderModelFile}", LlavaConfig.DecoderModelPath + ".onnx"),
            (LlavaConfig.VisionEncoderModelFile, $"onnx/{LlavaConfig.VisionEncoderModelFile}", LlavaConfig.VisionEncoderModelPath + ".onnx"),
            (LlavaConfig.EmbeddingModelFile, $"onnx/{LlavaConfig.EmbeddingModelFile}", LlavaConfig.EmbeddingModelPath + ".onnx"),
            (LlavaConfig.TokenizerModelFile, LlavaConfig.TokenizerModelFile, LlavaConfig.TokenizerConfigPath + ".json")
        };

        public ModelDownloader()
        {
            m_Downloader = new HfDownloader(
                m_ModelsPath,
                LlavaConfig.ModelId
            );
        }

        public async Task DownloadModels()
        {
            var downloadTasks = new List<Task>();
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
                AssetDatabase.Refresh();
            }
        }

        Task CreateDownloadTask(string fileName, string remotePath)
        {
            var progress = new Progress<float>(p => LogProgress(fileName, p));

            return Task.Run(() =>
                m_Downloader.Download(remotePath, progress)
            );
        }

        static void LogProgress(string file, float progress)
        {
            Debug.Log($"Downloading {file}: {progress * 10f}% complete.");
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

        static bool VerifyModelExist(string localPath)
        {
            var basePath = Application.dataPath + "/ChatLLM/Resources/";
            return File.Exists(basePath + localPath);
        }
    }
}
