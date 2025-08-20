using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using HuggingfaceHub;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat.LLM_Chat.Network
{
    public static class ModelDownloader
    {
        static readonly string k_ModelsPath = Application.dataPath + "/LLMChat/TestRes/";
        public static async Task DownloadModels()
        {
            var tasks = new List<Task<string>>();

            var decoderProgress = new Progress<int>();
            decoderProgress.ProgressChanged += (_, progress) =>
            {
                LogProgress(LlavaConfig.DecoderModelFile, progress);
            };
            var task = HFDownloader.DownloadFileAsync(LlavaConfig.ModelId, "onnx/"+ LlavaConfig.DecoderModelFile, progress:decoderProgress, cacheDir: k_ModelsPath, localDir:k_ModelsPath);
            tasks.Add(task);

            var visionEncoderProgress = new Progress<int>();
            visionEncoderProgress.ProgressChanged += (_, progress) =>
            {
                LogProgress(LlavaConfig.VisionEncoderModelFile, progress);
            };
            var task1 = HFDownloader.DownloadFileAsync(LlavaConfig.ModelId, "onnx/"+ LlavaConfig.VisionEncoderModelFile, progress:visionEncoderProgress, cacheDir: k_ModelsPath, localDir:k_ModelsPath);
            tasks.Add(task1);

            var embeddingProgress = new Progress<int>();
            embeddingProgress.ProgressChanged += (_, progress) =>
            {
                LogProgress(LlavaConfig.EmbeddingModelFile, progress);
            };
            var task2 = HFDownloader.DownloadFileAsync(LlavaConfig.ModelId, "onnx/"+ LlavaConfig.EmbeddingModelFile, progress:embeddingProgress, cacheDir: k_ModelsPath, localDir:k_ModelsPath);
            tasks.Add(task2);

            var tokenizerProgress = new Progress<int>();
            tokenizerProgress.ProgressChanged += (_, progress) =>
            {
                LogProgress(LlavaConfig.TokenizerModelFile, progress);
            };
            var task3 = HFDownloader.DownloadFileAsync(LlavaConfig.ModelId, LlavaConfig.TokenizerModelFile, progress:tokenizerProgress, cacheDir: k_ModelsPath, localDir:k_ModelsPath);
            tasks.Add(task3);

            foreach (var tsk in tasks)
            {
                try
                {
                    await tsk;
                }
                catch (Exception e)
                {
                    Debug.LogError($"Failed to download model: {e.Message}");
                }
            }
        }


        static void LogProgress(string file, int progress)
        {
            Debug.Log($"Downloading {file}: {progress}% complete.");
        }
    }
}
