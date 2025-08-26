using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    [UxmlElement]
    public partial class ModelDownloaderWindow: VisualElement
    {
        HfDownloader m_Downloader;
        static readonly string k_DownloadPath = Application.dataPath + LlavaConfig.DownloadPath;

        static readonly List<DownloadConfiguration> k_Configurations = new()
        {
            new(LlavaConfig.DecoderModelFile, $"onnx/{LlavaConfig.DecoderModelFile}", LlavaConfig.DecoderModelPath + ".onnx", LlavaConfig.ModelId),
            new(LlavaConfig.VisionEncoderModelFile, $"onnx/{LlavaConfig.VisionEncoderModelFile}", LlavaConfig.VisionEncoderModelPath + ".onnx", LlavaConfig.ModelId),
            new(LlavaConfig.EmbeddingModelFile, $"onnx/{LlavaConfig.EmbeddingModelFile}", LlavaConfig.EmbeddingModelPath + ".onnx", LlavaConfig.ModelId),
            new(LlavaConfig.TokenizerModelFile, LlavaConfig.TokenizerModelFile, LlavaConfig.TokenizerConfigPath + ".json", LlavaConfig.ModelId)
        };

        ScrollView m_ScrollView;

        public ModelDownloaderWindow()
        {
            m_Downloader = new(
                k_DownloadPath,
                LlavaConfig.ModelId
            );
            AddToClassList("model-downloader-window");

            m_ScrollView = new ScrollView();
            foreach (var config in k_Configurations)
            {
                var element = new ModelDownloaderElement(config, m_Downloader);
                m_ScrollView.Add(element);
            }
            m_ScrollView.AddToClassList("model-downloader-scrollview");
            Add(m_ScrollView);
        }

        public static bool VerifyModelsExist()
        {
            foreach (var config in k_Configurations)
            {
                if (!VerifyModelExist(config.localPath))
                    return false;
            }
            return true;
        }

        static bool VerifyModelExist(string localPath)
        {
            return File.Exists(k_DownloadPath + localPath);
        }
    }
}
