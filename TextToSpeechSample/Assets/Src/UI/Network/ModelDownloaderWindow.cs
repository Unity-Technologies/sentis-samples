using System;
using System.Collections.Generic;
using System.IO;
using Unity.AppUI.UI;
using Unity.InferenceEngine.Samples.TTS.State;
using Unity.InferenceEngine.Samples.TTS.Utils;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.TTS.UI.Network
{
    [UxmlElement]
    public partial class ModelDownloaderWindow: VisualElement
    {
        HfDownloader m_Downloader;
        static readonly string k_DownloadPath = Path.Join(Application.dataPath,AppState.DownloadPath);

        ScrollView m_ScrollView;

        public ModelDownloaderWindow()
        {
            m_Downloader = new(
                k_DownloadPath,
                AppState.ModelId
            );
            AddToClassList("model-downloader-window");
            var configurations = GetDownloadConfigurations();

            m_ScrollView = new ScrollView();
            foreach (var config in configurations)
            {
                var element = new ModelDownloaderElement(config, m_Downloader);
                m_ScrollView.Add(element);
            }
            m_ScrollView.AddToClassList("model-downloader-scrollview");
            Add(m_ScrollView);

            Add(new ActionButton(OnDownloadAllClicked)
            {
                label = "Download All",
                style = { flexShrink = 0}
            });
        }

        void OnDownloadAllClicked()
        {
            foreach (var element in m_ScrollView.Children())
            {
                if (element is ModelDownloaderElement { Status: ModelDownloaderElement.DownloadStatus.Missing } downloaderElement)
                    downloaderElement.OnDownloadClicked();
            }
        }

        public static List<DownloadConfiguration> GetDownloadConfigurations()
        {
            var voices = VoicesUtils.GetVoicesList();
            var configurations = new List<DownloadConfiguration>();
            configurations.Add(new DownloadConfiguration("model.onnx", AppState.OnnxPath, Path.Join(k_DownloadPath, AppState.OnnxPath), AppState.ModelId));
            foreach (var voice in voices)
            {
                configurations.Add(new DownloadConfiguration(voice + ".bin", Path.Join(AppState.VoicePath, voice + ".bin"), Path.Join(k_DownloadPath, AppState.VoicePath, voice + ".bin"), AppState.ModelId));
            }
            return configurations;
        }

        public static bool VerifyModelsExist(List<DownloadConfiguration> configurations)
        {
            foreach (var config in configurations)
            {
                if (!VerifyModelExist(config.remotePath))
                    return false;
            }
            return true;
        }

        static bool VerifyModelExist(string localPath)
        {
            return File.Exists(Path.Join(k_DownloadPath, localPath));
        }
    }
}
