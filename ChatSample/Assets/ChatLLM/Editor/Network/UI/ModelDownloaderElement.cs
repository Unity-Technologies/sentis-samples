using System;
using System.IO;
using System.Threading.Tasks;
using Unity.AppUI.UI;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using Progress = Unity.AppUI.UI.Progress;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    public class ModelDownloaderElement : VisualElement
    {
        DownloadStatus m_Status = DownloadStatus.Missing;
        readonly DownloadConfiguration m_Configuration;
        readonly HfDownloader m_Downloader;

        readonly Label m_Label;
        readonly IconButton m_DownloadButton;
        readonly Icon m_ErrorIcon;
        readonly Icon m_CheckIcon;
        readonly CircularProgress m_ProgressElement;
        readonly Text m_ProgressText;

        string DownloadPath => Path.Join(m_Downloader.DownloadPath,m_Configuration.remotePath);

        public ModelDownloaderElement(DownloadConfiguration configuration, HfDownloader downloader)
        {
            AddToClassList("model-downloader-element");
            m_Configuration = configuration;
            m_Downloader = downloader;

            m_Status = VerifyModelExist() ? DownloadStatus.Downloaded : DownloadStatus.Missing;

            m_Label = new Label(Path.Join("Assets", LlavaConfig.DownloadPath, configuration.fileName));
            Add(m_Label);

            m_DownloadButton = new IconButton
            {
                icon = "installs",
                quiet = true
            };
            m_DownloadButton.clicked += OnDownloadClicked;
            m_DownloadButton.AddToClassList("model-downloader-download-button");
            Add(m_DownloadButton);

            m_ErrorIcon = new Icon
            {
                iconName= "warning"
            };
            m_ErrorIcon.AddToClassList("model-downloader-error-icon");
            Add(m_ErrorIcon);

            m_CheckIcon = new Icon
            {
                iconName= "check"
            };
            m_CheckIcon.AddToClassList("model-downloader-check-icon");
            Add(m_CheckIcon);

            m_ProgressElement = new CircularProgress
            {
                value = 0f,
               variant = Progress.Variant.Determinate
            };
            m_ProgressElement.AddToClassList("model-downloader-progress");
            m_ProgressText = new Text
            {
                size = TextSize.XS
            };

            m_ProgressElement.Add(m_ProgressText);

            Add(m_ProgressElement);
            UpdateVisuals(m_Status);
        }

        void OnDownloadClicked()
        {
            _ = StartDownload();
        }

        async Task StartDownload()
        {
            m_Status = DownloadStatus.Downloading;
            UpdateVisuals(m_Status);
            var progress = new Progress<float>(p =>
            {
                m_ProgressElement.value = p;
                m_ProgressText.text = $"{Mathf.RoundToInt(m_ProgressElement.value * 100f)}%";
            });
            var task = Task.Run(() => m_Downloader.Download(m_Configuration.remotePath, progress));
            await task;
            m_Status = task.IsFaulted ? DownloadStatus.Error : DownloadStatus.Downloaded;
            UpdateVisuals(m_Status);
            AssetDatabase.ImportAsset(DownloadPath[(Application.dataPath.Length - "Assets".Length)..]);
        }

        void UpdateVisuals(DownloadStatus status)
        {
            m_DownloadButton.style.display = status == DownloadStatus.Missing ? DisplayStyle.Flex: DisplayStyle.None;
            m_ErrorIcon.style.display = status == DownloadStatus.Error ? DisplayStyle.Flex : DisplayStyle.None;
            m_CheckIcon.style.display = status == DownloadStatus.Downloaded ? DisplayStyle.Flex : DisplayStyle.None;
            m_ProgressElement.style.display = status == DownloadStatus.Downloading ? DisplayStyle.Flex : DisplayStyle.None;
        }

        enum DownloadStatus
        {
            Missing,
            Downloading,
            Downloaded,
            Error
        }

        bool VerifyModelExist()
        {
            return File.Exists(DownloadPath);
        }
    }
}
