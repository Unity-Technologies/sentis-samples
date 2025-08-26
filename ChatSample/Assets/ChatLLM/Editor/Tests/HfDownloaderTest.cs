using System;
using System.IO;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    public class HfDownloaderTest
    {
        const string k_FilePath = "config.json";
        const string k_RepoId = "llava-hf/llava-onevision-qwen2-0.5b-si-hf";
        const string k_Revision = "main";
        const string k_DownloadPath = "Downloads";

        [Test]
        public async Task DownloaderTest()
        {
            var downloader = new HfDownloader(
                Path.Join(Application.dataPath, k_DownloadPath),
                k_RepoId,
                k_Revision
            );

            var progress = new Progress<float>(p =>
            {
                // This callback runs on the Unity main thread by default
                Debug.Log($"Download progress: {p:P1}");
            });

            // Run the download on a background thread
            await Task.Run(() => downloader.Download(
                k_FilePath,
                progress
            ));

            Assert.IsTrue(File.Exists(Path.Join(
                Application.dataPath,
                k_DownloadPath,
                k_FilePath
            )));
        }

    }
}
