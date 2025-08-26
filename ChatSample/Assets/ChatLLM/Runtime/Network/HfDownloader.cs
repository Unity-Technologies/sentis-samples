using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace Unity.InferenceEngine.Samples.Chat
{
    public class HfDownloader
    {
        const string k_BaseUrl = "https://huggingface.co/";
        readonly string m_RepoId;
        readonly string m_Revision;
        readonly string m_DownloadPath;
        readonly SynchronizationContext m_UnityContext;

        /// <param name="downloadPath">The local path to store the downloaded files</param>
        /// <param name="repoId">The repo id on hf</param>
        /// <param name="revision">The branch name</param>
        public HfDownloader(string downloadPath, string repoId, string revision = "main")
        {
            m_DownloadPath = downloadPath;
            m_RepoId = repoId;
            m_Revision = revision;

            // Capture Unity's main thread SynchronizationContext at construction time
            m_UnityContext = SynchronizationContext.Current
                             ?? throw new InvalidOperationException("No SynchronizationContext found. Must be created on Unity's main thread.");
        }

        public async Task Download(string filePath, IProgress<float> progress = null)
        {
            var url = GetUrl(m_RepoId, m_Revision, filePath);
            using var client = new HttpClient();

            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? -1L;
            var canReportProgress = totalBytes != -1 && progress != null;

            var destinationPath = Path.Join(m_DownloadPath, filePath);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));

            await using var contentStream = await response.Content.ReadAsStreamAsync();
            await using var fileStream = File.Create(destinationPath);

            var buffer = new byte[8192];
            long totalRead = 0;
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
            {
                await fileStream.WriteAsync(buffer, 0, bytesRead);
                totalRead += bytesRead;

                if (canReportProgress)
                {
                    var progressValue = (float)totalRead / totalBytes;

                    // Post the progress update back to Unity's main thread
                    m_UnityContext.Post(_ => progress.Report(progressValue), null);
                }
            }
        }

        static string GetUrl(string repoId, string revision, string filePath)
        {
            return $"{k_BaseUrl}{repoId}/resolve/{revision}/{filePath}";
        }
    }
}
