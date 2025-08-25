using System;
using System.IO;
using System.Net;
using System.Text.RegularExpressions;

namespace HuggingfaceHub
{
    public static class HFGlobalConfig
    {
        public static string HUGGINGFACE_HEADER_X_REPO_COMMIT = "X-Repo-Commit";

        public static string HUGGINGFACE_HEADER_X_LINKED_ETAG = "X-Linked-Etag";

        public static string HUGGINGFACE_HEADER_X_LINKED_SIZE = "X-Linked-Size";

        public static string EndPoint { get; set; } = "https://huggingface.co";

        public static string Home { get; set; }

        public static string RepoIdSeparator { get; set; } = "--";

        public static Regex CommitHashRegex { get; set; }

        public static string DefaultCacheDir { get; set; }

        public static string DefaultAssetCachePath { get; set; }

        public static string DefaultRevision { get; set; } = "main";

        public static int DefaultEtagTimeout { get; set; } = 10;

        public static int DefaultDownloadTimeout { get; set; } = 10;

        public static int DefaultRequestTimeout { get; set; } = 10;
        
        /// <summary>
        /// The chunk size when downloading the file.
        /// </summary>
        public static int DownloadChunkSize { get; set; } = 1024 * 1024 * 10;

        /// <summary>
        /// Used if download to `localDir` and `localDirUseSymlinks=null`
        /// Files smaller than 5MB are copy-pasted while bigger files are symlinked. The idea is to save disk-usage by symlinking
        /// uge files (i.e. LFS files most of the time) while allowing small files to be manually edited in local folder.
        /// You could set it to another size if you want.
        /// </summary>
        public static int LocalDirAutoSymlinkThreshold { get; set; } = 5 * 1024 * 1024;

        static HFGlobalConfig()
        {
            Home = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
            DefaultCacheDir = Path.Combine(Home, "hub");
            DefaultAssetCachePath = Path.Combine(DefaultCacheDir, "assets");
            CommitHashRegex = new Regex(@"^[0-9a-f]{40}$");
        }
    }
}