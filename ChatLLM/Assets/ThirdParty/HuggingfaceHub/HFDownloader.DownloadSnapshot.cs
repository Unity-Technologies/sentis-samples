using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using HuggingfaceHub.Common;

namespace HuggingfaceHub
{
    public partial class HFDownloader
    {
        /// <summary>
        /// Download repo files. It adopts from https://github.com/huggingface/huggingface_hub/blob/v0.22.2/src/huggingface_hub/_snapshot_download.py#L35.
        ///
        /// Download a whole snapshot of a repo's files at the specified revision. This is useful when you want all files from
        /// a repo, because you don't know which ones you will need a priori. All files are nested inside a folder in order
        /// to keep their actual filename relative to that folder. You can also filter which files to download using
        /// `allowPatterns` and `ignorePatterns`.
        ///
        /// If `localDir` is provided, the file structure from the repo will be replicated in this location. You can configure
        /// how you want to move those files:
        ///   - If `localDirUseSymlinks=null` (default), files are downloaded and stored in the cache directory as blob
        ///     files. Small files (<5MB) are duplicated in `localDir` while a symlink is created for bigger files. The goal
        ///     is to be able to manually edit and save small files without corrupting the cache while saving disk space for
        ///     binary files. The 5MB threshold can be configured with the <see cref="HFGlobalConfig.LocalDirAutoSymlinkThreshold"/>.
        ///   - If `localDirUseSymlinks=True`, files are downloaded, stored in the cache directory and symlinked in `localDir`.
        ///     This is optimal in term of disk usage but files must not be manually edited.
        ///   - If `localDirUseSymlinks=False` and the blob files exist in the cache directory, they are duplicated in the
        ///     local dir. This means disk usage is not optimized.
        ///   - Finally, if `localDirUseSymlinks=False` and the blob files do not exist in the cache directory, then the
        ///     files are downloaded and directly placed under `localDir`. This means if you need to download them again later,
        ///     they will be re-downloaded entirely.
        ///
        /// An alternative would be to clone the repo but this requires git and git-lfs to be installed and properly
        /// configured. It is also not possible to filter which files to download when cloning a repository using git.
        /// </summary>
        /// <param name="repoId">A user or an organization name and a repo name separated by a `/`.</param>
        /// <param name="revision">An optional Git revision id which can be a branch name, a tag, or a commit hash.</param>
        /// <param name="cacheDir">Path to the folder where cached files are stored.</param>
        /// <param name="localDir">
        /// If provided, the downloaded files will be placed under this directory,
        /// either as symlinks (default) or regular files (see description for more details).
        /// </param>
        /// <param name="localDirUseSymlinks">
        /// To be used with `localDir`. If set to null, the cache directory will be used and the file will be either
        /// duplicated or symlinked to the local directory depending on its size. It set to `True`, a symlink will be
        /// created, no matter the file size. If set to `False`, the file will either be duplicated from cache (if
        /// already exists) or downloaded from the Hub and not cached. See description for more details.
        /// </param>
        /// <param name="userAgent">The user-agent info in the form of a dictionary or a string.</param>
        /// <param name="forceDownload">Whether the file should be downloaded even if it already exists in the local cache.</param>
        /// <param name="proxy">A string which contains the proxy info to be used when downloading.</param>
        /// <param name="etagTimeout">
        /// When fetching ETag, how many seconds to wait for the server to send
        /// data before giving up which is passed to the request.
        /// </param>
        /// <param name="token">A string used as the authentication token.</param>
        /// <param name="localFilesOnly">
        /// If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        /// </param>
        /// <param name="allowPatterns">If provided, only files matching at least one pattern are downloaded.</param>
        /// <param name="ignorePatterns">If provided, files matching any of the patterns are not downloaded.</param>
        /// <param name="maxWorkers">Number of concurrent threads to download files (1 thread = 1 file download). Defaults to 8.</param>
        /// <param name="progress">If provided the progress in percentage will be passed to the callback during the downloading.</param>
        /// <param name="endpoint">If provided, using the provided endpoint instead of the default one.</param>
        /// <returns></returns>
        public static async Task<string> DownloadSnapshotAsync(string repoId,  string? revision = null,
            string? cacheDir = null, string? localDir = null, bool? localDirUseSymlinks = null,
            IDictionary<string, string>? userAgent = null, bool forceDownload = false,
            string? proxy = null, int etagTimeout = -1, string? token = null, bool localFilesOnly = false,
            string[]? allowPatterns = null, string[]? ignorePatterns = null, int maxWorkers = 8,
            IGroupedProgress? progress = null, string? endpoint = null){
            if(cacheDir is null){
                cacheDir = HFGlobalConfig.DefaultCacheDir;
            }
            if(revision is null){
                revision = "main";
            }
            // Currently we only support model type repo.
            string repoType = "model";

            var storageFolder = Path.Combine(cacheDir, RepoFolderName(repoId, repoType));

            ModelInfo? repoInfo = null;
            if(!localFilesOnly){
                try{
                    repoInfo = await GetModelInfoAsync(repoId, revision, etagTimeout, token:token, endpoint:endpoint);
                }
                catch(Exception ex){
                    Logger?.LogWarning($"Got an error while getting model info: {ex.Message}");
                }
            }

            // At this stage, if `repoInfo` is None it means either:
            // - internet connection is down
            // - internet connection is deactivated (localFilesOnly=True or HF_HUB_OFFLINE=True)
            // - repo is private/gated and invalid/missing token sent
            // - Hub is down
            // => let's look if we can find the appropriate folder in the cache:
            //    - if the specified revision is a commit hash, look inside "snapshots".
            //    - f the specified revision is a branch or tag, look inside "refs".
            string? commitHash = null;
            string snapshotFolder;
            if(repoInfo is null){
                // Try to get which commit hash corresponds to the specified revision
                if(HFGlobalConfig.CommitHashRegex.IsMatch(revision)){
                    commitHash = revision;
                }
                else{
                    var refPath = Path.Combine(storageFolder, "refs", revision);
                    if(File.Exists(refPath)){
                        // retrieve commit_hash from refs file
                        commitHash = File.ReadAllText(refPath).Trim();
                    }
                }

                // Try to locate snapshot folder for this commit hash
                if(commitHash is not null){
                    snapshotFolder = Path.Combine(storageFolder, "snapshots", commitHash);
                    if(Directory.Exists(snapshotFolder)){
                        // Snapshot folder exists => let's return it
                        // (but we can't check if all the files are actually there)
                        return snapshotFolder;
                    }
                }

                // If we couldn't find the appropriate folder on disk, raise an error.
                if(localFilesOnly){
                    throw new DirectoryNotFoundException(
                        "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and " +
                        "outgoing traffic has been disabled. To enable repo look-ups and downloads online, pass " +
                        "'localFilesOnly=False' as input."
                    );
                }
                else{
                    // Otherwise: most likely a connection issue or Hub downtime => let's warn the user
                    throw new DirectoryNotFoundException(
                        "An error happened while trying to locate the files on the Hub and we cannot find the appropriate" +
                        " snapshot folder for the specified revision on the local disk. Please check your internet connection" +
                        " and try again."
                    );
                }
            }

            // At this stage, internet connection is up and running
            // => let's download the files!
            Debug.Assert(repoInfo is not null);
            Debug.Assert(repoInfo!.Sha is not null, "Repo info returned from server must have a revision sha.");
            Debug.Assert(repoInfo!.Siblings is not null, "Repo info returned from server must have a siblings list.");
            var filteredRepoFiles = Utils.FilterRepoObjects(repoInfo!.Siblings.Select(f => f.Filename), allowPatterns, ignorePatterns);
            commitHash = repoInfo!.Sha;
            snapshotFolder = Path.Combine(storageFolder, "snapshots", commitHash);
            // if passed revision is not identical to commit_hash
            // then revision has to be a branch name or tag name.
            // In that case store a ref.
            if(revision != commitHash){
                var refPath = Path.Combine(storageFolder, "refs", revision);
                Directory.CreateDirectory(Directory.GetParent(refPath)!.FullName);
                File.WriteAllText(refPath, commitHash);
            }

            // we pass the commit_hash to hf_hub_download
            // so no network call happens if we already
            // have the file locally.
            int totalTasks = filteredRepoFiles.Count();
            if(Logger is not null)
            {
                foreach (var filename in filteredRepoFiles)
                {
                    Logger.LogDebug($"Filtered file when downloading snapshot: {filename}");
                }
            }
            if(totalTasks > 0){
#if NET6_0_OR_GREATER
                await Parallel.ForEachAsync(filteredRepoFiles, new ParallelOptions() { MaxDegreeOfParallelism = maxWorkers }, async (repoFilename, state) =>
                {
                    var singleProgress = progress is null ? null : new ProgressForGrouping(repoFilename, progress);

                    var res = await DownloadFileAsync(repoId, repoFilename, revision: revision, cacheDir: cacheDir, localDir: localDir,
                        localDirUseSymlinks: localDirUseSymlinks, userAgent: userAgent, forceDownload: forceDownload,
                        proxy: proxy, etagTimeout: etagTimeout, token: token, endpoint: endpoint, progress: singleProgress);

                    if (res is null)
                    {
                        throw new Exception($"Got an error when downloading file {repoFilename}");
                    }
                    Logger?.LogDebug($"[Snapshot Download] Completed downloading {res}");
                });
#else
                // TODO: parallel downloading in netstandard2.0
                foreach (var repoFilename in filteredRepoFiles)
                {
                    var singleProgress = progress is null ? null : new ProgressForGrouping(repoFilename, progress);

                    var res = await DownloadFileAsync(repoId, repoFilename, revision: revision, cacheDir: cacheDir, localDir: localDir,
                        localDirUseSymlinks: localDirUseSymlinks, userAgent: userAgent, forceDownload: forceDownload,
                        proxy: proxy, etagTimeout: etagTimeout, token: token, endpoint: endpoint, progress: singleProgress);

                    if (res is null)
                    {
                        throw new Exception($"Got error when downloading file {repoFilename}");
                    }
                    Logger?.LogDebug($"[Snapshot Download] Completed downloading {res}");
                }
#endif
            }

#if NET6_0_OR_GREATER
            if (localDir is not null){
                return new DirectoryInfo(localDir).LinkTarget!;
            }
#endif
            return snapshotFolder;
        }

        class ProgressForGrouping: IProgress<int>
        {
            private string _filename;
            private IGroupedProgress _groupedProgress;
            public ProgressForGrouping(string filename, IGroupedProgress groupedProgress)
            {
                _filename = filename;
                _groupedProgress = groupedProgress;
            }
            public void Report(int value)
            {
                _groupedProgress.Report(_filename, value);
            }
        }
    }
}