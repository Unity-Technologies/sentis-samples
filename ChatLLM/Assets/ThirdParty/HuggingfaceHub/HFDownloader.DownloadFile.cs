using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Collections.Concurrent;
using System.Collections.Generic;
using HuggingfaceHub.Common;
using Microsoft.Extensions.Logging;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace HuggingfaceHub
{

    public partial class HFDownloader
    {
        private static readonly Dictionary<string, bool> _symlinksSupportedInDir = new Dictionary<string, bool>();

        /// <summary>
        /// Download a given file if it's not already present in the local cache.
        /// It adopts from https://github.com/huggingface/huggingface_hub/blob/v0.22.2/src/huggingface_hub/file_download.py#L1014.
        ///
        /// </summary>
        /// <param name="repoId">A user or an organization name and a repo name separated by a `/`.</param>
        /// <param name="filename">The name of the file in the repo.</param>
        /// <param name="subfolder">An optional value corresponding to a folder inside the model repo.</param>
        /// <param name="revision">An optional Git revision id which can be a branch name, a tag, or a commit hash.</param>
        /// <param name="cacheDir">Path to the folder where cached files are stored.</param>
        /// <param name="localDir">
        /// If provided, the downloaded file will be placed under this directory,
        /// either as a symlink (default) or a regular file (see description for more details).
        /// </param>
        /// <param name="localDirUseSymlinks">
        /// To be used with `local_dir`. If not set, the cache directory will be used and the file will be either
        /// duplicated or symlinked to the local directory depending on its size. It set to `True`, a symlink will be
        /// created, no matter the file size. If set to `False`, the file will either be duplicated from cache (if
        /// already exists) or downloaded from the Hub and not cached.
        /// </param>
        /// <param name="userAgent"> The user-agent info in the form of a dictionary.</param>
        /// <param name="forceDownload">
        /// Whether the file should be downloaded even if it already exists in the local cache.
        /// </param>
        /// <param name="proxy">String which contains the proxy address.</param>
        /// <param name="etagTimeout">
        /// When fetching ETag, how many seconds to wait for the server to send
        /// data before giving up which is passed to http request.
        /// </param>
        /// <param name="token">A token to be used for the download.</param>
        /// <param name="localFilesOnly">
        /// If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        /// </param>
        /// <param name="resumeDownload">If `true`, resume a previously interrupted download.</param>
        /// <param name="endpoint">If set, replace the default endpoint with this value.</param>
        /// <param name="progress">A callback used to show progress, which is passed a value from 0 to 100.</param>
        /// <returns></returns>
        public static async Task<string> DownloadFileAsync(string repoId, string filename, string? subfolder = null,
            string? revision = null, string? cacheDir = null, string? localDir = null,
            bool? localDirUseSymlinks = null, IDictionary<string, string>? userAgent = null, bool forceDownload = false,
            string? proxy = null, int etagTimeout = -1, string? token = null, bool localFilesOnly = false,
            bool resumeDownload = false, string? endpoint = null, IProgress<int>? progress = null)
        {
            if(etagTimeout == -1){
                etagTimeout = HFGlobalConfig.DefaultEtagTimeout;
            }

            if(cacheDir is null){
                cacheDir = HFGlobalConfig.DefaultCacheDir;
            }
            if(revision is null){
                revision = HFGlobalConfig.DefaultRevision;
            }
            if(subfolder is not null){
                filename = $"{subfolder}/{filename}";
            }
            var locksDir = Path.Combine(cacheDir, ".locks");

            string repoType = "model"; // currently only models are supported
            var storageFolder = Path.Combine(cacheDir, RepoFolderName(repoId, repoType));
            Directory.CreateDirectory(storageFolder);

            var relativeFilename = Path.Combine(filename.Split('/'));
            if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows)){
                if(relativeFilename.StartsWith("..\\") || relativeFilename.Contains("\\..\\")){
                    throw new Exception($"Invalid filename: cannot handle filename '{relativeFilename}' on Windows. " +
                    "Please ask the repository owner to rename this file.");
                }
            }

            string pointerPath;
            // if user provides a commit_hash and they already have the file on disk,
            // shortcut everything.
            if(HFGlobalConfig.CommitHashRegex.IsMatch(revision)){
                pointerPath = GetPointerPath(storageFolder, revision, relativeFilename);
#if NET7_0_OR_GREATER
                if(Path.Exists(pointerPath)){
#else
                if(File.Exists(pointerPath)){
#endif
                    if (localDir is not null)
                    {
                        return ToLocalDir(pointerPath, localDir, relativeFilename, localDirUseSymlinks);
                    }
                    return pointerPath;
                }
            }

            var url = GetHuggingfaceFileUrl(repoId, filename, subfolder, revision, endpoint);
            var uriToDownload = new Uri(url);

            var headers = BuildHFHeaders(token, userAgent: userAgent);
            System.Net.Http.Headers.EntityTagHeaderValue? etag = null;
            string? commitHash = null;
            long? expectedSize = null;

            if(!localFilesOnly){
                var metadata = await GetHfFileMetadata(uriToDownload, token, proxy, etagTimeout, userAgent);
                commitHash = metadata.CommitHash;
                if(commitHash is null){
                    throw new Exception("Distant resource does not seem to be on huggingface.co. It is possible that a configuration issue" +
                        " prevents you from downloading resources from https://huggingface.co. Please check your firewall" +
                        " and proxy settings and make sure your SSL certificates are updated.");
                }
                etag = metadata.Etag;
                if(etag is null){
                    throw new Exception("Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility.");
                }
                expectedSize = metadata.Size;
                // In case of a redirect, save an extra redirect on the request.get call,
                // and ensure we download the exact atomic version even if it changed
                // between the HEAD and the GET (unlikely, but hey).
                // Useful for lfs blobs that are stored on a CDN.
                if(metadata.Location is not null && metadata.Location.OriginalString != url){
                    uriToDownload = metadata.Location;
                    // Remove authorization header when downloading a LFS blob
                    headers.Remove("authorization");
                }
            }

            // etag can be None for several reasons:
            // 1. we passed local_files_only.
            // 2. we don't have a connection
            // 3. Hub is down (HTTP 500 or 504)
            // 4. repo is not found -for example private or gated- and invalid/missing token sent
            // 5. Hub is blocked by a firewall or proxy is not set correctly.
            // => Try to get the last downloaded one from the specified revision.
            //
            // If the specified revision is a commit hash, look inside "snapshots".
            // If the specified revision is a branch or tag, look inside "refs".
            if(etag is null){
                // In those cases, we cannot force download.
                if(forceDownload){
                    throw new Exception("We have no connection or you passed local_files_only, so force_download is not an accepted option.");
                }

                // Try to get "commit_hash" from "revision"
                if(HFGlobalConfig.CommitHashRegex.IsMatch(revision)){
                    commitHash = revision;
                }
                else
                {
                    var refPath = Path.Combine(storageFolder, "refs", revision);
                    if(File.Exists(refPath)){
                        commitHash = File.ReadAllText(refPath).Trim();
                    }
                }

                // Return pointer file if exists
                if(commitHash is not null){
                    pointerPath = GetPointerPath(storageFolder, commitHash, relativeFilename);
                    if(File.Exists(pointerPath)){
                        if(localDir is not null){
                            return ToLocalDir(pointerPath, localDir, relativeFilename, localDirUseSymlinks);
                        }
                        return pointerPath;
                    }
                }

                // If we couldn't find an appropriate file on disk, raise an error.
                // If files cannot be found and local_files_only=True,
                // the models might've been found if local_files_only=False
                // Notify the user about that
                if(localFilesOnly){
                    throw new FileNotFoundException("Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable" +
                        " hf.co look-ups and downloads online, set 'local_files_only' to False.");
                }
                else{
                    throw new Exception("An error happened while trying to locate the file on the Hub and we cannot find the requested files" +
                        " in the local cache. Please check your connection and try again or make sure your Internet connection is on.");
                }
            }

            // From now on, etag and commit_hash are not None.
            Debug.Assert(etag is not null, "etag must have been retrieved from server");
            Debug.Assert(commitHash is not null, "commit_hash must have been retrieved from server");
            Debug.Assert(relativeFilename is not null);
            Debug.Assert(commitHash is not null);
            Debug.Assert(storageFolder is not null);
            var blobPath = Path.Combine(storageFolder!, "blobs", etag!.Tag.Trim('\"'));
            pointerPath = GetPointerPath(storageFolder!, commitHash!, relativeFilename!);

            Directory.CreateDirectory(Path.GetDirectoryName(blobPath)!);
            Directory.CreateDirectory(Path.GetDirectoryName(pointerPath)!);
            // if passed revision is not identical to commit_hash
            // then revision has to be a branch name or tag name.
            // In that case store a ref.
            CacheCommitHashForSpecificVersion(storageFolder!, revision!, commitHash!);

            if(File.Exists(pointerPath) && !forceDownload)
            {
                if(localDir is not null){
                    return ToLocalDir(pointerPath, localDir, relativeFilename!, localDirUseSymlinks);
                }
                return pointerPath;
            }

            if(File.Exists(blobPath) && !forceDownload){
                if(localDir is not null){ // we have the blob already, but not the pointer
                    return ToLocalDir(blobPath, localDir, relativeFilename!, localDirUseSymlinks);
                }
                else{ // or in snapshot cache
                    CreateSymlink(blobPath, pointerPath, newBlob: false);
                    return pointerPath;
                }
            }

            // Prevent parallel downloads of the same file with a lock.
            // etag could be duplicated across repos,
            var locksPath = Path.Combine(locksDir, RepoFolderName(repoId, repoType), $"{etag!.Tag.Trim('\"')}.lock");

            // Some Windows versions do not allow for paths longer than 255 characters.
            // In this case, we must specify it is an extended path by using the "\\?\" prefix.
            if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Path.GetFullPath(blobPath).Length > 255){
                locksPath = $"\\\\?\\{Path.GetFullPath(locksPath)}";
            }
            if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) && Path.GetFullPath(blobPath).Length > 255){
                blobPath = $"\\\\?\\{Path.GetFullPath(blobPath)}";
            }

            // Directory.CreateDirectory(Directory.GetParent(locksPath).FullName);
            // File.Create(locksPath);
            // using(var fileStream = File.OpenRead(locksPath))
            // TODO: add lock here. Remove it now because of BUG.
            {
                // fileStream.Lock(0, fileStream.Length);

                // If the download just completed while the lock was activated.
                if(File.Exists(pointerPath) && !forceDownload){
                    if(localDir is not null){
                        return ToLocalDir(pointerPath, localDir, relativeFilename!, localDirUseSymlinks);
                    }
                    return pointerPath;
                }

                long resumeSize = 0;
                var tempFilePath = Path.GetTempFileName();
                if (resumeDownload)
                {
                    tempFilePath = blobPath + ".incomplete";
                    if(File.Exists(tempFilePath))
                    {
                        resumeSize = new FileInfo(tempFilePath).Length;
                    }
                }

                // TODO: resume download
                Logger?.LogInformation($"Downloading {url} to {tempFilePath}...");
                if(expectedSize is not null){
                    // Check tmp path
                    CheckDiskSpace(expectedSize.Value, Path.GetDirectoryName(tempFilePath)!);

                    // Check destination
                    CheckDiskSpace(expectedSize.Value, Path.GetDirectoryName(blobPath)!);
                    if(localDir is not null){
                        CheckDiskSpace(expectedSize.Value, localDir);
                    }
                }
                var client = GetClientForFileDownload(proxy);
                using(var fs = new FileStream(tempFilePath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    await HttpDownloadWithStreamAsync(client, uriToDownload, fs, proxy, resumeSize, headers, expectedSize is null ? -1 : expectedSize.Value, progress: progress);
                }

                if(localDir is null){
                    Logger?.LogDebug($"Storing {url} in cache at {blobPath}");
                    ChmodAndReplace(tempFilePath, blobPath);
                    CreateSymlink(blobPath, pointerPath, newBlob: true);
                }
                else{
                    var localDirFilepath = Path.Combine(localDir, relativeFilename);
                    Directory.CreateDirectory(Directory.GetParent(localDirFilepath)!.FullName);

                    bool isBigFile = new FileInfo(tempFilePath).Length > HFGlobalConfig.LocalDirAutoSymlinkThreshold;
                    if((localDirUseSymlinks is not null && localDirUseSymlinks.Value) || (localDirUseSymlinks is null && isBigFile)){
                        Logger?.LogDebug($"Storing {url} in cache at {blobPath}");
                        ChmodAndReplace(tempFilePath, blobPath);
                        Logger?.LogDebug("Create symlink to local dir");
                        // NOTE: MODIFIED --  Always copy for now
                        //CreateSymlink(blobPath, localDirFilepath, newBlob: false);
                        File.Copy(blobPath, localDirFilepath);
                    }
                    else if(localDirUseSymlinks is null && !isBigFile){
                        Logger?.LogDebug($"Storing {url} in cache at {blobPath}");
                        ChmodAndReplace(tempFilePath, blobPath);
                        Logger?.LogDebug("Duplicate in local dir (small file and use_symlink set to 'auto')");
                        File.Copy(blobPath, localDirFilepath);
                    }
                    else{
                        Logger?.LogDebug($"Storing {url} in local_dir at {localDirFilepath} (not cached).");
                        ChmodAndReplace(tempFilePath, localDirFilepath);
                    }
                    pointerPath = localDirFilepath;
                }
                // fileStream.Unlock(0, fileStream.Length);
            }
            // File.Delete(locksPath);

            return pointerPath;
        }

        /// <summary>
        /// Download a remote file. Do not gobble up errors, and will return errors tailored to the Hugging Face Hub.
        /// </summary>
        /// <param name="client"></param>
        /// <param name="uri"></param>
        /// <param name="fs"></param>
        /// <param name="proxy"></param>
        /// <param name="resumeSize"></param>
        /// <param name="headers"></param>
        /// <param name="expectedSize"></param>
        /// <param name="retryLeft"></param>
        /// <param name="progress"></param>
        private static async Task HttpDownloadWithStreamAsync(HttpClient client, Uri uri, FileStream fs, string? proxy = null, long resumeSize = 0,
            IDictionary<string, string>? headers = null, long expectedSize = -1, int retryLeft = 5, IProgress<int>? progress = null)
        {
            using (var request = new HttpRequestMessage(HttpMethod.Get, uri))
            {
                //Utils.AddDefaultHeaders(request.Headers);
                if(headers is not null)
                {
                    foreach (var k in headers.Keys)
                    {
                        request.Headers.TryAddWithoutValidation(k, headers[k]);
                    }
                }
                if(resumeSize > 0)
                {
                    request.Headers.TryAddWithoutValidation("Range", $"bytes={resumeSize}-");
                }
                using (var response = await Utils.HttpRequestWrapperAsync(client, request, false, true))
                {
                    response.EnsureSuccessStatusCode();
                    Debug.Assert(response.Content.Headers.TryGetValues("Content-Length", out var lengths));
                    Debug.Assert(lengths is not null && lengths!.Count() > 0);
                    // NOTE: 'total' is the total number of bytes to download, not the number of bytes in the file.
                    // If the file is compressed, the number of bytes in the saved file will be higher than 'total'.
                    if(long.TryParse(lengths!.First(), out var totalFileLengths))
                    {
                        totalFileLengths += resumeSize;
                    }
                    else
                    {
                        totalFileLengths = -1;
                    }
                    var newResumeSize = resumeSize;

                    var chunkSize = HFGlobalConfig.DownloadChunkSize;
                    var pieces = totalFileLengths / chunkSize;
                    using (var responseStream = await response.Content.ReadAsStreamAsync())
                    {
                        byte[] buffer = new byte[chunkSize];
                        int bytesRead;

                        try
                        {
                            while ((bytesRead = await responseStream.ReadAsync(buffer, 0, chunkSize)) > 0)
                            {
                                await fs.WriteAsync(buffer, 0, bytesRead);
                                newResumeSize += bytesRead;
                                progress?.Report((int)(newResumeSize * pieces / totalFileLengths));
                                // Some data has been downloaded from the server so we reset the number of retries.
                                retryLeft = 5;
                            }
                        }
                        catch(IOException ex)
                        {
                            if(retryLeft < 0)
                            {
                                Logger?.LogWarning($"Error while downloading from {uri.AbsoluteUri}: {ex.Message}\nMax retries exceeded.");
                                throw new HttpRequestException(ex.Message);
                            }
                            Logger?.LogWarning($"Error while downloading from {uri.AbsoluteUri}: {ex.Message}\nTrying to resume download...");
                            Thread.Sleep(1000);
                            // In case of SSLError it's best to reset the client.
                            await HttpDownloadWithStreamAsync(GetClientForFileDownload(proxy), uri, fs, proxy, newResumeSize, headers, expectedSize, retryLeft - 1, progress);
                            return;
                        }
                    }

                    if(expectedSize != -1 && expectedSize != fs.Position)
                    {
                        throw new Exception();
                    }
                }
            }
        }

        private static HttpClient GetClientForFileDownload(string? proxy = null)
        {
            // download the file
            var handler = new HttpClientHandler();
            if (proxy is not null)
            {
                handler.Proxy = new System.Net.WebProxy(proxy);
                handler.UseProxy = true;
            }
            handler.AllowAutoRedirect = true;
            handler.UseDefaultCredentials = true;
            //handler.MaxAutomaticRedirections = 30;
            //handler.UseCookies = true;

            var client = new System.Net.Http.HttpClient(handler);
            client.Timeout = TimeSpan.FromSeconds(HFGlobalConfig.DefaultDownloadTimeout);
            return client;
        }

        /// <summary>
        /// Fetch metadata of a file versioned on the Hub for a given url.
        /// </summary>
        /// <param name="url"></param>
        /// <param name="token"></param>
        /// <param name="proxy"></param>
        /// <param name="timeout"></param>
        /// <param name="userAgent"></param>
        /// <returns></returns>
        public static async Task<HfFileMetadata> GetHfFileMetadata(Uri uri, string? token = null, string? proxy = null,
            int timeout = -1, IDictionary<string, string>? userAgent = null){
            var headers = BuildHFHeaders(token, userAgent: userAgent);
            headers["Accept-Encoding"] = "identity"; // prevent any compression => we want to know the real size of the file

            var handler = new HttpClientHandler();
            handler.AllowAutoRedirect = false;
            if(proxy is not null){
                handler.Proxy = new System.Net.WebProxy(proxy);
                handler.UseProxy = true;
            }
            var client = new System.Net.Http.HttpClient(handler);
            client.Timeout = timeout == -1 ? TimeSpan.FromSeconds(HFGlobalConfig.DefaultEtagTimeout) : TimeSpan.FromSeconds(timeout);

            string? commitHash = null;
            System.Net.Http.Headers.EntityTagHeaderValue? etag = null;
            Uri? location;
            long? size = null;

            using(var request = new HttpRequestMessage(HttpMethod.Head, uri)){
                foreach(var header in headers)
                {
                    request.Headers.TryAddWithoutValidation(header.Key, header.Value);
                }

                var response = await Utils.HttpRequestWrapperAsync(client, request, true);
                Debug.Assert(response is not null);
                Debug.Assert(response!.Headers is not null);

                response!.Headers!.TryGetValues(HFGlobalConfig.HUGGINGFACE_HEADER_X_REPO_COMMIT, out var commitHashes);
                commitHash = commitHashes?.FirstOrDefault();
                if(!response.Headers.TryGetValues(HFGlobalConfig.HUGGINGFACE_HEADER_X_LINKED_ETAG, out var etags)){
                    etag = response.Headers.ETag;
                } else{
                    etag = new System.Net.Http.Headers.EntityTagHeaderValue(etags.First());
                }
                location = response.Headers.Location ?? response.RequestMessage!.RequestUri;
                response.Headers.TryGetValues(HFGlobalConfig.HUGGINGFACE_HEADER_X_LINKED_SIZE, out var sizes);
                if(sizes is not null && sizes.Count() > 0){
                    size = long.Parse(sizes.First());
                }
                if(size is null && response.Headers.TryGetValues("Content-Length", out sizes)){
                    size = long.Parse(sizes.First());
                }
            }

            return new HfFileMetadata(commitHash, etag, location, size);
        }

        /// <summary>
        /// Construct the URL of a file from the given information.
        /// </summary>
        /// <param name="repoId">A namespace (user or an organization) name and a repo name separated by a `/`.</param>
        /// <param name="filename">The name of the file in the repo.</param>
        /// <param name="subfolder">An optional value corresponding to a folder inside the repo.</param>
        /// <param name="revision">An optional Git revision id which can be a branch name, a tag, or a commit hash.</param>
        /// <param name="endpoint"></param>
        /// <returns></returns>
        public static string GetHuggingfaceFileUrl(string repoId, string filename, string? subfolder = null,
            string? revision = null, string? endpoint = null)
        {
            if(!string.IsNullOrEmpty(subfolder)){
                filename = $"{subfolder}/{filename}";
            }
            if(revision is null){
                revision = HFGlobalConfig.DefaultRevision;
            }
            var url = GetFileUrl(repoId, revision, filename, endpoint);
            return url;
        }

        /// <summary>
        /// Return a serialized version of a hf.co repo name and type, safe for disk storage
        /// as a single non-nested folder.
        /// </summary>
        /// <param name="repoId"></param>
        /// <param name="repoType"></param>
        /// <returns></returns>
        public static string RepoFolderName(string repoId, string repoType)
        {
            List<string> parts = new List<string>() { repoType };
            parts.AddRange(repoId.Split('/'));
            return string.Join(HFGlobalConfig.RepoIdSeparator, parts);
        }

        /// <summary>
        /// Cache reference between a revision (tag, branch or truncated commit hash) and the corresponding commit hash.
        ///
        /// Does nothing if `revision` is already a proper `commit_hash` or reference is already cached.
        /// </summary>
        /// <param name="storageFolder"></param>
        /// <param name="revision"></param>
        /// <param name="commitHash"></param>
        private static void CacheCommitHashForSpecificVersion(string storageFolder, string revision, string commitHash){
            if(revision != commitHash){
                var refPath = Path.Combine(storageFolder, "refs", revision);
                Directory.CreateDirectory(Directory.GetParent(refPath)!.FullName);
                if(!File.Exists(refPath) || commitHash != File.ReadAllText(refPath).Trim()){
                    File.WriteAllText(refPath, commitHash);
                }
            }
        }

        /// <summary>
        /// Check disk usage and log a warning if there is not enough disk space to download the file.
        /// </summary>
        /// <param name="expectedSize"></param>
        /// <param name="targetDir"></param>
        private static void CheckDiskSpace(long expectedSize, string targetDir){
            var drive = new DriveInfo(Directory.GetDirectoryRoot(targetDir));
            if(drive.AvailableFreeSpace < expectedSize){
                Logger?.LogWarning($"Not enough disk space to download the file: {expectedSize} bytes are needed, but only {drive.AvailableFreeSpace} bytes are available.");
            }
        }

        private static string GetPointerPath(string storageFolder, string revision, string relativeFilename){
            var snapshotPath = Path.Combine(storageFolder, "snapshots");
            var pointerPath = Path.Combine(snapshotPath, revision, relativeFilename);
            if(!Path.GetFullPath(pointerPath).Contains(Path.GetFullPath(snapshotPath))){
                throw new Exception("Invalid pointer path: cannot create pointer path in snapshot folder if" +
                    $" `storage_folder='{storageFolder}'`, `revision='{revision}'` and" +
                    $" `relative_filename='{relativeFilename}'`.");
            }
            return pointerPath;
        }

        /// <summary>
        /// Set correct permission before moving a blob from tmp directory to cache dir.
        ///
        /// Do not take into account the `umask` from the process as there is no convenient way
        /// to get it that is thread-safe.
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dst"></param>
        private static void ChmodAndReplace(string src, string dst){
            Logger?.LogWarning("Access control is ignored in current version when copying from temp file to target folder.");
            File.Move(src, dst);
// #if NET6_0_OR_GREATER
//             // Get umask by creating a temporary file in the cached repo folder.
//             var tmpFile = Path.Combine(Directory.GetParent(dst).Parent.FullName, Guid.NewGuid().ToString().Substring(0, 16));
//             try{
//                 File.Create(tmpFile);
//                 var cacheDirMode = new FileInfo(tmpFile).GetAccessControl();
//                 new FileInfo(src).SetAccessControl(cacheDirMode);
//             }
//             finally{
//                 File.Delete(tmpFile);
//             }
//
//             File.Move(src, dst);
// #else
//             Logger?.LogWarning("Cannot set file permissions with dotnet runtime version < 6. Just ignore it...");
// #endif
        }

        private static string ToLocalDir(string path, string localDir, string relativeFilename, bool? useSymlinks){
            var localDirFilepath = Path.Combine(localDir, relativeFilename);
            if(!Path.GetFullPath(localDirFilepath).Contains(Path.GetFullPath(localDir))){
                throw new Exception($"Cannot copy file '{relativeFilename}' to local dir '{localDir}': " +
                "file would not be in the local directory.");
            }
            Directory.CreateDirectory(Directory.GetParent(localDirFilepath)!.FullName);

#if NET6_0_OR_GREATER
            string? realBlobPath = File.ResolveLinkTarget(path, true)?.LinkTarget;
            realBlobPath ??= path;
            Debug.Assert(realBlobPath is not null);
            if(useSymlinks is null){
                useSymlinks = new FileInfo(realBlobPath).Length > HFGlobalConfig.LocalDirAutoSymlinkThreshold;
            }
            if(useSymlinks.Value){
                CreateSymlink(realBlobPath, localDirFilepath, false);
            }
            else{
                File.Copy(realBlobPath, localDirFilepath);
            }
            return localDirFilepath;
#else
            // NOTE: MODIFIED
            UnityEngine.Debug.Log($"Path info: path: {path} -- localFilePath: {localDirFilepath} -- relative: {relativeFilename} -- localdir: {localDir}");
            File.Copy(path, localDirFilepath);
            return localDirFilepath;
#endif
        }

        private static void CreateSymlink(string src, string dst, bool newBlob = false){
#if NET6_0_OR_GREATER
            Debug.Assert(File.Exists(src));
            try{
                File.Delete(dst);
            }
            catch (Exception)
            {
                Logger?.LogWarning(dst + " already exists and cannot be overwritten. Skipped the file copy.");
                return;
            }

            var absSrc = Path.GetFullPath(src);
            var absDst = Path.GetFullPath(dst);
            var absDstFolder = Directory.GetParent(absDst)!.FullName;

            string? relativeSrc;
            // Use relative_dst in priority
            try{
                relativeSrc = Utils.GetRelativePath(absDstFolder, absSrc);
            }
            catch (Exception){
                relativeSrc = null;
            }

            bool supportSymlinks = false;
            try{
                var commonPath = Utils.GetLongestCommonPath(absSrc, absDst);
                supportSymlinks = SymlinksSupported(commonPath);
            }
            catch(Exception){
                supportSymlinks = false;
            }

            if(supportSymlinks){
                var srcRelOrAbs = relativeSrc ?? absSrc;
                try{
                    File.CreateSymbolicLink(absDst, srcRelOrAbs);
                    return;
                }
                catch(Exception ex){
                    // # `absDst` already exists and is a symlink to the `absSrc` blob. It is most likely that the file has
                    // been cached twice concurrently. Do nothing.
                    if(new FileInfo(absDst).LinkTarget is not null && absSrc.Equals(absDst)){
                        return;
                    }
                    else{
                        throw new Exception($"Got an error when creating symlinks: {ex.Message}");
                    }
                }
            }

            // Symlinks are not supported => let's move or copy the file.
            if(newBlob){
                File.Move(absSrc, absDst);
            }
            else{
                File.Copy(absSrc, absDst);
            }
#else
            throw new Exception("Symlinks creation is not supported with dotnet runtime version < 6");
#endif
        }

        private static bool SymlinksSupported(string cacheDir){
#if NET6_0_OR_GREATER
            if (string.IsNullOrEmpty(cacheDir))
            {
                cacheDir = HFGlobalConfig.DefaultCacheDir;
            }

            cacheDir = Path.GetFullPath(Environment.ExpandEnvironmentVariables(cacheDir));

            if (!_symlinksSupportedInDir.TryGetValue(cacheDir, out bool supported))
            {
                try
                {
                    Directory.CreateDirectory(cacheDir);
                    string tempDirectoryPath = Path.Combine(cacheDir, Path.GetRandomFileName());
                    Directory.CreateDirectory(tempDirectoryPath);
                    string srcPath = Path.Combine(tempDirectoryPath, "dummy_file_src.txt");
                    string dstPath = Path.Combine(tempDirectoryPath, "dummy_file_dst.txt");
                    // create a temp file
                    File.WriteAllText(srcPath, string.Empty);
                    // try to create symlink
                    string relativeSrcPath = Utils.GetRelativePath(Path.GetDirectoryName(dstPath)!, srcPath);
                    bool symlinkCreated = File.CreateSymbolicLink(dstPath, relativeSrcPath).Exists;
                    File.Delete(srcPath);
                    Directory.Delete(tempDirectoryPath);
                    supported = symlinkCreated;
                }
                catch
                {
                    supported = false;
                }
                _symlinksSupportedInDir[cacheDir] = supported;
            }
            return supported;
#else
            // We don't support symlinks with dotnet runtime version < 6
            return false;
#endif
        }
    }
}
