using System;
using System.Net;
using System.Net.Http;
using System.Web;
using System.Collections.Specialized;
using System.Threading.Tasks;
using HuggingfaceHub.Common;
using HuggingfaceHub.Utilities;
using Newtonsoft.Json;

namespace HuggingfaceHub
{
    public partial class HFDownloader
    {
        /// <summary>
        /// Get info on one specific model on huggingface.co
        /// Model can be private if you pass an acceptable token.
        /// </summary>
        /// <param name="repoId">A namespace (user or an organization) and a repo name separated by a `/`.</param>
        /// <param name="revision">The revision of the model repository from which to get the information.</param>
        /// <param name="timeout">Whether to set a timeout for the request to the Hub.</param>
        /// <param name="securityStatus">Whether to retrieve the security status from the model repo as well.</param>
        /// <param name="filesMetadata">Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to `False`.</param>
        /// <param name="token">
        /// A valid authentication token (see https://huggingface.co/settings/token).
        /// If `None` or `True` and machine is logged in (through `huggingface-cli login`
        /// or [`~huggingface_hub.login`]), token will be retrieved from the cache.
        /// If `False`, token is not sent in the request header.
        /// </param>
        /// <param name="endpoint">The endpoint to use for the request. Defaults to the global endpoint.</param>
        /// <returns></returns>
        public static async Task<ModelInfo> GetModelInfoAsync(string repoId, string revision = "main", int? timeout = null,
            bool securityStatus = false, bool filesMetadata = false, string? token = null, string? endpoint = null)
        {
            endpoint ??= HFGlobalConfig.EndPoint;
            var header = BuildHFHeaders(token: token);
            string path = revision.Equals("main") ? $"{endpoint}/api/models/{repoId}"
             : $"{endpoint}/api/models/{repoId}/revision/{Uri.EscapeDataString(revision)}";
            UriBuilder baseUri = new UriBuilder(path);
            NameValueCollection queryString = HttpUtility.ParseQueryString(string.Empty);
            if(securityStatus){
                queryString["securityStatus"] = true.ToString();
            }
            if(filesMetadata){
                queryString["blobs"] = true.ToString();
            }
            baseUri.Query = queryString.ToString();

            using(var request = new HttpRequestMessage(HttpMethod.Get, baseUri.Uri)){
                foreach(var key in header.Keys){
                    request.Headers.TryAddWithoutValidation(key, header[key]);
                }
                var response = await HttpClient.SendAsync(request);
                if(response.StatusCode != HttpStatusCode.OK)
                {
                    throw new HttpRequestException($"Failed to get model info: {response.StatusCode}. " +
                    "Please check your input first, then try to use a mirror site.");
                }
                var content = await response.Content.ReadAsStringAsync();
                return JsonConvert.DeserializeObject<ModelInfo>(content);
            }
        }
    }
}