using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;

namespace HuggingfaceHub
{
    public static partial class HFDownloader
    {
        private static HttpClient HttpClient { get; set; }

        public static ILogger? Logger { get; set; }

        static HFDownloader()
        {
#if WINDOWS
        // In Windows, you can use HttpClientHandler to use the system's certificate store
        HttpClientHandler handler = new HttpClientHandler()
        {
            UseDefaultCredentials = true
        };
#else
            // If you're not in Windows, default handler can be used.
            HttpClientHandler handler = new HttpClientHandler();
#endif
            HttpClient = new HttpClient(handler);
        }

        private static string GetFileUrl(string repoId, string revision, string filename, string? endpoint = null)
        {
            if(endpoint is null){
                endpoint = HFGlobalConfig.EndPoint;
            }
            return $"{endpoint}/{repoId}/resolve/{revision}/{filename}";
        }

        /// <summary>
        /// Build headers dictionary to send in a HF Hub call.
        /// 
        /// By default, authorization token is always provided either from argument (explicit
        /// use) or retrieved from the cache (implicit use). To explicitly avoid sending the
        /// token to the Hub, set `token=False`.

        /// In case of an API call that requires write access, an error is thrown if token is
        /// `null` or token is an organization token (starting with `"api_org***"`).

        /// In addition to the auth header, a user-agent is added to provide information about
        /// the installed packages (versions of python, huggingface_hub, torch, tensorflow,
        /// fastai and fastcore).

        /// </summary>
        /// <param name="token"></param>
        /// <param name="isWriteAction"></param>
        /// <param name="userAgent"></param>
        private static IDictionary<string, string> BuildHFHeaders(string? token = null, bool isWriteAction = false, IDictionary<string, string>? userAgent = null)
        {
            Dictionary<string, string> headers = new();
            userAgent ??= new Dictionary<string, string>();
            headers["user-agent"] = GetHttpUserAgentStr(userAgent);
            if(token is not null){
                var tokenToSend = token;
                ValidateTokenToSend(tokenToSend, isWriteAction);
                headers["authorization"] = $"Bearer {tokenToSend}";
            }

            return headers;
        }

        private static string GetHttpUserAgentStr(IDictionary<string, string> userAgent){
            string res = "unknown/None";
            res += ";" + string.Join(";", userAgent.Select(kv => $"{kv.Key}/{kv.Value}").ToArray());
            return res;
        }

        private static string GetTokenToSend(string token){
            // TODO: deal with token cache here.
            return token;
        }

        private static void ValidateTokenToSend(string token, bool isWriteAction) { 
            if(isWriteAction){
                if(token.StartsWith("api_org")){
                    throw new ArgumentException("You must use your personal account token for write-access methods. To " +
                    " generate a write-access token, go to  https://huggingface.co/settings/tokens");
                }
            }
        }
    }
}