using System;
using System.Collections.Generic;
using System.Net;
using System.Text.RegularExpressions;
using System.Collections.Specialized;
using System.Net.Http.Headers;
using System.Web;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace HuggingfaceHub
{
    internal static class Utils
    {
        public static string GetRelativePath(string referencePath, string filePath)
        {
#if NETSTANDARD2_0
            if (!referencePath.EndsWith(Path.DirectorySeparatorChar.ToString()))
            {
                referencePath += Path.DirectorySeparatorChar;
            }

            Uri fileUri = new Uri(filePath);
            Uri referenceUri = new Uri(referencePath);

            Uri relativeUri = referenceUri.MakeRelativeUri(fileUri);

            string relativePath = Uri.UnescapeDataString(relativeUri.ToString());

            relativePath = relativePath.Replace('/', Path.DirectorySeparatorChar);

            return relativePath;
#else
            return Path.GetRelativePath(referencePath, filePath);
#endif
        }

        public static string GetLongestCommonPath(string path1, string path2)
        {
            string[] parts1 = path1.Split(Path.DirectorySeparatorChar);
            string[] parts2 = path2.Split(Path.DirectorySeparatorChar);

            List<string> commonParts = new List<string>();

            int minLength = Math.Min(parts1.Length, parts2.Length);
            for (int i = 0; i < minLength; i++)
            {
                if (parts1[i].Equals(parts2[i], StringComparison.OrdinalIgnoreCase))
                {
                    commonParts.Add(parts1[i]);
                }
                else
                {
                    break;
                }
            }

            string commonPath = String.Join(Path.DirectorySeparatorChar.ToString(), commonParts);

            return commonPath;
        }

        /// <summary>
        /// Filter repo objects based on an allowlist and a denylist.
        /// </summary>
        /// <param name="items"></param>
        /// <param name="allow_patterns"></param>
        /// <param name="ignorePatterns"></param>
        /// <returns></returns>
        public static IEnumerable<string> FilterRepoObjects(IEnumerable<string> items, string[]? allowPatterns, string[]? ignorePatterns){
            foreach(var item in items){
                // Skip if there's an allowlist and path doesn't match any
                if(allowPatterns is not null && allowPatterns.Length > 0 && allowPatterns.All(p => !FileNameMatches(item, p))){
                    continue;
                }
                // Skip if there's a denylist and path matches any
                if(ignorePatterns is not null && ignorePatterns.Length > 0 && ignorePatterns.Any(p => FileNameMatches(item, p))){
                    continue;
                }
                yield return item;
            }
        }

        public static bool FileNameMatches(string fileName, string pattern)
        {
            string regexPattern = "^" + Regex.Escape(pattern)
                                    .Replace("\\*", ".*")
                                    .Replace("\\?", ".") + "$";

            return Regex.IsMatch(fileName, regexPattern, RegexOptions.IgnoreCase);
        }

        public static void AddDefaultHeaders(HttpRequestHeaders headers)
        {
            headers.TryAddWithoutValidation("Accept-Encoding", "gzip, deflate");
            headers.TryAddWithoutValidation("Accept", "*/*");
            headers.TryAddWithoutValidation("Connection", "keep-alive");
        }

        /// <summary>
        /// Wrapper around requests methods to follow relative redirects if `follow_relative_redirects=True` even when
        /// `allow_redirection=False`.
        /// </summary>
        /// <param name="client"></param>
        /// <param name="request"></param>
        /// <param name="followRelativeRedirect">
        /// If True, relative redirection (redirection to the same site) will be resolved even when `allow_redirection`
        /// kwarg is set to False. Useful when we want to follow a redirection to a renamed repository without
        /// following redirection to a CDN.
        /// </param>
        /// <param name="isStream">Whether is a request for stream.</param>
        /// <returns></returns>
        public static async Task<HttpResponseMessage> HttpRequestWrapperAsync(HttpClient client, HttpRequestMessage request,
            bool followRelativeRedirect = false, bool isStream = false)
        {
            // if (isStream)
            // {
            //     var headers = request.Headers;
            //     var uri = new UriBuilder(request.RequestUri);
            //     NameValueCollection queryString = HttpUtility.ParseQueryString(string.Empty);
            //     queryString["stream"] = true.ToString();
            //     uri.Query = queryString.ToString();
            //     request = new HttpRequestMessage(request.Method, uri.ToString());
            //     foreach (var item in headers)
            //     {
            //         request.Headers.TryAddWithoutValidation(item.Key, item.Value);
            //     }
            // }
            if (followRelativeRedirect)
            {
                var response = await HttpRequestWrapperAsync(client, request, false);

                if (response.StatusCode is HttpStatusCode.Ambiguous or HttpStatusCode.Found or HttpStatusCode.Moved or
                    HttpStatusCode.Redirect or HttpStatusCode.Unused or HttpStatusCode.MovedPermanently or
                    HttpStatusCode.MultipleChoices or HttpStatusCode.NotModified or
                    HttpStatusCode.RedirectMethod or HttpStatusCode.SeeOther or HttpStatusCode.TemporaryRedirect or
                    HttpStatusCode.UseProxy or HttpStatusCode.RedirectKeepVerb)
                {
                    if (response.Headers.Location is not null)
                    {
                        var location = response.Headers.Location.ToString();

                        if (Uri.IsWellFormedUriString(location, UriKind.Relative))
                        {
                            var baseUri = request.RequestUri;
                            Debug.Assert(baseUri is not null);
                            var nextUrl = new Uri(baseUri, location).ToString();

                            var newRequest = new HttpRequestMessage(request.Method, nextUrl);
                            return await HttpRequestWrapperAsync(client, newRequest, followRelativeRedirect);
                        }
                    }
                }

                return response;
            }

            if (isStream)
            {
                return await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
            }
            else
            {
                return await client.SendAsync(request);
            }
        }
    }
}