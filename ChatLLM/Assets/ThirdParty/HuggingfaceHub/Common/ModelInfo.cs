using Newtonsoft.Json;

namespace HuggingfaceHub.Common
{
    /// <summary>
    /// Contains information about a model on the Hub.
    /// 
    /// Most attributes of this class are optional. This is because the data returned by the Hub depends on the query made.
    /// In general, the more specific the query, the more information is returned. On the contrary, when listing models
    /// using [`list_models`] only a subset of the attributes are returned.
    /// </summary>
    public class ModelInfo
    {
        [JsonProperty("id")]
        public string Id {get; set;}
        [JsonProperty("author")]
        public string Author {get; set;}
        [JsonProperty("sha")]
        public string Sha {get; set;}
        [JsonProperty("created_at")]
        public string CreatedAt {get; set;}
        [JsonProperty("last_modified")]
        public string LastModified {get; set;}
        [JsonProperty("private")]
        public bool Private {get; set;}
        [JsonProperty("disabled")]
        public bool Disabled {get; set;}
        [JsonProperty("downloads")]
        public int Downloads {get; set;}
        [JsonProperty("likes")]
        public int Likes {get; set;}
        [JsonProperty("library_name")]
        public string LibraryName {get; set;}
        [JsonProperty("tags")]
        public string[] Tags {get; set;}
        [JsonProperty("pipeline_tag")]
        public string PipelineTag {get; set;}
        [JsonProperty("mask_token")]
        public string MaskToken {get; set;}
        [JsonProperty("siblings")]
        public RepoSibling[] Siblings {get; set;}
        [JsonProperty("spaces")]
        public string[] Spaces {get; set;}
    }
}