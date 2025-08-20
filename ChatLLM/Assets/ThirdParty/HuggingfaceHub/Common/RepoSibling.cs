using Newtonsoft.Json;

namespace HuggingfaceHub.Common
{
    /// <summary>
    /// Contains basic information about a repo file inside a repo on the Hub.
    /// 
    /// All attributes of this class are optional except `rfilename`. This is because only the file names are returned when
    /// listing repositories on the Hub (with [`list_models`], [`list_datasets`] or [`list_spaces`]). If you need more
    /// information like file size, blob id or lfs details, you must request them specifically from one repo at a time
    /// (using [`model_info`], [`dataset_info`] or [`space_info`]) as it adds more constraints on the backend server to
    /// retrieve these.
    /// </summary>
    public class RepoSibling
    {
        [JsonProperty("rfilename")]
        public string Filename { get; set; }
        [JsonProperty("size")]
        public int Size { get; set; }
        [JsonProperty("blob_id")]
        public string BlobId { get; set; }
        [JsonProperty("lfs")]
        public BlobLfsInfo Lfs { get; set; }
    }

    public class BlobLfsInfo
    {
        [JsonProperty("oid")]
        public string Oid { get; set; }
        [JsonProperty("title")]
        public string Title { get; set; }
        [JsonProperty("date")]
        public string Date { get; set; }
    }
}