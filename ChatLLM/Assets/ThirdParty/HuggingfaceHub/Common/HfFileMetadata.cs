using System;
using System.Net.Http.Headers;

namespace HuggingfaceHub.Common
{
    public record HfFileMetadata(
        string? CommitHash,
        EntityTagHeaderValue? Etag,
        Uri? Location,
        long? Size
    );
}