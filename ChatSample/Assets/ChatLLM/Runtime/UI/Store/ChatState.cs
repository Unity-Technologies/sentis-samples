using System;
using System.Collections.Generic;

namespace Unity.InferenceEngine.Samples.Chat
{
    record ChatState
    {
        public string AttachmentPath = null;
        public List<ChatEntry> Entries = new();
    }
}
