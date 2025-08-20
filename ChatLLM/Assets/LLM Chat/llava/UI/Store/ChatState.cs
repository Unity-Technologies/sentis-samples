using System;
using System.Collections.Generic;

namespace Unity.InferenceEngine.Samples.Chat
{
    record ChatState
    {
        public List<ChatEntry> Entries = new();
    }
}
