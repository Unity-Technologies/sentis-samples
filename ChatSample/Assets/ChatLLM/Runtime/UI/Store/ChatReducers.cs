using System;
using System.Collections.Generic;
using Unity.AppUI.Redux;

namespace Unity.InferenceEngine.Samples.Chat
{
    class ChatReducers
    {
        public static ChatState AddChatEntry(ChatState state, IAction<ChatEntry> action)
        {
            var newState = state with { Entries = new List<ChatEntry>(state.Entries) { action.payload } };
            return newState;
        }

        public static ChatState UpdateChatEntry(ChatState state, IAction<ChatEntry> action)
        {
            var newState = state with { };
            var index = newState.Entries.FindIndex(e => e.Id == action.payload.Id);
            if (index >= 0)
            {
                newState.Entries[index] = action.payload;
                newState.Entries = new List<ChatEntry>(newState.Entries);
            }

            return newState;
        }

        public static ChatState SetAttachment(ChatState state, IAction<string> attachmentPath)
        {
            var newState = state with { AttachmentPath = attachmentPath.payload };
            return newState;
        }
    }
}
