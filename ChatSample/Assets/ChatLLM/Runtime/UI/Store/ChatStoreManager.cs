using System;
using Unity.AppUI.Redux;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    [Serializable]
    sealed class ChatStoreManager : IDisposable
    {
        public IStore<PartitionedState> Store;
        public ActionCreator<ChatEntry> AddChatEntry = new(ChatSlice.AddChatEntry);
        public ActionCreator<ChatEntry> UpdateChatEntry = new(ChatSlice.UpdateChatEntry);
        public ActionCreator<string> SetAttachment = new(ChatSlice.SetAttachment);

        public ChatStoreManager()
        {
            var state = new ChatState();

            var slice = StoreFactory.CreateSlice(
                ChatSlice.Name,
                state,
                builder =>
                {
                    builder.AddCase(AddChatEntry, ChatReducers.AddChatEntry);
                    builder.AddCase(UpdateChatEntry, ChatReducers.UpdateChatEntry);
                    builder.AddCase(SetAttachment, ChatReducers.SetAttachment);
                });

            Store = StoreFactory.CreateStore(new[] { slice });
        }
        public void Dispose()
        {
            Store?.Dispose();
            Store = null;
        }
    }
}
