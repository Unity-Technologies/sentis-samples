using System;
using System.Collections.Generic;
using Unity.AppUI.Redux;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat
{
    class HistoryHandler
    {
        ChatStoreManager m_StoreManager;
        ScrollView m_HistoryListView;

        List<ChatEntryElement> m_ChatEntryElements = new();

        public HistoryHandler(ChatStoreManager storeManager, ChatWindow ctxWindow)
        {
            m_StoreManager = storeManager;
            m_StoreManager.Store.Subscribe(ChatSlice.Name, (ChatState state) => OnStoreUpdate(state));

            m_HistoryListView = ctxWindow.rootVisualElement.Q<ScrollView>("Chat History");

            var currentState = m_StoreManager.Store.GetState<ChatState>(ChatSlice.Name);
            OnStoreUpdate(currentState);
        }

        void OnStoreUpdate(ChatState newState)
        {
            for (var i = 0; i < newState.Entries.Count; ++i)
            {
                if(i >= m_ChatEntryElements.Count)
                {
                    var newElement = new ChatEntryElement();
                    m_ChatEntryElements.Add(newElement);
                    m_HistoryListView.Add(newElement);
                }

                m_ChatEntryElements[i].SetChatEntry(newState.Entries[i]);

                m_HistoryListView.scrollOffset = new Vector2(0, int.MaxValue);
            }
        }
    }
}
