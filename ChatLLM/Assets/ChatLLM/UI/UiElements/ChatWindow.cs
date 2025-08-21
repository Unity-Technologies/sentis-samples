using System;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat
{
    [UxmlElement]
    public partial class ChatWindow : VisualElement, IDisposable
    {
        InputHandler m_InputHandler;
        HistoryHandler m_HistoryHandler;
        AssistantHandler m_AssistantHandler;
        ChatStoreManager m_StoreManager;
        public ChatWindow()
        {
            RegisterCallback<AttachToPanelEvent>(_ => Initialize());
            RegisterCallback<DetachFromPanelEvent>(_ => Clean());
        }
        void Initialize()
        {
            m_StoreManager ??= new ChatStoreManager();

            m_InputHandler = new InputHandler(m_StoreManager, this);
            m_HistoryHandler = new HistoryHandler(m_StoreManager, this);
            m_AssistantHandler = new AssistantHandler(m_StoreManager);
        }

        void Clean()
        {
            m_InputHandler = null;
            m_HistoryHandler = null;
            m_AssistantHandler?.Dispose();
            m_AssistantHandler = null;
            m_StoreManager?.Dispose();
            m_StoreManager = null;
        }

        public void Dispose()
        {
            Clean();
        }
    }
}
