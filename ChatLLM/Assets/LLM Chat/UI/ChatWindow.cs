using System;
using Unity.InferenceEngine.Samples.Chat.LLM_Chat.Network;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class ChatWindow : EditorWindow, IDisposable
    {
        const string k_VisualTreePath = "UI/ChatWindow";

        InputHandler m_InputHandler;
        HistoryHandler m_HistoryHandler;
        AssistantHandler m_AssistantHandler;

        ChatStoreManager m_StoreManager;

        public void CreateGUI()
        {
            rootVisualElement.Clear();

            var visualTree = Resources.Load<VisualTreeAsset>(k_VisualTreePath);
            if (visualTree != null)
            {
                visualTree.CloneTree(rootVisualElement);
            }
            else
            {
                Debug.LogError($"Failed to load VisualTreeAsset from path: {k_VisualTreePath}");
            }

            titleContent = new GUIContent("Chat");

            m_StoreManager ??= new ChatStoreManager();

            m_InputHandler = new InputHandler(m_StoreManager, this);
            m_HistoryHandler = new HistoryHandler(m_StoreManager, this);
            m_AssistantHandler = new AssistantHandler(m_StoreManager);
        }

        void OnDestroy()
        {
            Dispose();
        }

        [MenuItem("Inference Engine/Sample/Chat/Start Chat")]
        public static void OpenWindow()
        {
            var window = GetWindow<ChatWindow>();
            window.titleContent = new GUIContent("Chat");
            window.minSize = new Vector2(300, 400);
            window.Show();
        }

        [MenuItem("Inference Engine/Sample/Chat/Start Chat", true)]
        public static bool OpenWindowValidate()
        {
            return false;
        }

        [MenuItem("Inference Engine/Sample/Chat/Download Models")]
        public static async void DownloadModels()
        {
            await ModelDownloader.DownloadModels();
        }

        public void Dispose()
        {
            m_AssistantHandler?.Dispose();
            m_StoreManager?.Dispose();
        }
    }
}
