using System;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class ChatWindow : EditorWindow, IDisposable
    {
        const string k_VisualTreePath = "Packages/com.unity.ai.inference/Samples/LLM Chat/llava/UI/ChatWindow.uxml";

        InputHandler m_InputHandler;
        HistoryHandler m_HistoryHandler;
        AssistantHandler m_AssistantHandler;

        ChatStoreManager m_StoreManager;

        public void CreateGUI()
        {
            var visualTree = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>(k_VisualTreePath);
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

        [MenuItem("Inference Engine/Sample/Chat")]
        public static void OpenWindow()
        {
            var window = GetWindow<ChatWindow>();
            window.titleContent = new GUIContent("Chat");
            window.minSize = new Vector2(300, 400);
            window.Show();
        }

        public void Dispose()
        {
            m_AssistantHandler?.Dispose();
            m_StoreManager?.Dispose();
        }
    }
}
