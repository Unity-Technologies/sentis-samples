using System;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    sealed class EditorChatWindow : EditorWindow
    {
        const string k_VisualTreePath = "UI/ChatWindow";
        ChatWindow m_ChatWindow;

        public void CreateGUI()
        {
            rootVisualElement.Clear();

            var visualTree = Resources.Load<VisualTreeAsset>(k_VisualTreePath);
            if (visualTree != null)
            {
                visualTree.CloneTree(rootVisualElement);
                m_ChatWindow = rootVisualElement.Q<ChatWindow>();
            }
            else
            {
                Debug.LogError($"Failed to load VisualTreeAsset from path: {k_VisualTreePath}");
            }

            titleContent = new GUIContent("Chat");
        }

        void OnDestroy()
        {
            m_ChatWindow?.Dispose();
        }

        [MenuItem("Inference Engine/Sample/Chat/Start Chat")]
        public static void OpenWindow()
        {
            var window = GetWindow<EditorChatWindow>();
            window.titleContent = new GUIContent("Chat");
            window.minSize = new Vector2(300, 400);
            window.Show();
        }

        [MenuItem("Inference Engine/Sample/Chat/Start Chat", true)]
        public static bool OpenWindowValidate()
        {
            return ModelDownloader.VerifyModelsExist();
        }

        [MenuItem("Inference Engine/Sample/Chat/Download Models")]
        public static async Task DownloadModels()
        {
            var modelDownloader = new ModelDownloader();
            await modelDownloader.DownloadModels();
        }

        [MenuItem("Inference Engine/Sample/Chat/Download Models", true)]
        public static bool DownloadModelsValidate()
        {
            return !ModelDownloader.VerifyModelsExist();
        }
    }
}
