using System;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.TTS.Editor
{
    public class AppEditorWindow : EditorWindow
    {
        void CreateGUI()
        {
            rootVisualElement.Clear();
            rootVisualElement.AddToClassList("unity-editor");
            titleContent = new GUIContent("Kokoro Text-To-Speech");
            var visualTreeAsset = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>("Assets/Src/UI/App.uxml");
            visualTreeAsset.CloneTree(rootVisualElement);
        }

        [MenuItem("Inference Engine/Sample/Text-To-Speech/Start Kokoro")]
        public static void OpenWindow()
        {
            var window = GetWindow<AppEditorWindow>();
            window.minSize = new Vector2(300, 400);
            window.Show();
        }

        [MenuItem("Inference Engine/Sample/Text-To-Speech/Start Kokoro", true)]
        public static bool ValidateOpenWindow()
        {
            var configurations = UI.Network.ModelDownloaderWindow.GetDownloadConfigurations();
            return UI.Network.ModelDownloaderWindow.VerifyModelsExist(configurations);
        }
    }
}
