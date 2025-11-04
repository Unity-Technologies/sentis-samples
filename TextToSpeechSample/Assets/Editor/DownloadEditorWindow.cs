using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.TTS.Editor
{
    public class DownloadEditorWindow : EditorWindow
    {
        void CreateGUI()
        {
            rootVisualElement.Clear();
            rootVisualElement.AddToClassList("unity-editor");
            titleContent = new GUIContent("Kokoro Text-To-Speech - Download Files");
            var visualTreeAsset = AssetDatabase.LoadAssetAtPath<VisualTreeAsset>("Assets/Src/UI/Network/ModelDownloaderWindow.uxml");
            visualTreeAsset.CloneTree(rootVisualElement);
        }

        [MenuItem("Inference Engine/Sample/Text-To-Speech/Download Models")]
        public static void OpenWindow()
        {
            var window = GetWindow<DownloadEditorWindow>();
            window.minSize = new Vector2(400, 300);
            window.Show();
        }
    }
}
