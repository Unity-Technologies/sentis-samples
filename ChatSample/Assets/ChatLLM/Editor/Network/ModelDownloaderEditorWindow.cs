using System;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    public class ModelDownloaderEditorWindow: EditorWindow
    {
        void CreateGUI()
        {
            rootVisualElement.Clear();
            var visualTemplate = Resources.Load<VisualTreeAsset>("ModelDownloaderWindow");
            visualTemplate.CloneTree(rootVisualElement);
        }
    }
}
