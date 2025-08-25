using System;
using System.IO;
using Unity.AppUI.Redux;
using Unity.AppUI.UI;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat
{
    class AttachmentElement : VisualElement
    {
        IconButton m_RemoveButton;
        ChatStoreManager m_ChatStoreManager;

        internal AttachmentElement(ChatStoreManager storeManager)
        {
            m_ChatStoreManager = storeManager;
            var chatState = m_ChatStoreManager.Store.GetState<ChatState>(ChatSlice.Name);
            AddToClassList("attachment-element");

            var fileIcon = new Icon
            {
                iconName = "file"
            };
            Add(fileIcon);

            var fileNameLabel = new Label
            {
                text = Path.GetFileName(chatState.AttachmentPath)
            };
            Add(fileNameLabel);

            m_RemoveButton = new IconButton
            {
                icon = "x",
                tooltip = "Remove the attached image",
                quiet = true
            };

            m_RemoveButton.clickable.clicked += OnRemoveButtonClicked;

            Add(m_RemoveButton);
        }

        void OnRemoveButtonClicked()
        {
            m_ChatStoreManager.Store.Dispatch(m_ChatStoreManager.SetAttachment.Invoke(string.Empty));
        }
    }
}
