using System;
using System.IO;
using Unity.AppUI.Redux;
using Unity.AppUI.UI;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using TextField = Unity.AppUI.UI.TextField;

namespace Unity.InferenceEngine.Samples.Chat
{
    class InputHandler
    {
        ChatStoreManager m_StoreManager;
        TextField m_InputField;
        IconButton m_SendButton;
        IconButton m_AttachButton;
        VisualElement m_AttachmentContainer;

        public InputHandler(ChatStoreManager storeManager, ChatWindow ctxWindow)
        {
            m_StoreManager = storeManager;
            m_StoreManager.Store.Subscribe(ChatSlice.Name, (ChatState state) => OnStoreUpdate(state));

            m_InputField = ctxWindow.Q<TextField>("Input Field");

            m_SendButton = ctxWindow.Q<IconButton>("Send Button");
            m_SendButton.clickable.clicked += OnSendButtonClicked;

            m_AttachButton = ctxWindow.Q<IconButton>("Attach Button");
            m_AttachButton.clickable.clicked += OnAttachButtonClicked;

            m_AttachmentContainer = ctxWindow.Q<VisualElement>("Attachment Container");

            var currentState = m_StoreManager.Store.GetState<ChatState>(ChatSlice.Name);
            OnStoreUpdate(currentState);
        }

        void OnAttachButtonClicked()
        {
            #if UNITY_EDITOR
            var path = EditorUtility.OpenFilePanel("Select an image", "", "png,jpg,jpeg");
            m_StoreManager.Store.Dispatch(m_StoreManager.SetAttachment.Invoke(path));
            #endif
        }

        void OnStoreUpdate(ChatState state)
        {
            var sendEnabled = state.Entries?.Count == 0 || state.Entries[^1]?.Status == ChatEntry.EntryStatus.Completed;
            m_SendButton.SetEnabled(sendEnabled);
            m_SendButton.tooltip = sendEnabled ? "Send the message to chat" : "Disabled while processing the last message";

            var attachEnabled = state.Entries?.Count == 0 && string.IsNullOrEmpty(state.AttachmentPath);
            m_AttachButton.SetEnabled(attachEnabled);
            m_AttachButton.tooltip = attachEnabled ? "Attach an image to the message" : "Only available when the chat is empty and no image is attached";

            if (!string.IsNullOrEmpty(state.AttachmentPath))
            {
                m_AttachmentContainer.Add(new AttachmentElement(m_StoreManager));
            }
            else
            {
                m_AttachmentContainer.Clear();
            }
        }

        void OnSendButtonClicked()
        {
            var currentState = m_StoreManager.Store.GetState<ChatState>(ChatSlice.Name);

            var attachedImage = LoadImage(currentState.AttachmentPath);
            var message = m_InputField.value;
            var user = ChatEntry.Users.User;
            var newChatEntry = new ChatEntry(user, message, ChatEntry.EntryStatus.Completed, attachedImage);

            m_StoreManager.Store.Dispatch(m_StoreManager.AddChatEntry.Invoke(newChatEntry));
            m_StoreManager.Store.Dispatch(m_StoreManager.SetAttachment.Invoke(null));
            m_InputField.value = string.Empty;
        }

        Texture2D LoadImage(string path)
        {
            var fileData = File.ReadAllBytes(path);
            var tex = new Texture2D(2, 2);
            if (tex.LoadImage(fileData))
            {
                return tex;
            }

            Debug.LogError("Failed to load image from file.");
            return null;
        }
    }
}
