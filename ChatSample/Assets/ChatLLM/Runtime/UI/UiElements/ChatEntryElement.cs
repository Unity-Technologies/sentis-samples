using System;
using Unity.AppUI.UI;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Samples.Chat
{
    class ChatEntryElement : VisualElement
    {
        Text m_AuthorLabel;
        Text m_TimestampLabel;
        Text m_MessageLabel;
        Image m_AttachedImage;
        VisualElement m_MessageContainer;

        ChatEntry m_ChatEntry;
        public ChatEntryElement()
        {
            AddToClassList("chat-entry-element");

            m_AuthorLabel = new Text();
            m_AuthorLabel.AddToClassList("chat-entry-author");
            Add(m_AuthorLabel);

            m_MessageContainer = new VisualElement();
            m_MessageContainer.AddToClassList("chat-entry-message-container");
            Add(m_MessageContainer);

            m_AttachedImage = new Image();
            m_AttachedImage.AddToClassList("chat-entry-attachment");
            m_MessageContainer.Add(m_AttachedImage);

            m_MessageLabel = new Text();
            m_MessageLabel.AddToClassList("chat-entry-message");
            m_MessageContainer.Add(m_MessageLabel);

            m_TimestampLabel = new Text();
            m_TimestampLabel.AddToClassList("chat-entry-timestamp");
            Add(m_TimestampLabel);
        }

        public void SetChatEntry(ChatEntry entry)
        {
            if (entry == m_ChatEntry)
                return;

            m_AuthorLabel.text = entry.User.ToString();
            m_MessageLabel.text = FormatMessage(entry.Message);

            if (entry.User == ChatEntry.Users.User)
            {
                m_TimestampLabel.text = entry.StartTimestamp.ToString("HH:mm");
            }
            else if (entry.User == ChatEntry.Users.Assistant && entry.Status == ChatEntry.EntryStatus.Completed)
            {
                // Calculate the time difference
                var diff = entry.EndTimestamp - entry.StartTimestamp;

                if (diff.TotalSeconds < 60)
                {
                    // Less than a minute — return seconds
                    m_TimestampLabel.text = $"Took {(int)diff.TotalSeconds} seconds";
                }
                else
                {
                    // More than a minute — return minutes and seconds
                    m_TimestampLabel.text = $"Took {(int)diff.TotalMinutes} minutes {diff.Seconds} seconds";
                }
            }
            else
            {
                m_TimestampLabel.text = string.Empty;
            }

            m_AttachedImage.style.display =  entry.Attachments.Length > 0 ? DisplayStyle.Flex : DisplayStyle.None;
            if (entry.Attachments?.Length > 0)
            {
                m_AttachedImage.image = entry.Attachments[0];
            }

            EnableInClassList("user-entry", entry.User == ChatEntry.Users.User);
            EnableInClassList("assistant-entry", entry.User == ChatEntry.Users.Assistant);
            EnableInClassList("system-entry", entry.User == ChatEntry.Users.System);
        }

        static string FormatMessage(string message)
        {
            if(message == null)
                return string.Empty;

            return message.StartsWith("\n") ? message[1..] : message;
        }
    }
}
