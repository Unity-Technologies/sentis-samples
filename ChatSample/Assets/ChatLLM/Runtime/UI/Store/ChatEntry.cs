using System;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    record ChatEntry
    {
        public Guid Id;
        public Users User;
        public string Message;
        public DateTime StartTimestamp;
        public DateTime EndTimestamp;
        public Texture2D[] Attachments;
        public EntryStatus Status;

        public ChatEntry(Users user, string message, EntryStatus status, params Texture2D[] attachments)
        {
            Id = Guid.NewGuid();
            User = user;
            Message = message;
            StartTimestamp = DateTime.Now;
            Attachments = attachments;
            Status = status;
        }

        public enum EntryStatus
        {
            Pending,
            InProgress,
            Completed,
            Failed
        }

        public enum Users
        {
            User,
            Assistant,
            System
        }
    }
}
