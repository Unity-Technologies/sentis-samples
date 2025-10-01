using System;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Utils
{
    public static class VoicesUtils
    {
        const string k_IndexFilePath = "voicesIndex";

        public static string[]  GetVoicesList()
        {
            var voicesIndex = Resources.Load<TextAsset>(k_IndexFilePath);
            var voiceText = voicesIndex.text.Replace(".bin", string.Empty);
            voiceText = voiceText.TrimEnd('\n');
            return voiceText.Split('\n');
        }
    }
}
