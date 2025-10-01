using System;

namespace Unity.InferenceEngine.Samples.TTS.State
{
    public static class AppSlice
    {
        public const string Name = "AppSlice";
        public const string SetInputText = Name + "/SetInputText";
        public const string SetVoice = Name + "/SetVoice";
        public const string SetSpeed = Name + "/SetSpeed";
        public const string GenerateSpeech = Name + "/GenerateSpeech";
    }
}
