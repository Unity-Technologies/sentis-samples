using System;
using Unity.AppUI.Redux;
using Unity.InferenceEngine.Samples.TTS.Inference;

namespace Unity.InferenceEngine.Samples.TTS.State
{
    public static class AppReducers
    {
        public static AppState SetInputText(AppState state, IAction<string> inputText)
        {
            var newState = state with { InputText = inputText.payload };
            return newState;
        }

        public static AppState SetVoice(AppState state, IAction<KokoroHandler.Voice> voice)
        {
            var newState = state with { Voice = voice.payload };
            return newState;
        }

        public static AppState SetSpeed(AppState state, IAction<float> speed)
        {
            var newState = state with { Speed = speed.payload };
            return newState;
        }
    }
}
