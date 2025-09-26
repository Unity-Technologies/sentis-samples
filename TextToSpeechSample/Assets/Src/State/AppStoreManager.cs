using System;
using Unity.AppUI.Redux;
using Unity.InferenceEngine.Samples.TTS.Inference;
using Unity.InferenceEngine.Samples.TTS.Utils;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.State
{
    [Serializable]
    public class AppStoreManager
    {
        public IStore<PartitionedState> Store;

        public readonly ActionCreator<string> SetInputText = new(AppSlice.SetInputText);
        public readonly ActionCreator<KokoroHandler.Voice> SetVoice = new(AppSlice.SetVoice);
        public readonly ActionCreator<float> SetSpeed = new(AppSlice.SetSpeed);

        public readonly AsyncThunkCreator<Tensor<float>> GenerateSpeech = new(AppSlice.GenerateSpeech, AppAsyncThunks.GenerateSpeech);

        public AppStoreManager()
        {
            var state = new AppState
            {
                KokoroHandler = new KokoroHandler()
            };

            var slice = StoreFactory.CreateSlice(
                AppSlice.Name,
                state,
                builder =>
                {
                    builder.AddCase(SetInputText, AppReducers.SetInputText);
                    builder.AddCase(SetVoice, AppReducers.SetVoice);
                    builder.AddCase(SetSpeed, AppReducers.SetSpeed);
                    builder.AddCase(GenerateSpeech.pending, (state, _) =>
                    {
                        var newState = state with { Status = AppState.GenerationStatus.Pending };
                        newState.CleanGenerationData();

                        return newState;
                    });
                    builder.AddCase(GenerateSpeech.rejected, (state, _) =>
                    {
                        Debug.LogError("Speech generation failed.");
                        var newState = state with { Status = AppState.GenerationStatus.Failed };
                        return newState;
                    });
                    builder.AddCase(GenerateSpeech.fulfilled, (state, action) =>
                    {
                        var audioClip = AudioClipUtils.ToAudioClip(action.payload);
                        var newState = state with { Waveform = action.payload, Status = AppState.GenerationStatus.Completed, AudioClip = audioClip };
                        return newState;
                    });
                });

            Store = StoreFactory.CreateStore(new[] { slice });
        }
    }
}
