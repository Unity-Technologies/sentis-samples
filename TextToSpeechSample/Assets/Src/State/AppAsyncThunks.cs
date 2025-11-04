using System;
using System.Threading.Tasks;
using Unity.AppUI.Redux;
using Unity.InferenceEngine.Samples.TTS.Inference;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.State
{
    public static class AppAsyncThunks
    {
        public static async Task<Tensor<float>> GenerateSpeech(IThunkAPI<Tensor<float>> thunkAPI)
        {
            try
            {
                var store = thunkAPI.store as IStore<PartitionedState>;
                var state = store.GetState<AppState>(AppSlice.Name);
                var phonemes = MisakiSharp.TokenizeGraphemes(state.InputText);
                return await state.KokoroHandler.Execute(phonemes, state.Speed, state.Voice);
            }
            catch (Exception e)
            {
                Debug.LogError($"{e.Message}\n{e.StackTrace}");
                throw;
            }
        }
    }
}
