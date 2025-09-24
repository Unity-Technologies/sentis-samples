using System.Threading.Tasks;
using Unity.AppUI.Redux;
using Unity.InferenceEngine.Samples.TTS.Handlers;

namespace Unity.InferenceEngine.Samples.TTS.State
{
    public static class AppAsyncThunks
    {
        public static async Task<Tensor<float>> GenerateSpeech(IThunkAPI<Tensor<float>> thunkAPI)
        {
            var store = thunkAPI.store as IStore<PartitionedState>;
            var state = store.GetState<AppState>(AppSlice.Name);
            var phonemes = PhonemesHandler.TokenizeGraphemes(state.InputText);
            return await state.KokoroHandler.Execute(phonemes, state.Speed, state.Voice);
        }
    }
}
