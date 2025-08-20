using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.AppUI.Redux;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class AssistantHandler: IDisposable
    {
        ChatStoreManager m_StoreManager;
        LlavaRunner m_LlavaRunner;
        Tensor<float> m_InputTensor;

        public AssistantHandler(ChatStoreManager storeManager)
        {
            m_StoreManager = storeManager;
            m_StoreManager.Store.Subscribe(ChatSlice.Name, (ChatState state) => OnStoreUpdate(state));
        }

        void OnStoreUpdate(ChatState state)
        {
            if (state.Entries[^1].User != ChatEntry.Users.User)
                return;

            var newEntry = new ChatEntry(ChatEntry.Users.Assistant, string.Empty, ChatEntry.EntryStatus.Pending);
            m_StoreManager.Store.Dispatch(m_StoreManager.AddChatEntry.Invoke(newEntry));

            var currentState = m_StoreManager.Store.GetState<ChatState>(ChatSlice.Name);
            InferState(currentState);
        }

        async void InferState(ChatState state)
        {
            var currentAssistantEntry = state.Entries[^1];
            var latestUserEntry = state.Entries[^2];

            m_LlavaRunner ??= new LlavaRunner();
            LlavaDecoder.Input input = null;

            if (m_InputTensor == null)
            {
                input = await m_LlavaRunner.PrepareInputAsync(latestUserEntry.Attachments[0] ?? new Texture2D(256, 256), latestUserEntry.Message);
            }
            else
            {
                input = await ConcatMessagesToInput(m_InputTensor, state.Entries[^3],  latestUserEntry);
            }

            m_InputTensor?.Dispose();
            m_InputTensor = await input.inputsEmbeddings.ReadbackAndCloneAsync();

            var predictionTokens = m_LlavaRunner.GetPredictionTokenAsync(input, 256);
            var tokenList = new List<int>();

            await foreach (var token in predictionTokens)
            {
                tokenList.Add(token);
                var decodedOutput = m_LlavaRunner.Config.Tokenizer.Decode(new []{token});
                var message = currentAssistantEntry.Message + decodedOutput;
                currentAssistantEntry = currentAssistantEntry with { Message = message };
                m_StoreManager.Store.Dispatch(m_StoreManager.UpdateChatEntry.Invoke(currentAssistantEntry));
            }

            currentAssistantEntry = currentAssistantEntry with { Status = ChatEntry.EntryStatus.Completed, EndTimestamp = DateTime.Now };
            m_StoreManager.Store.Dispatch(m_StoreManager.UpdateChatEntry.Invoke(currentAssistantEntry));
            input.Dispose();
        }

        async Task<LlavaDecoder.Input> ConcatMessagesToInput(Tensor<float> inputTensor, ChatEntry latestAssistantEntry, ChatEntry latestUserEntry)
        {
            var newPromptData = latestAssistantEntry.Message + "<|im_end|>" + "<|im_start|>user " + latestUserEntry.Message + "<|im_end|>" + "<|im_start|>assistant";
            using var output = m_LlavaRunner.EmbedTextTokensAsync(newPromptData);
            using var cpuOutput = await output.ReadbackAndCloneAsync();

            var graph = new FunctionalGraph();
            var inputShape = new DynamicTensorShape(-1, -1, 896);
            var textInput01 = graph.AddInput<float>(inputShape, "text_01_embeddings");
            var textInput02 = graph.AddInput<float>(inputShape, "text_02_embeddings");

            var concat = Functional.Concat(new[] {textInput01, textInput02 }, 1);

            graph.AddOutput(concat, "output");
            var model = graph.Compile();
            using var worker = new Worker(model, m_LlavaRunner.Config.BackendType);
            worker.Schedule(inputTensor, cpuOutput);

            using var outputTensor = worker.PeekOutput() as Tensor<float>;
            var cpuConcatOutput = await outputTensor.ReadbackAndCloneAsync();

            return new LlavaDecoder.Input(cpuConcatOutput);
        }

        public void Dispose()
        {
            m_StoreManager?.Dispose();
            m_LlavaRunner?.Dispose();
            m_InputTensor?.Dispose();
        }
    }
}
