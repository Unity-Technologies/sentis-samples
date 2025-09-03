using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Unity.InferenceEngine.Samples.Chat
{
    class LlavaRunner : IDisposable
    {
        LlavaDecoder m_Decoder;
        LlavaVisionEncoder m_VisionEncoder;
        LlavaEmbedder m_Embedder;
        LlavaConfig m_Config;

        public LlavaConfig Config => m_Config;

        public LlavaRunner(bool lazyInit = false)
        {
            m_Config = new LlavaConfig(BackendType.GPUCompute);
            m_Decoder = new LlavaDecoder(m_Config, lazyInit);
            m_VisionEncoder = new LlavaVisionEncoder(m_Config, lazyInit);
            m_Embedder = new LlavaEmbedder(m_Config, lazyInit);
        }

        public Tensor<float> EmbedTextTokensAsync(string text)
        {
            return m_Embedder.Schedule(new LlavaEmbedder.Input(text)).embedding;
        }

        public async Task<LlavaDecoder.Input> PrepareInputAsync(Texture2D image, string userPrompt)
        {
            using var visionEncoderInput = new LlavaVisionEncoder.Input(new List<Texture2D> { image });
            using var visionEmbeddingOutput = m_VisionEncoder.Schedule(visionEncoderInput);
            using var cpuVisionEmbedding = await visionEmbeddingOutput.features.ReadbackAndCloneAsync();

            // Prompt Encoding
            var prompt = LlavaConfig.ApplyChatTemplate(userPrompt);
            using var textEmbeddingOutput = m_Embedder.Schedule(new LlavaEmbedder.Input(prompt));
            using var cpuTextEmbedding = await textEmbeddingOutput.embedding.ReadbackAndCloneAsync();

            return new LlavaDecoder.Input(textEmbeddingOutput.encoding, cpuTextEmbedding, cpuVisionEmbedding);
        }

        public async IAsyncEnumerable<int> GetPredictionTokenAsync(LlavaDecoder.Input input, int maxTokens = 512)
        {
            var stepCount = 0;

            // Track the total sequence length including image embeddings
            var totalSequenceLength = input.inputsEmbeddings.shape[1];

            bool endTokenReached;

            do
            {
                var output = m_Decoder.Schedule(input);
                using var logits = await output.logits.ReadbackAndCloneAsync();
                output.logits.Dispose();

                var token = m_Decoder.InterpretLogits(logits);

                endTokenReached = token == LlavaConfig.TokenIdEndOfText;

                if(!endTokenReached)
                    yield return token;

                if (!endTokenReached)
                {
                    using var tokenEmbedding = m_Embedder.Schedule(new[] { token });
                    var tokenEmbeddingCpu = await tokenEmbedding.ReadbackAndCloneAsync();

                    var currentSequenceLength = totalSequenceLength + stepCount + 1;
                    var attentionMaskArray = new int[currentSequenceLength];
                    for (var i = 0; i < currentSequenceLength; i++)
                    {
                        attentionMaskArray[i] = 1;
                    }
                    var attentionMask = new Tensor<int>(new TensorShape(1, currentSequenceLength), attentionMaskArray);

                    var currentPosition = totalSequenceLength + stepCount;
                    var positionId = new Tensor<int>(new TensorShape(1, 1), new[] { currentPosition });

                    input.Dispose();
                    input.attentionMask = attentionMask;
                    input.positionIds = positionId;
                    input.inputsEmbeddings = tokenEmbeddingCpu;

                    input.pastKeys = output.presentKeys;
                    input.pastKeysValues = output.presentValues;

                    stepCount++;
                }
                else
                {
                    output.Dispose();
                }
            }
            while (!endTokenReached && stepCount < maxTokens);

            if (stepCount >= maxTokens)
            {
                Debug.Log($"[LlavaRunner] Reached maximum token limit of {maxTokens}");
            }
        }

        public async IAsyncEnumerable<int> GetPredictionTokenAsync(Texture2D image, string userPrompt, int maxTokens = 512)
        {
            using var input = await PrepareInputAsync(image, userPrompt);
            await foreach (var token in GetPredictionTokenAsync(input, maxTokens))
            {
                yield return token;
            }
        }

        public void Dispose()
        {
            m_Decoder?.Dispose();
            m_VisionEncoder?.Dispose();
            m_Embedder?.Dispose();
            m_Decoder = null;
            m_VisionEncoder = null;
            m_Embedder = null;
            m_Config = null;
        }
    }
}
