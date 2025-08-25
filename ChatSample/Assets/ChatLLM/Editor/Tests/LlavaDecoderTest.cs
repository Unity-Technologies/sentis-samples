using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    class LlavaDecoderTest
    {
        [Test]
        public async Task TestLlavaDecoder()
        {
            var config = new LlavaConfig(BackendType.GPUCompute);
            using var embedder = new LlavaEmbedder(config);
            using var visionEncoder = new LlavaVisionEncoder(config);
            using var decoder = new LlavaDecoder(config);

            var testTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/ChatLLM/Editor/Tests/Test-Image.png");

            var prompt = "You are a labeling tool. Only provide descriptions. Here is a Texture2D asset. Describe its content.";
            var chatTemplatePrompt = LlavaConfig.ApplyChatTemplate(prompt);

            using var textEmbeddingOutput = embedder.Schedule(new LlavaEmbedder.Input(chatTemplatePrompt));
            using var visionEmbeddingOutput = visionEncoder.Schedule(new LlavaVisionEncoder.Input(new List<Texture2D> { testTexture }));

            using var cpuTextEmbedding = await textEmbeddingOutput.embedding.ReadbackAndCloneAsync();
            using var cpuVisionEmbedding = await visionEmbeddingOutput.features.ReadbackAndCloneAsync();

            using var input = new LlavaDecoder.Input(textEmbeddingOutput.encoding, cpuTextEmbedding, cpuVisionEmbedding);
            using var output = decoder.Schedule(input);
            using var logits = await output.logits.ReadbackAndCloneAsync();

            var pastKeys = new Tensor[output.presentKeys.Length];
            for (var i = 0; i < output.presentKeys.Length; i++)
            {
                pastKeys[i] = await output.presentKeys[i].ReadbackAndCloneAsync();
            }

            var pastValues = new Tensor[output.presentValues.Length];
            for (var i = 0; i < output.presentValues.Length; i++)
            {
                pastValues[i] = await output.presentValues[i].ReadbackAndCloneAsync();
            }


            var goodLogits = decoder.InterpretLogits(logits);
            Debug.Log($"Good logits: {string.Join(", ", goodLogits)}");
            var decoded = config.Tokenizer.Decode(new[] { goodLogits });
            Debug.Log($"Decoded output: {decoded}");

            Assert.IsNotNull(logits);
            Assert.IsTrue(logits.shape[0] > 0);
            Assert.IsTrue(logits.shape[0] > 0);
            Assert.IsNotNull(decoded);

            // Ensure all tensors in arrays are disposed
            foreach (var key in pastKeys)
            {
                key.Dispose();
            }

            foreach (var value in pastValues)
            {
                value.Dispose();
            }
        }
    }
}
