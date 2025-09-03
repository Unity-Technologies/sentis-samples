using System;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    class LlavaEmbedderTest
    {
        [Test]
        public async Task TestLlavaEmbedder()
        {
            var config = new LlavaConfig(BackendType.GPUCompute);
            // Arrange
            var embedder = new LlavaEmbedder(config);
            var inputText = "This is a test input for LlavaEmbedder.";

            // Act
            var input = new LlavaEmbedder.Input(inputText);
            var outputs = embedder.Schedule(input);

            var output = await outputs.embedding.ReadbackAndCloneAsync();

            // Assert
            Assert.IsNotNull(output);

            output.Dispose();
            outputs.Dispose();
            embedder.Dispose();
        }
    }
}
