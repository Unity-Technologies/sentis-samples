using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    class LlavaRunnerTest
    {
        [Test]
        public async Task TestLlavaRunner()
        {
            var runner = new LlavaRunner();

            var testTexture = AssetDatabase.LoadAssetAtPath<Texture2D>("Assets/LLM Chat/Tests/Test-Image.png");
            var predictionTokens = runner.GetPredictionTokenAsync(testTexture, "You are a labeling tool. Only provide descriptions. Here is a Texture2D asset. Describe its content.", 256);
            var tokenList = new List<int>();
            await foreach (var token in predictionTokens)
            {
                tokenList.Add(token);
            }

            var decodedOutput = runner.Config.Tokenizer.Decode(tokenList.ToArray());
            Debug.Log($"Decoded output: {decodedOutput}");
            Assert.IsNotNull(decodedOutput);
            Assert.IsTrue(tokenList.Count > 0, "No tokens were generated.");
            Assert.IsTrue(decodedOutput.Length > 0, "Decoded output is empty.");
            runner.Dispose();
        }
    }
}
