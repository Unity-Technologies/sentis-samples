using NUnit.Framework;
using Unity.InferenceEngine.Samples.TTS.Inference;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Tests
{
    public class PhonemesTest
    {
        const string k_TestText = "Papa";

        [Test]
        public void TokenizePhonemes()
        {
            var tokens =MisakiSharp.TokenizeGraphemes(k_TestText);
            Assert.IsNotNull(tokens, "Tokenization returned null.");
            Assert.IsNotEmpty(tokens, "Tokenization returned empty array.");

            Debug.Log($"Text: {k_TestText}");
            Debug.Log($"Tokens: {string.Join(", ", tokens)}");
        }
    }
}
