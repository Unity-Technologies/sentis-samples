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
            var phonemes = PhonemesHandler.TextToPhonemes(k_TestText);
            Assert.IsNotNull(phonemes, "Phoneme conversion returned null.");
            Assert.IsNotEmpty(phonemes, "Phoneme conversion returned empty string.");

            var tokens = PhonemesHandler.Tokenize(phonemes);
            Assert.IsNotNull(tokens, "Tokenization returned null.");
            Assert.IsNotEmpty(tokens, "Tokenization returned empty array.");

            Debug.Log($"Text: {k_TestText}");
            Debug.Log($"Phonemes: {phonemes}");
            Debug.Log($"Tokens: {string.Join(", ", tokens)}");

        }
    }
}
