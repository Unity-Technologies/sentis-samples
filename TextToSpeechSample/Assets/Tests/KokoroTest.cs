using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NUnit.Framework;
using Unity.InferenceEngine.Samples.TTS.Inference;
using Unity.InferenceEngine.Samples.TTS.Utils;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Tests
{
    public class KokoroTest
    {
        const string k_OutputWavPath = "Assets/Tests/Output/kokoro_output.wav";
        const string k_OutputWavPathShort = "Assets/Tests/Output/kokoro_short.wav";
        const string k_OutputWavPathSpeed = "Assets/Tests/Output/kokoro_speed.wav";

        static readonly int[] k_TestTokens = { 50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4 };
        static readonly string k_TestSentence = "The quick brown fox darts over the lazy dog at dawn, whispering hello through the mist as church bells chime, rain patters softly, and distant violins hum beneath a humming streetlamp.";
        static readonly string k_ShortTestSentence = "Hello, world!";

        KokoroHandler m_KokoroHandler;

        [SetUp]
        public void KokoroSetup()
        {
            m_KokoroHandler = new KokoroHandler(lazyLoadModel: false, backendType: BackendType.GPUCompute);
        }

        [Test]
        public async Task RunKokoroWithToken()
        {
            var voices = KokoroHandler.GetVoices();
            using var output = await m_KokoroHandler.Execute(k_TestTokens, speed: 1.0f, voices[0]);

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }

            WavUtils.WriteFloatWav(Path.GetFullPath(k_OutputWavPath), output.DownloadToArray());
            Assert.IsNotNull(output, "Failed to get output from Kokoro model.");

            var audioData = output.DownloadToArray();
            Assert.IsTrue(audioData.Length > 0, "Audio output should not be empty.");
            Debug.Log($"Generated audio with {audioData.Length} samples from predefined tokens.");
        }

        [Test]
        public async Task RunKokoroWithText()
        {
            var phonemes = MisakiSharp.TokenizeGraphemes(k_TestSentence);
            var voices = KokoroHandler.GetVoices();
            using var output = await m_KokoroHandler.Execute(phonemes, speed: 1.0f, voices[0]);

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }

            WavUtils.WriteFloatWav(Path.GetFullPath(k_OutputWavPath), output.DownloadToArray());
            Assert.IsNotNull(output, "Failed to get output from Kokoro model.");

            var audioData = output.DownloadToArray();
            Assert.IsTrue(audioData.Length > 0, "Audio output should not be empty.");
            Debug.Log($"Generated audio with {audioData.Length} samples from text: '{k_TestSentence}'");
        }

        [Test]
        public async Task RunKokoroWithShortText()
        {
            var phonemes = MisakiSharp.TokenizeGraphemes(k_ShortTestSentence);
            var voices = KokoroHandler.GetVoices();
            using var output = await m_KokoroHandler.Execute(phonemes, speed: 1.0f, voices[0]);

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }

            WavUtils.WriteFloatWav(Path.GetFullPath(k_OutputWavPathShort), output.DownloadToArray());
            Assert.IsNotNull(output, "Failed to get output from Kokoro model for short text.");

            var audioData = output.DownloadToArray();
            Assert.IsTrue(audioData.Length > 0, "Audio output should not be empty for short text.");
            Debug.Log($"Generated short audio with {audioData.Length} samples from: '{k_ShortTestSentence}'");
        }

        [Test]
        public async Task RunKokoroWithDifferentSpeeds()
        {
            var phonemes = MisakiSharp.TokenizeGraphemes(k_ShortTestSentence);
            var voices = KokoroHandler.GetVoices();

            var speeds = new[] { 0.5f, 1.0f, 1.5f, 2.0f };
            var audioLengths = new int[speeds.Length];

            for (int i = 0; i < speeds.Length; i++)
            {
                using var output = await m_KokoroHandler.Execute(phonemes, speed: speeds[i], voices[0]);
                Assert.IsNotNull(output, $"Failed to get output at speed {speeds[i]}.");

                var audioData = output.DownloadToArray();
                audioLengths[i] = audioData.Length;
                Assert.IsTrue(audioData.Length > 0, $"Audio output should not be empty at speed {speeds[i]}.");

                if (i == 1) // Save normal speed for reference
                {
                    WavUtils.WriteFloatWav(Path.GetFullPath(k_OutputWavPathSpeed), audioData);
                }

                Debug.Log($"Speed {speeds[i]}: Generated {audioData.Length} samples");
            }

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }

            // Verify that different speeds produce different length outputs
            Assert.AreNotEqual(audioLengths[0], audioLengths[3], "Slow and fast speeds should produce different length audio.");
        }

        [Test]
        public async Task RunKokoroWithMultipleVoices()
        {
            var phonemes = MisakiSharp.TokenizeGraphemes(k_ShortTestSentence);
            var voices = KokoroHandler.GetVoices();

            Assert.IsTrue(voices.Count > 0, "Should have at least one voice available.");

            for (int i = 0; i < Math.Min(voices.Count, 3); i++) // Test up to 3 voices
            {
                using var output = await m_KokoroHandler.Execute(phonemes, speed: 1.0f, voices[i]);
                Assert.IsNotNull(output, $"Failed to get output from voice {i}.");

                var audioData = output.DownloadToArray();
                Assert.IsTrue(audioData.Length > 0, $"Audio output should not be empty for voice {i}.");

                Debug.Log($"Voice {i}: Generated {audioData.Length} samples");
            }

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }
        }

        [Test]
        public async Task RunKokoroWithSpecialCharacters()
        {
            var testTexts = new[]
            {
                "Hello, world! How are you?",
                "It's a beautiful day, isn't it?",
                "The year 2023 was amazing.",
                "She said, \"Hello there!\"",
                "Numbers: 1, 2, 3, 100, 1000."
            };

            var voices = KokoroHandler.GetVoices();

            foreach (var testText in testTexts)
            {
                var phonemes = MisakiSharp.TokenizeGraphemes(testText);
                Assert.IsTrue(phonemes.Length > 0, $"Should generate phonemes for: '{testText}'");

                using var output = await m_KokoroHandler.Execute(phonemes, speed: 1.0f, voices[0]);
                Assert.IsNotNull(output, $"Failed to get output for text: '{testText}'");

                var audioData = output.DownloadToArray();
                Assert.IsTrue(audioData.Length > 0, $"Audio output should not be empty for: '{testText}'");

                Debug.Log($"Special chars test '{testText}': {audioData.Length} samples");
            }

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }
        }

        [Test]
        public async Task RunKokoroEndToEndPipeline()
        {
            // Test the complete pipeline: text -> phonemes -> tokens -> audio
            var inputText = "Testing the complete text-to-speech pipeline with MisakiSharp.";

            // Step 1: Convert text to phonemes
            var phonemesString = MisakiSharp.TextToPhonemes(inputText);
            Assert.IsNotNull(phonemesString, "TextToPhonemes should not return null.");
            Assert.IsNotEmpty(phonemesString, "TextToPhonemes should not return empty string.");
            Debug.Log($"Generated phonemes: '{phonemesString}'");

            // Step 2: Convert phonemes to tokens
            var tokens = MisakiSharp.TokenizeGraphemes(inputText);
            Assert.IsNotNull(tokens, "TokenizeGraphemes should not return null.");
            Assert.IsTrue(tokens.Length > 0, "TokenizeGraphemes should return non-empty array.");
            Debug.Log($"Generated {tokens.Length} tokens: [{string.Join(", ", tokens)}]");

            // Step 3: Generate audio from tokens
            var voices = KokoroHandler.GetVoices();
            using var output = await m_KokoroHandler.Execute(tokens, speed: 1.0f, voices[0]);
            Assert.IsNotNull(output, "Kokoro execution should not return null.");

            var audioData = output.DownloadToArray();
            Assert.IsTrue(audioData.Length > 0, "Generated audio should not be empty.");

            Debug.Log($"End-to-end pipeline complete: '{inputText}' -> {tokens.Length} tokens -> {audioData.Length} audio samples");

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }
        }

        [Test]
        public void ValidateVoicesAvailability()
        {
            var voices = KokoroHandler.GetVoices();
            Assert.IsNotNull(voices, "GetVoices should not return null.");
            Assert.IsTrue(voices.Count > 0, "Should have at least one voice available.");

            Debug.Log($"Available voices: {voices.Count}");

            foreach (var voice in voices)
            {
                Assert.IsNotNull(voice, "Each voice should not be null.");
                voice?.Dispose();
            }
        }

        [Test]
        public async Task RunKokoroStressTest()
        {
            // Test with a longer text to stress test the system
            var longText = string.Join(" ", Enumerable.Repeat(k_ShortTestSentence, 10));

            var phonemes = MisakiSharp.TokenizeGraphemes(longText);
            Assert.IsTrue(phonemes.Length > 100, "Long text should generate many tokens.");

            var voices = KokoroHandler.GetVoices();
            using var output = await m_KokoroHandler.Execute(phonemes, speed: 1.0f, voices[0]);

            Assert.IsNotNull(output, "Should handle long text without failing.");
            var audioData = output.DownloadToArray();
            Assert.IsTrue(audioData.Length > 10000, "Long text should generate substantial audio.");

            Debug.Log($"Stress test: {longText.Length} chars -> {phonemes.Length} tokens -> {audioData.Length} samples");

            foreach (var voice in voices)
            {
                voice?.Dispose();
            }
        }

        [TearDown]
        public void KokoroTearDown()
        {
            m_KokoroHandler?.Dispose();
            m_KokoroHandler = null;
        }
    }
}
