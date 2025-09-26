using System;
using System.IO;
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
        const string k_OutputWavPath = "Packages/com.unity.ai.inference/Samples/TTS/Tests/Output/kokoro_output.wav";

        static readonly int[] k_TestTokens = { 50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4 };
        static readonly string k_TestSentence = "The quick brown fox darts over the lazy dog at dawn, whispering hello through the mist as church bells chime, rain patters softly, and distant violins hum beneath a humming streetlamp.";

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
        }

        [TearDown]
        public void KokoroTearDown()
        {
            m_KokoroHandler?.Dispose();
            m_KokoroHandler = null;
        }
    }
}
