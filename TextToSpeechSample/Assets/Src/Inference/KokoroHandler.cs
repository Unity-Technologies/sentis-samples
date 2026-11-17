using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Unity.InferenceEngine.Samples.TTS.Assets;
using Unity.InferenceEngine.Samples.TTS.Utils;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Inference
{
    public class KokoroHandler: IDisposable
    {
        const string k_KokoroModelPath = "onnx/model";
        const string k_VoicesFolderPath = "Voices/";

        Model m_Model;
        Worker m_Worker;
        readonly BackendType m_BackendType;

        public KokoroHandler(BackendType backendType = BackendType.GPUCompute, bool lazyLoadModel = true)
        {
            m_BackendType = backendType;

            if (!lazyLoadModel)
                LoadModelIfMissing();
        }

        void LoadModelIfMissing()
        {
            if (m_Model != null)
                return;

            var modelAsset = Resources.Load<ModelAsset>(k_KokoroModelPath);

            m_Model = ModelLoader.Load(modelAsset);

            m_Worker = new Worker(m_Model, m_BackendType);
        }

        // Kokoro's context is 512 tokens including the two pad tokens.
        const int k_MaxModelTokens = 510;

        // A single generation longer than ~2^19 samples (~21.8 s at 24 kHz) degrades into whisper-like noise,
        // so generations are kept below this, with margin.
        const int k_MaxOutputSamples = 500_000;

        // Initial chunk budget at speed 1.0 (~1,500 samples per token => ~15 s), scaled by speed.
        const int k_ChunkTokensAtSpeed1 = 250;

        const int k_StyleDim = 256;

        // Token ids (see MisakiSharp.k_PhonemeVocab) used to choose natural chunk boundaries.
        static readonly HashSet<int> k_SentenceEndTokens = new() { 4, 5, 6, 10 }; // . ! ? …
        static readonly HashSet<int> k_ClauseEndTokens = new() { 1, 2, 3, 9 };    // ; : , —
        const int k_SpaceToken = 16;

        /// <summary>
        /// Synthesizes speech for any number of tokens. Long input is split into chunks at sentence
        /// (then clause, then word) boundaries; each chunk is generated separately and the audio concatenated.
        /// </summary>
        public async Task<Tensor<float>> Execute(int[] inputIds, float speed, Voice voice)
        {
            var maxChunkTokens = Math.Clamp((int)(k_ChunkTokensAtSpeed1 * speed), 32, k_MaxModelTokens);
            var chunks = SplitIntoChunks(inputIds, maxChunkTokens);

            // Nothing to split (empty/whitespace-only input): run once, as before, so callers still get audio.
            if (chunks.Count == 0)
                return await ExecuteSingle(inputIds, speed, voice);

            var audio = new List<float>();
            foreach (var chunk in chunks)
                await GenerateChunk(chunk, speed, voice, audio);

            return new Tensor<float>(new TensorShape(1, audio.Count), audio.ToArray());
        }

        async Task GenerateChunk(int[] tokens, float speed, Voice voice, List<float> audio)
        {
            using var output = await ExecuteSingle(tokens, speed, voice);

            // Slow voices/speeds can still exceed the reliable output length: split further and retry.
            if (output.count > k_MaxOutputSamples && tokens.Length > 1)
            {
                foreach (var half in SplitIntoChunks(tokens, (tokens.Length + 1) / 2))
                    await GenerateChunk(half, speed, voice, audio);
                return;
            }

            audio.AddRange(output.DownloadToArray());
        }

        async Task<Tensor<float>> ExecuteSingle(int[] inputIds, float speed, Voice voice)
        {
            // Add the pad ids
            var paddedInputIds = new int[inputIds.Length + 2];
            paddedInputIds[0] = 0;
            Array.Copy(inputIds, 0, paddedInputIds, 1, inputIds.Length);
            paddedInputIds[^1] = 0;

            using var inputIdsTensor = new Tensor<int>(new TensorShape(1, paddedInputIds.Length), paddedInputIds);
            using var speedTensor = new Tensor<float>(new TensorShape(1), new[] { speed });
            using var voiceTensor = GetVoiceVector(inputIds.Length, voice.Tensor);

            return await Execute(inputIdsTensor, voiceTensor, speedTensor);
        }

        /// <summary>
        /// Greedily packs tokens into chunks of at most <paramref name="maxTokens"/>, ending each chunk at the
        /// last sentence end that fits, else the last clause break, else the last space, else a hard cut.
        /// </summary>
        public static List<int[]> SplitIntoChunks(int[] tokens, int maxTokens)
        {
            var chunks = new List<int[]>();
            var start = 0;

            while (start < tokens.Length)
            {
                while (start < tokens.Length && tokens[start] == k_SpaceToken)
                    start++;
                if (start >= tokens.Length)
                    break;

                var end = tokens.Length - start <= maxTokens ? tokens.Length : FindSplit(tokens, start, start + maxTokens);
                chunks.Add(tokens[start..end]);
                start = end;
            }

            return chunks;
        }

        // Returns the exclusive end index of the chunk beginning at start, no later than limit.
        static int FindSplit(int[] tokens, int start, int limit)
        {
            for (var i = limit - 1; i > start; i--)
                if (k_SentenceEndTokens.Contains(tokens[i])) return i + 1;
            for (var i = limit - 1; i > start; i--)
                if (k_ClauseEndTokens.Contains(tokens[i])) return i + 1;
            for (var i = limit - 1; i > start; i--)
                if (tokens[i] == k_SpaceToken) return i;
            return limit;
        }

        public async Task<Tensor<float>> Execute(Tensor<int> inputIdsTensor, Tensor<float> voiceTensor, Tensor<float> speedTensor)
        {
            LoadModelIfMissing();

            m_Worker.Schedule(inputIdsTensor, voiceTensor, speedTensor);
            using var result = m_Worker.PeekOutput() as Tensor<float>;
            using var output = await result.ReadbackAndCloneAsync();

            var processedOutput = KokoroOutputProcessor.Apply2NotchFiltering(output);
            return processedOutput;
        }

        public static List<Voice> GetVoices()
        {
            var voices = new List<Voice>();
            var voicesList = VoicesUtils.GetVoicesList();

            foreach (var file in voicesList)
            {
                var voiceAsset = Resources.Load<RawBytesAsset>(Path.Join(k_VoicesFolderPath, file));

                if (voiceAsset == null)
                    continue;

                var voiceData = voiceAsset.bytes;

                var voiceArray = new float[voiceData.Length / sizeof(float)];
                Buffer.BlockCopy(voiceData, 0, voiceArray, 0, voiceData.Length);

                var styleShape = new TensorShape(voiceArray.Length / 256, 1, 256);
                var tensor = new Tensor<float>(styleShape, voiceArray);
                var voice = new Voice(file, tensor);
                voices.Add(voice);
            }

            return voices;
        }

        /// <summary>
        /// Selects the style vector for an utterance. Voice packs hold one row per token count; the row is
        /// indexed by the unpadded token count (as in the reference kokoro-js implementation), clamped to the pack.
        /// </summary>
        static Tensor<float> GetVoiceVector(int tokenCount, Tensor<float> voice)
        {
            var rows = voice.shape[0];
            var index = Math.Clamp(tokenCount, 0, rows - 1);

            var data = voice.DownloadToArray();
            var style = new float[k_StyleDim];
            Array.Copy(data, index * k_StyleDim, style, 0, k_StyleDim);
            return new Tensor<float>(new TensorShape(1, k_StyleDim), style);
        }

        public void Dispose()
        {
            m_Worker?.Dispose();
            m_Worker = null;
        }

        public class Voice: IDisposable
        {
            public string Name;
            public Tensor<float> Tensor;

            public Voice(string name, Tensor<float> data)
            {
                Name = name;
                Tensor = data;
            }
            public void Dispose()
            {
                Tensor?.Dispose();
            }
        }
    }
}
