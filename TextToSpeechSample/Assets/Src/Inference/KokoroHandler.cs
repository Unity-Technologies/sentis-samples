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

        public async Task<Tensor<float>> Execute(int[] inputIds, float speed, Voice voice)
        {
            // Add the pad ids
            var paddedInputIds = new int[inputIds.Length + 2];
            paddedInputIds[0] = 0;
            Array.Copy(inputIds, 0, paddedInputIds, 1, inputIds.Length);
            paddedInputIds[^1] = 0;

            using var inputIdsTensor = new Tensor<int>(new TensorShape(1, paddedInputIds.Length), paddedInputIds);
            using var speedTensor = new Tensor<float>(new TensorShape(1), new[] { speed });
            using var voiceTensor = await GetVoiceVector(inputIdsTensor, voice.Tensor);

            return await Execute(inputIdsTensor, voiceTensor, speedTensor);
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

        async Task<Tensor<float>> GetVoiceVector(Tensor<int> inputIds, Tensor<float> voice)
        {
            var graph = new FunctionalGraph();
            var tokenInput = graph.AddInput<float>(voice.shape, "voice");
            var output = tokenInput[inputIds.count];
            graph.AddOutput(output, "output");
            var model = graph.Compile();

            using var worker = new Worker(model, m_BackendType);
            worker.Schedule(voice);
            using var result = worker.PeekOutput() as Tensor<float>;
            return await result.ReadbackAndCloneAsync();
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
