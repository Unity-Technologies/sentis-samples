using System;
using System.Linq;
using Unity.ML.Tokenization;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class LlavaDecoder : ModelScheduler<LlavaDecoder.Input, LlavaDecoder.Output>, IDisposable
    {
        protected override string ModelPath => LlavaConfig.DecoderModelPath;
        Worker m_Worker;
        public LlavaDecoder(LlavaConfig config, bool lazyInit = false)
            : base(config, lazyInit) { }

        public override Output Schedule(Input input)
        {
            m_Worker ??= new Worker(m_Model, m_Config.BackendType);
            m_Worker.SetInput("attention_mask", input.attentionMask);
            m_Worker.SetInput("position_ids", input.positionIds);
            m_Worker.SetInput("inputs_embeds", input.inputsEmbeddings);

            for (int i = 0; i < input.pastKeys.Length; i++)
            {
                m_Worker.SetInput($"past_key_values.{i}.key", input.pastKeys[i]);
                m_Worker.SetInput($"past_key_values.{i}.value", input.pastKeysValues[i]);
            }

            m_Worker.Schedule();

            var logitsTensor = m_Worker.PeekOutput("logits") as Tensor<float>;
            var presentKeys = new Tensor[input.pastKeys.Length];
            var presentValues = new Tensor[input.pastKeysValues.Length];

            for (var i = 0; i < input.pastKeys.Length; i++)
            {
                m_Worker.CopyOutput($"present.{i}.key", ref presentKeys[i]);
                m_Worker.CopyOutput($"present.{i}.value", ref presentValues[i]);
            }

            return new Output(logitsTensor, presentKeys, presentValues);
        }

        public int InterpretLogits(Tensor<float> logits)
        {
            var logitsInterpreter = BuildLogitsGraph();
            using var worker = new Worker(logitsInterpreter, BackendType.GPUCompute);
            worker.Schedule(logits);
            using var outputTensor = worker.PeekOutput() as Tensor<int>;
            using var cpuOutput = outputTensor.ReadbackAndClone();
            var array = cpuOutput.DownloadToArray();

            return array[^1];
        }

        Model BuildLogitsGraph()
        {
            var graph = new FunctionalGraph();
            var inputShape = new DynamicTensorShape(-1, -1, 152000);
            var logitsInput = graph.AddInput<float>(inputShape, "logits");

            var output = Functional.ArgMax(logitsInput, 2);
            graph.AddOutput(output, "output");

            return graph.Compile();
        }

        public override void Dispose()
        {
            base.Dispose();
            m_Worker?.Dispose();
            m_Worker = null;
        }

        public sealed class Input : IDisposable
        {
            const int k_MaxPastKeys = 24;
            public Tensor<int> attentionMask { get; set; }
            public Tensor<int> positionIds { get; set; }
            public Tensor<float> inputsEmbeddings { get; set; }
            public Tensor[] pastKeys { get; set; }
            public Tensor[] pastKeysValues { get; set; }

            public Input(Tensor<int> attentionMask, Tensor<int> positionIds, Tensor<float> inputsEmbeddings, Tensor<float>[] pastKeys, Tensor<float>[] pastKeysValues)
            {
                this.attentionMask = attentionMask;
                this.positionIds = positionIds;
                this.inputsEmbeddings = inputsEmbeddings;
                this.pastKeys = pastKeys;
                this.pastKeysValues = pastKeysValues;
            }

            public Input(Encoding encoding, Tensor<float> textEmbeddings, Tensor<float> imageEmbeddings)
            {
                var inputsEmbeddings = InsertImageEmbeddingInTextEmbedding(encoding, textEmbeddings, imageEmbeddings);
                Initialize(inputsEmbeddings);
            }

            public Input(Tensor<float> inputsEmbeddings)
            {
                Initialize(inputsEmbeddings);
            }

            void Initialize(Tensor<float> inputsEmbeddings)
            {
                this.inputsEmbeddings = inputsEmbeddings;
                var sequenceLength = this.inputsEmbeddings.shape[1];

                var attentionMaskArray = Enumerable.Repeat(1, sequenceLength).ToArray();
                attentionMask = new Tensor<int>(new TensorShape(1, sequenceLength), attentionMaskArray);

                var positionIdsArray = Enumerable.Range(0, sequenceLength).ToArray();
                positionIds = new Tensor<int>(new TensorShape(1, sequenceLength), positionIdsArray);

                pastKeys = new Tensor<float>[k_MaxPastKeys];
                pastKeysValues = new Tensor<float>[k_MaxPastKeys];

                var batchSize = 1;
                var numHeads = 2;
                var headDim = 64;

                for (var i = 0; i < k_MaxPastKeys; i++)
                {
                    pastKeys[i] = new Tensor<float>(new TensorShape(batchSize, numHeads, 0, headDim));
                    pastKeysValues[i] = new Tensor<float>(new TensorShape(batchSize, numHeads, 0, headDim));
                }
            }

            Tensor<float> InsertImageEmbeddingInTextEmbedding(Encoding encoding, Tensor<float> textEmbeddings, Tensor<float> imageEmbeddings)
            {
                var imagePosition = -1;
                var encodingArray = encoding.Ids.ToArray();
                for (var i = 0; i < encoding.Length; ++i)
                {
                    if (encodingArray[i] == LlavaConfig.TokenIdImage)
                    {
                        imagePosition = i;
                        break;
                    }
                }

                Debug.Assert(imagePosition != -1, "Could not find image token in text embedding.");

                var graph = new FunctionalGraph();
                var inputShape = new DynamicTensorShape(-1, -1, 896);
                var textInput = graph.AddInput<float>(inputShape, "text_embeddings");
                var imageInput = graph.AddInput<float>(inputShape, "image_embedding");

                var firstPart = textInput[.., ..imagePosition];
                var secondPart = textInput[.., (imagePosition + 1)..];
                var concat = Functional.Concat(new[] { firstPart, imageInput, secondPart }, 1);

                graph.AddOutput(concat, "output");
                var model = graph.Compile();
                using var worker = new Worker(model, BackendType.GPUCompute);
                worker.Schedule(textEmbeddings, imageEmbeddings);

                using var outputTensor = worker.PeekOutput() as Tensor<float>;
                var cpuOutput = outputTensor.ReadbackAndClone();

                return cpuOutput;
            }

            public void Dispose()
            {
                attentionMask?.Dispose();
                positionIds?.Dispose();
                inputsEmbeddings?.Dispose();
                foreach (var key in pastKeys)
                {
                    key?.Dispose();
                }
                foreach (var value in pastKeysValues)
                {
                    value?.Dispose();
                }
            }
        }

        public sealed record Output(Tensor<float> logits, Tensor[] presentKeys, Tensor[] presentValues) : IDisposable
        {
            public void Dispose()
            {
                logits?.Dispose();
                foreach (var key in presentKeys)
                {
                    key?.Dispose();
                }
                foreach (var value in presentValues)
                {
                    value?.Dispose();
                }
            }
        }
    }
}
