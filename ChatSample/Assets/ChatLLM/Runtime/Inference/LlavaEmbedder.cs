using System;
using System.Collections.Generic;
using Unity.InferenceEngine.Tokenization;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class LlavaEmbedder : ModelScheduler<LlavaEmbedder.Input, LlavaEmbedder.Output>, IDisposable
    {
        protected override string ModelPath => LlavaConfig.EmbeddingModelPath;
        Worker m_Worker;

        public LlavaEmbedder(LlavaConfig config, bool lazyInit = false)
            : base(config, lazyInit)
        {
        }

        public override Output Schedule(Input input)
        {
            LoadModelIfMissing();

            var inputEncoding = m_Config.Tokenizer.Encode(input.text);
            var inputIds = new List<int>();
            inputEncoding.GetIds(inputIds);
            var inputTensor = new Tensor<int>(new TensorShape(1, inputEncoding.Length), inputIds.ToArray());

            m_Worker ??= new Worker(m_Model, m_Config.BackendType);
            m_Worker.Schedule(inputTensor);

            var outputTensor = m_Worker.PeekOutput() as Tensor<float>;

            inputTensor.Dispose();
            return new Output(inputEncoding, outputTensor);
        }

        public Tensor<float> Schedule(int[] inputIds)
        {
            LoadModelIfMissing();

            var inputTensor = new Tensor<int>(new TensorShape(1, inputIds.Length), inputIds);
            m_Worker ??= new Worker(m_Model, m_Config.BackendType);
            m_Worker.Schedule(inputTensor);

            var outputTensor = m_Worker.PeekOutput() as Tensor<float>;
            inputTensor.Dispose();
            return outputTensor;
        }

        public record Input(string text);

        public sealed record Output(IEncoding encoding, Tensor<float> embedding) : IDisposable
        {
            public void Dispose()
            {
                embedding?.Dispose();
            }
        }

        public override void Dispose()
        {
            base.Dispose();
            m_Worker?.Dispose();
            m_Worker = null;
        }
    }
}
