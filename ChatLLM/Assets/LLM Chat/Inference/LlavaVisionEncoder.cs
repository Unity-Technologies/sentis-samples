using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class LlavaVisionEncoder : ModelScheduler<LlavaVisionEncoder.Input, LlavaVisionEncoder.Output>, IDisposable
    {
        protected override string ModelPath => LlavaConfig.VisionEncoderModelPath;
        Worker m_Worker;

        public LlavaVisionEncoder(LlavaConfig config, bool lazyInit = false)
            : base(config, lazyInit) { }

        public override Output Schedule(Input input)
        {
            LoadModelIfMissing();


            m_Worker ??= new Worker(m_Model, m_Config.BackendType);
            m_Worker.Schedule(input.features);

            var outputTensor = m_Worker.PeekOutput() as Tensor<float>;
            return new Output(outputTensor);
        }

        public class Input : IDisposable
        {
            public Texture2D image;
            public Tensor<float> features;
            readonly TensorShape m_InputShape = new(1, 3, 384, 384);
            public Input(List<Texture2D> images)
            {
                var batchShape = m_InputShape;

                features = new Tensor<float>(batchShape);
                TextureConverter.ToTensor(images[0], features);

                if (images.Count > 1)
                {
                    var concatModel = BuildLogitsGraph();
                    using var worker = new Worker(concatModel, BackendType.GPUCompute);

                    for (int i = 1; i < images.Count; ++i)
                    {
                        using var tempTensor = new Tensor<float>(m_InputShape);
                        TextureConverter.ToTensor(images[i], tempTensor);
                        worker.Schedule(features, tempTensor);
                        using var outputTensor = worker.PeekOutput() as Tensor<float>;

                        features.Dispose();
                        features = null;

                        features = outputTensor.ReadbackAndClone();
                    }
                }
            }

            Model BuildLogitsGraph()
            {
                var graph = new FunctionalGraph();
                var input0 = graph.AddInput<float>(new DynamicTensorShape(-1, 3, 384, 384), "input0");
                var input1 = graph.AddInput<float>(m_InputShape, "input1");

                var output = Functional.Concat(new[] { input0, input1 });
                graph.AddOutput(output, "output");

                return graph.Compile();
            }

            public void Dispose()
            {
                features?.Dispose();
            }
        }

        public sealed record Output(Tensor<float> features) : IDisposable
        {
            public void Dispose()
            {
                features?.Dispose();
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

namespace System.Runtime.CompilerServices
{
    static class IsExternalInit { }
}
