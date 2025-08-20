using System;
using UnityEditor;

namespace Unity.InferenceEngine.Samples.Chat
{
    abstract class ModelScheduler<TInput, TOutput>
    {
        protected LlavaConfig m_Config;
        protected Model m_Model;
        protected abstract string ModelPath { get; }

        protected ModelScheduler(LlavaConfig config, bool lazyInit = false)
        {
            m_Config = config;

            if (!lazyInit)
            {
                LoadModelIfMissing();
            }
        }

        protected void LoadModelIfMissing()
        {
            if (m_Model != null) return;

            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);

            if (modelAsset == null)
            {
                throw new Exception($"Model asset not found at {ModelPath}");
            }

            m_Model = ModelLoader.Load(modelAsset);

            if (m_Model == null)
            {
                throw new Exception($"Failed to load model from {ModelPath}");
            }
        }

        public abstract TOutput Schedule(TInput input);

        public virtual void Dispose()
        {
            m_Model = null;
            m_Config = null;
        }
    }
}
