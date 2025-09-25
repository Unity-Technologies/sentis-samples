using System;
using Unity.InferenceEngine.Samples.TTS.Handlers;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Unity.InferenceEngine.Samples.TTS.State
{
    public record AppState: IDisposable
    {
        public string InputText = null;
        public KokoroHandler KokoroHandler;
        public KokoroHandler.Voice Voice = null;
        public float Speed = 1.0f;
        public Tensor<float> Waveform = null;
        public AudioClip AudioClip = null;
        public GenerationStatus Status = GenerationStatus.Idle;
        public string Error = null;

        public void CleanGenerationData()
        {
            Waveform?.Dispose();
            Waveform = null;
            Error = null;

            if (AudioClip != null)
            {
                if (Application.isPlaying)
                {
                    Object.Destroy(AudioClip);
                }
                else
                {
                    Object.DestroyImmediate(AudioClip);
                }
            }
        }

        public enum GenerationStatus
        {
            Idle,
            Pending,
            Completed,
            Failed
        }

        public void Dispose()
        {
            KokoroHandler?.Dispose();
            Voice?.Dispose();
            Waveform?.Dispose();
        }
    }
}
