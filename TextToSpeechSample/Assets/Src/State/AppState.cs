using System;
using Unity.InferenceEngine.Samples.TTS.Inference;
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
        public Tensor<float> Waveform;
        public AudioClip AudioClip = null;
        public GenerationStatus Status = GenerationStatus.Idle;
        public string Error;

        public const string ModelId = "onnx-community/Kokoro-82M-v1.0-ONNX";
        public const string OnnxPath = "onnx/model.onnx";
        public const string VoicePath = "voices";
        public const string DownloadPath = "Resources";

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
