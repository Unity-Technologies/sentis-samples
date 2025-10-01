using System;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Utils
{
    public static class AudioClipUtils
    {
        public static AudioClip ToAudioClip(Tensor<float> waveform, int sampleRate = 24000, int channels = 1, string name = "AudioClip")
        {
            var audioClip = AudioClip.Create(name, waveform.count / channels, channels, sampleRate, false);
            audioClip.SetData(waveform.DownloadToArray(), 0);
            return audioClip;
        }
    }
}
