using System;
using System.IO;
using System.Text;

namespace Unity.InferenceEngine.Samples.TTS.Utils
{
    public static class WavUtils
    {
        public static void WriteFloatWav(string path, float[] samples, int sampleRate = 24000)
        {
            if (samples == null) throw new ArgumentNullException(nameof(samples));

            var numChannels = 1;
            var bitsPerSample = 32;
            var blockAlign = numChannels * (bitsPerSample / 8);
            var byteRate = sampleRate * blockAlign;

            // Convert float[] to little-endian bytes without scaling
            var dataBytes = new byte[samples.Length * sizeof(float)];
            Buffer.BlockCopy(samples, 0, dataBytes, 0, dataBytes.Length);

            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

            // RIFF header
            bw.Write(Encoding.ASCII.GetBytes("RIFF"));
            bw.Write(36 + dataBytes.Length);         // ChunkSize
            bw.Write(Encoding.ASCII.GetBytes("WAVE"));

            // fmt chunk
            bw.Write(Encoding.ASCII.GetBytes("fmt "));
            bw.Write(16);                            // Subchunk1Size (PCM)
            bw.Write((short)3);                      // AudioFormat = 3 (IEEE float)
            bw.Write((short)numChannels);            // NumChannels
            bw.Write(sampleRate);                    // SampleRate
            bw.Write(byteRate);                      // ByteRate
            bw.Write((short)blockAlign);             // BlockAlign
            bw.Write((short)bitsPerSample);          // BitsPerSample

            // data chunk
            bw.Write(Encoding.ASCII.GetBytes("data"));
            bw.Write(dataBytes.Length);
            bw.Write(dataBytes);
        }

    }
}
