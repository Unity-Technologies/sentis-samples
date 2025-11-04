using System;
using System.Collections.Generic;

namespace Unity.InferenceEngine.Samples.TTS.Inference
{
    public static class KokoroOutputProcessor
    {
        // We post-process Kokoro’s output today because the model is extremely sensitive to the STFT method.
        // Until our STFT matches the training-time FFT behavior, the two biquad notches suppress the learned “FFT-specific” artifact
        // tones and restore acceptable audio.
        public static Tensor<float> Apply2NotchFiltering(Tensor<float> kokoroOutput)
        {
            var signal = kokoroOutput.DownloadToArray();
            double r = 24000;
            double[] freqs = { 4810, 9600 };

            var chain = CreateNotchChain(r, freqs, octaveWidth: 0.02, targetAttenuation: -40);

            var finalOutput = new float[signal.Length];
            chain.ProcessBuffer(signal, finalOutput);
            return new Tensor<float>(new TensorShape(1, finalOutput.Length), finalOutput);
        }

        // Create a chain for multiple notch frequencies; cascade stages per notch if deeper attenuation desired
        static BiquadChain CreateNotchChain(double R, double[] freqs, double octaveWidth = 0.02, double targetAttenuation = -30)
        {
            var list = new List<Biquad>();

            // -30 dB => 1 stage; -40 dB => 2 stages (wider, deeper rejection across band)
            var stagesPerNotch = (targetAttenuation <= -40) ? 2 : 1;

            foreach (var nf in freqs)
            {
                var notch = DesignNotch(R, nf, octaveWidth);
                for (var s = 0; s < stagesPerNotch; s++)
                    list.Add(notch);
            }

            return new BiquadChain(list);
        }

        // Design one notch biquad at nf with octave width and target rejection (~band coverage)
        static Biquad DesignNotch(double R, double nf, double octaveWidth)
        {
            var omega0 = 2.0 * Math.PI * nf / R;
            var cos0 = Math.Cos(omega0);

            // Convert octave width to approximate 3 dB bandwidth
            var bwHz = nf * (Math.Pow(2, octaveWidth / 2.0) - Math.Pow(2, -octaveWidth / 2.0));

            // Pole radius from bandwidth (narrow notch => r near 1)
            var r = 1.0 - (Math.PI * bwHz / R);
            if (r < 0.0) r = 0.0;      // clamp
            if (r > 0.999999) r = 0.999999; // avoid exact 1.0

            // Standard notch coefficients (zeros on unit circle, poles at radius r)
            var b0 = 1.0;
            var b1 = -2.0 * cos0;
            var b2 = 1.0;
            var a1 = -2.0 * r * cos0;
            var a2 = r * r;

            return new Biquad(b0, b1, b2, a1, a2);
        }



        struct Biquad
        {
            readonly double b0;
            readonly double b1;
            readonly double b2;
            readonly double a1;
            readonly double a2;
            double w1, w2; // DF-II state

            public Biquad(double b0, double b1, double b2, double a1, double a2)
            {
                this.b0 = b0; this.b1 = b1; this.b2 = b2;
                this.a1 = a1; this.a2 = a2;
                w1 = 0.0; w2 = 0.0;
            }

            public float ProcessSample(float x)
            {
                // Direct Form II Transposed (stable and cache-friendly)
                double w0 = x - a1 * w1 - a2 * w2;
                double y = b0 * w0 + b1 * w1 + b2 * w2;
                w2 = w1; w1 = w0;
                return (float)y;
            }

            public void Reset()
            {
                w1 = 0.0; w2 = 0.0;
            }
        }

        class BiquadChain
        {
            readonly Biquad[] stages;

            public BiquadChain(IEnumerable<Biquad> biquads)
            {
                stages = new List<Biquad>(biquads).ToArray();
            }

            // Per-sample chain processing (streaming-friendly)
            float ProcessSample(float x)
            {
                float y = x;
                for (int i = 0; i < stages.Length; i++)
                    y = stages[i].ProcessSample(y);
                return y;
            }

            // Single-pass buffer processing using the per-sample chain
            public void ProcessBuffer(float[] input, float[] output)
            {
                for (int n = 0; n < input.Length; n++)
                    output[n] = ProcessSample(input[n]);
            }
        }
    }
}
