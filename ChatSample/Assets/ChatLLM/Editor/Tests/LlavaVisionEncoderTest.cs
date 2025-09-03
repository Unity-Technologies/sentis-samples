using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Unity.InferenceEngine.Samples.Chat.Editor
{
    class LlavaVisionEncoderTest
    {
        [Test]
        public async Task TestLlavaVisionEncoder()
        {
            var textures = new List<Texture2D>();
            for (int i = 0; i < 5; i++)
            {
                textures.Add(GetRandomTexture(224, 224));
            }
            using var input = new LlavaVisionEncoder.Input(textures);

            var config = new LlavaConfig(BackendType.GPUCompute);
            // Create an instance of LlavaVisionEncoder
            using var encoder = new LlavaVisionEncoder(config);

            // Execute the model
            using var outputs = encoder.Schedule(input);

            using var features = await outputs.features.ReadbackAndCloneAsync();

            // Assert that the output is not null
            Assert.IsNotNull(features);
            using var array = features.DownloadToNativeArray();
            Assert.IsTrue(array.Length > 0, "Output features should not be empty.");

            foreach (var texture in textures)
            {
                Object.DestroyImmediate(texture);
            }
        }

        public static Texture2D GetRandomTexture(int width, int height)
        {
            // Create a new texture with specified dimensions
            var randomTexture = new Texture2D(width, height);

            // Create an array to hold pixel colors
            Color[] pixels = new Color[width * height];

            // Fill the array with random colors
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = new Color(
                    UnityEngine.Random.Range(0f, 1f), // Red
                    UnityEngine.Random.Range(0f, 1f), // Green
                    UnityEngine.Random.Range(0f, 1f), // Blue
                    1f                    // Alpha (fully opaque)
                );
            }

            // Apply the pixel data to the texture
            randomTexture.SetPixels(pixels);
            randomTexture.Apply();

            return randomTexture;
        }
    }
}
