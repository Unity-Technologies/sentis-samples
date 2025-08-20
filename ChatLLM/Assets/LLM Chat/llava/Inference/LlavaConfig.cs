using System;

namespace Unity.InferenceEngine.Samples.Chat
{
    class LlavaConfig
    {
        public const string DecoderModelPath = "Packages/com.unity.ai.inference/Samples/LLM Chat/llava/Models/decoder_model_merged.onnx";
        public const string VisionEncoderModelPath = "Packages/com.unity.ai.inference/Samples/LLM Chat/llava/Models/vision_encoder.onnx";
        public const string EmbeddingModelPath = "Packages/com.unity.ai.inference/Samples/LLM Chat/llava/Models/embed_tokens.onnx";
        public const string TokenizerConfigPath = "Packages/com.unity.ai.inference/Samples/LLM Chat/llava/Models/tokenizer.json";
        public const int TokenIdEndOfText = 151645;
        public const int TokenIdImage = 151646;

        public BackendType BackendType { get; private set; } = BackendType.GPUCompute;
        public LlavaTokenizer Tokenizer { get; private set; }

        public LlavaConfig(BackendType backendType)
        {
            BackendType = backendType;
            Tokenizer = new LlavaTokenizer();
        }

        public static string ApplyChatTemplate(string userPrompt)
        {
            return $"<|im_start|>user <image>\n{userPrompt}<|im_end|><|im_start|>assistant\n";
        }
    }
}
