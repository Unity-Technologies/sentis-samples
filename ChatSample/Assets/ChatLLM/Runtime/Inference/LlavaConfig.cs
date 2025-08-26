using System;

namespace Unity.InferenceEngine.Samples.Chat
{
    class LlavaConfig
    {
        public const string ModelId = "llava-hf/llava-onevision-qwen2-0.5b-si-hf";
        public const string DownloadPath = "/ChatLLM/Resources/Models";

        public const string DecoderModelName = "decoder_model_merged";
        public const string DecoderModelFile =  DecoderModelName + ".onnx";
        public const string DecoderModelPath = "Models/onnx/" + DecoderModelName;

        public const string VisionEncoderModelName = "vision_encoder";
        public const string VisionEncoderModelFile = VisionEncoderModelName + ".onnx";
        public const string VisionEncoderModelPath = "Models/onnx/" + VisionEncoderModelName;

        public const string EmbeddingModelName = "embed_tokens";
        public const string EmbeddingModelFile = EmbeddingModelName + ".onnx";
        public const string EmbeddingModelPath = "Models/onnx/" + EmbeddingModelName;

        public const string TokenizerModelName = "tokenizer";
        public const string TokenizerModelFile = TokenizerModelName + ".json";
        public const string TokenizerConfigPath = "Models/" + TokenizerModelName;

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
