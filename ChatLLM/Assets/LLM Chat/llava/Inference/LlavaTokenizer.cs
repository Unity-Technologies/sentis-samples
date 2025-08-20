using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using Unity.ML.Tokenization;
using Unity.ML.Tokenization.Decoders;
using Unity.ML.Tokenization.Padding;
using Unity.ML.Tokenization.PostProcessors;
using Unity.ML.Tokenization.PreTokenizers;
using Unity.ML.Tokenization.Tokenizers;
using UnityEditor;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class LlavaTokenizer : IDisposable
    {
        IVocabulary m_Vocabulary;
        TokenizationPipeline m_Tokenizer;

        public LlavaTokenizer()
        {
            var config = GetTokenizerConfig();

            m_Vocabulary = BuildVocabulary(config);

            var merges = (config["model"]["merges"] as JArray)
                .Select(t => t.Value<string>())
                .Select(s => s.Split(" "))
                .Select(sa => (sa[0], sa[1]));

            m_Vocabulary.TryGetToken("<|endoftext|>", out var padToken);

            m_Tokenizer = new TokenizationPipeline(
                new BpeTokenizer(m_Vocabulary, merges: merges),
                preTokenizer: new SequencePreTokenizer(
                    new SplitPreTokenizer(
                        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                        SplitDelimiterBehavior.Isolated), new ByteLevelPreTokenizer(false, false)),
                postProcessor: new ByteLevelPostProcessor(),
                paddingProcessor: new RightPadding(new BatchLongestSizeProvider(), padToken),
                decoder: new ByteLevelDecoder(),
                vocabulary: m_Vocabulary);
        }

        static IVocabulary BuildVocabulary(JObject config)
        {
            var builder = new VocabularyBuilder();

            var addedTokens = config["added_tokens"] as JArray;
            foreach (var addedToken in addedTokens)
            {
                var id = addedToken["id"].Value<int>();
                var key = addedToken["content"].Value<string>();
                builder.Add(id, key, key, true);
            }

            var vocab = config["model"]["vocab"] as JObject;
            foreach (var (key, value) in vocab)
                if(!addedTokens.Contains(key))
                    builder.Add(value.Value<int>(), key, key);

            return builder.Build();
        }
        static JObject GetTokenizerConfig()
        {
            var asset = AssetDatabase.LoadAssetAtPath<TextAsset>(LlavaConfig.TokenizerConfigPath);
            return JObject.Parse(asset.text);
        }

        public Encoding Encode(string text)
        {
            var tokens = m_Tokenizer.Encode(text);
            return tokens;
        }

        public string Decode(int[] inputs)
        {
            return m_Tokenizer.Decode(inputs);
        }

        public void Dispose()
        {
            m_Vocabulary = null;
            m_Tokenizer = null;
        }
    }
}
