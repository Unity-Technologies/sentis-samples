using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using Newtonsoft.Json.Linq;
using Unity.InferenceEngine.Tokenization;
using Unity.InferenceEngine.Tokenization.Decoders;
using Unity.InferenceEngine.Tokenization.Mappers;
using Unity.InferenceEngine.Tokenization.Padding;
using Unity.InferenceEngine.Tokenization.PostProcessors;
using Unity.InferenceEngine.Tokenization.PreTokenizers;

using UnityEngine;

namespace Unity.InferenceEngine.Samples.Chat
{
    sealed class LlavaTokenizer : IDisposable
    {
        Tokenizer m_Tokenizer;

        public LlavaTokenizer()
        {
            var config = GetTokenizerConfig();

            var vocabulary = BuildVocabulary(config);
            var addedTokens = GetAddedTokens(config);
            var merges = GetMerges(config);

            const string k_EotValue = "<|endoftext|>";
            vocabulary.TryGetValue(k_EotValue, out var eotId);

            m_Tokenizer = new Tokenizer(
                new BpeMapper(vocabulary, merges: merges),
                preTokenizer: new SequencePreTokenizer(
                    new RegexSplitPreTokenizer(
                        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                        SplitDelimiterBehavior.Isolated), new ByteLevelPreTokenizer(false, false)),
                postProcessor: new ByteLevelPostProcessor(),
                paddingProcessor: new RightPadding(new BatchLongestSizeProvider(),new(eotId, k_EotValue)),
                decoder: new ByteLevelDecoder(), addedVocabulary: addedTokens);
        }

        static Dictionary<string, int> BuildVocabulary(JObject config)
        {
            var output = new Dictionary<string, int>();
            var vocab = config["model"]["vocab"] as JObject;

            foreach (var (value, id) in vocab)
                output[value] = id?.Value<int>() ?? throw new DataException($"No id for value {value}");
            return output;
        }

        static IEnumerable<TokenConfiguration> GetAddedTokens(JObject config)
        {
            var addedTokens = config["added_tokens"] as JArray;
            foreach (var addedToken in addedTokens)
            {
                var id = addedToken["id"].Value<int>();
                var value = addedToken["content"].Value<string>();
                var wholeWord = addedToken["single_word"].Value<bool>();
                var strip = (addedToken["lstrip"].Value<bool>() ? Direction.Left : Direction.None) |
                    (addedToken["rstrip"].Value<bool>() ? Direction.Right : Direction.None);
                var normalized = addedToken["normalized"].Value<bool>();
                var special = addedToken["special"].Value<bool>();

                yield return new(id, value, wholeWord, strip, normalized, special);
            }
        }

        static MergePair[] GetMerges(JObject config)
        {
            var model = config["model"] as JObject;
            var modelMerges =  model["merges"] as JArray;

            var test = modelMerges[0];

            if (test.Type == JTokenType.String)
            {
                return modelMerges
                    .Select(t => t.Value<string>())
                    .Select(s => s.Split(" "))
                    .Select(sa => new MergePair(sa[0], sa[1]))
                    .ToArray();
            }

            if(test.Type == JTokenType.Array)
            {
                return modelMerges
                    .Select(t => t as JArray)
                    .Select(sa => new MergePair(sa[0].Value<string>(), sa[1].Value<string>()))
                    .ToArray();
            }

            throw new DataException($"Unexpected type {test.Type}");
        }


        static JObject GetTokenizerConfig()
        {
            var asset = Resources.Load<TextAsset>(LlavaConfig.TokenizerConfigPath);
            return JObject.Parse(asset.text);
        }

        public IEncoding Encode(string text)
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
            m_Tokenizer = null;
        }
    }
}
