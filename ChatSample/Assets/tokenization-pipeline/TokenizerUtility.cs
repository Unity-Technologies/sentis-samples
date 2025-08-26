using System.Collections.Generic;
using Newtonsoft.Json;

namespace Unity.ML.Tokenization
{
    static class TokenizerUtility
    {
        /// <summary>
        ///     Load a JSON containing vocabulary and merge rules
        /// </summary>
        /// <param name="data">A JSON string containing the vocabulary and merge rules.</param>
        /// <returns></returns>
        public static TokenizerData Load(string data)
        {
            var result = new TokenizerData();
            var jsonData = JsonConvert.DeserializeObject<JsonTokenizerData>(data);

            // TODO: remove this once tokenizer update is complete.
            if (jsonData?.model.merges != null)
                foreach (var merge in jsonData.model.merges)
                {
                    var mergePair = merge.Split(" ");
                    result.Merges.Add((CleanString(mergePair[0]), CleanString(mergePair[1])));
                }

            foreach (var vocab in jsonData.model.vocab)
                result.Vocab.Add(CleanString(vocab.Key), vocab.Value);

            if (jsonData.added_tokens != null)
                foreach (var token in jsonData.added_tokens)
                    if (!token.normalized)
                        result.AddedTokens.Add(CleanString(token.content), token.id);

            return result;
        }

        /// <summary>
        ///     Load a JSON containing a Piper model's phoneme to id map.
        /// </summary>
        /// <param name="data">A JSON string containing the phoneme to id map.</param>
        /// <returns></returns>
        public static TokenizerData LoadPiperTokenizer(string data)
        {
            var result = new TokenizerData();
            var jsonData = JsonConvert.DeserializeObject<PiperTokenizerData>(data);

            foreach (var vocab in jsonData.phoneme_id_map)
                result.Vocab.Add(vocab.Key, vocab.Value[0]);

            return result;
        }

        static string CleanString(string text)
        {
            text = text.Replace('Ġ', ' ');
            text = text.Replace('Ċ', '\n');

            return text;
        }

        /// <summary>
        ///     Temporary storage based on HuggingFace model JSON data structure
        ///     Used for deserialization
        /// </summary>
        public class JsonTokenizerData
        {
            public List<AddedToken> added_tokens;
            public Model model;

            public class Model
            {
                public List<string> merges;
                public Dictionary<string, int> vocab;
            }

            public class AddedToken
            {
                public string content;
                public int id;
                public bool lstrip;
                public bool normalized;
                public bool rstrip;
                public bool single_word;
                public bool special;
            }
        }

        /// <summary>
        ///     Temporary storage based on Piper model JSON data structure. Used for deserialization.
        /// </summary>
        public class PiperTokenizerData
        {
            public Dictionary<string, int[]> phoneme_id_map;
        }

        /// <summary>
        ///     TokenizerData used to store Vocab, Merges, and AddedTokens for tokenization pipeline.
        /// </summary>
        public class TokenizerData
        {
            public Dictionary<string, int> AddedTokens = new();
            public List<(string, string)> Merges = new();
            public Dictionary<string, int> Vocab = new();
        }
    }
}
