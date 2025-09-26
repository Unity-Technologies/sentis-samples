using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Globalization;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Inference
{
    public class MToken
    {
        public string Text { get; set; } = "";
        public string Tag { get; set; } = "";
        public string Whitespace { get; set; } = "";
        public string Phonemes { get; set; }
        public float StartTs { get; set; }
        public float EndTs { get; set; }
        public int Rating { get; set; }
        public TokenUnderscore _ { get; set; } = new TokenUnderscore();

        public class TokenUnderscore
        {
            public bool IsHead { get; set; } = true;
            public string Alias { get; set; }
            public float? Stress { get; set; }
            public string Currency { get; set; }
            public string NumFlags { get; set; } = "";
            public bool Prespace { get; set; }
            public int? Rating { get; set; }
        }
    }

    public class TokenContext
    {
        public bool? FutureVowel { get; set; }
        public bool FutureTo { get; set; }
    }

    public static class MisakiSharp
    {
        // Character sets
        private static readonly HashSet<char> DIPHTHONGS = new HashSet<char>("AIOQWYʤʧ".ToCharArray());
        private static readonly HashSet<char> CONSONANTS = new HashSet<char>("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ".ToCharArray());
        private static readonly HashSet<char> VOWELS = new HashSet<char>("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ".ToCharArray());
        private static readonly HashSet<char> PUNCTS = new HashSet<char>(";:,.!?—…\"\"\"".ToCharArray());
        private static readonly HashSet<char> NON_QUOTE_PUNCTS = new HashSet<char>(PUNCTS.Where(p => p != '\"' && p != '\"' && p != '\"').ToArray());
        private static readonly HashSet<string> PUNCT_TAGS = new HashSet<string> { ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP" };

        // Stress markers
        private const char PRIMARY_STRESS = 'ˈ';
        private const char SECONDARY_STRESS = 'ˌ';
        private static readonly string STRESSES = new string(new[] { SECONDARY_STRESS, PRIMARY_STRESS });

        // Common word dictionaries (simplified subset)
        private static readonly Dictionary<string, string> COMMON_WORDS = new Dictionary<string, string>
        {
            { "the", "ðə" }, { "The", "ðə" },
            { "a", "ɐ" }, { "A", "ɐ" },
            { "an", "ɐn" }, { "An", "ɐn" },
            { "to", "tə" }, { "To", "tə" },
            { "in", "ɪn" }, { "In", "ɪn" },
            { "of", "əv" }, { "Of", "əv" },
            { "and", "ænd" }, { "And", "ænd" },
            { "is", "ɪz" }, { "Is", "ɪz" },
            { "it", "ɪt" }, { "It", "ɪt" },
            { "you", "ju" }, { "You", "ju" },
            { "that", "ðæt" }, { "That", "ðæt" },
            { "he", "hi" }, { "He", "hi" },
            { "was", "wəz" }, { "Was", "wəz" },
            { "for", "fɔɹ" }, { "For", "fɔɹ" },
            { "are", "ɑɹ" }, { "Are", "ɑɹ" },
            { "with", "wɪθ" }, { "With", "wɪθ" },
            { "his", "hɪz" }, { "His", "hɪz" },
            { "they", "ðeɪ" }, { "They", "ðeɪ" },
            { "at", "æt" }, { "At", "æt" },
            { "be", "bi" }, { "Be", "bi" },
            { "this", "ðɪs" }, { "This", "ðɪs" },
            { "have", "hæv" }, { "Have", "hæv" },
            { "from", "fɹəm" }, { "From", "fɹəm" },
            { "or", "ɔɹ" }, { "Or", "ɔɹ" },
            { "one", "wʌn" }, { "One", "wʌn" },
            { "had", "hæd" }, { "Had", "hæd" },
            { "by", "baɪ" }, { "By", "baɪ" },
            { "not", "nɑt" }, { "Not", "nɑt" },
            { "what", "wʌt" }, { "What", "wʌt" },
            { "all", "ɔl" }, { "All", "ɔl" },
            { "were", "wɜɹ" }, { "Were", "wɜɹ" },
            { "we", "wi" }, { "We", "wi" },
            { "when", "wen" }, { "When", "wen" },
            { "your", "jʊɹ" }, { "Your", "jʊɹ" },
            { "can", "kæn" }, { "Can", "kæn" },
            { "said", "sed" }, { "Said", "sed" },
            { "there", "ðeɹ" }, { "There", "ðeɹ" },
            { "each", "itʃ" }, { "Each", "itʃ" },
            { "which", "wɪtʃ" }, { "Which", "wɪtʃ" },
            { "do", "du" }, { "Do", "du" },
            { "how", "haʊ" }, { "How", "haʊ" },
            { "their", "ðeɹ" }, { "Their", "ðeɹ" },
            { "if", "ɪf" }, { "If", "ɪf" },
            { "will", "wɪl" }, { "Will", "wɪl" },
            { "up", "ʌp" }, { "Up", "ʌp" },
            { "other", "ʌðɜɹ" }, { "Other", "ʌðɜɹ" },
            { "about", "əbaʊt" }, { "About", "əbaʊt" },
            { "out", "aʊt" }, { "Out", "aʊt" },
            { "many", "mɛni" }, { "Many", "mɛni" },
            { "then", "ðɛn" }, { "Then", "ðɛn" },
            { "them", "ðɛm" }, { "Them", "ðɛm" },
            { "these", "ðiz" }, { "These", "ðiz" },
            { "so", "soʊ" }, { "So", "soʊ" },
            { "some", "sʌm" }, { "Some", "sʌm" },
            { "her", "hɜɹ" }, { "Her", "hɜɹ" },
            { "would", "wʊd" }, { "Would", "wʊd" },
            { "make", "meɪk" }, { "Make", "meɪk" },
            { "like", "laɪk" }, { "Like", "laɊk" },
            { "into", "ɪntu" }, { "Into", "ɪntu" },
            { "him", "hɪm" }, { "Him", "hɪm" },
            { "has", "hæz" }, { "Has", "hæz" },
            { "two", "tu" }, { "Two", "tu" },
            { "more", "mɔɹ" }, { "More", "mɔɹ" },
            { "go", "goʊ" }, { "Go", "goʊ" },
            { "no", "noʊ" }, { "No", "noʊ" },
            { "way", "weɪ" }, { "Way", "weɪ" },
            { "could", "kʊd" }, { "Could", "kʊd" },
            { "my", "maɪ" }, { "My", "maɪ" },
            { "than", "ðæn" }, { "Than", "ðæn" },
            { "first", "fɜɹst" }, { "First", "fɜɹst" },
            { "been", "bɪn" }, { "Been", "bɪn" },
            { "call", "kɔl" }, { "Call", "kɔl" },
            { "who", "hu" }, { "Who", "hu" },
            { "its", "ɪts" }, { "Its", "ɪts" },
            { "now", "naʊ" }, { "Now", "naʊ" },
            { "find", "faɪnd" }, { "Find", "faɪnd" },
            { "long", "lɔŋ" }, { "Long", "lɔŋ" },
            { "down", "daʊn" }, { "Down", "daʊn" },
            { "day", "deɪ" }, { "Day", "deɪ" },
            { "did", "dɪd" }, { "Did", "dɪd" },
            { "get", "ɡɛt" }, { "Get", "ɡɛt" },
            { "come", "kʌm" }, { "Come", "kʌm" },
            { "made", "meɪd" }, { "Made", "meɪd" },
            { "may", "meɪ" }, { "May", "meɪ" },
            { "part", "pɑɹt" }, { "Part", "pɑɹt" }
        };

        private static readonly Dictionary<string, string> SYMBOLS = new Dictionary<string, string>
        {
            { "%", "percent" },
            { "&", "and" },
            { "+", "plus" },
            { "@", "at" }
        };

        public static int[] TokenizeGraphemes(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return new int[0];

            try
            {
                var phonemes = TextToPhonemes(inputText);
                return PhonemesHandler.Tokenize(phonemes);
            }
            catch (Exception ex)
            {
                Debug.LogError($"MisakiSharp tokenization failed: {ex.Message}");
                // Fallback to original implementation
                return PhonemesHandler.TokenizeGraphemes(inputText);
            }
        }

        public static string TextToPhonemes(string text)
        {
            if (string.IsNullOrEmpty(text))
                return string.Empty;

            var tokens = SimpleTokenize(text);
            var result = new List<string>();

            var context = new TokenContext();

            foreach (var token in tokens)
            {
                var phonemes = GetWordPhonemes(token, context);
                if (!string.IsNullOrEmpty(phonemes))
                {
                    result.Add(phonemes);
                    UpdateContext(context, phonemes, token);
                }
            }

            return string.Join(" ", result);
        }

        private static List<MToken> SimpleTokenize(string text)
        {
            var tokens = new List<MToken>();
            var words = Regex.Split(text, @"(\s+|[.,!?;:])")
                .Where(w => !string.IsNullOrEmpty(w))
                .ToArray();

            for (int i = 0; i < words.Length; i++)
            {
                var word = words[i];
                var token = new MToken
                {
                    Text = word.Trim(),
                    Whitespace = (i < words.Length - 1 && char.IsWhiteSpace(word.LastOrDefault())) ? " " : "",
                    Tag = GetSimpleTag(word)
                };

                if (!string.IsNullOrWhiteSpace(token.Text))
                {
                    tokens.Add(token);
                }
            }

            return tokens;
        }

        private static string GetSimpleTag(string word)
        {
            if (string.IsNullOrWhiteSpace(word))
                return "";

            word = word.Trim();

            if (PUNCT_TAGS.Contains(word) || PUNCTS.Contains(word.FirstOrDefault()))
                return "PUNCT";

            if (Regex.IsMatch(word, @"^\d+$"))
                return "CD";

            if (char.IsUpper(word.FirstOrDefault()) && word.Length > 1)
                return "NNP";

            return "NN";
        }

        private static string GetWordPhonemes(MToken token, TokenContext context)
        {
            var word = token.Text;

            if (string.IsNullOrEmpty(word))
                return "";

            // Handle punctuation
            if (PUNCTS.Contains(word.FirstOrDefault()))
            {
                return GetPunctuationPhonemes(word);
            }

            // Handle symbols
            if (SYMBOLS.ContainsKey(word))
            {
                return GetWordPhonemes(new MToken { Text = SYMBOLS[word], Tag = "NN" }, context);
            }

            // Handle numbers
            if (Regex.IsMatch(word, @"^\d+$"))
            {
                return GetNumberPhonemes(word);
            }

            // Handle common words
            if (COMMON_WORDS.ContainsKey(word))
            {
                return ApplyContextualChanges(COMMON_WORDS[word], word, context);
            }

            // Handle contractions
            var contractedPhonemes = HandleContractions(word);
            if (!string.IsNullOrEmpty(contractedPhonemes))
            {
                return contractedPhonemes;
            }

            // Fallback to eSpeak for unknown words
            try
            {
                return PhonemesHandler.TextToPhonemes(word).Trim();
            }
            catch
            {
                // Ultimate fallback - return empty to skip
                Debug.LogWarning($"Could not convert word to phonemes: {word}");
                return "";
            }
        }

        private static string GetPunctuationPhonemes(string punct)
        {
            switch (punct)
            {
                case ".":
                case "!":
                case "?":
                    return ".";
                case ",":
                case ";":
                case ":":
                    return ",";
                case "—":
                case "-":
                    return "—";
                default:
                    return "";
            }
        }

        private static string GetNumberPhonemes(string number)
        {
            // Simple number to word conversion for basic cases
            var dict = new Dictionary<string, string>
            {
                { "0", "ziɹoʊ" }, { "1", "wʌn" }, { "2", "tu" }, { "3", "θɹi" },
                { "4", "fɔɹ" }, { "5", "faɪv" }, { "6", "sɪks" }, { "7", "sɛvən" },
                { "8", "eɪt" }, { "9", "naɪn" }, { "10", "tɛn" }
            };

            if (dict.ContainsKey(number))
                return dict[number];

            // For larger numbers, fall back to eSpeak
            try
            {
                return PhonemesHandler.TextToPhonemes(number).Trim();
            }
            catch
            {
                return "";
            }
        }

        private static string HandleContractions(string word)
        {
            var contractions = new Dictionary<string, string>
            {
                { "don't", "doʊnt" }, { "Don't", "doʊnt" },
                { "can't", "kænt" }, { "Can't", "kænt" },
                { "won't", "woʊnt" }, { "Won't", "woʊnt" },
                { "shouldn't", "ʃʊdənt" }, { "Shouldn't", "ʃʊdənt" },
                { "wouldn't", "wʊdənt" }, { "Wouldn't", "wʊdənt" },
                { "couldn't", "kʊdənt" }, { "Couldn't", "kʊdənt" },
                { "isn't", "ɪzənt" }, { "Isn't", "ɪzənt" },
                { "aren't", "ɑɹənt" }, { "Aren't", "ɑɹənt" },
                { "wasn't", "wʌzənt" }, { "Wasn't", "wʌzənt" },
                { "weren't", "wɜɹənt" }, { "Weren't", "wɜɹənt" },
                { "haven't", "hævənt" }, { "Haven't", "hævənt" },
                { "hasn't", "hæzənt" }, { "Hasn't", "hæzənt" },
                { "hadn't", "hædənt" }, { "Hadn't", "hædənt" },
                { "I'm", "aɪm" }, { "you're", "jʊɹ" }, { "You're", "jʊɹ" },
                { "he's", "hiz" }, { "He's", "hiz" },
                { "she's", "ʃiz" }, { "She's", "ʃiz" },
                { "it's", "ɪts" }, { "It's", "ɪts" },
                { "we're", "wiɹ" }, { "We're", "wiɹ" },
                { "they're", "ðeɪɹ" }, { "They're", "ðeɪɹ" },
                { "I've", "aɪv" }, { "you've", "juv" }, { "You've", "juv" },
                { "we've", "wiv" }, { "We've", "wiv" },
                { "they've", "ðeɪv" }, { "They've", "ðeɪv" },
                { "I'll", "aɪl" }, { "you'll", "jul" }, { "You'll", "jul" },
                { "he'll", "hil" }, { "He'll", "hil" },
                { "she'll", "ʃil" }, { "She'll", "ʃil" },
                { "it'll", "ɪtəl" }, { "It'll", "ɪtəl" },
                { "we'll", "wil" }, { "We'll", "wil" },
                { "they'll", "ðeɪl" }, { "They'll", "ðeɪl" },
                { "I'd", "aɪd" }, { "you'd", "jud" }, { "You'd", "jud" },
                { "he'd", "hid" }, { "He'd", "hid" },
                { "she'd", "ʃid" }, { "She'd", "ʃid" },
                { "we'd", "wid" }, { "We'd", "wid" },
                { "they'd", "ðeɪd" }, { "They'd", "ðeɪd" }
            };

            return contractions.ContainsKey(word) ? contractions[word] : null;
        }

        private static string ApplyContextualChanges(string phonemes, string word, TokenContext context)
        {
            // Apply context-sensitive changes similar to Misaki
            switch (word.ToLower())
            {
                case "the":
                    return context.FutureVowel == true ? "ði" : "ðə";
                case "a":
                    return "ə"; // Usually reduced
                case "to":
                    if (context.FutureVowel == null)
                        return "tu";
                    return context.FutureVowel == true ? "tʊ" : "tə";
                default:
                    return phonemes;
            }
        }

        private static void UpdateContext(TokenContext context, string phonemes, MToken token)
        {
            // Update context based on current phonemes
            if (!string.IsNullOrEmpty(phonemes))
            {
                var firstSoundChar = phonemes.FirstOrDefault(c =>
                    VOWELS.Contains(c) || CONSONANTS.Contains(c) || NON_QUOTE_PUNCTS.Contains(c));

                if (firstSoundChar != default)
                {
                    context.FutureVowel = VOWELS.Contains(firstSoundChar) ? (bool?)true :
                                        NON_QUOTE_PUNCTS.Contains(firstSoundChar) ? null : false;
                }
            }

            var word = token.Text.ToLower();
            context.FutureTo = word == "to" || (word == "TO" && (token.Tag == "TO" || token.Tag == "IN"));
        }

        private static int StressWeight(string phonemes)
        {
            if (string.IsNullOrEmpty(phonemes))
                return 0;

            return phonemes.Sum(c => DIPHTHONGS.Contains(c) ? 2 : 1);
        }

        private static string ApplyStress(string phonemes, float? stress)
        {
            if (string.IsNullOrEmpty(phonemes) || stress == null)
                return phonemes;

            // Simplified stress application
            if (stress < -1)
            {
                return phonemes.Replace(PRIMARY_STRESS.ToString(), "").Replace(SECONDARY_STRESS.ToString(), "");
            }
            else if (stress == -1 || (stress >= 0 && stress < 1 && phonemes.Contains(PRIMARY_STRESS)))
            {
                return phonemes.Replace(SECONDARY_STRESS.ToString(), "").Replace(PRIMARY_STRESS.ToString(), SECONDARY_STRESS.ToString());
            }
            else if (stress >= 1 && !phonemes.Contains(PRIMARY_STRESS) && phonemes.Contains(SECONDARY_STRESS))
            {
                return phonemes.Replace(SECONDARY_STRESS.ToString(), PRIMARY_STRESS.ToString());
            }

            return phonemes;
        }
    }
}
