using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Globalization;
using UnityEngine;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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

        // Lexicon dictionaries
        private static Dictionary<string, object> _goldDictionary;
        private static Dictionary<string, string> _silverDictionary;
        private static bool _dictionariesLoaded = false;

        // Lexicon constants
        private static readonly HashSet<int> LEXICON_ORDS = new HashSet<int>(new[] { 39, 45 }.Concat(Enumerable.Range(65, 26)).Concat(Enumerable.Range(97, 26)));
        private static readonly HashSet<char> US_TAUS = new HashSet<char>("AIOWYiuæɑəɛɪɹʊʌ".ToCharArray());
        private static readonly Dictionary<string, (string, string)> CURRENCIES = new Dictionary<string, (string, string)>
        {
            { "$", ("dollar", "cent") },
            { "£", ("pound", "pence") },
            { "€", ("euro", "cent") }
        };
        private static readonly HashSet<string> ORDINALS = new HashSet<string> { "st", "nd", "rd", "th" };
        private static readonly Dictionary<string, string> ADD_SYMBOLS = new Dictionary<string, string> { { ".", "dot" }, { "/", "slash" } };

        // Stress markers
        private const char PRIMARY_STRESS = 'ˈ';
        private const char SECONDARY_STRESS = 'ˌ';
        private static readonly string STRESSES = new string(new[] { SECONDARY_STRESS, PRIMARY_STRESS });

        // Phoneme vocabulary for tokenization
        private static readonly Dictionary<char, int> PhonemeVocab = new Dictionary<char, int>
        {
            ['\n'] = -1, ['$'] = 0, [';'] = 1, [':'] = 2, [','] = 3, ['.'] = 4, ['!'] = 5, ['?'] = 6, ['¡'] = 7, ['¿'] = 8, ['—'] = 9, ['…'] = 10, ['\"'] = 11, ['('] = 12, [')'] = 13, ['"'] = 14, ['"'] = 15, [' '] = 16, ['\u0303'] = 17, ['ʣ'] = 18, ['ʥ'] = 19, ['ʦ'] = 20, ['ʨ'] = 21, ['ᵝ'] = 22, ['\uAB67'] = 23, ['A'] = 24, ['I'] = 25, ['O'] = 31, ['Q'] = 33, ['S'] = 35, ['T'] = 36, ['W'] = 39, ['Y'] = 41, ['ᵊ'] = 42, ['a'] = 43, ['b'] = 44, ['c'] = 45, ['d'] = 46, ['e'] = 47, ['f'] = 48, ['h'] = 50, ['i'] = 51, ['j'] = 52, ['k'] = 53, ['l'] = 54, ['m'] = 55, ['n'] = 56, ['o'] = 57, ['p'] = 58, ['q'] = 59, ['r'] = 60, ['s'] = 61, ['t'] = 62, ['u'] = 63, ['v'] = 64, ['w'] = 65, ['x'] = 66, ['y'] = 67, ['z'] = 68, ['ɑ'] = 69, ['ɐ'] = 70, ['ɒ'] = 71, ['æ'] = 72, ['β'] = 75, ['ɔ'] = 76, ['ɕ'] = 77, ['ç'] = 78, ['ɖ'] = 80, ['ð'] = 81, ['ʤ'] = 82, ['ə'] = 83, ['ɚ'] = 85, ['ɛ'] = 86, ['ɜ'] = 87, ['ɟ'] = 90, ['ɡ'] = 92, ['ɥ'] = 99, ['ɨ'] = 101, ['ɪ'] = 102, ['ʝ'] = 103, ['ɯ'] = 110, ['ɰ'] = 111, ['ŋ'] = 112, ['ɳ'] = 113, ['ɲ'] = 114, ['ɴ'] = 115, ['ø'] = 116, ['ɸ'] = 118, ['θ'] = 119, ['œ'] = 120, ['ɹ'] = 123, ['ɾ'] = 125, ['ɻ'] = 126, ['ʁ'] = 128, ['ɽ'] = 129, ['ʂ'] = 130, ['ʃ'] = 131, ['ʈ'] = 132, ['ʧ'] = 133, ['ʊ'] = 135, ['ʋ'] = 136, ['ʌ'] = 138, ['ɣ'] = 139, ['ɤ'] = 140, ['χ'] = 142, ['ʎ'] = 143, ['ʒ'] = 147, ['ʔ'] = 148, ['ˈ'] = 156, ['ˌ'] = 157, ['ː'] = 158, ['ʰ'] = 162, ['ʲ'] = 164, ['↓'] = 169, ['→'] = 171, ['↗'] = 172, ['↘'] = 173, ['ᵻ'] = 177
        };

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

        private static void LoadDictionaries()
        {
            if (_dictionariesLoaded) return;

            try
            {
                // Load gold dictionary
                var goldTextAsset = Resources.Load<TextAsset>("us_gold");
                if (goldTextAsset != null)
                {
                    _goldDictionary = JsonConvert.DeserializeObject<Dictionary<string, object>>(goldTextAsset.text);
                }
                else
                {
                    Debug.LogError("Failed to load us_gold.json from Resources");
                    _goldDictionary = new Dictionary<string, object>();
                }

                // Load silver dictionary
                var silverTextAsset = Resources.Load<TextAsset>("us_silver");
                if (silverTextAsset != null)
                {
                    _silverDictionary = JsonConvert.DeserializeObject<Dictionary<string, string>>(silverTextAsset.text);
                }
                else
                {
                    Debug.LogError("Failed to load us_silver.json from Resources");
                    _silverDictionary = new Dictionary<string, string>();
                }

                _dictionariesLoaded = true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to load dictionaries: {ex.Message}");
                _goldDictionary = new Dictionary<string, object>();
                _silverDictionary = new Dictionary<string, string>();
                _dictionariesLoaded = true;
            }
        }

        public static int[] TokenizeGraphemes(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return new int[0];

            var phonemes = TextToPhonemes(inputText);
            return Tokenize(phonemes);
        }

        public static int[] Tokenize(string phonemes)
        {
            if (string.IsNullOrEmpty(phonemes))
                return new int[0];

            var tokens = new List<int>();
            foreach (var character in phonemes.ToCharArray())
            {
                if (!PhonemeVocab.TryGetValue(character, out var value))
                {
                    Debug.LogWarning($"Character '{character}' not in vocabulary, skipping.");
                    continue;
                }
                tokens.Add(value);
            }
            return tokens.ToArray();
        }

        public static string TextToPhonemes(string text)
        {
            if (string.IsNullOrEmpty(text))
                return string.Empty;

            LoadDictionaries();

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

            // More sophisticated regex to handle punctuation, contractions, and spacing
            var words = Regex.Split(text, @"(\s+|[.,!?;:—…""'()]+)")
                .Where(w => !string.IsNullOrEmpty(w))
                .ToArray();

            for (int i = 0; i < words.Length; i++)
            {
                var word = words[i];
                var trimmedWord = word.Trim();

                if (string.IsNullOrWhiteSpace(trimmedWord))
                    continue;

                // Handle words with attached punctuation (like "Hello,", "Today's", etc.)
                var cleanedWords = SplitWordWithPunctuation(trimmedWord);

                foreach (var cleanWord in cleanedWords)
                {
                    if (!string.IsNullOrWhiteSpace(cleanWord))
                    {
                        var token = new MToken
                        {
                            Text = cleanWord,
                            Whitespace = (i == words.Length - 1 && cleanWord == cleanedWords.Last()) ? "" : " ",
                            Tag = GetSimpleTag(cleanWord)
                        };
                        tokens.Add(token);
                    }
                }
            }

            return tokens;
        }

        private static List<string> SplitWordWithPunctuation(string word)
        {
            var result = new List<string>();

            // Handle leading quotes and punctuation
            var match = Regex.Match(word, @"^([""''""\u2018\u2019\u201C\u201D\(\[\{]*)(.*?)([""''""\u2018\u2019\u201C\u201D\)\]\}.,!?;:—…]*)$");

            if (match.Success)
            {
                var leadingPunct = match.Groups[1].Value;
                var coreWord = match.Groups[2].Value;
                var trailingPunct = match.Groups[3].Value;

                // Add leading punctuation as separate tokens
                foreach (char c in leadingPunct)
                {
                    if (!char.IsWhiteSpace(c))
                        result.Add(c.ToString());
                }

                if (!string.IsNullOrEmpty(coreWord))
                    result.Add(coreWord);

                // Add trailing punctuation as separate tokens
                foreach (char c in trailingPunct)
                {
                    if (!char.IsWhiteSpace(c))
                        result.Add(c.ToString());
                }
            }
            else
            {
                result.Add(word);
            }

            return result.Where(s => !string.IsNullOrEmpty(s)).ToList();
        }

        private static string GetSimpleTag(string word)
        {
            if (string.IsNullOrWhiteSpace(word))
                return "";

            word = word.Trim();

            // Check for punctuation - including quotes and other symbols
            if (PUNCT_TAGS.Contains(word) ||
                PUNCTS.Contains(word.FirstOrDefault()) ||
                word.All(c => !char.IsLetterOrDigit(c))) // Any non-alphanumeric character
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
            if (word.Length == 1 && PUNCTS.Contains(word[0]))
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

            // Try lexicon lookup
            var (phonemes, rating) = LookupWord(word, token.Tag, null, context);
            if (!string.IsNullOrEmpty(phonemes))
            {
                return phonemes;
            }

            // Handle contractions as fallback
            var contractedPhonemes = HandleContractions(word);
            if (!string.IsNullOrEmpty(contractedPhonemes))
            {
                return contractedPhonemes;
            }

            // If no phonemes found, return empty (no eSpeak fallback)
            // Only warn for actual words, not punctuation
            if (token.Tag != "PUNCT")
            {
                Debug.LogWarning($"Could not find phonemes for word: '{word}' (tag: '{token.Tag}')");
            }
            return "";
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

            // For larger numbers, try to convert to words and lookup each word
            var numberWord = ConvertNumberToWords(int.Parse(number));
            if (!string.IsNullOrEmpty(numberWord))
            {
                // Split multi-word numbers and look up each part
                var words = numberWord.Split(' ');
                var phonemeParts = new List<string>();

                foreach (var word in words)
                {
                    if (string.IsNullOrEmpty(word)) continue;

                    var (phonemes, _) = LookupWord(word, "CD", null, new TokenContext());
                    if (!string.IsNullOrEmpty(phonemes))
                    {
                        phonemeParts.Add(phonemes);
                    }
                    else
                    {
                        // If any word fails, return empty
                        return "";
                    }
                }

                if (phonemeParts.Count > 0)
                {
                    return string.Join(" ", phonemeParts);
                }
            }

            return "";
        }

        private static string ConvertNumberToWords(int number)
        {
            if (number < 0) return "minus " + ConvertNumberToWords(-number);
            if (number == 0) return "zero";
            if (number < 20) return new[] { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" }[number];
            if (number < 100) return new[] { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" }[number / 10] + (number % 10 != 0 ? " " + ConvertNumberToWords(number % 10) : "");
            if (number < 1000) return ConvertNumberToWords(number / 100) + " hundred" + (number % 100 != 0 ? " " + ConvertNumberToWords(number % 100) : "");
            if (number < 1000000) return ConvertNumberToWords(number / 1000) + " thousand" + (number % 1000 != 0 ? " " + ConvertNumberToWords(number % 1000) : "");
            return "";
        }

        private static string HandleContractions(string word)
        {
            // Normalize Unicode apostrophes to ASCII apostrophes
            word = word.Replace("\u2019", "'").Replace("'", "'");

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

        private static (string, int) LookupWord(string word, string tag, float? stress, TokenContext context)
        {
            if (string.IsNullOrEmpty(word))
                return ("", 0);

            // Handle special cases first
            var specialCase = GetSpecialCase(word, tag, stress, context);
            if (!string.IsNullOrEmpty(specialCase.Item1))
                return specialCase;

            var isNNP = false;
            var originalWord = word;
            var lookupWord = word;

            // Handle case variations - try original case first, then lowercase
            var casesToTry = new List<string>();
            casesToTry.Add(word);
            if (word != word.ToLower())
                casesToTry.Add(word.ToLower());
            if (word != word.ToUpper())
                casesToTry.Add(word.ToUpper());

            foreach (var caseVariant in casesToTry)
            {
                // Try gold dictionary first
                if (_goldDictionary.ContainsKey(caseVariant))
                {
                    var entry = _goldDictionary[caseVariant];
                    string phonemes = null;
                    int rating = 4;

                    if (entry is string simplePhonemes)
                    {
                        phonemes = simplePhonemes;
                    }
                    else if (entry is JObject complexEntry)
                    {
                        // Handle complex entries with multiple POS tags
                        if (!string.IsNullOrEmpty(tag) && complexEntry.ContainsKey(tag))
                        {
                            phonemes = complexEntry[tag]?.ToString();
                        }
                        else if (complexEntry.ContainsKey("DEFAULT"))
                        {
                            phonemes = complexEntry["DEFAULT"]?.ToString();
                        }
                        else
                        {
                            // If no specific tag match, try the first available entry
                            var firstEntry = complexEntry.Properties().FirstOrDefault();
                            phonemes = firstEntry?.Value?.ToString();
                        }
                    }

                    if (!string.IsNullOrEmpty(phonemes))
                    {
                        return (ApplyStress(phonemes, stress), rating);
                    }
                }

                // Try silver dictionary if not found in gold and not a proper noun
                if (!isNNP && _silverDictionary.ContainsKey(caseVariant))
                {
                    var phonemes = _silverDictionary[caseVariant];
                    return (ApplyStress(phonemes, stress), 3);
                }
            }

            // Handle possessives (like "Today's" -> "Today" + "'s")
            if (word.EndsWith("'s") || word.EndsWith("'s") || word.EndsWith("\u2019s"))
            {
                var baseWord = word.EndsWith("'s") ? word.Substring(0, word.Length - 2) :
                              word.EndsWith("'s") ? word.Substring(0, word.Length - 2) :
                              word.Substring(0, word.Length - 2);
                var (basePhonemes, baseRating) = LookupWord(baseWord, tag, stress, context);
                if (!string.IsNullOrEmpty(basePhonemes))
                {
                    return (basePhonemes + "s", baseRating);
                }
            }

            // Handle morphological variants
            var morphResult = TryMorphologicalLookup(word, tag, stress, context);
            if (!string.IsNullOrEmpty(morphResult.Item1))
                return morphResult;

            return ("", 0);
        }

        private static (string, int) GetSpecialCase(string word, string tag, float? stress, TokenContext context)
        {
            // Handle ADD symbols
            if (tag == "ADD" && ADD_SYMBOLS.ContainsKey(word))
            {
                return LookupWord(ADD_SYMBOLS[word], null, -0.5f, context);
            }

            // Handle general symbols
            if (SYMBOLS.ContainsKey(word))
            {
                return LookupWord(SYMBOLS[word], null, null, context);
            }

            // Handle contextual words like "the", "a", "to"
            switch (word.ToLower())
            {
                case "the":
                case "The":
                    return (context.FutureVowel == true ? "ði" : "ðə", 4);
                case "a":
                case "A":
                    return (tag == "DT" ? "ɐ" : "ˈA", 4);
                case "to":
                case "To":
                    if (context.FutureVowel == null) return ("ˈtu", 4);
                    return (context.FutureVowel == true ? "tʊ" : "tə", 4);
                case "an":
                case "An":
                    return ("ɐn", 4);
                case "in":
                case "In":
                    var stress_marker = context.FutureVowel == null || tag != "IN" ? "ˈ" : "";
                    return (stress_marker + "ɪn", 4);
            }

            return ("", 0);
        }

        private static (string, int) TryMorphologicalLookup(string word, string tag, float? stress, TokenContext context)
        {
            // Try -s suffix (plurals, verb conjugations)
            if (word.Length > 2 && word.EndsWith("s") && !word.EndsWith("ss"))
            {
                var stem = word.Substring(0, word.Length - 1);
                if (IsKnown(stem, tag))
                {
                    var (stemPhonemes, rating) = LookupWord(stem, tag, stress, context);
                    if (!string.IsNullOrEmpty(stemPhonemes))
                    {
                        return (ApplySSuffix(stemPhonemes), rating);
                    }
                }
            }

            // Try -ed suffix
            if (word.Length > 3 && word.EndsWith("ed") && !word.EndsWith("eed"))
            {
                var stem = word.Substring(0, word.Length - 2);
                if (IsKnown(stem, tag))
                {
                    var (stemPhonemes, rating) = LookupWord(stem, tag, stress, context);
                    if (!string.IsNullOrEmpty(stemPhonemes))
                    {
                        return (ApplyEdSuffix(stemPhonemes), rating);
                    }
                }
            }

            // Try -ing suffix
            if (word.Length > 4 && word.EndsWith("ing"))
            {
                var stem = word.Substring(0, word.Length - 3);
                if (IsKnown(stem, tag))
                {
                    var (stemPhonemes, rating) = LookupWord(stem, tag, 0.5f, context);
                    if (!string.IsNullOrEmpty(stemPhonemes))
                    {
                        return (ApplyIngSuffix(stemPhonemes), rating);
                    }
                }
            }

            return ("", 0);
        }

        private static bool IsKnown(string word, string tag)
        {
            if (_goldDictionary.ContainsKey(word) || SYMBOLS.ContainsKey(word) || _silverDictionary.ContainsKey(word))
                return true;

            if (!word.All(c => char.IsLetter(c) && LEXICON_ORDS.Contains(c)))
                return false;

            if (word.Length == 1)
                return true;

            if (word == word.ToUpper() && _goldDictionary.ContainsKey(word.ToLower()))
                return true;

            return word.Substring(1) == word.Substring(1).ToUpper();
        }

        private static string ApplySSuffix(string stem)
        {
            if (string.IsNullOrEmpty(stem)) return null;

            var lastChar = stem.LastOrDefault();
            if ("ptkfθ".Contains(lastChar))
                return stem + "s";
            else if ("szʃʒʧʤ".Contains(lastChar))
                return stem + "ɪz"; // Using US pronunciation
            else
                return stem + "z";
        }

        private static string ApplyEdSuffix(string stem)
        {
            if (string.IsNullOrEmpty(stem)) return null;

            var lastChar = stem.LastOrDefault();
            if ("pkfθʃsʧ".Contains(lastChar))
                return stem + "t";
            else if (lastChar == 'd')
                return stem + "ɪd"; // Using US pronunciation
            else if (lastChar != 't')
                return stem + "d";
            else if (stem.Length >= 2 && US_TAUS.Contains(stem[stem.Length - 2]))
                return stem.Substring(0, stem.Length - 1) + "ɾᵻd";
            else
                return stem + "ᵻd";
        }

        private static string ApplyIngSuffix(string stem)
        {
            if (string.IsNullOrEmpty(stem)) return null;

            if (stem.Length > 1 && stem.LastOrDefault() == 't' && US_TAUS.Contains(stem[stem.Length - 2]))
                return stem.Substring(0, stem.Length - 1) + "ɾɪŋ";
            else
                return stem + "ɪŋ";
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

            // Simplified stress application based on Misaki logic
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
            else if (stress > 1 && !phonemes.Contains(PRIMARY_STRESS) && !phonemes.Contains(SECONDARY_STRESS))
            {
                if (VOWELS.Any(v => phonemes.Contains(v)))
                {
                    return PRIMARY_STRESS + phonemes;
                }
            }

            return phonemes;
        }
    }
}
