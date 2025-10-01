using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Inference
{
    public class MToken
    {
        public string Text { get; init; } = "";
        public string Tag { get; init; } = "";
    }

    public class TokenContext
    {
        public bool? FutureVowel { get; set; }
    }

    public static class MisakiSharp
    {
        // Character sets
        static readonly HashSet<char> k_Consonants = new("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ".ToCharArray());
        static readonly HashSet<char> k_Vowels = new("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ".ToCharArray());
        static readonly HashSet<char> k_Puncts = new(";:,.!?—…\"\"\"".ToCharArray());
        static readonly HashSet<char> k_NonQuotePuncts = new(k_Puncts.Where(p => p != '\"' && p != '\"' && p != '\"').ToArray());
        static readonly HashSet<string> k_PunctTags = new() { ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP" };

        // Lexicon dictionaries
        static Dictionary<string, object> s_GoldDictionary;
        static Dictionary<string, string> s_SilverDictionary;
        static bool s_DictionariesLoaded;

        // Lexicon constants
        static readonly HashSet<int> k_LexiconOrds = new(new[] { 39, 45 }.Concat(Enumerable.Range(65, 26)).Concat(Enumerable.Range(97, 26)));
        static readonly HashSet<char> k_UsTaus = new("AIOWYiuæɑəɛɪɹʊʌ".ToCharArray());
        static readonly Dictionary<string, string> k_AddSymbols = new() { { ".", "dot" }, { "/", "slash" } };

        // Stress markers
        const char k_PrimaryStress = 'ˈ';
        const char k_SecondaryStress = 'ˌ';

        // Phoneme vocabulary for tokenization
        static readonly Dictionary<char, int> k_PhonemeVocab = new()
        {
            ['\n'] = -1, ['$'] = 0, [';'] = 1, [':'] = 2, [','] = 3, ['.'] = 4, ['!'] = 5, ['?'] = 6, ['¡'] = 7, ['¿'] = 8, ['—'] = 9, ['…'] = 10, ['\"'] = 11, ['('] = 12, [')'] = 13, ['"'] = 14, ['"'] = 15, [' '] = 16, ['\u0303'] = 17, ['ʣ'] = 18, ['ʥ'] = 19, ['ʦ'] = 20, ['ʨ'] = 21, ['ᵝ'] = 22, ['\uAB67'] = 23, ['A'] = 24, ['I'] = 25, ['O'] = 31, ['Q'] = 33, ['S'] = 35, ['T'] = 36, ['W'] = 39, ['Y'] = 41, ['ᵊ'] = 42, ['a'] = 43, ['b'] = 44, ['c'] = 45, ['d'] = 46, ['e'] = 47, ['f'] = 48, ['h'] = 50, ['i'] = 51, ['j'] = 52, ['k'] = 53, ['l'] = 54, ['m'] = 55, ['n'] = 56, ['o'] = 57, ['p'] = 58, ['q'] = 59, ['r'] = 60, ['s'] = 61, ['t'] = 62, ['u'] = 63, ['v'] = 64, ['w'] = 65, ['x'] = 66, ['y'] = 67, ['z'] = 68, ['ɑ'] = 69, ['ɐ'] = 70, ['ɒ'] = 71, ['æ'] = 72, ['β'] = 75, ['ɔ'] = 76, ['ɕ'] = 77, ['ç'] = 78, ['ɖ'] = 80, ['ð'] = 81, ['ʤ'] = 82, ['ə'] = 83, ['ɚ'] = 85, ['ɛ'] = 86, ['ɜ'] = 87, ['ɟ'] = 90, ['ɡ'] = 92, ['ɥ'] = 99, ['ɨ'] = 101, ['ɪ'] = 102, ['ʝ'] = 103, ['ɯ'] = 110, ['ɰ'] = 111, ['ŋ'] = 112, ['ɳ'] = 113, ['ɲ'] = 114, ['ɴ'] = 115, ['ø'] = 116, ['ɸ'] = 118, ['θ'] = 119, ['œ'] = 120, ['ɹ'] = 123, ['ɾ'] = 125, ['ɻ'] = 126, ['ʁ'] = 128, ['ɽ'] = 129, ['ʂ'] = 130, ['ʃ'] = 131, ['ʈ'] = 132, ['ʧ'] = 133, ['ʊ'] = 135, ['ʋ'] = 136, ['ʌ'] = 138, ['ɣ'] = 139, ['ɤ'] = 140, ['χ'] = 142, ['ʎ'] = 143, ['ʒ'] = 147, ['ʔ'] = 148, ['ˈ'] = 156, ['ˌ'] = 157, ['ː'] = 158, ['ʰ'] = 162, ['ʲ'] = 164, ['↓'] = 169, ['→'] = 171, ['↗'] = 172, ['↘'] = 173, ['ᵻ'] = 177
        };

        static readonly Dictionary<string, string> k_Symbols = new()
        {
            { "%", "percent" },
            { "&", "and" },
            { "+", "plus" },
            { "@", "at" }
        };

        static void LoadDictionaries()
        {
            if (s_DictionariesLoaded) return;

            try
            {
                // Load gold dictionary
                var goldTextAsset = Resources.Load<TextAsset>("us_gold");
                if (goldTextAsset != null)
                {
                    s_GoldDictionary = JsonConvert.DeserializeObject<Dictionary<string, object>>(goldTextAsset.text);
                }
                else
                {
                    Debug.LogError("Failed to load us_gold.json from Resources");
                    s_GoldDictionary = new Dictionary<string, object>();
                }

                // Load silver dictionary
                var silverTextAsset = Resources.Load<TextAsset>("us_silver");
                if (silverTextAsset != null)
                {
                    s_SilverDictionary = JsonConvert.DeserializeObject<Dictionary<string, string>>(silverTextAsset.text);
                }
                else
                {
                    Debug.LogError("Failed to load us_silver.json from Resources");
                    s_SilverDictionary = new Dictionary<string, string>();
                }

                s_DictionariesLoaded = true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to load dictionaries: {ex.Message}");
                s_GoldDictionary = new Dictionary<string, object>();
                s_SilverDictionary = new Dictionary<string, string>();
                s_DictionariesLoaded = true;
            }
        }

        public static int[] TokenizeGraphemes(string inputText)
        {
            if (string.IsNullOrEmpty(inputText))
                return Array.Empty<int>();

            var phonemes = TextToPhonemes(inputText);
            return Tokenize(phonemes);
        }

        static int[] Tokenize(string phonemes)
        {
            if (string.IsNullOrEmpty(phonemes))
                return Array.Empty<int>();

            var tokens = new List<int>();
            foreach (var character in phonemes)
            {
                if (!k_PhonemeVocab.TryGetValue(character, out var value))
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
                    UpdateContext(context, phonemes);
                }
            }

            return string.Join(" ", result);
        }

        static List<MToken> SimpleTokenize(string text)
        {
            var tokens = new List<MToken>();

            // More sophisticated regex to handle punctuation, contractions, and spacing
            // Note: Don't split on apostrophes to preserve contractions
            var words = Regex.Split(text, @"(\s+|[.,!?;:—…""()]+)")
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
                            Tag = GetSimpleTag(cleanWord)
                        };
                        tokens.Add(token);
                    }
                }
            }

            return tokens;
        }

        static List<string> SplitWordWithPunctuation(string word)
        {
            var result = new List<string>();

            // Don't split contractions - keep them as single tokens
            if (IsContraction(word))
            {
                result.Add(word);
                return result;
            }

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

        static bool IsContraction(string word)
        {
            // Check if the word contains apostrophe and looks like a contraction/possessive
            return (word.Contains("'") || word.Contains("'") || word.Contains("\u2019")) &&
                   word.Length > 2 &&
                   !word.StartsWith('\'') && !word.StartsWith('\'') && !word.StartsWith('’') &&
                   !word.EndsWith('\'') && !word.EndsWith('\'') && !word.EndsWith('’');
        }

        static string GetSimpleTag(string word)
        {
            if (string.IsNullOrWhiteSpace(word))
                return "";

            word = word.Trim();

            // Check for punctuation - including quotes and other symbols
            if (k_PunctTags.Contains(word) ||
                k_Puncts.Contains(word.FirstOrDefault()) ||
                word.All(c => !char.IsLetterOrDigit(c))) // Any non-alphanumeric character
                return "PUNCT";

            if (Regex.IsMatch(word, @"^\d+$"))
                return "CD";

            if (char.IsUpper(word.FirstOrDefault()) && word.Length > 1)
                return "NNP";

            return "NN";
        }

        static string GetWordPhonemes(MToken token, TokenContext context)
        {
            var word = token.Text;

            if (string.IsNullOrEmpty(word))
                return "";

            // Handle punctuation
            if (word.Length == 1 && k_Puncts.Contains(word[0]))
            {
                return GetPunctuationPhonemes(word);
            }

            // Handle symbols
            if (k_Symbols.TryGetValue(word, out var symbol))
            {
                return GetWordPhonemes(new MToken { Text = symbol, Tag = "NN" }, context);
            }

            // Handle numbers
            if (Regex.IsMatch(word, @"^\d+$"))
            {
                return GetNumberPhonemes(word);
            }

            // Handle contractions first (before dictionary lookup)
            var contractedPhonemes = HandleContractions(word);
            if (!string.IsNullOrEmpty(contractedPhonemes))
            {
                return contractedPhonemes;
            }

            // Try lexicon lookup
            var (phonemes, _) = LookupWord(word, token.Tag, null, context);
            if (!string.IsNullOrEmpty(phonemes))
            {
                return phonemes;
            }

            // If no phonemes found, return empty (no eSpeak fallback)
            // Only warn for actual words, not punctuation
            if (token.Tag != "PUNCT")
            {
                Debug.LogWarning($"Could not find phonemes for word: '{word}' (tag: '{token.Tag}')");
            }
            return "";
        }

        static string GetPunctuationPhonemes(string punct)
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

        static string GetNumberPhonemes(string number)
        {
            // Simple number to word conversion for basic cases
            var dict = new Dictionary<string, string>
            {
                { "0", "ziɹoʊ" }, { "1", "wʌn" }, { "2", "tu" }, { "3", "θɹi" },
                { "4", "fɔɹ" }, { "5", "faɪv" }, { "6", "sɪks" }, { "7", "sɛvən" },
                { "8", "eɪt" }, { "9", "naɪn" }, { "10", "tɛn" }
            };

            if (dict.TryGetValue(number, out var numberPhonemes))
                return numberPhonemes;

            // For larger numbers, try to convert to words and look up each word
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

        static string ConvertNumberToWords(int number)
        {
            if (number < 0) return "minus " + ConvertNumberToWords(-number);
            if (number == 0) return "zero";
            if (number < 20) return new[] { "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" }[number];
            if (number < 100) return new[] { "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" }[number / 10] + (number % 10 != 0 ? " " + ConvertNumberToWords(number % 10) : "");
            if (number < 1000) return ConvertNumberToWords(number / 100) + " hundred" + (number % 100 != 0 ? " " + ConvertNumberToWords(number % 100) : "");
            if (number < 1000000) return ConvertNumberToWords(number / 1000) + " thousand" + (number % 1000 != 0 ? " " + ConvertNumberToWords(number % 1000) : "");
            return "";
        }

        static string HandleContractions(string word)
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
                { "they'd", "ðeɪd" }, { "They'd", "ðeɪd" },
                { "what's", "wʌts" }, { "What's", "wʌts" },
                { "that's", "ðæts" }, { "That's", "ðæts" },
                { "there's", "ðɛɹz" }, { "There's", "ðɛɹz" },
                { "here's", "hɪɹz" }, { "Here's", "hɪɹz" },
                { "where's", "wɛɹz" }, { "Where's", "wɛɹz" },
                { "who's", "huz" }, { "Who's", "huz" },
                { "how's", "haʊz" }, { "How's", "haʊz" },
                { "when's", "wɛnz" }, { "When's", "wɛnz" },
                { "why's", "waɪz" }, { "Why's", "waɪz" }
            };

            // Check direct contraction first
            if (contractions.TryGetValue(word, out var handleContractions))
                return handleContractions;

            // Handle general's pattern for words not in the contraction list
            if (word.EndsWith("'s") || word.EndsWith("'s") || word.EndsWith("\u2019s"))
            {
                var baseWord = word.Substring(0, word.Length - 2);

                // Try to look up the base word and add "s" sound
                var (basePhonemes, _) = LookupWord(baseWord, "NN", null, new TokenContext());
                if (!string.IsNullOrEmpty(basePhonemes))
                {
                    return basePhonemes + "s";
                }
            }

            return null;
        }

        static (string, int) LookupWord(string word, string tag, float? stress, TokenContext context)
        {
            if (string.IsNullOrEmpty(word))
                return ("", 0);

            // Handle special cases first
            var specialCase = GetSpecialCase(word, tag, context);
            if (!string.IsNullOrEmpty(specialCase.Item1))
                return specialCase;

            // Handle case variations - try an original case first, then lowercase
            var casesToTry = new List<string>();
            casesToTry.Add(word);
            if (word != word.ToLower())
                casesToTry.Add(word.ToLower());
            if (word != word.ToUpper())
                casesToTry.Add(word.ToUpper());

            foreach (var caseVariant in casesToTry)
            {
                // Try gold dictionary first
                if (s_GoldDictionary.TryGetValue(caseVariant, out var entry))
                {
                    string phonemes = null;
                    int rating = 4;

                    if (entry is string simplePhonemes)
                    {
                        phonemes = simplePhonemes;
                    }
                    else if (entry is JObject complexEntry)
                    {
                        // Handle complex entries with multiple POS tags
                        if (!string.IsNullOrEmpty(tag) && complexEntry.TryGetValue(tag, out var value))
                        {
                            phonemes = value?.ToString();
                        }
                        else if (complexEntry.TryGetValue("DEFAULT", out var value1))
                        {
                            phonemes = value1?.ToString();
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
                if (s_SilverDictionary.TryGetValue(caseVariant, out var phonemes1))
                {
                    return (ApplyStress(phonemes1, stress), 3);
                }
            }


            // Handle morphological variants
            var morphResult = TryMorphologicalLookup(word, tag, stress, context);
            if (!string.IsNullOrEmpty(morphResult.Item1))
                return morphResult;

            return ("", 0);
        }

        static (string, int) GetSpecialCase(string word, string tag, TokenContext context)
        {
            // Handle ADD symbols
            if (tag == "ADD" && k_AddSymbols.TryGetValue(word, out var symbol))
            {
                return LookupWord(symbol, null, -0.5f, context);
            }

            // Handle general symbols
            if (k_Symbols.TryGetValue(word, out var symbol1))
            {
                return LookupWord(symbol1, null, null, context);
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
                    var stressMarker = context.FutureVowel == null || tag != "IN" ? "ˈ" : "";
                    return (stressMarker + "ɪn", 4);
            }

            return ("", 0);
        }

        static (string, int) TryMorphologicalLookup(string word, string tag, float? stress, TokenContext context)
        {
            // Try -s suffix (plurals, verb conjugations)
            if (word.Length > 2 && word.EndsWith('s') && !word.EndsWith("ss"))
            {
                var stem = word.Substring(0, word.Length - 1);
                if (IsKnown(stem))
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
                if (IsKnown(stem))
                {
                    var (stemPhonemes, rating) = LookupWord(stem, tag, stress, context);
                    if (!string.IsNullOrEmpty(stemPhonemes))
                    {
                        return (ApplyEdSuffix(stemPhonemes), rating);
                    }
                }
            }

            // Try-ing suffix
            if (word.Length > 4 && word.EndsWith("ing"))
            {
                var stem = word.Substring(0, word.Length - 3);
                if (IsKnown(stem))
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

        static bool IsKnown(string word)
        {
            if (s_GoldDictionary.ContainsKey(word) || k_Symbols.ContainsKey(word) || s_SilverDictionary.ContainsKey(word))
                return true;

            if (!word.All(c => char.IsLetter(c) && k_LexiconOrds.Contains(c)))
                return false;

            if (word.Length == 1)
                return true;

            if (word == word.ToUpper() && s_GoldDictionary.ContainsKey(word.ToLower()))
                return true;

            return word.Substring(1) == word.Substring(1).ToUpper();
        }

        static string ApplySSuffix(string stem)
        {
            if (string.IsNullOrEmpty(stem)) return null;

            var lastChar = stem.LastOrDefault();
            if ("ptkfθ".Contains(lastChar))
                return stem + "s";
            if ("szʃʒʧʤ".Contains(lastChar))
                return stem + "ɪz"; // Using US pronunciation
            return stem + "z";
        }

        static string ApplyEdSuffix(string stem)
        {
            if (string.IsNullOrEmpty(stem)) return null;

            var lastChar = stem.LastOrDefault();
            if ("pkfθʃsʧ".Contains(lastChar))
                return stem + "t";
            if (lastChar == 'd')
                return stem + "ɪd"; // Using US pronunciation
            if (lastChar != 't')
                return stem + "d";
            if (stem.Length >= 2 && k_UsTaus.Contains(stem[^2]))
                return stem.Substring(0, stem.Length - 1) + "ɾᵻd";
            return stem + "ᵻd";
        }

        static string ApplyIngSuffix(string stem)
        {
            if (string.IsNullOrEmpty(stem)) return null;

            if (stem.Length > 1 && stem.LastOrDefault() == 't' && k_UsTaus.Contains(stem[^2]))
                return stem.Substring(0, stem.Length - 1) + "ɾɪŋ";
            return stem + "ɪŋ";
        }

        static void UpdateContext(TokenContext context, string phonemes)
        {
            // Update context based on current phonemes
            if (!string.IsNullOrEmpty(phonemes))
            {
                var firstSoundChar = phonemes.FirstOrDefault(c =>
                    k_Vowels.Contains(c) || k_Consonants.Contains(c) || k_NonQuotePuncts.Contains(c));

                if (firstSoundChar != 0)
                {
                    if (k_Vowels.Contains(firstSoundChar))
                        context.FutureVowel = true;
                    else
                        context.FutureVowel = k_NonQuotePuncts.Contains(firstSoundChar) ? null : false;
                }
            }
        }

        static string ApplyStress(string phonemes, float? stress)
        {
            if (string.IsNullOrEmpty(phonemes) || stress == null)
                return phonemes;

            // Simplified stress application based on Misaki logic
            if (stress < -1)
            {
                return phonemes.Replace(k_PrimaryStress.ToString(), "").Replace(k_SecondaryStress.ToString(), "");
            }

            if (stress == -1 || (stress is >= 0 and < 1 && phonemes.Contains(k_PrimaryStress)))
            {
                return phonemes.Replace(k_SecondaryStress.ToString(), "").Replace(k_PrimaryStress.ToString(), k_SecondaryStress.ToString());
            }

            if (stress >= 1 && !phonemes.Contains(k_PrimaryStress) && phonemes.Contains(k_SecondaryStress))
            {
                return phonemes.Replace(k_SecondaryStress.ToString(), k_PrimaryStress.ToString());
            }

            if (stress > 1 && !phonemes.Contains(k_PrimaryStress) && !phonemes.Contains(k_SecondaryStress) && k_Vowels.Any(phonemes.Contains))
            {
                return k_PrimaryStress + phonemes;
            }

            return phonemes;
        }
    }
}
