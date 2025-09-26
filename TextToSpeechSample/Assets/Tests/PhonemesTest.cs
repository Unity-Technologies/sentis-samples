using NUnit.Framework;
using Unity.InferenceEngine.Samples.TTS.Inference;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Tests
{
    public class PhonemesTest
    {
        const string k_TestText = "Papa";

        [Test]
        public void TokenizePhonemes()
        {
            var tokens = MisakiSharp.TokenizeGraphemes(k_TestText);
            Assert.IsNotNull(tokens, "Tokenization returned null.");
            Assert.IsNotEmpty(tokens, "Tokenization returned empty array.");

            Debug.Log($"Text: {k_TestText}");
            Debug.Log($"Tokens: {string.Join(", ", tokens)}");
        }

        [Test]
        public void TextToPhonemesBasicWords()
        {
            var phonemes = MisakiSharp.TextToPhonemes("hello world");
            Assert.IsNotNull(phonemes, "TextToPhonemes returned null.");
            Assert.IsNotEmpty(phonemes, "TextToPhonemes returned empty string.");

            Debug.Log($"Text: 'hello world' -> Phonemes: '{phonemes}'");

            // Should contain recognizable phoneme patterns
            Assert.IsTrue(phonemes.Length > 5, "Phonemes string too short for input text.");
        }

        [Test]
        public void TextToPhonemesContractions()
        {
            var testCases = new[]
            {
                "don't",
                "can't",
                "you've",
                "I'm",
                "it's"
            };

            foreach (var testCase in testCases)
            {
                var phonemes = MisakiSharp.TextToPhonemes(testCase);
                Assert.IsNotNull(phonemes, $"TextToPhonemes returned null for '{testCase}'.");
                Assert.IsNotEmpty(phonemes, $"TextToPhonemes returned empty string for '{testCase}'.");

                Debug.Log($"Contraction: '{testCase}' -> Phonemes: '{phonemes}'");
            }
        }

        [Test]
        public void TextToPhonemesPossessives()
        {
            var testCases = new[]
            {
                "John's",
                "cat's",
                "Today's",
                "children's"
            };

            foreach (var testCase in testCases)
            {
                var phonemes = MisakiSharp.TextToPhonemes(testCase);
                Assert.IsNotNull(phonemes, $"TextToPhonemes returned null for '{testCase}'.");
                Assert.IsNotEmpty(phonemes, $"TextToPhonemes returned empty string for '{testCase}'.");

                Debug.Log($"Possessive: '{testCase}' -> Phonemes: '{phonemes}'");
            }
        }

        [Test]
        public void TextToPhonemesPunctuation()
        {
            var testText = "Hello, world! How are you? I'm fine.";
            var phonemes = MisakiSharp.TextToPhonemes(testText);

            Assert.IsNotNull(phonemes, "TextToPhonemes returned null for punctuated text.");
            Assert.IsNotEmpty(phonemes, "TextToPhonemes returned empty string for punctuated text.");

            Debug.Log($"Punctuated text: '{testText}' -> Phonemes: '{phonemes}'");

            // Should contain punctuation markers
            Assert.IsTrue(phonemes.Contains(","), "Should contain comma phoneme.");
            Assert.IsTrue(phonemes.Contains("."), "Should contain period phoneme.");
        }

        [Test]
        public void TextToPhonemesNumbers()
        {
            var testCases = new[]
            {
                "1",
                "42",
                "100",
                "2023"
            };

            foreach (var testCase in testCases)
            {
                var phonemes = MisakiSharp.TextToPhonemes(testCase);
                Assert.IsNotNull(phonemes, $"TextToPhonemes returned null for number '{testCase}'.");
                Assert.IsNotEmpty(phonemes, $"TextToPhonemes returned empty string for number '{testCase}'.");

                Debug.Log($"Number: '{testCase}' -> Phonemes: '{phonemes}'");
            }
        }

        [Test]
        public void TextToPhonemesUnicodeQuotes()
        {
            var testText = "\"Hello,\" she said, 'How are you?'";
            var phonemes = MisakiSharp.TextToPhonemes(testText);

            Assert.IsNotNull(phonemes, "TextToPhonemes returned null for Unicode quotes.");
            Assert.IsNotEmpty(phonemes, "TextToPhonemes returned empty string for Unicode quotes.");

            Debug.Log($"Unicode quotes: '{testText}' -> Phonemes: '{phonemes}'");
        }

        [Test]
        public void TextToPhonemesMorphology()
        {
            var testCases = new[]
            {
                "cats",      // plural
                "walked",    // past tense
                "running",   // present participle
                "books",     // plural
                "played"     // past tense
            };

            foreach (var testCase in testCases)
            {
                var phonemes = MisakiSharp.TextToPhonemes(testCase);
                Assert.IsNotNull(phonemes, $"TextToPhonemes returned null for '{testCase}'.");
                Assert.IsNotEmpty(phonemes, $"TextToPhonemes returned empty string for '{testCase}'.");

                Debug.Log($"Morphology: '{testCase}' -> Phonemes: '{phonemes}'");
            }
        }

        [Test]
        public void TextToPhonemesCapitalization()
        {
            var testCases = new[]
            {
                ("hello", "Hello"),
                ("world", "WORLD"),
                ("test", "Test")
            };

            foreach (var (lower, upper) in testCases)
            {
                var lowerPhonemes = MisakiSharp.TextToPhonemes(lower);
                var upperPhonemes = MisakiSharp.TextToPhonemes(upper);

                Assert.IsNotNull(lowerPhonemes, $"TextToPhonemes returned null for '{lower}'.");
                Assert.IsNotNull(upperPhonemes, $"TextToPhonemes returned null for '{upper}'.");

                Debug.Log($"Case comparison: '{lower}' -> '{lowerPhonemes}' | '{upper}' -> '{upperPhonemes}'");
            }
        }

        [Test]
        public void TokenizeGraphemesConsistency()
        {
            var testText = "The quick brown fox jumps over the lazy dog.";

            // Test multiple calls return same result
            var tokens1 = MisakiSharp.TokenizeGraphemes(testText);
            var tokens2 = MisakiSharp.TokenizeGraphemes(testText);

            Assert.AreEqual(tokens1.Length, tokens2.Length, "TokenizeGraphemes should return consistent length.");

            for (int i = 0; i < tokens1.Length; i++)
            {
                Assert.AreEqual(tokens1[i], tokens2[i], $"Token at index {i} should be consistent.");
            }

            Debug.Log($"Consistency test passed for {tokens1.Length} tokens.");
        }

        [Test]
        public void EmptyAndNullInputHandling()
        {
            // Test empty string
            var emptyTokens = MisakiSharp.TokenizeGraphemes("");
            Assert.IsNotNull(emptyTokens, "Should return non-null array for empty string.");
            Assert.AreEqual(0, emptyTokens.Length, "Should return empty array for empty string.");

            var emptyPhonemes = MisakiSharp.TextToPhonemes("");
            Assert.IsNotNull(emptyPhonemes, "Should return non-null string for empty input.");
            Assert.AreEqual("", emptyPhonemes, "Should return empty string for empty input.");

            // Test null string
            var nullTokens = MisakiSharp.TokenizeGraphemes(null);
            Assert.IsNotNull(nullTokens, "Should return non-null array for null string.");
            Assert.AreEqual(0, nullTokens.Length, "Should return empty array for null string.");

            var nullPhonemes = MisakiSharp.TextToPhonemes(null);
            Assert.IsNotNull(nullPhonemes, "Should return non-null string for null input.");
            Assert.AreEqual("", nullPhonemes, "Should return empty string for null input.");

            Debug.Log("Empty and null input handling tests passed.");
        }
    }
}
