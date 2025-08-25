using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using UnityEngine.Assertions;

namespace Unity.ML.Tokenization.Tokenizers
{
    /// <summary>
    ///     Converts a string into a sequence of phonemes using the Piper Phonemizer strategy.
    /// </summary>
    public class PiperTokenizer : TokenizerBase
    {
        ITokenDefinition m_EndOfSequence;
        ITokenDefinition m_Separator;
        IVocabulary m_Vocabulary;

        /// <summary>
        ///     Creates a new Piper Tokenizer.
        /// </summary>
        /// <param name="vocabulary">
        ///     The map associating phoneme string representations with their ids.
        /// </param>
        /// <param name="separator">
        ///     Optional separator.
        /// </param>
        /// <param name="endOfSequence">
        ///     Optional end of sequence.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="vocabulary"/> cannot be <c>null</c>.
        /// </exception>
        public PiperTokenizer(
            [NotNull] IVocabulary vocabulary,
            string separator = "_",
            string endOfSequence = "$")
        {
            if (vocabulary is null)
                throw new ArgumentNullException(nameof(vocabulary));

            vocabulary.TryGetToken(separator, out var separatorToken);
            Assert.IsNotNull(separatorToken, $"Separator {separator} not found in vocabulary.");

            vocabulary.TryGetToken(endOfSequence, out var endOfSequenceToken);
            Assert.IsNotNull(
                endOfSequenceToken, $"End of Sequence {endOfSequence} not found in vocabulary.");

            m_Vocabulary = vocabulary;
            m_Separator = separatorToken;
            m_EndOfSequence = endOfSequenceToken;
        }

        /// <inheritdoc />
        protected override void TokenizeInternal(IEnumerable<SubString> input, IOutput<int> output)
        {
            foreach (var single in input)
            {
                output.Add(1);
                foreach (var phoneme in single)
                {
                    var found = m_Vocabulary.TryGetToken(char.ToString(phoneme), out var result) && !result.IsSpecial;
                    if (!found)
                        throw new KeyNotFoundException($"Phoneme: {phoneme} not found.");

                    output.Add(result.Id);
                    output.Add(m_Separator.Id);
                }

                output.Add(m_EndOfSequence.Id);
            }
        }

        /// <inheritdoc />
        protected override void
            DeTokenizeInternal(IEnumerable<int> input, bool skipSpecialTokens, IOutput<string> output)
        {
            foreach (var id in input)
            {
                var found = m_Vocabulary.TryGetToken(id, out var token);
                if (!found || skipSpecialTokens && token.IsSpecial)
                    continue;

                output.Add(token.Value);
            }
        }
    }
}
