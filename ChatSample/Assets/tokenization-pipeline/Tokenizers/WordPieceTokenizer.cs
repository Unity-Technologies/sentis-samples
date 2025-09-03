using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Tokenizers
{
    /// <summary>
    ///     Turns an input string into a sequence of token ids using the Word Piece strategy.
    /// </summary>
    public class WordPieceTokenizer : TokenizerBase
    {
        IVocabulary m_Vocabulary;

        /// <summary>
        ///     Initializes a new instance of the <see cref="WordPieceTokenizer" /> type.
        /// </summary>
        /// <param name="vocabulary">
        ///     The value->ids map for token definitions.
        /// </param>
        /// <param name="unknownToken">
        ///     The value of the unknown token.
        /// </param>
        /// <param name="continuingSubWordPrefix">
        ///     The prefix to add to inner subwords (not at the beginning of a word).
        /// </param>
        /// <param name="maxInputCharsPerWord">
        ///     Maximum length of a tokenizable word.
        /// </param>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="maxInputCharsPerWord" /> is negative or <c>0</c>.
        /// </exception>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="vocabulary" /> cannot be <see langword="null" />.
        /// </exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="unknownToken" /> not found in the vocabulary.
        /// </exception>
        public WordPieceTokenizer(
            [NotNull] IVocabulary vocabulary,
            SubString unknownToken,
            string continuingSubWordPrefix = "##",
            int maxInputCharsPerWord = 100)
        {
            if (maxInputCharsPerWord <= 0)
                throw new ArgumentOutOfRangeException(
                    nameof(maxInputCharsPerWord), maxInputCharsPerWord, null);

            if (vocabulary is null)
                throw new ArgumentNullException(nameof(vocabulary));

            if (unknownToken.IsEmpty)
                throw new ArgumentNullException(nameof(unknownToken), "Cannot be empty");

            if (!vocabulary.TryGetToken(unknownToken, out var unk) || !unk.IsSpecial)
                throw new ArgumentException(
                    $"Cannot find the unknown token {unknownToken} in the vocabulary",
                    nameof(unknownToken));
            m_Vocabulary = vocabulary;

            ContinuingSubWordPrefix = continuingSubWordPrefix;
            MaxInputCharsPerWord = maxInputCharsPerWord;
            UnknownToken = unk;
        }

        /// <summary>
        ///     The definition to use for unknown token.
        /// </summary>
        public ITokenDefinition UnknownToken { get; }

        /// <summary>
        ///     The prefix to add to inner subwords (not at the beginning of a word).
        /// </summary>
        public string ContinuingSubWordPrefix { get; }

        /// <summary>
        ///     Maximum length of a tokenizable word.
        /// </summary>
        public int MaxInputCharsPerWord { get; }

        /// <inheritdoc />
        protected override void TokenizeInternal(IEnumerable<SubString> inputs, IOutput<int> output)
        {
            foreach (var src in inputs)
            {
                var input = src;

                if (input.UtfLength > MaxInputCharsPerWord)
                {
                    output.Add(UnknownToken.Id);
                    return;
                }

                using var _ = PoolUtility.GetListOfIntPool().Get(out var tokens);

                SubString prefix = ContinuingSubWordPrefix;

                var @continue = false;
                while (input.UtfLength > 0)
                {
                    var searchInput = input;
                    var utfLength = searchInput.UtfLength;

                    var found = m_Vocabulary.TryGetToken(
                            searchInput, out var result,
                            prefix: @continue ? prefix : (SubString?) null)
                        && !result.IsSpecial;

                    while (!found && utfLength > 1)
                    {
                        utfLength--;
                        searchInput = input.UtfSub(0, utfLength);

                        found = m_Vocabulary.TryGetToken(
                                searchInput, out result,
                                prefix: @continue ? prefix : (SubString?) null)
                            && !result.IsSpecial;
                    }

                    if (!found)
                    {
                        output.Add(UnknownToken.Id);
                        return;
                    }

                    tokens.Add(result.Id);

                    if (input.UtfLength - utfLength == 0)
                        break;

                    input = input.UtfSub(utfLength);
                    @continue = true;
                }

                output.Add(tokens);
            }
        }

        /// <inheritdoc />
        protected override void DeTokenizeInternal(
            IEnumerable<int> input,
            bool skipSpecialTokens,
            IOutput<string> output)
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
