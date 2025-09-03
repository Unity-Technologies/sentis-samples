using System.Collections.Generic;

namespace Unity.ML.Tokenization.Tokenizers
{
    partial class BpeTokenizer
    {
        /// <inheritdoc />
        internal class Utf8CharToTokenConverter
            : IOneToOneConverter<(SubString character, bool first, bool last), string>
        {
            /// <summary>
            ///     Stores previously generated string representations in order to save heap
            ///     allocations.
            /// </summary>
            readonly Dictionary<SubString, (string single, string first, string last, string inner)>
                m_Cache = new();

            /// <summary>
            ///     String prepended to a token value representing a piece of a word while not being
            ///     at the beginning of it.
            /// </summary>
            readonly string m_SubWordPrefix;

            /// <summary>
            ///     String appended to a token value representing a piece of a word while being at
            ///     the end of it.
            /// </summary>
            readonly string m_WordSuffix;

            /// <summary>
            ///     Initializes a new instance of the <see cref="Utf8CharToTokenConverter" /> type.
            /// </summary>
            /// <param name="subWordPrefix">
            ///     String prepended to a token value representing a piece of a word while not being
            ///     at the beginning of it.
            /// </param>
            /// <param name="wordSuffix">
            ///     String appended to a token value representing a piece of a word while being at
            ///     the end of it.
            /// </param>
            public Utf8CharToTokenConverter(string subWordPrefix, string wordSuffix)
            {
                m_SubWordPrefix = subWordPrefix;
                m_WordSuffix = wordSuffix;
            }

            /// <summary>
            ///     Gets the token value of a <paramref name="character" /> considering its position
            ///     in the word.
            /// </summary>
            /// <param name="character">
            ///     The character to get the token value from.
            /// </param>
            /// <param name="first">
            ///     Tells whether it is the first <paramref name="character" /> of the word.
            /// </param>
            /// <param name="last">
            ///     Tells whether it is the last <paramref name="character" /> of the word.
            /// </param>
            /// <returns>
            ///     The token value of the <paramref name="character" /> considering its position in
            ///     the word.
            /// </returns>
            public string GetToken(SubString character, bool first, bool last)
            {
                var found = m_Cache.TryGetValue(character, out var tuple);
                if (!found)
                {
                    // Generates all forms of the token representation of the character.
                    tuple = (single: $"{character}{m_WordSuffix}", first: $"{character}",
                        last: $"{m_SubWordPrefix}{character}{m_WordSuffix}",
                        inner: $"{m_SubWordPrefix}{character}");

                    m_Cache.Add(character.Apply(), tuple);
                }

                return first
                    ? last ? tuple.single : tuple.first
                    : last
                        ? tuple.last
                        : tuple.inner;
            }

            string IOneToOneConverter<(SubString character, bool first, bool last), string>.Convert(
                (SubString character, bool first, bool last) input) =>
                GetToken(input.character, input.first, input.last);
        }
    }
}
