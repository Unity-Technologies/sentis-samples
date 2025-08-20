using System.Collections.Generic;

namespace Unity.ML.Tokenization.Tokenizers
{
    partial class BpeTokenizer
    {
        /// <summary>
        ///     Default char->token converter for <see cref="BpeTokenizer" />.
        /// </summary>
        internal class InternalTokenizer : IOneToManyConverter<SubString, ITokenDefinition>
        {
            /// <summary>
            ///     The helper instance responsible for turning a single <see cref="byte" /> into a
            ///     token value.
            /// </summary>
            IOneToOneConverter<byte, string> m_ByteToToken;

            /// <summary>
            ///     The helper instance responsible for turning a <see cref="char" /> into a
            ///     <see cref="byte" /> array.
            /// </summary>
            IOneToOneConverter<SubString, IReadOnlyCollection<byte>> m_CharToByte;

            /// <summary>
            ///     Splits the input string into a sequence of unicode characters.
            /// </summary>
            IOneToManyConverter<SubString, SubString> m_StringToUtfChars;

            /// <summary>
            ///     The helper instance responsible for turning a single <see cref="char" /> into a
            ///     token value, considering its position in the word.
            /// </summary>
            IOneToOneConverter<(SubString character, bool first, bool last), string> m_CharToToken;

            /// <summary>
            ///     Configuration for unknown token.
            ///     <ul>
            ///         <li><c>token</c>: the token to use to represent an unknown token.</li>
            ///         <li>
            ///             <c>fuse</c>: tells whether a sequence of unknown tokens should be merged
            ///             into a single token.
            ///         </li>
            ///         <li>
            ///             <c>byteFallback</c>: Tells whether an unknown token should be
            ///             represented by its byte array representation.
            ///         </li>
            ///     </ul>
            /// </summary>
            (ITokenDefinition token, bool fuse, bool byteFallback) m_Unknown;

            /// <summary>
            ///     The value->ids map of token definitions.
            /// </summary>
            IVocabulary m_Vocabulary;

            /// <summary>
            ///     Initializes a new instance of the <see cref="InternalTokenizer" /> type.
            /// </summary>
            /// <param name="vocabulary">
            ///     The value->ids map of token definitions.
            /// </param>
            /// <param name="unknown">
            ///     Configuration for unknown token.
            ///     <ul>
            ///         <li><c>token</c>: the token to use to represent an unknown token.</li>
            ///         <li>
            ///             <c>fuse</c>: tells whether a sequence of unknown tokens should be merged
            ///             into a single token.
            ///         </li>
            ///         <li>
            ///             <c>byteFallback</c>: Tells whether an unknown token should be
            ///             represented by its byte array representation.
            ///         </li>
            ///     </ul>
            /// </param>
            /// <param name="decorator">
            ///     Configuration for the token string representation.
            ///     <ul>
            ///         <li>
            ///             <c>subWordPrefix</c>: string prepended to a token value representing a
            ///             piece of a word while not being at the beginning of it.
            ///         </li>
            ///         <li>
            ///             <c>wordSuffix</c>: string appended to a token value representing a piece
            ///             of a word which being at the end of it.
            ///         </li>
            ///     </ul>
            /// </param>
            public InternalTokenizer(
                IVocabulary vocabulary,
                (ITokenDefinition token, bool fuse, bool byteFallback) unknown,
                (string subWordPrefix, string wordSuffix) decorator)
            {
                Init(
                    new OneToOneCachedConverter<SubString, IReadOnlyCollection<byte>>(new UtfCharToByteConverter()),
                    ByteToTokenConverter.Instance,
                    Utf8CharSplitter.Instance,
                    new Utf8CharToTokenConverter(
                        decorator.subWordPrefix,
                        decorator.wordSuffix),
                    vocabulary, unknown);
            }

            /// <summary>
            ///     Constructor for unit testing purpose.
            /// </summary>
            /// <param name="charToByte">
            ///     The helper instance responsible for turning a <see cref="char" /> into a
            ///     <see cref="byte" /> array.
            /// </param>
            /// <param name="byteToToken">
            ///     The helper instance responsible for turing a single <see cref="byte" /> into a
            ///     token value.
            /// </param>
            /// <param name="stringToUtf8Char">
            /// </param>
            /// <param name="charToToken">
            ///     The helper instance responsible for turning a single <see cref="char" /> into a
            ///     token value, considering its position in the word.
            /// </param>
            /// <param name="vocabulary">
            ///     The value->ids map of token definitions.
            /// </param>
            /// <param name="unknown">
            ///     Configuration for unknown token.
            ///     <ul>
            ///         <li><c>token</c>: the token to use to represent an unknown token.</li>
            ///         <li>
            ///             <c>fuse</c>: tells whether a sequence of unknown tokens should be merged
            ///             into a single token.
            ///         </li>
            ///         <li>
            ///             <c>byteFallback</c>: Tells whether an unknown token should be
            ///             represented by its byte array representation.
            ///         </li>
            ///     </ul>
            /// </param>
            internal InternalTokenizer(
                IOneToOneConverter<SubString, IReadOnlyCollection<byte>> charToByte,
                IOneToOneConverter<byte, string> byteToToken,
                IOneToManyConverter<SubString, SubString> stringToUtf8Char,
                IOneToOneConverter<(SubString, bool, bool), string> charToToken,
                IVocabulary vocabulary,
                (ITokenDefinition token, bool fuse, bool byteFallback) unknown)
            {
                Init(charToByte, byteToToken, stringToUtf8Char, charToToken, vocabulary, unknown);
            }

            /// <summary>
            ///     Gets the sequence of token ids for each <c>char</c> of the
            ///     <paramref name="input" />.
            /// </summary>
            /// <param name="input">
            ///     The <c>char</c> sequence.
            /// </param>
            /// <param name="output">
            /// </param>
            /// <returns>
            ///     The sequence of token ids.
            /// </returns>
            public void Convert(SubString input, IOutput<ITokenDefinition> output)
            {
                var previousIsUnk = false;

                using var _ = PoolUtility.GetOutputOfSubStringPool().Get(out var utfChars);
                m_StringToUtfChars.Convert(input, utfChars);

                for (var i = 0; i < utfChars.Count; i++)
                {
                    var @char = utfChars[i];

                    var repr = m_CharToToken.Convert((@char, i == 0, i == utfChars.Count - 1));

                    // token representation found
                    if (m_Vocabulary.TryGetToken(repr, out var definition) && !definition.IsSpecial)
                    {
                        output.Add(definition);
                        previousIsUnk = false;
                    }

                    // token representation not found, but byte fallback allowed
                    else if (m_Unknown.byteFallback)
                    {
                        previousIsUnk = false;
                        var bytes = m_CharToByte.Convert(@char);
                        foreach (var b in bytes)
                            if (m_Vocabulary.TryGetToken(m_ByteToToken.Convert(b), out definition) && !definition.IsSpecial)
                                output.Add(definition);
                        // else should with warn?
                    }

                    // unknown
                    else if (!m_Unknown.fuse || !previousIsUnk)
                    {
                        if (m_Unknown.token is not null)
                            output.Add(m_Unknown.token);

                        previousIsUnk = true;
                    }
                }
            }

            /// <summary>
            ///     Initializes the <see cref="InternalTokenizer" /> instance.
            /// </summary>
            /// <param name="charToByte">
            ///     The helper instance responsible for turning a <see cref="char" /> into a
            ///     <see cref="byte" /> array.
            /// </param>
            /// <param name="byteToToken">
            ///     The helper instance responsible for turing a single <see cref="byte" /> into a
            ///     token value.
            /// </param>
            /// <param name="stringToUtf8Chars">
            /// </param>
            /// <param name="charToToken">
            ///     The helper instance responsible for turning a single <see cref="char" /> into a
            ///     token value, considering its position in the word.
            /// </param>
            /// <param name="vocabulary">
            ///     The value->ids map of token definitions.
            /// </param>
            /// <param name="unknown">
            ///     Configuration for unknown token.
            ///     <ul>
            ///         <li><c>token</c>: the token to use to represent an unknown token.</li>
            ///         <li>
            ///             <c>fuse</c>: tells whether a sequence of unknown tokens should be merged
            ///             into a single token.
            ///         </li>
            ///         <li>
            ///             <c>byteFallback</c>: Tells whether an unknown token should be
            ///             represented by its byte array representation.
            ///         </li>
            ///     </ul>
            /// </param>
            void Init(
                IOneToOneConverter<SubString, IReadOnlyCollection<byte>> charToByte,
                IOneToOneConverter<byte, string> byteToToken,
                IOneToManyConverter<SubString, SubString> stringToUtf8Chars,
                IOneToOneConverter<(SubString, bool, bool), string> charToToken,
                IVocabulary vocabulary,
                (ITokenDefinition token, bool fuse, bool byteFallback) unknown)
            {
                m_CharToByte = charToByte;
                m_ByteToToken = byteToToken;
                m_StringToUtfChars = stringToUtf8Chars;
                m_CharToToken = charToToken;

                m_Vocabulary = vocabulary;
                m_Unknown = unknown;
            }
        }
    }
}
