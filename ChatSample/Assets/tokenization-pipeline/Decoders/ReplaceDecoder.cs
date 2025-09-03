using System.Collections.Generic;

namespace Unity.ML.Tokenization.Decoders
{
    /// <summary>
    ///     Replace Decoder replaces certain char to another char from the tokens in the list.
    /// </summary>
    public class ReplaceDecoder : DecoderBase
    {
        readonly string m_Content;
        readonly string m_Pattern;

        internal ReplaceDecoder(string pattern, string content)
        {
            m_Pattern = pattern;
            m_Content = content;
        }

        /// <inheritdoc />
        protected override void DecodeInternal(IEnumerable<string> tokens, IOutput<string> output)
        {
            foreach (var token in tokens)
                output.Add(token.Replace(m_Pattern, m_Content));
        }
    }
}
