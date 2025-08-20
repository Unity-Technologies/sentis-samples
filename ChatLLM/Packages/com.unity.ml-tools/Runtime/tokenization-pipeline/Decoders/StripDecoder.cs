using System;
using System.Collections.Generic;

namespace Unity.ML.Tokenization.Decoders
{
    /// <summary>
    ///     Strip Decoder removes certain char from the substring of the token in the list.
    /// </summary>
    public class StripDecoder : DecoderBase
    {
        readonly char m_Content;
        readonly int m_Stop;
        readonly int m_Start;

        /// <summary>
        ///     Initializes a new instance of the <see cref="StripDecoder"/> type.
        /// </summary>
        /// <param name="content">
        ///     The character to remove.
        /// </param>
        /// <param name="start">
        ///     Lowerbound of the portion of the input to keep.
        /// </param>
        /// <param name="stop">
        ///     Upperbound of the portion of the input to keep.
        /// </param>
        public StripDecoder(char content, int start, int stop)
        {
            m_Content = content;
            m_Start = start;
            m_Stop = stop;
        }

        /// <inheritdoc />
        protected override void DecodeInternal(IEnumerable<string> tokens, IOutput<string> output)
        {
            foreach (var token in tokens)
            {
                var startCut = 0;
                for (var i = 0; i < Math.Min(m_Start, token.Length); i++)
                    if (token[i] == m_Content)
                        startCut = i + 1;
                    else
                        break;

                var stopCut = token.Length;
                for (var i = 0; i < m_Stop; i++)
                {
                    var index = token.Length - i - 1;
                    if (index < 0)
                        break;

                    if (token[index] == m_Content)
                        stopCut = index;
                    else
                        break;
                }

                output.Add(token.Substring(startCut, stopCut - startCut));
            }
        }
    }
}
