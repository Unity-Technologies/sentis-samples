using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.ML.Tokenization.Tokenizers;

namespace Unity.ML.Tokenization.Decoders
{
    /// <summary>
    ///     Base implementation for token decoding.
    /// </summary>
    public abstract class DecoderBase : IDecoder
    {
        /// <summary>
        ///     Decodes string chunks from <see cref="ITokenizer"/> to strings and postprocess the
        ///     string.
        /// </summary>
        /// <param name="input">
        ///     Sequence of tokens to decode.
        /// </param>
        /// <param name="output">
        ///     String chunks.
        /// </param>
        public void Decode([NotNull] IEnumerable<string> input, IOutput<string> output)
        {
            if (input is null)
                throw new ArgumentNullException(nameof(input));

            DecodeInternal(input, output);
        }

        /// <inheritdoc cref="Decode" />
        protected abstract void DecodeInternal(IEnumerable<string> input, IOutput<string> output);

        void IDecoder.Decode(IEnumerable<string> input, IOutput<string> output) =>
            Decode(input, output);
    }
}
