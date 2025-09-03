using System.Collections.Generic;

namespace Unity.ML.Tokenization.Decoders
{
    /// <summary>
    ///     Default decoder.
    ///     Doe not change the input chunks.
    /// </summary>
    public class DefaultDecoder : DecoderBase
    {
        /// <inheritdoc />
        protected override void DecodeInternal(IEnumerable<string> input, IOutput<string> output) => output.Add(input);
    }
}
