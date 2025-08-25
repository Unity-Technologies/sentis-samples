using System.Collections.Generic;

namespace Unity.ML.Tokenization.Decoders
{
    /// <summary>
    ///     Fuse Decoder combine the tokens in list into a single large token.
    /// </summary>
    public class FuseDecoder : DecoderBase
    {
        /// <inheritdoc />
        protected override void DecodeInternal(IEnumerable<string> input, IOutput<string> output) =>
            output.Add(string.Concat(input));
    }
}
