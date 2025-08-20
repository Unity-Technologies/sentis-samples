using System.Collections.Generic;

namespace Unity.ML.Tokenization.Tokenizers
{
    partial class BpeTokenizer
    {
        /// <summary>
        ///     The merger type used when no merge rules are given to the <see cref="BpeTokenizer" />
        ///     constructor.
        ///     It is a typical passthrough which doesn't modify the input sequence of tokens.
        /// </summary>
        internal class DefaultMerger : IManyToManyConverter<ITokenDefinition, ITokenDefinition>
        {
            public void Convert(
                IEnumerable<ITokenDefinition> input,
                IOutput<ITokenDefinition> output) =>
                output.Add(input);
        }
    }
}
