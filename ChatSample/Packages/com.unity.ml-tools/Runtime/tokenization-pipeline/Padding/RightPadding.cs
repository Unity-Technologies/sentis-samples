using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Padding
{
    /// <summary>
    ///     Pads the sequences of tokens by adding tokens to the right.
    /// </summary>
    public class RightPadding : DirectionalPaddingBase
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="RightPadding" /> type.
        /// </summary>
        /// <param name="paddingSizeProvider">
        ///     Computes the target length of the padded sequences.
        /// </param>
        /// <param name="padToken">
        ///     The token to use to pad a sequence of token.
        /// </param>
        public RightPadding(
            [NotNull] IPaddingSizeProvider paddingSizeProvider,
            ITokenDefinition padToken) : base(paddingSizeProvider, padToken)
        { }

        /// <inheritdoc />
        protected override IEnumerable<(int id, int attention)> Pad(
            IReadOnlyCollection<int> input,
            int padSize)
        {
            foreach (var token in input)
                yield return (token, 1);

            for (int i = 0, limit = padSize - input.Count; i < limit; i++)
                yield return (PadToken.Id, 0);
        }
    }
}
