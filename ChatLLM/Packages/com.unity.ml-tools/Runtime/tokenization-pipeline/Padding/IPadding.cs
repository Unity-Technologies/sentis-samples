using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Padding
{
    /// <summary>
    ///     Applies a padding to sequences of tokens.
    /// </summary>
    public interface IPadding
    {
        /// <summary>
        ///     Applies a padding to sequences of tokens.
        /// </summary>
        /// <param name="input">
        ///     The sequences of tokens to pad.
        /// </param>
        /// <param name="output">
        ///     The target container of padded sequences of tokens.
        /// </param>
        void Pad(
            [NotNull] IEnumerable<IEnumerable<int>> input,
            [NotNull] IOutput<IEnumerable<(int id, int attention)>> output);
    }
}
