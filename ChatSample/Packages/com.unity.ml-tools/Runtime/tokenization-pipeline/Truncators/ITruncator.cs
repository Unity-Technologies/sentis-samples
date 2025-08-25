using System.Collections.Generic;
using JetBrains.Annotations;
using Unity.ML.Tokenization.PostProcessors;

namespace Unity.ML.Tokenization.Truncators
{
    /// <summary>
    ///     Splits sequences of tokens into smaller collection of sequences.
    /// </summary>
    public interface ITruncator
    {
        /// <summary>
        ///     Splits sequences of tokens into smaller collection of sequences.
        /// </summary>
        /// <param name="inputA">
        ///     The primary sequence of tokens (mandatory).
        /// </param>
        /// <param name="inputB">
        ///     The optional secondary sequence of tokens.
        /// </param>
        /// <param name="numAddedTokens">
        ///     The number of tokens that the <see cref="IPostProcessor" /> steps will add.
        /// </param>
        /// <param name="outputA">
        ///     The target container of the truncated subsequences of <paramref name="inputA" />.
        /// </param>
        /// <param name="outputB">
        ///     The target container of the truncated subsequences of <paramref name="inputB" />.
        /// </param>
        void Truncate(
            [NotNull] IEnumerable<int> inputA,
            [CanBeNull] IEnumerable<int> inputB,
            int numAddedTokens,
            [NotNull] IOutput<IEnumerable<int>> outputA,
            [CanBeNull] IOutput<IEnumerable<int>> outputB);
    }
}
