using System.Collections.Generic;
using Unity.ML.Tokenization.Tokenizers;

namespace Unity.ML.Tokenization.PostProcessors
{
    /// <summary>
    ///     Transforms the sequences of tokens from the truncated output of <see cref="ITokenizer" />
    ///     and merges it into a single sequence.
    /// </summary>
    public interface IPostProcessor
    {
        /// <summary>
        ///     Determines the number of tokens that this <see cref="IPostProcessor" /> will add to
        ///     the sequence of tokens.
        /// </summary>
        /// <param name="isPair">
        ///     Tells if we want the number of added tokens for a pair of sequences of tokens
        ///     (<see langword="true" />), of a single sequence (<see langword="false" />).
        /// </param>
        /// <returns>
        ///     Number of tokens that this <see cref="IPostProcessor" /> will add to the sequence of
        ///     tokens
        /// </returns>
        int GetNumAddedTokens(bool isPair);

        void PostProcess(IEnumerable<IEnumerable<int>> sequenceA,
            IEnumerable<IEnumerable<int>> sequenceB, bool addSpecialTokens, IOutput<IEnumerable<int>> output);
    }
}
