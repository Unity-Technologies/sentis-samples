using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.PostProcessors
{
    /// <summary>
    ///     Base implementation for builtin <see cref="IPostProcessor" /> types.
    ///     Runs the necessary parameter validation before calling
    ///     <see cref="PostProcessInternal" />.
    /// </summary>
    public abstract class PostProcessorBase : IPostProcessor
    {
        /// <summary>
        ///     Returns the number of token added by this post processor.
        /// </summary>
        /// <param name="isPair">
        ///     Tells whether the post processor is applied to a pair of sequences.
        /// </param>
        /// <returns>
        ///     The number of token added by this post processor.
        /// </returns>
        public abstract int GetNumAddedTokens(bool isPair);

        /// <summary>
        ///     Transforms the sequence of tokens.
        /// </summary>
        /// <param name="sequenceA">
        ///     The primary sequence of tokens (mandatory).
        /// </param>
        /// <param name="sequenceB">
        ///     An optional sequence of tokens.
        /// </param>
        /// <param name="addSpecialTokens">
        ///     Tells whether to add the special tokens when transforming.
        /// </param>
        /// <param name="output">
        ///     The target container to receive the processed sequence.
        /// </param>
        public void PostProcess(
            [NotNull] IEnumerable<IEnumerable<int>> sequenceA,
            [CanBeNull] IEnumerable<IEnumerable<int>> sequenceB,
            bool addSpecialTokens,
            IOutput<IEnumerable<int>> output)
        {
            if (sequenceA == null)
                throw new ArgumentNullException(nameof(sequenceA));

            using var enumA = sequenceA.GetEnumerator();
            using var enumB = sequenceB?.GetEnumerator();

            var (nextA, nextB) = (enumA.MoveNext(), enumB?.MoveNext() ?? false);

            while (nextA || nextB)
            {
                var tokensA = nextA ? enumA.Current : default;
                var tokensB = nextB ? enumB!.Current : default;

                using var postProcessedHandle =
                    PoolUtility.GetOutputOfIntPool().Get(out var postProcessed);

                PostProcessInternal(tokensA, tokensB, addSpecialTokens, postProcessed);
                output.Add(postProcessed.ToArray());

                (nextA, nextB) = (enumA.MoveNext(), enumB?.MoveNext() ?? false);
            }
        }

        /// <inheritdoc cref="PostProcess" />
        protected abstract void PostProcessInternal(
            [CanBeNull] IEnumerable<int> tokensA,
            [CanBeNull] IEnumerable<int> tokensB,
            bool addSpecialTokens,
            IOutput<int> output);

        int IPostProcessor.GetNumAddedTokens(bool isPair) => GetNumAddedTokens(isPair);

        void IPostProcessor.PostProcess(
            IEnumerable<IEnumerable<int>> sequenceA,
            IEnumerable<IEnumerable<int>> sequenceB,
            bool addSpecialTokens,
            IOutput<IEnumerable<int>> output) =>
            PostProcess(sequenceA, sequenceB, addSpecialTokens, output);
    }
}
