using System.Collections.Generic;

namespace Unity.ML.Tokenization.PostProcessors
{
    /// <summary>
    ///     Interlaces the primary and secondary sequences of tokens.
    /// </summary>
    public class DefaultPostProcessor : PostProcessorBase
    {
        /// <inheritdoc />
        protected override void PostProcessInternal(
            IEnumerable<int> tokensA,
            IEnumerable<int> tokensB,
            bool _,
            IOutput<int> output)
        {
            if(tokensA != null)
                output.Add(tokensA);

            if(tokensB != null)
                output.Add(tokensB);
        }

        /// <inheritdoc />
        public override int GetNumAddedTokens(bool _) => 0;
    }
}
