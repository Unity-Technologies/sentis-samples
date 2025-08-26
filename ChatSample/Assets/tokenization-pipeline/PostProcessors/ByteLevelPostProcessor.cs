using System.Collections.Generic;

namespace Unity.ML.Tokenization.PostProcessors
{
    public class ByteLevelPostProcessor : PostProcessorBase
    {
        bool m_TrimOffsets = false;

        public ByteLevelPostProcessor(bool trimOffsets = false) => m_TrimOffsets = trimOffsets;

        protected override void PostProcessInternal(
            IEnumerable<int> tokensA,
            IEnumerable<int> tokensB,
            bool addSpecialTokens,
            IOutput<int> output)
        {
            if (m_TrimOffsets)
            {
                // TODO> trim offsets
            }

            if(tokensA != null)
                output.Add(tokensA);

            if(tokensB != null)
                output.Add(tokensB);
        }

        public override int GetNumAddedTokens(bool isPair) => 0;
    }
}
