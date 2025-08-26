using System.Collections.Generic;

namespace Unity.ML.Tokenization.PostProcessors
{
    public class RobertaPostProcessor : PostProcessorBase
    {
        ITokenDefinition m_SepToken;
        ITokenDefinition m_ClsToken;

        public RobertaPostProcessor(ITokenDefinition sep, ITokenDefinition cls)
        {
            m_SepToken = sep;
            m_ClsToken = cls;
        }

        protected override void PostProcessInternal(
            IEnumerable<int> tokensA,
            IEnumerable<int> tokensB,
            bool addSpecialTokens,
            IOutput<int> output)
        {
            AddSequence(tokensA, addSpecialTokens, output);

            if(tokensB is not null)
                AddSequence(tokensB, addSpecialTokens, output);

            return;

            void AddSequence(IEnumerable<int> tokens, bool addSpecial, IOutput<int> o)
            {
                if (addSpecial)
                {
                    output.Add(m_ClsToken.Id);
                }

                output.Add(tokens);

                if (addSpecial)
                {
                    o.Add(m_SepToken.Id);
                }
            }
        }

        public override int GetNumAddedTokens(bool isPair) => isPair ? 4 : 2;
    }
}
