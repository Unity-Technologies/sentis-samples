using System.Collections.Generic;

namespace Unity.ML.Tokenization.Tokenizers
{
    /// <summary>
    ///     Turns an input string into a sequence of token ids.
    /// </summary>
    public interface ITokenizer
    {
        void Tokenize(IEnumerable<SubString> input, IOutput<int> output);
        void DeTokenize(IEnumerable<int> input, bool skipSpecialTokens, IOutput<string> output);
    }
}
