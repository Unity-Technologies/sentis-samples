using Unity.ML.Tokenization.Tokenizers;

namespace Unity.ML.Tokenization.PreTokenizers
{
    /// <summary>
    ///     Pre-cuts the input <see cref="string" /> into smaller parts.
    ///     Those parts will be passed to the <see cref="ITokenizer" /> for tokenization.
    /// </summary>
    public interface IPreTokenizer
    {
        void PreTokenize(SubString input, IOutput<SubString> output);
    }
}
