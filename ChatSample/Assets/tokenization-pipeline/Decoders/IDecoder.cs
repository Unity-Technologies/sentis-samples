using System.Collections.Generic;

namespace Unity.ML.Tokenization.Decoders
{
    public interface IDecoder
    {
        void Decode(IEnumerable<string> input, IOutput<string> output);
    }
}
