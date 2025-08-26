namespace Unity.ML.Tokenization.PreTokenizers
{
    partial class ByteLevelPreTokenizer
    {
        internal class DefaultSplitter : IOneToManyConverter<SubString, SubString>
        {
            public void Convert(SubString input, IOutput<SubString> output)
            {
                if (input.IsNull)
                    return;

                output.Add(input);
            }
        }
    }
}
