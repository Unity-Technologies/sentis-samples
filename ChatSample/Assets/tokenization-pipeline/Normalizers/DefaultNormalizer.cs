namespace Unity.ML.Tokenization.Normalizers
{
    public class DefaultNormalizer : INormalizer
    {
        public SubString Normalize(SubString input) => input;
    }
}
