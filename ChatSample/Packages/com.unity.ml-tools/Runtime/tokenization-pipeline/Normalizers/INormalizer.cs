namespace Unity.ML.Tokenization.Normalizers
{
    public interface INormalizer
    {
        SubString Normalize(SubString input);
    }
}
