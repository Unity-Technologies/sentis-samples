namespace Unity.ML.Tokenization.PreTokenizers
{
    /// <summary>
    ///     Default placeholder implementation of a pretokenizer.
    ///     Does not pre-cut the input.
    /// </summary>
    public class DefaultPreTokenizer : PreTokenizerBase
    {
        /// <inheritdoc />
        protected override void PreTokenizeInternal(SubString input, IOutput<SubString> output) =>
            output.Add(input);
    }
}
