namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Separates each UTF-8 character from a <see cref="SubString" /> input.
    /// </summary>
    class Utf8CharSplitter : IOneToManyConverter<SubString, SubString>
    {
        /// <summary>
        ///     Gets the singleton instance of the <see cref="Utf8CharSplitter" /> type.
        /// </summary>
        public static Utf8CharSplitter Instance { get; } = new();

        public void Convert(SubString input, IOutput<SubString> output)
        {
            var (source, offset, length) = input;
            var to = offset + length;

            while (offset < to)
            {
                if (!char.IsSurrogate(source[offset]))
                {
                    output.Add(new SubString(source, offset, 1));
                    offset++;
                }

                // Simple character
                else
                {
                    var end = offset + 1;
                    while (end < to && char.IsSurrogate(source[end]))
                        end++;
                    output.Add(SubString.FromTo(source, offset, end));
                    offset = end;
                }
            }
        }

        void IOneToManyConverter<SubString, SubString>.Convert(
            SubString input,
            IOutput<SubString> output) =>
            Convert(input, output);
    }
}
