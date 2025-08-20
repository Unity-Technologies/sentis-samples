namespace Unity.ML.Tokenization.Tokenizers
{
    partial class BpeTokenizer
    {
        /// <inheritdoc />
        internal class ByteToTokenConverter : IOneToOneConverter<byte, string>
        {
            /// <summary>
            ///     Stores previously generated string representations in order to save heap
            ///     allocations.
            /// </summary>
            static readonly string[] k_Cache = new string[256];

            /// <summary>
            ///     Initializes the string representation cache.
            /// </summary>
            static ByteToTokenConverter()
            {
                for (var i = 0; i <= 255; i++)
                    k_Cache[i] = $"<0x{i:X2}>";
            }

            /// <summary>
            ///     Gets a singleton of the <see cref="ByteToTokenConverter" /> type.
            /// </summary>
            public static ByteToTokenConverter Instance { get; } = new();

            /// <inheritdoc />
            public string Convert(byte @byte) => k_Cache[@byte];
        }
    }
}
