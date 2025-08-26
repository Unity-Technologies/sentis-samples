using System.Collections.Generic;

namespace Unity.ML.Tokenization.Padding
{
    /// <summary>
    ///     Gives a fixed padding length.
    /// </summary>
    public class FixedPaddingSizeProvider : IPaddingSizeProvider
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="FixedPaddingSizeProvider" /> type.
        /// </summary>
        /// <param name="size">
        ///     The target padding size.
        /// </param>
        public FixedPaddingSizeProvider(int size) => Size = size;

        /// <summary>
        ///     The target padding size.
        /// </summary>
        public int Size { get; }

        int IPaddingSizeProvider.GetPaddingSize(IEnumerable<int> _) => Size;
    }
}
