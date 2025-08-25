using System.Collections.Generic;

namespace Unity.ML.Tokenization.Truncators
{
    /// <summary>
    ///     Generates a sequence of <see cref="Range" />s.
    /// </summary>
    public interface IRangeGenerator
    {
        /// <summary>
        ///     Generates a sequence of chunks based on the <paramref name="length" /> of the source,
        ///     the maximum size of those chunks, and a <paramref name="stride" />.
        /// </summary>
        /// <param name="length">
        ///     The length of the source.
        /// </param>
        /// <param name="rangeMaxLength">
        ///     The maximum size of the resulting chunks.
        /// </param>
        /// <param name="stride">
        ///     The stride controls how the generator goes along the <paramref name="length" /> of
        ///     the source.
        /// </param>
        /// <returns>
        ///     The sequence of chunks, each represented by a <see cref="Range" />.
        /// </returns>
        IEnumerable<Range> GetRanges(int length, int rangeMaxLength, int stride);
    }
}
