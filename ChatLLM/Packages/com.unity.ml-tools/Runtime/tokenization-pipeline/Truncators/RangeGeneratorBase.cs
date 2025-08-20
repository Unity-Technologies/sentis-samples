using System;
using System.Collections.Generic;

namespace Unity.ML.Tokenization.Truncators
{
    /// <summary>
    ///     Base implementation of the builtin <see cref="IRangeGenerator" />.
    ///     Runs the necessary parameter validation before calling <see cref="GetRangesInternal" />.
    /// </summary>
    public abstract class RangeGeneratorBase : IRangeGenerator
    {
        IEnumerable<Range> IRangeGenerator.GetRanges(int length, int rangeMaxLength, int stride)
        {
            return GetRanges(length, rangeMaxLength, stride);
        }

        /// <inheritdoc cref="IRangeGenerator.GetRanges" />
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <list type="bullet">
        ///         <item>
        ///             <term>
        ///                 <paramref name="length" /> is not positive or <c>0</c>.
        ///             </term>
        ///         </item>
        ///         <item>
        ///             <term>
        ///                 <paramref name="rangeMaxLength" /> is not positive or <c>0</c>.
        ///             </term>
        ///         </item>
        ///         <item>
        ///             <term>
        ///                 <paramref name="stride" /> lowe than <c>0</c> or higher than
        ///                 <paramref name="rangeMaxLength" />.
        ///             </term>
        ///         </item>
        ///     </list>
        /// </exception>
        public IEnumerable<Range> GetRanges(int length, int rangeMaxLength, int stride)
        {
            if (length <= 0)
                throw new ArgumentOutOfRangeException(nameof(length), length, null);

            if (rangeMaxLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(rangeMaxLength), rangeMaxLength, null);

            if (stride < 0)
                throw new ArgumentOutOfRangeException(nameof(stride), stride, null);

            if (stride >= rangeMaxLength)
                throw new ArgumentOutOfRangeException(nameof(stride), stride,
                    $"Must be strictly less than {rangeMaxLength}");

            return GetRangesInternal(length, rangeMaxLength, stride);
        }

        /// <inheritdoc cref="IRangeGenerator.GetRanges" />
        protected abstract IEnumerable<Range> GetRangesInternal(int length, int rangeMaxLength,
            int stride);
    }
}
