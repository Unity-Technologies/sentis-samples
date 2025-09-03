using System;
using System.Collections.Generic;

namespace Unity.ML.Tokenization.Truncators
{
    /// <summary>
    ///     Generates a sequence of <see cref="Range" /> starting from the left (<c>0</c>, the lower
    ///     bound of the source).
    /// </summary>
    public class LeftDirectionRangeGenerator : RangeGeneratorBase
    {
        /// <inheritdoc />
        protected override IEnumerable<Range> GetRangesInternal(int length, int rangeMaxLength,
            int stride)
        {
            var offset = rangeMaxLength - stride;

            for (var to = length; to > 0; to -= offset)
            {
                var from = Math.Max(to - rangeMaxLength, 0);
                yield return new Range(from, to - from);
                if (from == 0)
                    yield break;
            }
        }
    }
}
