using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.ML.Tokenization.Truncators
{
    /// <summary>
    ///     This truncation strategy truncates the longest sequence of tokens first.
    ///     In case a secondary sequence of tokens is not provided, it doesn't remove any token.
    /// </summary>
    public class LongestFirstTruncator : StrategicTruncator
    {
        /// <inheritdoc cref="StrategicTruncator(IRangeGenerator, int, int)" />
        /// <summary>
        ///     Initializes a new instance of the <see cref="LongestFirstTruncator" /> type.
        /// </summary>
        public LongestFirstTruncator(IRangeGenerator rangeGenerator, int maxLength, int stride)
            : base(rangeGenerator, maxLength, stride)
        {
        }

        /// <inheritdoc />
        protected override void Truncate(
            ICollection<int> tokensA,
            ICollection<int> tokensB,
            int maxLength,
            int toRemove,
            IOutput<IEnumerable<int>> outputA,
            IOutput<IEnumerable<int>> outputB)
        {
            if (tokensB.Count == 0)
            {
                foreach (var range in GetRanges(tokensA.Count, tokensA.Count - toRemove))
                    outputA.Add(tokensA.Skip(range.Offset).Take(range.Length));

                return;
            }

            var (n1, n2) = (tokensA.Count, tokensB.Count);
            var swap = n1 > n2;
            if (swap) n1 = n2;

            n2 = n1 > maxLength
                ? n1
                : Math.Max(n1, maxLength - n1);

            if (n1 + n2 > maxLength)
                (n1, n2) = (maxLength / 2, n1 + maxLength % 2);

            if (swap)
                (n1, n2) = (n2, n1);

            foreach (var range in GetRanges(tokensA.Count, n1))
                outputA.Add(tokensA.Skip(range.Offset).Take(range.Length));

            foreach (var range in GetRanges(tokensB.Count, n2))
                outputB.Add(tokensB.Skip(range.Offset).Take(range.Length));
        }
    }
}
