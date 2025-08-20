using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using JetBrains.Annotations;
using Unity.ML.Tokenization.SplitDelimiterBehaviors;

namespace Unity.ML.Tokenization.PreTokenizers
{
    /// <summary>
    ///     Splits the input based on a regular expression.
    /// </summary>
    public class SplitPreTokenizer : PreTokenizerBase
    {
        Regex m_Regex;
        ISplitDelimiterBehavior m_Behavior;

        /// <summary>
        ///     Initializes a new instance of the <see cref="SplitPreTokenizer"/> type.
        /// </summary>
        /// <param name="pattern">
        /// </param>
        /// <param name="behavior">
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// </exception>
        public SplitPreTokenizer([NotNull] string pattern, SplitDelimiterBehavior behavior)
        {
            if (pattern == null)
                throw new ArgumentNullException(nameof(pattern));

            var regex = new Regex(pattern);

            ISplitDelimiterBehavior behaviorImpl = behavior switch
            {
                SplitDelimiterBehavior.Removed => SplitDelimiterRemove.Instance,
                SplitDelimiterBehavior.Isolated => SplitDelimiterIsolate.Instance,
                SplitDelimiterBehavior.MergedWithPrevious => SplitDelimiterMergeWithPrevious
                    .Instance,
                SplitDelimiterBehavior.MergedWithNext => SplitDelimiterMergeWithNext.Instance,
                SplitDelimiterBehavior.Contiguous => SplitDelimiterContiguous.Instance,
                _ => throw new ArgumentOutOfRangeException(nameof(behavior), behavior, null)
            };

            Init(regex, behaviorImpl);
        }

        internal SplitPreTokenizer(Regex pattern, ISplitDelimiterBehavior behavior) =>
            Init(pattern, behavior);

        void Init(Regex pattern, ISplitDelimiterBehavior behavior)
        {
            m_Regex = pattern;
            m_Behavior = behavior;
        }

        protected override void PreTokenizeInternal(SubString input, IOutput<SubString> output)
        {
            var copy = input.ToString();
            var matches = m_Regex.Matches(copy);

            var splits = matches
                .Select(m => m.Groups[0])
                .Select(g => (offset: g.Index, length: g.Length, isContent: true));

            splits = AddDelimiters(splits, input.Length);

            m_Behavior.Apply(input, splits, output);

            return;

            IEnumerable<(int offset, int length, bool isContent)> AddDelimiters(
                IEnumerable<(int offset, int length, bool isContent)> contentSplits,
                int inputLength)
            {
                var expectedOffset = 0;
                foreach (var contentSplit in contentSplits)
                {
                    if (contentSplit.offset > expectedOffset)
                        yield return (expectedOffset, contentSplit.offset - expectedOffset, false);

                    yield return contentSplit;
                    expectedOffset = contentSplit.offset + contentSplit.length;
                }

                if (expectedOffset < inputLength)
                    yield return (expectedOffset, inputLength - expectedOffset, false);
            }
        }
    }
}
