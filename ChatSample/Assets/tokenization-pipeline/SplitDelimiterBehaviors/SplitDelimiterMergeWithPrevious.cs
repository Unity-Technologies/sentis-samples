using System.Collections.Generic;
using UnityEngine;

namespace Unity.ML.Tokenization.SplitDelimiterBehaviors
{
    /// <summary>
    ///     Merges each single delimiter split with its next content split.
    /// </summary>
    public class SplitDelimiterMergeWithPrevious : ISplitDelimiterBehavior
    {
        public static SplitDelimiterMergeWithPrevious Instance { get; } = new();

        public void Apply(
            SubString source,
            IEnumerable<(int offset, int length, bool isContent)> splits,
            IOutput<SubString> output)
        {
            (int offset, int length)? content = default;
            foreach (var (offset, length, isContent) in splits)
            {
                if (isContent)
                {
                    if(content.HasValue)
                        output.Add(source.Sub(content.Value.offset, content.Value.length));

                    content = (offset, length);
                }
                else if (content.HasValue)
                {
                    output.Add(
                        source.Sub(content.Value.offset, offset + length - content.Value.offset));
                    content = default;
                }
                else
                    output.Add(source.Sub(offset, length));
            }

            if(content.HasValue)
                output.Add(source.Sub(content.Value.offset, content.Value.length));
        }
    }
}
