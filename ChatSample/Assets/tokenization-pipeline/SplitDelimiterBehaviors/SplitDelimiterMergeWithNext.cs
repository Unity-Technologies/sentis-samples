using System.Collections.Generic;

namespace Unity.ML.Tokenization.SplitDelimiterBehaviors
{
    /// <summary>
    ///     Merges each single delimiter split with its preceding content split.
    /// </summary>
    public class SplitDelimiterMergeWithNext : ISplitDelimiterBehavior
    {
        public static SplitDelimiterMergeWithNext Instance { get; } = new();

        public void Apply(
            SubString source,
            IEnumerable<(int offset, int length, bool isContent)> splits,
            IOutput<SubString> output)
        {
            (int offset, int length)? delim = default;
            foreach (var (offset, length, isContent) in splits)
            {
                if (!isContent)
                {
                    if(delim.HasValue)
                        output.Add(source.Sub(delim.Value.offset, delim.Value.length));

                    delim = (offset, length);
                }
                else if (delim.HasValue)
                {
                    output.Add(
                        source.Sub(delim.Value.offset, offset + length - delim.Value.offset));
                    delim = default;
                }
                else
                    output.Add(source.Sub(offset, length));
            }

            if(delim.HasValue)
                output.Add(source.Sub(delim.Value.offset, delim.Value.length));
        }
    }
}
