using System.Collections.Generic;

namespace Unity.ML.Tokenization.SplitDelimiterBehaviors
{
    /// <summary>
    ///     Aggregates all the successive delimiters into a single split and keeps all the content
    ///     splits as is.
    /// </summary>
    public class SplitDelimiterContiguous : ISplitDelimiterBehavior
    {
        public static SplitDelimiterContiguous Instance { get; } = new();

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
                    delim = delim.HasValue
                        ? (delim.Value.offset, offset + length - delim.Value.offset)
                        : (offset, length);
                }
                else
                {
                    if (delim.HasValue)
                    {
                        output.Add(source.Sub(delim.Value.offset, offset - delim.Value.offset));
                        delim = default;
                    }
                    output.Add(source.Sub(offset, length));
                }
            }

            if(delim.HasValue)
                output.Add(source.Sub(delim.Value.offset, delim.Value.length));
        }
    }
}
