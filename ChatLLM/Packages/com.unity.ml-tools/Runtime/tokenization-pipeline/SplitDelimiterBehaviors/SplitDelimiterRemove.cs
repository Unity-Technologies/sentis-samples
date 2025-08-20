using System.Collections.Generic;
using System.Linq;

namespace Unity.ML.Tokenization.SplitDelimiterBehaviors
{
    /// <summary>
    ///     Removes the delimiter splits, only the content splits only.
    /// </summary>
    public class SplitDelimiterRemove : ISplitDelimiterBehavior
    {
        public static SplitDelimiterRemove Instance { get; } = new();

        public void Apply(
            SubString source,
            IEnumerable<(int offset, int length, bool isContent)> splits,
            IOutput<SubString> output)
        {
            foreach (var (offset, length, _) in splits.Where(s => s.isContent))
                output.Add(source.Sub(offset, length));
        }
    }
}
