using System.Collections.Generic;
using UnityEngine;

namespace Unity.ML.Tokenization.SplitDelimiterBehaviors
{
    /// <summary>
    ///     Keeps both the content splits and the delimiter splits, separated.
    /// </summary>
    public class SplitDelimiterIsolate : ISplitDelimiterBehavior
    {
        public static SplitDelimiterIsolate Instance { get; } = new();

        public void Apply(
            SubString source,
            IEnumerable<(int offset, int length, bool isContent)> splits,
            IOutput<SubString> output)
        {
            foreach (var (offset, length, _) in splits)
                output.Add(source.Sub(offset, length));
        }
    }
}
