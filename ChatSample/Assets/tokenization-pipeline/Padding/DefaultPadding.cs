using System.Collections.Generic;
using System.Linq;

namespace Unity.ML.Tokenization.Padding
{
    /// <summary>
    ///     Placeholder padding processor.
    ///     Does not apply in padding rules.
    /// </summary>
    public class DefaultPadding : PaddingBase
    {
        /// <inheritdoc />
        protected override void PadInternal(
            IEnumerable<IEnumerable<int>> input,
            IOutput<IEnumerable<(int id, int attention)>> output)
        {
            foreach (var sequence in input)
                output.Add(sequence.Select(t => (t, 1)));
        }
    }
}
