using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Padding
{
    /// <summary>
    ///     Base implementation for builtin padding processors.
    ///     Runs the necessary parameter validation before calling <see cref="PadInternal" />.
    /// </summary>
    public abstract class PaddingBase : IPadding
    {
        /// <summary>
        ///     Applies a padding to sequences of tokens.
        /// </summary>
        /// <param name="input">
        ///     The sequences of tokens to pad.
        /// </param>
        /// <param name="output">
        ///     The target container of padded sequences of tokens.
        /// </param>
        public void Pad(
            [NotNull] IEnumerable<IEnumerable<int>> input,
            [NotNull] IOutput<IEnumerable<(int id, int attention)>> output)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            if (output == null)
                throw new ArgumentNullException(nameof(output));

            PadInternal(input, output);
        }

        /// <inheritdoc cref="Pad" />
        protected abstract void PadInternal(
            [NotNull] IEnumerable<IEnumerable<int>> input,
            [NotNull] IOutput<IEnumerable<(int id, int attention)>> output);

        void IPadding.Pad(
            IEnumerable<IEnumerable<int>> input,
            IOutput<IEnumerable<(int id, int attention)>> output) =>
            Pad(input, output);
    }
}
