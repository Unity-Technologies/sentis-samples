using System;

namespace Unity.ML.Tokenization.PreTokenizers
{
    /// <summary>
    ///     Base implementation for the builtin <see cref="IPreTokenizer" />s.
    ///     Runs the necessary parameter validation before calling
    ///     <see cref="PreTokenizeInternal" />.
    /// </summary>
    public abstract class PreTokenizerBase : IPreTokenizer
    {
        /// <summary>
        ///     Pre-cuts the <paramref name="input" /> into smaller parts.
        /// </summary>
        /// <param name="input">
        ///     The source to pre-cut.
        /// </param>
        /// <param name="output">
        ///     Target collection of generated pretokenized strings.
        /// </param>
        public void PreTokenize(SubString input, IOutput<SubString> output)
        {
            if (input.IsNull)
                throw new ArgumentNullException(nameof(input));

            PreTokenizeInternal(input, output);
        }

        /// <inheritdoc cref="PreTokenize" />
        protected abstract void PreTokenizeInternal(SubString input, IOutput<SubString> output);

        void IPreTokenizer.PreTokenize(SubString input, IOutput<SubString> output) =>
            PreTokenize(input, output);
    }
}
