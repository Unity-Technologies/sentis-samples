using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Tokenizers
{
    /// <summary>
    ///     Base type for builtin implementations of <see cref="ITokenizer" />.
    ///     The <see cref="Tokenize" /> methods do the necessary parameter validations before calling
    ///     a protected <see cref="TokenizeInternal" /> method.
    /// </summary>
    public abstract class TokenizerBase : ITokenizer
    {
        /// <summary>
        ///     Turns an input string into a sequence of token ids.
        /// </summary>
        /// <param name="input">
        ///     The source string to tokenize.
        /// </param>
        /// <param name="output">
        ///     Target sequence of token ids.
        /// </param>
        public void Tokenize(IEnumerable<SubString> input, [NotNull] IOutput<int> output)
        {
            if (output == null)
                throw new ArgumentNullException(nameof(output));

            TokenizeInternal(input, output);
        }

        /// <summary>
        ///     Turns a sequence token ids into a sequence of string.
        /// </summary>
        /// <param name="input">
        ///     The source string to tokenize.
        /// </param>
        /// <param name="skipSpecialTokens">
        ///     Tells whether skipping the special tokens.
        /// </param>
        /// <param name="output">
        ///     Target sequence of token ids.
        /// </param>
        public void DeTokenize(
            [NotNull] IEnumerable<int> input,
            bool skipSpecialTokens,
            [NotNull] IOutput<string> output)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            if (output == null)
                throw new ArgumentNullException(nameof(output));

            DeTokenizeInternal(input, skipSpecialTokens, output);
        }

        /// <inheritdoc cref="Tokenize" />
        protected abstract void TokenizeInternal(IEnumerable<SubString> input, IOutput<int> output);

        /// <inheritdoc cref="DeTokenize" />
        protected abstract void DeTokenizeInternal(
            IEnumerable<int> input,
            bool skipSpecialTokens,
            IOutput<string> output);

        void ITokenizer.Tokenize(IEnumerable<SubString> input, IOutput<int> output) =>
            Tokenize(input, output);

        void ITokenizer.DeTokenize(
            IEnumerable<int> input,
            bool skipSpecialTokens,
            IOutput<string> output) =>
            DeTokenize(input, skipSpecialTokens, output);
    }
}
