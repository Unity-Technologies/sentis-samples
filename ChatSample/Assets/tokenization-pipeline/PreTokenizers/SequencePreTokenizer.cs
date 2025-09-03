using System;
using System.Linq;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.PreTokenizers
{
    /// <summary>
    ///     Applies a sequence of pre tokenizers.
    /// </summary>
    public class SequencePreTokenizer : PreTokenizerBase
    {
        readonly IPreTokenizer[] m_PreTokenizers;

        /// <summary>
        ///     Initializes a new instance of the <see cref="SequencePreTokenizer"/> type.
        /// </summary>
        /// <param name="preTokenizers">
        ///     Sequence of pretokenizers to apply.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="preTokenizers"/> cannot be null.
        /// </exception>
        /// <exception cref="ArgumentException">
        ///     <paramref name="preTokenizers"/> cannot be empty.
        /// </exception>
        public SequencePreTokenizer(
            [NotNull] params IPreTokenizer[] preTokenizers)
        {
            if (preTokenizers == null)
                throw new ArgumentNullException(nameof(preTokenizers));

            if (preTokenizers.Length == 0)
                throw new ArgumentException(
                    "At least one preTokenizer is required", nameof(preTokenizers));

            if (preTokenizers.Any(t => t is null))
                throw new ArgumentNullException(
                    nameof(preTokenizers), $"None of the {nameof(preTokenizers)} can be null.");

            m_PreTokenizers = preTokenizers.ToArray();
        }

        /// <inheritdoc />
        protected override void PreTokenizeInternal(SubString input, IOutput<SubString> output)
        {
            using var handleA = PoolUtility.GetOutputOfSubStringPool().Get(out var preTokInput);
            using var handleB = PoolUtility.GetOutputOfSubStringPool().Get(out var preTokOutput);

            preTokOutput.Add(input);

            foreach (var preTokenizer in m_PreTokenizers)
            {
                (preTokInput, preTokOutput) = (preTokOutput, preTokInput);

                foreach (var subString in preTokInput)
                    preTokenizer.PreTokenize(subString, preTokOutput);

                preTokInput.Reset();
            }

            output.Add(preTokOutput);
        }
    }
}
