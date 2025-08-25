using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization.Padding
{
    /// <summary>
    ///     Base type for directional padding processor.
    /// </summary>
    public abstract class DirectionalPaddingBase : PaddingBase
    {
        readonly Pool<List<int>> m_ListOfIntPool;
        readonly Pool<List<List<int>>> m_ListOfListOfIntPool;

        readonly IPaddingSizeProvider m_SizeProvider;


        /// <summary>
        ///     Initializes a new instance of the <see cref="DirectionalPaddingBase" />
        ///     type.
        /// </summary>
        /// <param name="paddingSizeProvider">
        ///     When applying the padding, this object provide the final size of the padded
        ///     sequence.
        /// </param>
        /// <param name="padToken">
        ///     The token to use to pad a sequence of token.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="paddingSizeProvider" /> cannot be null.
        /// </exception>
        protected DirectionalPaddingBase(
            [NotNull] IPaddingSizeProvider paddingSizeProvider,
            ITokenDefinition padToken)
        {
            PadToken = padToken;
            m_SizeProvider = paddingSizeProvider ??
                             throw new ArgumentNullException(nameof(paddingSizeProvider));

            m_ListOfIntPool = PoolUtility.GetListOfIntPool();

            m_ListOfListOfIntPool = new Pool<List<List<int>>>(
                () => new List<List<int>>(),
                list =>
                {
                    foreach (var item in list)
                        m_ListOfIntPool.Release(item);

                    list.Clear();
                });
        }

        /// <summary>
        ///     The token to use to fill the final sequence with.
        /// </summary>
        public ITokenDefinition PadToken { get; }

        /// <summary>
        ///     Apply the padding to sequences of tokens and add the result to the
        ///     <paramref name="output" />.
        /// </summary>
        /// <param name="input">
        ///     The collection of sequences of tokens to pad.
        /// </param>
        /// <param name="output">
        ///     The target container of padded sequences.
        /// </param>
        protected override void PadInternal(
            IEnumerable<IEnumerable<int>> input,
            IOutput<IEnumerable<(int id, int attention)>> output)
        {
            using var handleInput = m_ListOfListOfIntPool.Get(out var sequenceList);
            Copy(input, sequenceList);

            var padSize =
                m_SizeProvider.GetPaddingSize(sequenceList.Select(sequence => sequence.Count));

            foreach (var sequence in sequenceList)
                output.Add(sequence.Count >= padSize
                    ? sequence.Select(token => (token, 1))
                    : Pad(sequence, padSize));
        }

        /// <summary>
        ///     Copies the sequences of tokens into a list of countable lists.
        ///     It is used by <see cref="PadInternal" /> in order to manipulate the input easily.
        /// </summary>
        /// <param name="input">
        ///     The sequences of tokens.
        /// </param>
        /// <param name="target">
        ///     The target container.
        /// </param>
        void Copy(IEnumerable<IEnumerable<int>> input, ICollection<List<int>> target)
        {
            foreach (var sequence in input)
            {
                var tokens = m_ListOfIntPool.Get();
                tokens.AddRange(sequence);
                target.Add(tokens);
            }
        }

        /// <summary>
        ///     Pads the <paramref name="input" /> sequence of tokens to reach the
        ///     <paramref name="padSize" />.
        /// </summary>
        /// <param name="input">
        ///     The sequence of tokens to pad.
        /// </param>
        /// <param name="padSize">
        ///     The target size.
        /// </param>
        /// <returns>
        ///     The padded sequence of tokens.
        /// </returns>
        protected abstract IEnumerable<(int id, int attention)> Pad(IReadOnlyCollection<int> input,
            int padSize);
    }
}
