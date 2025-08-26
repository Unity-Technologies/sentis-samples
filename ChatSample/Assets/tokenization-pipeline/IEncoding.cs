using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Describes the result of a tokenization pipeline execution.
    /// </summary>
    public interface IEncoding
    {
        /// <summary>
        ///     The number of tokens.
        /// </summary>
        int Length { get; }

        /// <summary>
        ///     The list of token ids.
        /// </summary>
        IReadOnlyCollection<int> Ids { get; }

        /// <summary>
        ///     The attention mask.
        ///     When a tokenization requires truncation and padding, this mask indicates which
        ///     tokens are the most relevant.
        /// </summary>
        IReadOnlyCollection<int> Attention { get; }

        /// <summary>
        ///     In case the tokenization pipeline produces more tokens than the expected size, the
        ///     following tokens are stored into another <see cref="IEncoding" /> instance.
        ///     This overflow can also define its own overflow, similarly to a linked list.
        /// </summary>
        IEncoding Overflow { get; }

        /// <summary>
        ///     The type ids. Not supported yet.
        /// </summary>
        IReadOnlyCollection<int> TypeIds { get; }
    }
}
