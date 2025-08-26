using System;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Definition of a token, its <see cref="Value" /> and its <see cref="Id" />.
    /// </summary>
    public interface ITokenDefinition : IEquatable<ITokenDefinition>, IComparable<ITokenDefinition>
    {
        /// <summary>
        ///     The id of the token.
        /// </summary>
        int Id { get; }

        /// <summary>
        ///     The string identifier of the token.
        /// </summary>
        string Key { get; }

        /// <summary>
        ///     The string representation of the token.
        ///     The <see cref="string" /> value of the token.
        ///     For special tokens, it is empty.
        ///     For tokens key with prefix, the prefix is removed ("_blah" becomes "blah").
        /// </summary>
        string Value { get; }

        /// <summary>
        ///     Tells whether this token is special.
        ///     Being special means it is probably used for structuring the sequence of tokens, and
        ///     it shouldn't be part of the detokenized <see cref="string" /> value.
        /// </summary>
        bool IsSpecial { get; }
    }
}
