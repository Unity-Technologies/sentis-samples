using System;
using System.Runtime.InteropServices;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Represents the data of a token in a sequence.
    /// </summary>
    [StructLayout(LayoutKind.Explicit, CharSet = CharSet.Ansi, Pack = 0, Size = sizeof(int) * 3)]
    public readonly struct Token
    {
        /// <summary>
        ///     ID of the token.
        /// </summary>
        [FieldOffset(0)]
        public readonly int Id;

        /// <summary>
        ///     Attention of the token, expressed by 0 or 1.
        /// </summary>
        [FieldOffset(8)]
        public readonly int Attention;

        /// <summary>
        ///     Identifies the sub-sequence this token belongs to, in a sequence of token computed
        ///     from a pair of input.
        /// </summary>
        [FieldOffset(16)]
        public readonly int TypeId;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Token"/> type.
        /// </summary>
        /// <param name="id">
        ///     ID of the token.
        /// </param>
        /// <param name="attention">
        ///     Attention of the token, expressed by 0 or 1.
        /// </param>
        /// <param name="typeId">
        ///     Identifies the sub-sequence this token belongs to, in a sequence of token computed
        ///     from a pair of input.
        /// </param>
        public Token(int id, int attention = 1, int typeId = 0)
        {
            Id = id;
            Attention = attention;
            TypeId = typeId;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="Token"/> type.
        /// </summary>
        /// <param name="definition">
        ///     Definition of the token.
        ///     Provides the ID.
        /// </param>
        /// <param name="attention">
        ///     Attention of the token, expressed by 0 or 1.
        /// </param>
        /// <param name="typeId">
        ///     Identifies the sub-sequence this token belongs to, in a sequence of token computed
        ///     from a pair of input.
        /// </param>
        public Token([NotNull] ITokenDefinition definition, int attention = 1, int typeId = 0) :
            this(definition.Id, attention, typeId)
        {
            if (definition == null)
                throw new ArgumentNullException(nameof(definition));
        }
    }
}
