using System;

namespace Unity.ML.Tokenization.PostProcessors.Templating
{
    /// <summary>
    ///     Represents a special token in a <see cref="Template" />.
    /// </summary>
    public class SpecialToken : Piece, IEquatable<SpecialToken>
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="SpecialToken" /> type.
        /// </summary>
        /// <param name="value">
        ///     The value of the token.
        /// </param>
        /// <param name="sequenceId">
        ///     Identifies the sequence to link this special token to:
        ///     <list type="bullet">
        ///         <item>
        ///             <term><see cref="SequenceIdentifier.A" /></term>
        ///             <description>
        ///                 for the primary sequence.
        ///             </description>
        ///         </item>
        ///         <item>
        ///             <term><see cref="SequenceIdentifier.B" /></term>
        ///             <description>
        ///                 for the secondary sequence.
        ///             </description>
        ///         </item>
        ///     </list>
        /// </param>
        public SpecialToken(string value, SequenceIdentifier sequenceId)
        {
            Value = value;
            SequenceId = sequenceId;
        }

        /// <summary>
        ///     The value of the token.
        /// </summary>
        public string Value { get; }

        /// <summary>
        ///     Identifies the sequence to link this special token to:
        ///     <list type="bullet">
        ///         <item>
        ///             <term>
        ///                 <see cref="SequenceIdentifier.A" /> for the primary sequence.
        ///             </term>
        ///         </item>
        ///         <item>
        ///             <term>
        ///                 <see cref="SequenceIdentifier.B" /> for the secondary sequence.
        ///             </term>
        ///         </item>
        ///     </list>
        /// </summary>
        public SequenceIdentifier SequenceId { get; }

        /// <inheritdoc />
        public bool Equals(SpecialToken other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Value == other.Value && SequenceId == other.SequenceId;
        }

        /// <inheritdoc />
        protected override bool PieceEquals(Piece other) => other is SpecialToken token && Equals(token);

        /// <inheritdoc />
        protected override int GetPieceHashCode() => HashCode.Combine(base.GetHashCode(), Value, SequenceId);

        /// <inheritdoc />
        public override string ToString() => $"{Value}:{SequenceId}";
    }
}
