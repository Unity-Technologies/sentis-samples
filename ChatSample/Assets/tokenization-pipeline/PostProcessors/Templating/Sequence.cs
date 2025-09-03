using System;

namespace Unity.ML.Tokenization.PostProcessors.Templating
{
    /// <summary>
    ///     Represents a sequence of tokens in a <see cref="Template" />.
    /// </summary>
    public class Sequence : Piece, IEquatable<Sequence>
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Sequence" /> type.
        /// </summary>
        /// <param name="identifier">
        ///     Identifies the sequence:
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
        /// </param>
        /// <param name="sequenceId">
        ///     The type id of the sequence.
        /// </param>
        public Sequence(SequenceIdentifier identifier, int sequenceId)
        {
            Identifier = identifier;
            SequenceId = sequenceId;
        }

        /// <summary>
        ///     Identifies the sequence:
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
        public SequenceIdentifier Identifier { get; }

        /// <summary>
        ///     The type id of the sequence.
        /// </summary>
        public int SequenceId { get; }

        /// <inheritdoc />
        public bool Equals(Sequence other)
        {
            if (ReferenceEquals(null, other))
                return false;
            if (ReferenceEquals(this, other))
                return true;
            return Identifier == other.Identifier && SequenceId == other.SequenceId;
        }

        /// <inheritdoc />
        protected override bool PieceEquals(Piece other) =>
            other is Sequence sequence && Equals(sequence);

        /// <inheritdoc />
        protected override int GetPieceHashCode() =>
            HashCode.Combine(base.GetHashCode(), (int) Identifier, SequenceId);

        /// <inheritdoc />
        public override string ToString() => $"${Identifier}:{SequenceId}";
    }
}
