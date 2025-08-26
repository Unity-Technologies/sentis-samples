using System;

namespace Unity.ML.Tokenization.PostProcessors.Templating
{
    /// <summary>
    ///     A element of a template.
    ///     It can be either a <see cref="Sequence" /> or a <see cref="SpecialToken" />.
    /// </summary>
    public abstract class Piece : IEquatable<Piece>
    {
        /// <inheritdoc />
        public bool Equals(Piece other) => PieceEquals(other);

        /// <inheritdoc />
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            return
                obj.GetType() == GetType()
                && Equals((Piece)obj);
        }

        /// <inheritdoc cref="object.GetHashCode"/>
        public override int GetHashCode() => GetPieceHashCode();

        /// <inheritdoc cref="object.ToString"/>
        public abstract override string ToString();

        /// <summary>
        ///     Tell whether this <see cref="Piece" /> equals the <paramref name="other" /> one.
        /// </summary>
        /// <param name="other">
        ///     The other <see cref="Piece" /> to compare.
        /// </param>
        /// <returns>
        ///     Whether this <see cref="Piece" /> equals the <paramref name="other" /> one.
        /// </returns>
        protected abstract bool PieceEquals(Piece other);

        /// <summary>
        ///     Gets the hash code of this <see cref="Piece" />.
        /// </summary>
        /// <returns>
        ///     The hash code of this <see cref="Piece" />.
        /// </returns>
        protected abstract int GetPieceHashCode();
    }
}
