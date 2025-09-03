using Unity.ML.Tokenization.PostProcessors;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Identifies a sequence.
    ///     It is used in the <see cref="TemplatePostProcessor" />.
    /// </summary>
    public enum SequenceIdentifier
    {
        /// <summary>
        ///     First sequence.
        /// </summary>
        A,

        /// <summary>
        ///     Second sequence.
        /// </summary>
        B
    }
}
