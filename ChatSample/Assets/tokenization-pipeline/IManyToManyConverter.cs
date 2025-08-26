using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Converts a sequence of <typeparamref name="TFrom" /> into a sequence of
    ///     <typeparamref name="TTo" /> instance.
    /// </summary>
    /// <typeparam name="TFrom">
    ///     The type of the source instance.
    /// </typeparam>
    /// <typeparam name="TTo">
    ///     The type of the converted instances.
    /// </typeparam>
    interface IManyToManyConverter<in TFrom, out TTo>
    {
        /// <summary>
        ///     Converts a sequence of <typeparamref name="TFrom" /> into a sequence of
        ///     <typeparamref name="TTo" /> instance.
        /// </summary>
        /// <param name="input">
        ///     The sequence of objects to convert.
        /// </param>
        /// <param name="output">
        ///     The target container of converted objects.
        /// </param>
        void Convert(IEnumerable<TFrom> input, IOutput<TTo> output);
    }
}
