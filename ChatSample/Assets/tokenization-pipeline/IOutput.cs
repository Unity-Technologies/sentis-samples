using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Target container for <typeparamref name="T"/> values.
    /// </summary>
    /// <typeparam name="T">
    ///     The type of the added values.
    /// </typeparam>
    public interface IOutput<in T>
    {
        /// <summary>
        ///     Adds a single value.
        /// </summary>
        /// <param name="value">
        ///     The value to add.
        /// </param>
        void Add(T value);

        /// <summary>
        ///     Adds a sequence of values.
        /// </summary>
        /// <param name="values">
        ///     The values to add.
        /// </param>
        void Add(IEnumerable<T> values);
    }
}
