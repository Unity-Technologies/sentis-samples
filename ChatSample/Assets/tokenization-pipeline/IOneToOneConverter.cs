namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Converts a input of type <typeparamref name="TFrom" /> into a value of type
    ///     <typeparamref name="TTo" />.
    /// </summary>
    /// <typeparam name="TFrom">
    ///     Source type.
    /// </typeparam>
    /// <typeparam name="TTo">
    ///     Target type.
    /// </typeparam>
    interface IOneToOneConverter<in TFrom, out TTo>
    {
        /// <summary>
        ///     Converts a input of type <typeparamref name="TFrom" /> into a value of type
        ///     <typeparamref name="TTo" />.
        /// </summary>
        /// <param name="input">
        ///     The source to convert.
        /// </param>
        /// <returns>
        ///     <typeparamref name="TTo" /> representation of <paramref name="input" />.
        /// </returns>
        TTo Convert(TFrom input);
    }
}
