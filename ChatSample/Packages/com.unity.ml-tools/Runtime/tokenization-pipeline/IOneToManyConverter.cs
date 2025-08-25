namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Converts a <typeparamref name="TFrom"/> object into a sequence of
    ///     <typeparamref name="TTo"/> objects.
    /// </summary>
    /// <typeparam name="TFrom">
    ///     Type of the object to convert.
    /// </typeparam>
    /// <typeparam name="TTo">
    ///     Type of the converted objects.
    /// </typeparam>
    interface IOneToManyConverter<in TFrom, out TTo>
    {
        /// <summary>
        ///     Converts a <typeparamref name="TFrom"/> object into a sequence of
        ///     <typeparamref name="TTo"/> objects.
        /// </summary>
        /// <param name="input">
        ///     The object to convert.
        /// </param>
        /// <param name="output">
        ///     The target container for converted objects.
        /// </param>
        void Convert(TFrom input, IOutput<TTo> output);
    }
}
