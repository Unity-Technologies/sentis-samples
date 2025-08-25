namespace Unity.ML.Tokenization.Normalizers
{
    /// <summary>
    ///     Converts the input into its phoneme representation, using ESpeak as conversion backend.
    /// </summary>
    public class Phonemizer : INormalizer
    {
        /// <summary>
        ///     Initializes a new instance of the <see cref="Phonemizer" /> type.
        /// </summary>
        /// <param name="lang">
        ///     The language to use.
        /// </param>
        public Phonemizer(string lang = "en-US") => Lang = lang;

        /// <summary>
        ///     The language used for the conversion to phonemes.
        /// </summary>
        public string Lang { get; }

        /// <inheritdoc />
        public SubString Normalize(SubString input)
        {
            if (!EspeakWrapper.Initialized)
                EspeakWrapper.Initialize();

            EspeakWrapper.SetLang(Lang);
            return EspeakWrapper.TextToPhonemes(input);
        }
    }
}
