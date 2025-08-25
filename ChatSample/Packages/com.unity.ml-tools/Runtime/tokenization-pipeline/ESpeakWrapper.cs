using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Unity.ML.Tokenization
{
    static class EspeakWrapper
    {
        const int k_ClauseIntonationFullStop = 0x00000000;
        const int k_ClauseIntonationComma = 0x00001000;
        const int k_ClauseIntonationQuestion = 0x00002000;
        const int k_ClauseIntonationExclamation = 0x00003000;
        const int k_ClauseIntonationNone = 0x00004000;

        const int k_ClauseTypeNone = 0x00000000;
        const int k_ClauseTypeEof = 0x00010000;
        const int k_ClauseTypeClause = 0x00040000;
        const int k_ClauseTypeSentence = 0x00080000;

        const int k_ClauseNone = 0 | k_ClauseIntonationNone | k_ClauseTypeNone;
        const int k_ClauseParagraph = 70 | k_ClauseIntonationFullStop | k_ClauseTypeSentence;

        const int k_ClauseEof =
            40 | k_ClauseIntonationFullStop | k_ClauseTypeSentence | k_ClauseTypeEof;

        const int k_ClausePeriod = 40 | k_ClauseIntonationFullStop | k_ClauseTypeSentence;
        const int k_ClauseComma = 20 | k_ClauseIntonationComma | k_ClauseTypeClause;
        const int k_ClauseQuestion = 40 | k_ClauseIntonationQuestion | k_ClauseTypeSentence;
        const int k_ClauseExclamation = 45 | k_ClauseIntonationExclamation | k_ClauseTypeSentence;
        const int k_ClauseColon = 30 | k_ClauseIntonationFullStop | k_ClauseTypeClause;
        const int k_ClauseSemicolon = 30 | k_ClauseIntonationComma | k_ClauseTypeClause;

        const int k_EspeakPhonemesIpa = 0x02;
        const int k_EspeakCharsAuto = 0;

        /// <summary>
        ///     Tells whether the plugin is initialized.
        /// </summary>
        public static bool Initialized { get; private set; }

        /// <summary>
        ///     Initializes the plugin if not already.
        ///     It takes the Espeak data from the StreamingAssets folder of the project.
        /// </summary>
        public static void Initialize()
        {
            if (Initialized)
                return;

            var path = $"{Application.streamingAssetsPath}/espeak-ng-data";
            var pathPtr = Marshal.StringToHGlobalAuto(path);
            PInvoke.Initialize(0x02, 0, pathPtr, 0);
            Marshal.FreeHGlobal(pathPtr);

            Initialized = true;
        }

        /// <summary>
        ///     Sets the language of the phonemizer.
        /// </summary>
        /// <param name="lang">
        ///     The target language.
        /// </param>
        public static void SetLang(string lang)
        {
            var voicePtr = Marshal.StringToHGlobalAuto(lang);
            PInvoke.SetVoiceByName(voicePtr);
            Marshal.FreeHGlobal(voicePtr);
        }

        /// <summary>
        ///     Converts the specified
        ///     <param name="input" />
        ///     text into its phoneme representation.
        /// </summary>
        /// <param name="input">
        ///     The text to convert.
        /// </param>
        /// <returns>
        ///     The phoneme representation of
        ///     <param name="input" />
        ///     considering the language set
        ///     with <see cref="SetLang" />.
        /// </returns>
        public static unsafe string TextToPhonemes(string input)
        {
            var inputPtr = Marshal.StringToHGlobalAuto(input);
            var localPtr = new IntPtr(inputPtr.ToPointer());
            var inputPtrs = new IntPtr(&localPtr);

            var phonemes = string.Empty;

            while (localPtr != IntPtr.Zero)
            {
                var phonemesPtr = PInvoke.TextToPhonemes(inputPtrs, k_EspeakCharsAuto,
                    k_EspeakPhonemesIpa, out var terminator);

                phonemes += Marshal.PtrToStringAuto(phonemesPtr);

                switch (terminator)
                {
                    case k_ClauseExclamation:
                        phonemes += "!";
                        break;
                    case k_ClauseQuestion:
                        phonemes += "?";
                        break;
                    case k_ClauseComma:
                        phonemes += ",";
                        break;
                    case k_ClauseColon:
                        phonemes += ":";
                        break;
                    case k_ClauseSemicolon:
                        phonemes += ";";
                        break;
                    case k_ClausePeriod:
                        phonemes += ".";
                        break;
                }

                if (localPtr != IntPtr.Zero)
                {
                    if ((terminator & k_ClauseTypeSentence) == k_ClauseTypeSentence)
                        phonemes += "\n";
                    else
                        phonemes += " ";
                }
            }

            Marshal.FreeHGlobal(inputPtr);
            return phonemes;
        }

        public static void Dispose()
        {
            PInvoke.Terminate();
        }

        static class PInvoke
        {
            const string k_LibraryName = "libespeak-ng";

            [DllImport(k_LibraryName, EntryPoint = "espeak_Initialize",
                CallingConvention = CallingConvention.Cdecl)]
            public static extern int Initialize(int mode, int bufLength, IntPtr path, int options);

            [DllImport(k_LibraryName, EntryPoint = "espeak_SetVoiceByName",
                CallingConvention = CallingConvention.Cdecl)]
            public static extern int SetVoiceByName(IntPtr name);

            [DllImport(k_LibraryName, EntryPoint = "espeak_TextToPhonemesWithTerminator",
                CallingConvention = CallingConvention.Cdecl)]
            public static extern IntPtr TextToPhonemes(IntPtr input, int textMode, int phonemeMode,
                out int terminator);

            [DllImport(k_LibraryName, EntryPoint = "espeak_Terminate",
                CallingConvention = CallingConvention.Cdecl)]
            public static extern void Terminate();
        }
    }
}
