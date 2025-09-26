using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Unity.InferenceEngine.Samples.TTS.Inference
{
    public static class PhonemesHandler
    {
        static string eSpeakNGPath => GetEspeakPathForCurrentPlatform();
        static readonly string esPeakDataPath = $"{Application.streamingAssetsPath}/espeak-ng-data";
        static Dictionary<char, int> vocab = new() { ['\n'] = -1, ['$'] = 0, [';'] = 1, [':'] = 2, [','] = 3, ['.'] = 4, ['!'] = 5, ['?'] = 6, ['¡'] = 7, ['¿'] = 8, ['—'] = 9, ['…'] = 10, ['\"'] = 11, ['('] = 12, [')'] = 13, ['“'] = 14, ['”'] = 15, [' '] = 16, ['\u0303'] = 17, ['ʣ'] = 18, ['ʥ'] = 19, ['ʦ'] = 20, ['ʨ'] = 21, ['ᵝ'] = 22, ['\uAB67'] = 23, ['A'] = 24, ['I'] = 25, ['O'] = 31, ['Q'] = 33, ['S'] = 35, ['T'] = 36, ['W'] = 39, ['Y'] = 41, ['ᵊ'] = 42, ['a'] = 43, ['b'] = 44, ['c'] = 45, ['d'] = 46, ['e'] = 47, ['f'] = 48, ['h'] = 50, ['i'] = 51, ['j'] = 52, ['k'] = 53, ['l'] = 54, ['m'] = 55, ['n'] = 56, ['o'] = 57, ['p'] = 58, ['q'] = 59, ['r'] = 60, ['s'] = 61, ['t'] = 62, ['u'] = 63, ['v'] = 64, ['w'] = 65, ['x'] = 66, ['y'] = 67, ['z'] = 68, ['ɑ'] = 69, ['ɐ'] = 70, ['ɒ'] = 71, ['æ'] = 72, ['β'] = 75, ['ɔ'] = 76, ['ɕ'] = 77, ['ç'] = 78, ['ɖ'] = 80, ['ð'] = 81, ['ʤ'] = 82, ['ə'] = 83, ['ɚ'] = 85, ['ɛ'] = 86, ['ɜ'] = 87, ['ɟ'] =  90, ['ɡ'] = 92, ['ɥ'] = 99, ['ɨ'] = 101, ['ɪ'] = 102, ['ʝ'] = 103, ['ɯ'] = 110, ['ɰ'] = 111, ['ŋ'] = 112, ['ɳ'] = 113, ['ɲ'] = 114, ['ɴ'] = 115, ['ø'] = 116, ['ɸ'] = 118, ['θ'] = 119, ['œ'] = 120, ['ɹ'] = 123, ['ɾ'] = 125, ['ɻ'] = 126, ['ʁ'] = 128, ['ɽ'] = 129, ['ʂ'] = 130, ['ʃ'] = 131, ['ʈ'] = 132, ['ʧ'] = 133, ['ʊ'] = 135, ['ʋ'] = 136, ['ʌ'] = 138, ['ɣ'] = 139, ['ɤ'] = 140, ['χ'] = 142, ['ʎ'] = 143, ['ʒ'] = 147, ['ʔ'] = 148, ['ˈ'] = 156, ['ˌ'] = 157, ['ː'] = 158, ['ʰ'] = 162, ['ʲ'] = 164, ['↓'] = 169, ['→'] = 171, ['↗'] = 172, ['↘'] = 173, ['ᵻ'] = 177 };

        public static string TextToPhonemes(string text, string langCode = "en-us")
        {
            if (string.IsNullOrEmpty(text))
                return string.Empty;

            try
            {
                var executablePath = Path.GetFullPath(eSpeakNGPath);
                var dataPath = Path.GetFullPath(esPeakDataPath);

                using var process = new Process {
                    StartInfo = new ProcessStartInfo {
                        FileName = executablePath,
                        WorkingDirectory = null,
                        Arguments = $"--ipa=3 -b 1 -q -v {langCode} -x \"{text}\"",
                        RedirectStandardInput = true,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        UseShellExecute = false,
                        StandardInputEncoding = Encoding.UTF8,
                        StandardOutputEncoding = Encoding.UTF8
                    }
                };

                process.StartInfo.EnvironmentVariables.Add("ESPEAK_DATA_PATH", dataPath);

                process.Start();

                var output = process.StandardOutput.ReadToEnd();
                var error = process.StandardError.ReadToEnd();

                process.WaitForExit();

                if (process.ExitCode != 0)
                {
                    throw new Exception($"eSpeak NG failed with exit code {process.ExitCode}. Error: {error}");
                }

                return output.Trim();
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to convert text to phonemes: {ex.Message}");
                throw;
            }
        }

        public static int[] Tokenize(string phonemes)
        {
            var tokens = new List<int>();
            foreach (var @char in phonemes.ToCharArray())
            {
                if (!vocab.TryGetValue(@char, out var value))
                {
                    Debug.LogWarning($"Character '{@char}' not in vocabulary, skipping.");
                    continue;
                }
                tokens.Add(value);
            }
            return tokens.ToArray();
        }

        public static int[] TokenizeGraphemes(string graphemes, string langCode = "en-us")
        {
            var phonemes = TextToPhonemes(graphemes, langCode);
            return Tokenize(phonemes);
        }

        static string GetEspeakPathForCurrentPlatform()
        {
            if (Application.platform == RuntimePlatform.OSXEditor || Application.platform == RuntimePlatform.OSXPlayer)
                return $"{Application.streamingAssetsPath}/Libs/MacOs/espeak-ng";

            throw new PlatformNotSupportedException($"eSpeak NG is not supported on platform {Application.platform}");
        }
    }
}
