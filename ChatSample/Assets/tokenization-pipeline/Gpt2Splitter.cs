namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     A fast, low-allocation implementation of the GPT2 regex:
    ///     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    /// </summary>
    class Gpt2Splitter : IOneToManyConverter<SubString, SubString>
    {
        public static Gpt2Splitter Instance { get; } = new();

        static bool HandleSpecialCase(string source, ref int index, int limit, out SubString output)
        {
            output = default;
            if (source[index] != '\'') return false;

            // 's, 't, 'm, 'd
            if (index + 1 < limit && source[index + 1] is 's' or 't' or 'm' or 'd')
            {
                output = new SubString(source, index, 2);
                index += 2;
                return true;
            }

            // 're, 've
            if (
                index + 2 < limit
                && source[index + 1] is 'r' or 'v'
                && source[index + 2] is 'e')
            {
                output = new SubString(source, index, 3);
                index += 3;
                return true;
            }

            // 'll
            if (
                index + 2 < limit
                && source[index + 1] is 'l'
                && source[index + 2] is 'l')
            {
                output = new SubString(source, index, 3);
                index += 3;
                return true;
            }

            return false;
        }

        static bool HandleWord(string source, ref int index, int limit, out SubString output)
        {
            output = default;
            var wordStart = index;

            if (!char.IsLetter(source[wordStart])
                && (source[wordStart] is not ' '
                    || wordStart + 1 == limit
                    || !char.IsLetter(source[wordStart + 1])))
                return false;

            var wordEnd = wordStart + 1;
            while (wordEnd < limit && char.IsLetter(source[wordEnd]))
                wordEnd++;

            output = SubString.FromTo(source, wordStart, wordEnd);
            index = wordEnd;
            return true;
        }

        static bool HandleNumber(string source, ref int index, int limit, out SubString output)
        {
            output = default;
            var numberStart = index;

            if (!char.IsDigit(source[numberStart])
                && (source[numberStart] is not ' '
                    || numberStart + 1 == limit
                    || !char.IsLetter(source[numberStart + 1])))
                return false;

            var numberEnd = numberStart + 1;
            while (numberEnd < limit && char.IsDigit(source[numberEnd]))
                numberEnd++;

            output = SubString.FromTo(source, numberStart, numberEnd);
            index = numberEnd;
            return true;
        }

        static bool IsSymbol(char c)
        {
            return !char.IsWhiteSpace(c)
                   && !char.IsLetter(c)
                   && !char.IsDigit(c);
        }

        static bool HandleSymbols(string source, ref int index, int limit, out SubString output)
        {
            output = default;
            var symbolStart = index;

            if (!IsSymbol(source[symbolStart])
                && (source[symbolStart] is not ' '
                    || symbolStart + 1 == limit
                    || !IsSymbol(source[symbolStart + 1])))
                return false;

            var symbolEnd = symbolStart + 1;
            while (symbolEnd < limit && IsSymbol(source[symbolEnd]))
                symbolEnd++;

            output = SubString.FromTo(source, symbolStart, symbolEnd);
            index = symbolEnd;
            return true;
        }

        static bool HandleWhiteSpaces(string source, ref int index, int limit, out SubString output)
        {
            output = default;
            var whiteSpaceStart = index;

            if (!char.IsWhiteSpace(source[whiteSpaceStart]))
                return false;

            var whiteSpaceEnd = whiteSpaceStart + 1;
            while (whiteSpaceEnd < limit
                   && char.IsWhiteSpace(source[whiteSpaceEnd])
                   && (whiteSpaceEnd + 1 == limit ||
                       char.IsWhiteSpace(source[whiteSpaceEnd + 1])))
                whiteSpaceEnd++;

            output = SubString.FromTo(source, whiteSpaceStart, whiteSpaceEnd);
            index = whiteSpaceEnd;
            return true;
        }

        public void Split(SubString input, IOutput<SubString> output)
        {
            if (input.IsNull)
                return;

            var (source, offset, length) = input;
            var (index, limit) = (offset, offset + length);

            while (index < limit && (HandleSpecialCase(source, ref index, limit, out var match)
                || HandleWord(source, ref index, limit, out match)
                || HandleNumber(source, ref index, limit, out match)
                || HandleSymbols(source, ref index, limit, out match)
                || HandleWhiteSpaces(source, ref index, limit, out match)))
            {
                output.Add(match);
            }
        }

        void IOneToManyConverter<SubString, SubString>.Convert(
            SubString input, IOutput<SubString> output) =>
            Split(input, output);
    }
}
