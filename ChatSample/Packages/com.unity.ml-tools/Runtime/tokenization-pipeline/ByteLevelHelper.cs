using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Unity.ML.Tokenization
{
    static class ByteLevelHelper
    {
        public static readonly ReadOnlyDictionary<byte, char> BytesChars;
        public static readonly ReadOnlyDictionary<char, byte> CharsBytes;

        static ByteLevelHelper()
        {
            BytesChars = new ReadOnlyDictionary<byte, char>(BuildByteCharMap());

            CharsBytes = new ReadOnlyDictionary<char, byte>(
                BytesChars.ToDictionary(
                    kvp => kvp.Value,
                    kvp => kvp.Key));
        }

        static Dictionary<byte, char> BuildByteCharMap()
        {
            var ints = new List<int>();

            for (var i = '!'; i <= '~'; i++)
                ints.Add(i);

            for (var i = 0xA1; i <= 0xAC; i++)
                ints.Add(i);

            for (var i = 0xAE; i <= 0xFF; i++)
                ints.Add(i);

            var bytes = ints
                .Select(b => (byte)b)
                .ToList();

            {
                byte b = 0, n = 0;
                do
                {
                    if (bytes.Contains(b)) continue;
                    bytes.Add(b);
                    ints.Add(256 + n++);
                } while (++b != 0);
            }

            return bytes
                .Zip(ints, (@byte, @int) => (@int, @byte))
                .ToDictionary(
                    t => t.@byte,
                    t => char.ConvertFromUtf32(t.@int)[0]);
        }
    }
}
