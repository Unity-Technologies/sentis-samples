using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Unity.ML.Tokenization.Decoders
{
    /// <summary>
    ///     ByteFallBack converts tokens looking like "&lt;0x61>" to character, and attempts to
    ///     concatenate them into a string.
    ///     If the tokens cannot be decoded, '�' is used instead for each inconvertible byte token.
    /// </summary>
    public class ByteFallbackDecoder : DecoderBase
    {
        readonly Pool<List<byte>> m_ByteListPool = PoolUtility.GetListOfBytePool();

        /// <inheritdoc />
        protected override void DecodeInternal(IEnumerable<string> tokens, IOutput<string> output)
        {
            using var byteTokenHandle = m_ByteListPool.Get(out var previousByteTokens);

            foreach (var token in tokens)
                if (token.Length == 6 && token.StartsWith("<0x") && token.EndsWith(">"))
                {
                    // Convert the hex string to a byte. If it fails, clear the previous byte tokens
                    // and add a '�' character.
                    if (byte.TryParse(
                        token.AsSpan(3, 2), NumberStyles.HexNumber, null, out var bytes))
                        previousByteTokens.Add(bytes);
                    else
                        output.Add("�");
                }
                else
                {
                    if (previousByteTokens.Count > 0)
                    {
                        output.Add(ConvertByteToToken(previousByteTokens));
                        previousByteTokens = null;
                    }

                    output.Add(token);
                }

            if (previousByteTokens is {Count: > 0})
                output.Add(ConvertByteToToken(previousByteTokens));
        }

        IEnumerable<string> ConvertByteToToken(List<byte> previousByteTokens)
        {
            var returnTokens = new List<string>();

            try
            {
                // TODO improve allocation
                var str = System.Text.Encoding.UTF8.GetString(previousByteTokens.ToArray());
                returnTokens.Add(str);
                if (str.Equals("�"))
                    for (var i = 0; i < previousByteTokens.Count - 1; i++)
                        returnTokens.Add("�");
            }
            catch (DecoderFallbackException)
            {
                foreach (var b in previousByteTokens)
                    returnTokens.Add("�");
            }

            return returnTokens;
        }
    }
}
