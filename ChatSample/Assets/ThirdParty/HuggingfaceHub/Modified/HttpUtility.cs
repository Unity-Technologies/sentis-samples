using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;
using System.Text;

#nullable disable
namespace HuggingfaceHub.Utilities
{

  public static class HttpUtility
  {
    public static void HtmlAttributeEncode(string s, TextWriter output)
    {
      if (output == null)
        throw new ArgumentNullException(nameof(output));
      HttpEncoder.Current.HtmlAttributeEncode(s, output);
    }

    public static string HtmlAttributeEncode(string s)
    {
      if (s == null)
        return (string) null;
      using (StringWriter output = new StringWriter())
      {
        HttpEncoder.Current.HtmlAttributeEncode(s, (TextWriter) output);
        return output.ToString();
      }
    }

    public static string UrlDecode(string str) => HttpUtility.UrlDecode(str, Encoding.UTF8);

    private static char[] GetChars(MemoryStream b, Encoding e)
    {
      return e.GetChars(b.GetBuffer(), 0, (int) b.Length);
    }

    private static void WriteCharBytes(IList buf, char ch, Encoding e)
    {
      if (ch > 'ÿ')
      {
        Encoding encoding = e;
        char[] chars = new char[1] {ch};
        foreach (byte num in encoding.GetBytes(chars))
          buf.Add((object) num);
      }
      else
        buf.Add((object) (byte) ch);
    }

    public static string UrlDecode(string s, Encoding e)
    {
      if (null == s)
        return (string) null;
      if (s.IndexOf('%') == -1 && s.IndexOf('+') == -1)
        return s;
      if (e == null)
        e = Encoding.UTF8;
      long length = (long) s.Length;
      List<byte> buf = new List<byte>();
      for (int index = 0; (long) index < length; ++index)
      {
        char ch = s[index];
        if (ch == '%' && (long) (index + 2) < length && s[index + 1] != '%')
        {
          if (s[index + 1] == 'u' && (long) (index + 5) < length)
          {
            int num = HttpUtility.GetChar(s, index + 2, 4);
            if (num != -1)
            {
              HttpUtility.WriteCharBytes((IList) buf, (char) num, e);
              index += 5;
            }
            else
              HttpUtility.WriteCharBytes((IList) buf, '%', e);
          }
          else
          {
            int num;
            if ((num = HttpUtility.GetChar(s, index + 1, 2)) != -1)
            {
              HttpUtility.WriteCharBytes((IList) buf, (char) num, e);
              index += 2;
            }
            else
              HttpUtility.WriteCharBytes((IList) buf, '%', e);
          }
        }
        else if (ch == '+')
          HttpUtility.WriteCharBytes((IList) buf, ' ', e);
        else
          HttpUtility.WriteCharBytes((IList) buf, ch, e);
      }

      byte[] array = buf.ToArray();
      return e.GetString(array);
    }

    public static string UrlDecode(byte[] bytes, Encoding e)
    {
      return bytes == null ? (string) null : HttpUtility.UrlDecode(bytes, 0, bytes.Length, e);
    }

    private static int GetInt(byte b)
    {
      char ch = (char) b;
      if (ch >= '0' && ch <= '9')
        return (int) ch - 48 /*0x30*/;
      if (ch >= 'a' && ch <= 'f')
        return (int) ch - 97 + 10;
      return ch >= 'A' && ch <= 'F' ? (int) ch - 65 + 10 : -1;
    }

    private static int GetChar(byte[] bytes, int offset, int length)
    {
      int num1 = 0;
      int num2 = length + offset;
      for (int index = offset; index < num2; ++index)
      {
        int num3 = HttpUtility.GetInt(bytes[index]);
        if (num3 == -1)
          return -1;
        num1 = (num1 << 4) + num3;
      }

      return num1;
    }

    private static int GetChar(string str, int offset, int length)
    {
      int num1 = 0;
      int num2 = length + offset;
      for (int index = offset; index < num2; ++index)
      {
        char b = str[index];
        if (b > '\u007F')
          return -1;
        int num3 = HttpUtility.GetInt((byte) b);
        if (num3 == -1)
          return -1;
        num1 = (num1 << 4) + num3;
      }

      return num1;
    }

    public static string UrlDecode(byte[] bytes, int offset, int count, Encoding e)
    {
      if (bytes == null)
        return (string) null;
      if (count == 0)
        return string.Empty;
      if (bytes == null)
        throw new ArgumentNullException(nameof(bytes));
      if (offset < 0 || offset > bytes.Length)
        throw new ArgumentOutOfRangeException(nameof(offset));
      if (count < 0 || offset + count > bytes.Length)
        throw new ArgumentOutOfRangeException(nameof(count));
      StringBuilder stringBuilder = new StringBuilder();
      MemoryStream b = new MemoryStream();
      int num1 = count + offset;
      for (int index = offset; index < num1; ++index)
      {
        if (bytes[index] == (byte) 37 && index + 2 < count && bytes[index + 1] != (byte) 37)
        {
          if (bytes[index + 1] == (byte) 117 && index + 5 < num1)
          {
            if (b.Length > 0L)
            {
              stringBuilder.Append(HttpUtility.GetChars(b, e));
              b.SetLength(0L);
            }

            int num2 = HttpUtility.GetChar(bytes, index + 2, 4);
            if (num2 != -1)
            {
              stringBuilder.Append((char) num2);
              index += 5;
              continue;
            }
          }
          else
          {
            int num3;
            if ((num3 = HttpUtility.GetChar(bytes, index + 1, 2)) != -1)
            {
              b.WriteByte((byte) num3);
              index += 2;
              continue;
            }
          }
        }

        if (b.Length > 0L)
        {
          stringBuilder.Append(HttpUtility.GetChars(b, e));
          b.SetLength(0L);
        }

        if (bytes[index] == (byte) 43)
          stringBuilder.Append(' ');
        else
          stringBuilder.Append((char) bytes[index]);
      }

      if (b.Length > 0L)
        stringBuilder.Append(HttpUtility.GetChars(b, e));
      return stringBuilder.ToString();
    }

    public static byte[] UrlDecodeToBytes(byte[] bytes)
    {
      return bytes == null ? (byte[]) null : HttpUtility.UrlDecodeToBytes(bytes, 0, bytes.Length);
    }

    public static byte[] UrlDecodeToBytes(string str)
    {
      return HttpUtility.UrlDecodeToBytes(str, Encoding.UTF8);
    }

    public static byte[] UrlDecodeToBytes(string str, Encoding e)
    {
      if (str == null)
        return (byte[]) null;
      return e != null ? HttpUtility.UrlDecodeToBytes(e.GetBytes(str)) : throw new ArgumentNullException(nameof(e));
    }

    public static byte[] UrlDecodeToBytes(byte[] bytes, int offset, int count)
    {
      if (bytes == null)
        return (byte[]) null;
      if (count == 0)
        return new byte[0];
      int length = bytes.Length;
      if (offset < 0 || offset >= length)
        throw new ArgumentOutOfRangeException(nameof(offset));
      if (count < 0 || offset > length - count)
        throw new ArgumentOutOfRangeException(nameof(count));
      MemoryStream memoryStream = new MemoryStream();
      int num1 = offset + count;
      for (int index = offset; index < num1; ++index)
      {
        char ch = (char) bytes[index];
        int num2;
        switch (ch)
        {
          case '%':
            num2 = index >= num1 - 2 ? 1 : 0;
            break;
          case '+':
            ch = ' ';
            goto label_17;
          default:
            num2 = 1;
            break;
        }

        if (num2 == 0)
        {
          int num3 = HttpUtility.GetChar(bytes, index + 1, 2);
          if (num3 != -1)
          {
            ch = (char) num3;
            index += 2;
          }
        }

        label_17:
        memoryStream.WriteByte((byte) ch);
      }

      return memoryStream.ToArray();
    }

    public static string UrlEncode(string str) => HttpUtility.UrlEncode(str, Encoding.UTF8);

    public static string UrlEncode(string s, Encoding Enc)
    {
      if (s == null)
        return (string) null;
      if (s == string.Empty)
        return string.Empty;
      bool flag = false;
      int length = s.Length;
      for (int index = 0; index < length; ++index)
      {
        char c = s[index];
        if ((c < '0' || c < 'A' && c > '9' || c > 'Z' && c < 'a' || c > 'z') && !HttpEncoder.NotEncoded(c))
        {
          flag = true;
          break;
        }
      }

      if (!flag)
        return s;
      byte[] bytes1 = new byte[Enc.GetMaxByteCount(s.Length)];
      int bytes2 = Enc.GetBytes(s, 0, s.Length, bytes1, 0);
      return Encoding.ASCII.GetString(HttpUtility.UrlEncodeToBytes(bytes1, 0, bytes2));
    }

    public static string UrlEncode(byte[] bytes)
    {
      if (bytes == null)
        return (string) null;
      return bytes.Length == 0 ? string.Empty : Encoding.ASCII.GetString(HttpUtility.UrlEncodeToBytes(bytes, 0, bytes.Length));
    }

    public static string UrlEncode(byte[] bytes, int offset, int count)
    {
      if (bytes == null)
        return (string) null;
      return bytes.Length == 0 ? string.Empty : Encoding.ASCII.GetString(HttpUtility.UrlEncodeToBytes(bytes, offset, count));
    }

    public static byte[] UrlEncodeToBytes(string str)
    {
      return HttpUtility.UrlEncodeToBytes(str, Encoding.UTF8);
    }

    public static byte[] UrlEncodeToBytes(string str, Encoding e)
    {
      switch (str)
      {
        case null:
          return (byte[]) null;
        case "":
          return new byte[0];
        default:
          byte[] bytes = e.GetBytes(str);
          return HttpUtility.UrlEncodeToBytes(bytes, 0, bytes.Length);
      }
    }

    public static byte[] UrlEncodeToBytes(byte[] bytes)
    {
      if (bytes == null)
        return (byte[]) null;
      return bytes.Length == 0 ? new byte[0] : HttpUtility.UrlEncodeToBytes(bytes, 0, bytes.Length);
    }

    public static byte[] UrlEncodeToBytes(byte[] bytes, int offset, int count)
    {
      return bytes == null ? (byte[]) null : HttpEncoder.Current.UrlEncode(bytes, offset, count);
    }

    public static string UrlEncodeUnicode(string str)
    {
      return str == null ? (string) null : Encoding.ASCII.GetString(HttpUtility.UrlEncodeUnicodeToBytes(str));
    }

    public static byte[] UrlEncodeUnicodeToBytes(string str)
    {
      switch (str)
      {
        case null:
          return (byte[]) null;
        case "":
          return new byte[0];
        default:
          MemoryStream result = new MemoryStream(str.Length);
          foreach (char c in str)
            HttpEncoder.UrlEncodeChar(c, (Stream) result, true);
          return result.ToArray();
      }
    }

    public static string HtmlDecode(string s)
    {
      if (s == null)
        return (string) null;
      using (StringWriter output = new StringWriter())
      {
        HttpEncoder.Current.HtmlDecode(s, (TextWriter) output);
        return output.ToString();
      }
    }

    public static void HtmlDecode(string s, TextWriter output)
    {
      if (output == null)
        throw new ArgumentNullException(nameof(output));
      if (string.IsNullOrEmpty(s))
        return;
      HttpEncoder.Current.HtmlDecode(s, output);
    }

    public static string HtmlEncode(string s)
    {
      if (s == null)
        return (string) null;
      using (StringWriter output = new StringWriter())
      {
        HttpEncoder.Current.HtmlEncode(s, (TextWriter) output);
        return output.ToString();
      }
    }

    public static void HtmlEncode(string s, TextWriter output)
    {
      if (output == null)
        throw new ArgumentNullException(nameof(output));
      if (string.IsNullOrEmpty(s))
        return;
      HttpEncoder.Current.HtmlEncode(s, output);
    }

    public static string HtmlEncode(object value)
    {
      return value == null ? (string) null : HttpUtility.HtmlEncode(value.ToString());
    }

    public static string JavaScriptStringEncode(string value)
    {
      return HttpUtility.JavaScriptStringEncode(value, false);
    }

    public static string JavaScriptStringEncode(string value, bool addDoubleQuotes)
    {
      if (string.IsNullOrEmpty(value))
        return addDoubleQuotes ? "\"\"" : string.Empty;
      int length = value.Length;
      bool flag = false;
      for (int index = 0; index < length; ++index)
      {
        char ch = value[index];
        if (ch >= char.MinValue && ch <= '\u001F' || ch == '"' || ch == '\'' || ch == '<' || ch == '>' || ch == '\\')
        {
          flag = true;
          break;
        }
      }

      if (!flag)
        return addDoubleQuotes ? $"\"{value}\"" : value;
      StringBuilder stringBuilder = new StringBuilder();
      if (addDoubleQuotes)
        stringBuilder.Append('"');
      for (int index = 0; index < length; ++index)
      {
        char ch = value[index];
        if (ch >= char.MinValue && ch <= '\a' || ch == '\v' || ch >= '\u000E' && ch <= '\u001F' || ch == '\'' || ch == '<' || ch == '>')
        {
          stringBuilder.AppendFormat("\\u{0:x4}", (object) (int) ch);
        }
        else
        {
          switch (ch)
          {
            case '\b':
              stringBuilder.Append("\\b");
              continue;
            case '\t':
              stringBuilder.Append("\\t");
              continue;
            case '\n':
              stringBuilder.Append("\\n");
              continue;
            case '\f':
              stringBuilder.Append("\\f");
              continue;
            case '\r':
              stringBuilder.Append("\\r");
              continue;
            case '"':
              stringBuilder.Append("\\\"");
              continue;
            case '\\':
              stringBuilder.Append("\\\\");
              continue;
            default:
              stringBuilder.Append(ch);
              continue;
          }
        }
      }

      if (addDoubleQuotes)
        stringBuilder.Append('"');
      return stringBuilder.ToString();
    }

    public static string UrlPathEncode(string s) => HttpEncoder.Current.UrlPathEncode(s);

    public static NameValueCollection ParseQueryString(string query)
    {
      return HttpUtility.ParseQueryString(query, Encoding.UTF8);
    }

    public static NameValueCollection ParseQueryString(string query, Encoding encoding)
    {
      if (query == null)
        throw new ArgumentNullException(nameof(query));
      if (encoding == null)
        throw new ArgumentNullException(nameof(encoding));
      if (query.Length == 0 || query.Length == 1 && query[0] == '?')
        return (NameValueCollection) new HttpUtility.HttpQSCollection();
      if (query[0] == '?')
        query = query.Substring(1);
      NameValueCollection result = (NameValueCollection) new HttpUtility.HttpQSCollection();
      HttpUtility.ParseQueryString(query, encoding, result);
      return result;
    }

    internal static void ParseQueryString(
      string query,
      Encoding encoding,
      NameValueCollection result)
    {
      if (query.Length == 0)
        return;
      string str1 = HttpUtility.HtmlDecode(query);
      int length = str1.Length;
      int num1 = 0;
      bool flag = true;
      while (num1 <= length)
      {
        int startIndex = -1;
        int num2 = -1;
        for (int index = num1; index < length; ++index)
        {
          if (startIndex == -1 && str1[index] == '=')
            startIndex = index + 1;
          else if (str1[index] == '&')
          {
            num2 = index;
            break;
          }
        }

        if (flag)
        {
          flag = false;
          if (str1[num1] == '?')
            ++num1;
        }

        string name;
        if (startIndex == -1)
        {
          name = (string) null;
          startIndex = num1;
        }
        else
          name = HttpUtility.UrlDecode(str1.Substring(num1, startIndex - num1 - 1), encoding);

        if (num2 < 0)
        {
          num1 = -1;
          num2 = str1.Length;
        }
        else
          num1 = num2 + 1;

        string str2 = HttpUtility.UrlDecode(str1.Substring(startIndex, num2 - startIndex), encoding);
        result.Add(name, str2);
        if (num1 == -1)
          break;
      }
    }

    private sealed class HttpQSCollection : NameValueCollection
    {
      public override string ToString()
      {
        int count = this.Count;
        if (count == 0)
          return "";
        StringBuilder stringBuilder = new StringBuilder();
        string[] allKeys = this.AllKeys;
        for (int index = 0; index < count; ++index)
          stringBuilder.AppendFormat("{0}={1}&", (object) allKeys[index], (object) HttpUtility.UrlEncode(this[allKeys[index]]));
        if (stringBuilder.Length > 0)
          --stringBuilder.Length;
        return stringBuilder.ToString();
      }
    }
  }
}