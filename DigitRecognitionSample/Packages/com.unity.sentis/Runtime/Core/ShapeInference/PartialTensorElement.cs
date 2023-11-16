using System;
using UnityEngine;

namespace Unity.Sentis
{
    [Serializable]
    enum ElementType
    {
        Unknown = 0,
        IntValue,
        FloatValue,
        Param
    }

    /// <summary>
    /// Represents a single element of a SymbolicTensorShape, can be an int value, float value, char param or unknown
    /// </summary>
    [Serializable]
    struct PartialTensorElement
    {
        ElementType m_ElementType;
        char m_Param;
        int m_IntValue;
        float m_FloatValue;

        public static PartialTensorElement Unknown => new PartialTensorElement();

        public PartialTensorElement(int intValue)
        {
            m_ElementType = ElementType.IntValue;
            m_Param = default;
            m_IntValue = intValue;
            m_FloatValue = default;
        }

        public PartialTensorElement(char param)
        {
            Logger.AssertIsTrue(param >= 0, "Element param cannot be negative");
            m_ElementType = ElementType.Param;
            m_Param = param;
            m_IntValue = default;
            m_FloatValue = default;
        }

        public PartialTensorElement(float value)
        {
            m_ElementType = ElementType.FloatValue;
            m_Param = default;
            m_IntValue = default;
            m_FloatValue = value;
        }

        public bool isUnknown => m_ElementType == ElementType.Unknown;
        public bool isIntValue => m_ElementType == ElementType.IntValue;
        public bool isParam => m_ElementType == ElementType.Param;
        public bool isFloatValue => m_ElementType == ElementType.FloatValue;

        public static PartialTensorElement Zero => new PartialTensorElement(0);
        public static PartialTensorElement One => new PartialTensorElement(1);

        public int intValue
        {
            get
            {
                Logger.AssertIsTrue(m_ElementType == ElementType.IntValue, "Cannot get value of element with type != ElementType.Value");
                return m_IntValue;
            }
        }

        public float floatValue
        {
            get
            {
                Logger.AssertIsTrue(m_ElementType == ElementType.FloatValue, "Cannot get floatValue of element with type != ElementType.FloatValue");
                return m_FloatValue;
            }
        }

        public char param
        {
            get
            {
                Logger.AssertIsTrue(m_ElementType == ElementType.Param, "Cannot get param of element with type != ElementType.Param");
                return m_Param;
            }
        }

        /// <summary>
        /// Whether the current 'PartialTensorElement' is 'ElementType.Value' and has a value equal to the specified value.
        /// </summary>
        public bool EqualsIntValue(int v)
        {
            return m_ElementType == ElementType.IntValue && v == m_IntValue;
        }

        /// <summary>
        /// Whether the current 'SymbolicTensorDim' is 'ElementType.Value' and is equal to the specified element.
        /// </summary>
        public bool EqualsIntValue(PartialTensorElement other)
        {
            return m_ElementType == ElementType.IntValue && other.m_ElementType == ElementType.IntValue && m_IntValue == other.m_IntValue;
        }

        /// <summary>
        /// Whether the current 'PartialTensorElement' is 'ElementType.Param' and is equal to the specified element.
        /// </summary>
        public bool EqualsParam(PartialTensorElement other)
        {
            return m_ElementType == ElementType.Param && other.m_ElementType == ElementType.Param && m_Param == other.m_Param;
        }

        public static explicit operator SymbolicTensorDim(PartialTensorElement v)
        {
            return v.m_ElementType switch
            {
                ElementType.Unknown => SymbolicTensorDim.Unknown,
                ElementType.IntValue => v.m_IntValue < 0 ? SymbolicTensorDim.Unknown : new SymbolicTensorDim(v.m_IntValue),
                ElementType.Param => new SymbolicTensorDim(v.m_Param),
                ElementType.FloatValue => SymbolicTensorDim.Unknown,
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        public static explicit operator PartialTensorElement(SymbolicTensorDim d)
        {
            if (d.isValue)
                return new PartialTensorElement(d.value);
            if (d.isParam)
                return new PartialTensorElement(d.param);
            return Unknown;
        }

        /// <summary>
        /// Returns a string that represents the `PartialTensorElement`.
        /// </summary>
        public override string ToString()
        {
            return m_ElementType switch
            {
                ElementType.Unknown => "?",
                ElementType.IntValue => intValue.ToString(),
                ElementType.Param => param.ToString(),
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        public bool Equals(PartialTensorElement other)
        {
            return m_ElementType == other.m_ElementType && m_IntValue == other.m_IntValue && m_Param == other.m_Param;
        }

        /// <summary>
        /// Determines whether the specified object is equal to the current `PartialTensorElement`.
        /// </summary>
        public override bool Equals(object obj)
        {
            return obj is PartialTensorElement other && Equals(other);
        }

        /// <summary>
        ///
        /// Compares element to element
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.m_ElementType != b.m_ElementType)
                return false;
            if (a.m_IntValue != b.m_IntValue)
                return false;
            if (a.m_FloatValue != b.m_FloatValue)
                return false;
            if (a.m_Param != b.m_Param)
                return false;
            return true;
        }

        /// <summary>
        ///
        /// Compares element to element
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   T   T   T
        /// 3 | T   T   T   T   T
        /// A | T   T   F   T   T
        /// ? | T   T   T   T   T
        ///
        /// </summary>
        public static bool operator !=(PartialTensorElement a, PartialTensorElement b)
        {
            return !(a == b);
        }

        /// <summary>
        ///
        /// Compares element to int
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(PartialTensorElement a, int b)
        {
            if (!a.isIntValue)
                return false;
            return a.m_IntValue == b;
        }

        /// <summary>
        ///
        /// Compares element to int
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | T   T   F   F   F
        ///
        /// </summary>
        public static bool operator !=(PartialTensorElement a, int b)
        {
            if (!a.isIntValue)
                return false;
            return a.m_IntValue != b;
        }

        /// <summary>
        ///
        /// Compares int to element
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(int a, PartialTensorElement b)
        {
            return b == a;
        }

        /// <summary>
        ///
        /// Compares int to element
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   T   T   T
        /// 3 | T   T   T   T   T
        ///
        /// </summary>
        public static bool operator !=(int a, PartialTensorElement b)
        {
            return b != a;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// ==
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   F   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator ==(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isIntValue && b.isValue)
                return a.intValue == b.value;
            return false;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// !=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   T   T   T
        /// 3 | T   T   T   T   T
        /// A | T   T   F   T   T
        /// ? | T   T   T   T   T
        ///
        /// </summary>
        public static bool operator !=(PartialTensorElement a, SymbolicTensorDim b)
        {
            return !(a == b);
        }

        /// <summary>
        ///
        /// Compares element to int
        /// >
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | T   T   F   F   F
        ///
        /// </summary>
        public static bool operator >(PartialTensorElement a, int b)
        {
            if (!a.isIntValue)
                return false;
            return a.m_IntValue > b;
        }

        /// <summary>
        ///
        /// Compares element to int
        /// <
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <(PartialTensorElement a, int b)
        {
            if (!a.isIntValue)
                return false;
            return a.m_IntValue < b;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// >
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | T   T   F   F   F
        /// A | F   F   F   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator >(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (!a.isIntValue || !b.isValue)
                return false;
            return a.m_IntValue > b.value;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// <
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   T   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   F   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (!a.isIntValue || !b.isValue)
                return false;
            return a.m_IntValue < b.value;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// >=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   F   F   F   F
        /// 3 | T   T   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator >=(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isIntValue && b.isValue)
                return a.m_IntValue >= b.value;
            return false;
        }

        /// <summary>
        ///
        /// Compares element to symbolic tensor dim
        /// <=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   T   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <=(PartialTensorElement a, SymbolicTensorDim b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isIntValue && b.isValue)
                return a.m_IntValue <= b.value;
            return false;
        }

        /// <summary>
        ///
        /// Compares symbolic tensor dim to element
        /// >=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | F   F   F   F   F
        /// 3 | T   T   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator >=(SymbolicTensorDim a, PartialTensorElement b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isValue && b.isIntValue)
                return a.value >= b.intValue;
            if (a.isValue && b.isIntValue)
                return a.value >= b.intValue;
            return false;
        }

        /// <summary>
        ///
        /// Compares symbolic tensor dim to element
        /// <=
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | T   T   F   F   F
        /// 3 | F   F   F   F   F
        /// A | F   F   T   F   F
        /// ? | F   F   F   F   F
        ///
        /// </summary>
        public static bool operator <=(SymbolicTensorDim a, PartialTensorElement b)
        {
            if (a.isParam && b.isParam)
                return a.param == b.param;
            if (a.isValue && b.isIntValue)
                return a.value <= b.intValue;
            return false;
        }

        /// <summary>
        ///
        /// Subtracts element
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        ///   | 0   -1  ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(PartialTensorElement a)
        {
            if (a.isIntValue)
                return new PartialTensorElement(-a.intValue);
            if (a.isFloatValue)
                return new PartialTensorElement(-a.floatValue);
            return Unknown;
        }

        /// <summary>
        ///
        /// Adds element to element
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 0 | 0   1   A   B   ?
        /// 3 | 3   4   ?   ?   ?
        /// A | A   ?   ?   ?   ?
        /// ? | ?   ?   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator +(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isIntValue)
                return a.intValue + b;
            if (b.isIntValue)
                return b.intValue + a;
            return Unknown;
        }

        /// <summary>
        ///
        /// Adds element to int
        ///
        ///   | 0   1   A   ?
        /// --|-----------------
        /// 0 | 0   4   A   ?
        /// 3 | 3   4   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator +(int a, PartialTensorElement b)
        {
            if (b.isIntValue)
                return new PartialTensorElement(a + b.intValue);
            if (a == 0)
                return b;
            return Unknown;
        }

        /// <summary>
        ///
        /// Adds int to element
        ///
        ///   | 0   1
        /// --|--------
        /// 0 | 0   1
        /// 3 | 3   4
        /// A | A   ?
        /// ? | ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator +(PartialTensorElement a, int b)
        {
            return b + a;
        }

        /// <summary>
        ///
        /// Subtracts element from element
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 3 | 3   2   ?   ?   ?
        /// A | A   ?   0   ?   ?
        /// ? | ?   ?   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isIntValue)
                return a.intValue - b;
            if (b.isIntValue)
                return a - b.intValue;
            if (a.isParam && b.isParam && a.param == b.param)
                return Zero;
            return Unknown;
        }

        /// <summary>
        /// Subtracts element from int
        ///
        ///   | 0   1   A   B   ?
        /// --|---------------------
        /// 3 | 3   2   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(int a, PartialTensorElement b)
        {
            if (b.isIntValue)
                return new PartialTensorElement(a - b.intValue);
            return Unknown;
        }

        /// <summary>
        ///
        /// Subtracts int from element
        ///
        ///   | 0   1
        /// --|---------
        /// 3 | 3   2
        /// A | A   ?
        /// ? | ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator -(PartialTensorElement a, int b)
        {
            if (a.isIntValue)
                return new PartialTensorElement(a.intValue - b);
            if (b == 0)
                return a;
            return Unknown;
        }

        /// <summary>
        /// Multiplies element by element
        ///
        ///   | 0   1   3   A   B   ?
        /// --|-----------------------
        /// 0 | 0   0   0   0   0   0
        /// 2 | 0   2   6   ?   ?   ?
        /// A | 0   A   ?   ?   ?   ?
        /// ? | 0   ?   ?   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator *(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isIntValue)
                return a.intValue * b;
            if (b.isIntValue)
                return b.intValue * a;
            return Unknown;
        }

        /// <summary>
        /// Multiplies int by element
        ///
        ///   | 1   3   A   B   ?
        /// --|--------------------
        /// 0 | 0   0   0   0   0
        /// 2 | 2   6   ?   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator *(int a, PartialTensorElement b)
        {
            if (b.isIntValue)
                return new PartialTensorElement(a * b.intValue);
            if (a == 1)
                return b;
            if (a == 0)
                return Zero;
            return Unknown;
        }

        /// <summary>
        /// Multiplies element by int
        ///
        ///   | 0   1   3
        /// --|-----------
        /// 2 | 0   2   6
        /// A | 0   A   ?
        /// ? | 0   ?   ?
        ///
        /// </summary>
        public static PartialTensorElement operator *(PartialTensorElement a, int b)
        {
            return b * a;
        }

        /// <summary>
        /// Returns the better known of two elements known to be equal, throws error if both elements are values and not equal
        ///
        ///   | 2   3   A   B   ?
        /// --|-------------------
        /// 2 | 2  Err  2   2   2
        /// A | 2   3   A   A   A
        /// ? | 2   3   A   B   ?
        ///
        /// </summary>
        public static PartialTensorElement MaxDefinedElement(PartialTensorElement a, PartialTensorElement b)
        {
            if (a.isUnknown)
                return b;
            if (b.isUnknown)
                return a;
            if (b.isIntValue)
            {
                Logger.AssertIsTrue(!a.isIntValue || b == a, "ValueError: value elements must be equal");
                return b;
            }
            if (b.isFloatValue)
            {
                Logger.AssertIsTrue(!a.isFloatValue || b == a, "ValueError: float value elements must be equal");
                return b;
            }

            return a;
        }

        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        public override int GetHashCode()
        {
            return m_ElementType.GetHashCode() ^ m_Param.GetHashCode() ^ m_IntValue.GetHashCode() ^ m_FloatValue.GetHashCode();
        }
    }
}
