using System;
using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    class Output<T> : IOutput<T>, IEnumerable<T>
    {
        readonly List<T> m_Target = new();

        public int Count => m_Target.Count;

        public void Add(T value) => m_Target.Add(value);

        public void Add([NotNull] IEnumerable<T> values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            m_Target.AddRange(values);
        }

        public List<T>.Enumerator GetEnumerator() => m_Target.GetEnumerator();

        public T this[int i] => m_Target[i];

        public void Reset() => m_Target.Clear();

        void IOutput<T>.Add(T value) => Add(value);

        void IOutput<T>.Add([NotNull] IEnumerable<T> values) => Add(values);

        IEnumerator<T> IEnumerable<T>.GetEnumerator() => m_Target.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => m_Target.GetEnumerator();
    }
}
