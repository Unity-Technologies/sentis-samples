using System;
using System.Collections.Generic;

namespace Unity.ML.Tokenization
{
    class OutputCollection<TValue> : IOutput<IEnumerable<TValue>>
    {
        Func<List<TValue>> m_GetList;

        ICollection<List<TValue>> m_Target;

        public void Add(IEnumerable<TValue> value)
        {
            var collection = m_GetList.Invoke();
            collection.AddRange(value);
            m_Target.Add(collection);
        }

        public void Add(IEnumerable<IEnumerable<TValue>> values)
        {
            foreach (var value in values)
                Add(value);
        }

        public void Init(ICollection<List<TValue>> target, Func<List<TValue>> getList)
        {
            m_Target = target;
            m_GetList = getList;
        }

        public void Reset()
        {
            m_Target = default;
            m_GetList = default;
        }
    }
}
