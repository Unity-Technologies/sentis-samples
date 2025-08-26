using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Simple implementation of a priority queue as it is not yet available with the current
    ///     version of C#/.NET Unity supports.
    /// </summary>
    /// <typeparam name="T">
    ///     The type of stored elements.
    /// </typeparam>
    class PriorityQueue<T>
    {
        /// <summary>
        ///     The comparison method used to compare two <typeparamref name="T" /> elements.
        /// </summary>
        readonly Func<T, T, int> m_Compare;

        /// <summary>
        ///     The list of element to store ordered.
        /// </summary>
        readonly List<T> m_Storage = new();

        /// <summary>
        ///     Initializes a new instance of the <see cref="PriorityQueue{T}" /> type.
        /// </summary>
        /// <param name="compare">
        ///     The comparison method used to compare two <typeparamref name="T" /> elements.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="compare" /> cannot be <c>null</c>.
        /// </exception>
        public PriorityQueue([NotNull] Func<T, T, int> compare)
        {
            m_Compare = compare ?? throw new ArgumentNullException(nameof(compare));
        }

        /// <summary>
        ///     Gives the number of stored elements.
        /// </summary>
        public int Count => m_Storage.Count;

        /// <summary>
        ///     Adds a new <paramref name="element" /> to a queue, in order.
        /// </summary>
        /// <param name="element">
        ///     The element to add.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="element" /> cannot be <c>null</c>.
        /// </exception>
        public void Push([NotNull] T element)
        {
            if (element == null)
                throw new ArgumentNullException(nameof(element));

            if (m_Storage.Count == 0 || m_Compare(element, m_Storage[^1]) >= 0)
            {
                m_Storage.Add(element);
                return;
            }

            if (m_Compare(element, m_Storage[0]) <= 0)
            {
                m_Storage.Insert(0, element);
                return;
            }

            int l = 0, r = m_Storage.Count - 1;

            while (r - l > 1)
            {
                var m = (l + r) / 2;
                if (m_Compare.Invoke(element, m_Storage[m]) <= 0)
                    r = m;
                else
                    l = m;
            }

            m_Storage.Insert(r, element);
        }

        /// <summary>
        ///     Gets the next <paramref name="element" /> and removes it from the queue.
        /// </summary>
        /// <param name="element">
        ///     The next element of the queue.
        /// </param>
        /// <returns>
        ///     Tells whether an element has been retrieved.
        /// </returns>
        public bool TryPop(out T element)
        {
            if (m_Storage.Count == 0)
            {
                element = default;
                return false;
            }

            element = m_Storage[0];
            m_Storage.RemoveAt(0);
            return true;
        }

        /// <summary>
        ///     Removes all the elements from the queue.
        /// </summary>
        public void Clear()
        {
            m_Storage.Clear();
        }
    }
}
