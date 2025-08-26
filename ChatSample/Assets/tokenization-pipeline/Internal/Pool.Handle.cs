using System;

namespace Unity.ML.Tokenization
{
    partial class Pool<T>
    {
        /// <summary>
        ///     A disposable handle linked to a pooled element which releases it when calling
        ///     <see cref="Handle.Dispose" />.
        /// </summary>
        public readonly struct Handle : IDisposable
        {
            readonly Pool<T> m_Pool;
            readonly T m_Instance;

            /// <summary>
            ///     Initializes a new instance of the <see cref="Handle" /> type.
            /// </summary>
            /// <param name="pool">
            ///     The pool to release the <paramref name="instance" /> to.
            /// </param>
            /// <param name="instance">
            ///     The pooled element to release when disposing the handle.
            /// </param>
            internal Handle(Pool<T> pool, T instance)
            {
                m_Pool = pool;
                m_Instance = instance;
            }

            /// <inheritdoc />
            public void Dispose()
            {
                m_Pool.Release(m_Instance);
            }
        }
    }
}
