using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;

namespace Unity.ML.Tokenization
{
    /// <summary>
    ///     Simple implementation of a flexible pooling feature.
    /// </summary>
    /// <typeparam name="T">
    ///     The type of the pooled elements.
    /// </typeparam>
    partial class Pool<T> : IDisposable where T : class
    {
        /// <summary>
        ///     Stores the pooled elements.
        /// </summary>
        readonly Stack<T> m_Cache;

        /// <summary>
        ///     The lambda function called to generate a brand new instance of
        ///     <typeparamref name="T" />.
        /// </summary>
        Func<T> m_Create;

        /// <summary>
        ///     Tells whether this <see cref="Pool{T}" /> instance is disposed.
        /// </summary>
        volatile bool m_Disposed;

        /// <summary>
        ///     The lambda function called when a pooled instance is put back to the cache.
        /// </summary>
        Action<T> m_Release;

        /// <summary>
        ///     Initializes a new instance of the <see cref="Pool{T}" /> type.
        /// </summary>
        /// <param name="create">
        ///     The lambda function called to generate a brand new instance of
        ///     <typeparamref name="T" />.
        /// </param>
        /// <param name="release">
        ///     The lambda function called when an pooled instance is put back to the cache.
        /// </param>
        /// <exception cref="ArgumentNullException">
        ///     <paramref name="create" /> cannot be null.
        /// </exception>
        public Pool([JetBrains.Annotations.NotNull] Func<T> create, [CanBeNull] Action<T> release = default)
        {
            m_Cache = new Stack<T>();
            m_Create = create ?? throw new ArgumentNullException(nameof(create));
            m_Release = release;
        }

        /// <inheritdoc />
        public void Dispose()
        {
            DisposeObject();
            GC.SuppressFinalize(this);
        }

        ~Pool()
        {
            DisposeObject();
        }

        /// <summary>
        ///     Get a pooled element.
        ///     If the pool is empty, it creates a new one.
        /// </summary>
        /// <returns>
        ///     A pooled element, or a new one.
        /// </returns>
        /// <exception cref="ObjectDisposedException">
        ///     This instance of the <see cref="Pool{T}" /> type is disposed.
        /// </exception>
        public T Get()
        {
            lock (this)
            {
                CheckDisposed();

                return m_Cache.Count > 0
                    ? m_Cache.Pop()
                    : m_Create.Invoke();
            }
        }

        /// <summary>
        ///     Get a pooled element.
        ///     If the pool is empty, it creates a new one.
        ///     It returns a <see cref="Handle" /> so it can be called in a <see langword="using" />
        ///     block for easy release.
        /// </summary>
        /// <param name="instance">
        ///     A pooled element, or a new one.
        /// </param>
        /// <returns>
        ///     a <see cref="Handle" /> so it can be called in a <see langword="using" /> block for
        ///     easy release.
        /// </returns>
        /// <exception cref="ObjectDisposedException">
        ///     This instance of the <see cref="Pool{T}" /> type is disposed.
        /// </exception>
        public Handle Get(out T instance)
        {
            instance = Get();
            return new(this, instance);
        }

        /// <summary>
        ///     Releases the <paramref name="instance" /> into the pool.
        ///     When released, the <c>release</c> lambda function specified in the constructor will
        ///     be applied to the object.
        /// </summary>
        /// <param name="instance">
        ///     The element to put back to the pool.
        /// </param>
        /// <exception cref="ObjectDisposedException">
        ///     This instance of the <see cref="Pool{T}" /> type is disposed.
        /// </exception>
        public void Release(T instance)
        {
            lock (this)
            {
                CheckDisposed();
                m_Release?.Invoke(instance);
                m_Cache.Push(instance);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        void CheckDisposed()
        {
            if (m_Disposed)
                throw new ObjectDisposedException(GetType().ToString());
        }

        /// <summary>
        ///     Calls <see cref="IDisposable.Dispose" /> on each pooled element if necessary.
        /// </summary>
        void DisposeObject()
        {
            lock (this)
            {
                if (m_Disposed)
                    return;

                foreach (var instance in m_Cache)
                    if (instance is IDisposable disposable)
                        disposable.Dispose();

                m_Cache.Clear();

                m_Create = default;
                m_Release = default;

                m_Disposed = true;
            }
        }
    }
}
