using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine.Assertions;

namespace Unity.Sentis {

static class Logger
{
    //TODO handle context (execution/import/model/layer) + log it along error/assert (warning think of thread safety vs context)
    //TODO is it valuable to have a way collect many errors before asserting/throw?
    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertAreEqual<T>(T expected, T actual, string msg)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            Assert.AreEqual(expected, actual, msg);
    }
    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertAreEqual<T, TP0>(T expected, T actual, string msg, TP0 msgParam)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            Assert.AreEqual(expected, actual, string.Format(msg, msgParam));
    }
    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertAreEqual<T, TP0, TP1>(T expected, T actual, string msg, TP0 msgParam0, TP1 msgParam1)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            Assert.AreEqual(expected, actual, string.Format(msg, msgParam0, msgParam1));
    }
    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertAreEqual<T, TP0, TP1, TP2>(T expected, T actual, string msg, TP0 msgParam0, TP1 msgParam1, TP2 msgParam2)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            Assert.AreEqual(expected, actual, string.Format(msg, msgParam0, msgParam1, msgParam2));
    }

    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertIsFalse(bool condition, string msg)
    {
        if (condition)
            Assert.IsFalse(condition, msg);
    }

    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertIsTrue(bool condition, string msg)
    {
        if (!condition)
            Assert.IsTrue(condition, msg);
    }
    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertIsTrue<TP0>(bool condition, string msg, TP0 msgParam0)
    {
        if (!condition)
            Assert.IsTrue(condition, string.Format(msg, msgParam0));
    }
    [Conditional("UNITY_ASSERTIONS")]
    public static void AssertIsTrue<TP0, TP1>(bool condition, string msg, TP0 msgParam0, TP1 msgParam1)
    {
        if (!condition)
            Assert.IsTrue(condition, string.Format(msg, msgParam0, msgParam1));
    }
}
} // namespace Unity.Sentis
