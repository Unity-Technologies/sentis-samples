#define SENTIS_LOG_ENABLED

using System;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Unity.Sentis
{
    /// <summary>
    /// Sentis debug logging utility
    /// </summary>
    class D
    {
        /// <summary>
        /// Warning stack trace collection enabling flag
        /// </summary>
        static bool warningStackTraceEnabled = Application.isEditor;

        /// <summary>
        /// Error stack trace collection enabling flag
        /// </summary>
        static bool errorStackTraceEnabled = true;

        /// <summary>
        /// Debug log stack trace collection enabling flag
        /// </summary>
        static bool logStackTraceEnabled = false;

        /// <summary>
        /// Warning logging enabled flag
        /// </summary>
        static bool warningEnabled = true;

        /// <summary>
        /// Error logging enabled flag
        /// </summary>
        static bool errorEnabled = true;

        /// <summary>
        /// Debug logging enabled flag
        /// </summary>
        static bool logEnabled = true;

#if SENTIS_LOG_ENABLED

        /// <summary>
        /// Log warning
        /// </summary>
        /// <param name="message">message</param>
        internal static void LogWarning(object message)
        {
            if (!warningEnabled)
                return;

            if (!warningStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
                    UnityEngine.Debug.LogWarning(message);
                    Application.SetStackTraceLogType(LogType.Warning, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogWarning(message);
                }

            }
            else
            {
                UnityEngine.Debug.LogWarning(message);
            }
        }

        /// <summary>
        /// Log warning
        /// </summary>
        /// <param name="message">message</param>
        /// <param name="context">context</param>
        internal static void LogWarning(object message, Object context)
        {
            if (!warningEnabled)
                return;

            if (!warningStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
                    UnityEngine.Debug.LogWarning(message, context);
                    Application.SetStackTraceLogType(LogType.Warning, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogWarning(message, context);
                }
            }
            else
            {
                UnityEngine.Debug.LogWarning(message, context);
            }
        }

        /// <summary>
        /// Log error
        /// </summary>
        /// <param name="message">message</param>
        internal static void LogError(object message)
        {
            if (!errorEnabled)
                return;

            if (!errorStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Error, StackTraceLogType.None);
                    UnityEngine.Debug.LogError(message);
                    Application.SetStackTraceLogType(LogType.Error, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogError(message);
                }
            }
            else
            {
                UnityEngine.Debug.LogError(message);
            }
        }

        /// <summary>
        /// Log error
        /// </summary>
        /// <param name="message">message</param>
        /// <param name="context">context</param>
        internal static void LogError(object message, Object context)
        {
            if (!errorEnabled)
                return;

            if (!errorStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Error, StackTraceLogType.None);
                    UnityEngine.Debug.LogError(message, context);
                    Application.SetStackTraceLogType(LogType.Error, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogError(message, context);
                }
            }
            else
            {
                UnityEngine.Debug.LogError(message, context);
            }
        }

        /// <summary>
        /// Log debug info
        /// </summary>
        /// <param name="message">message</param>
        internal static void Log(object message)
        {
            if (!logEnabled)
                return;

            if (!logStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);
                    UnityEngine.Debug.Log(message);
                    Application.SetStackTraceLogType(LogType.Log, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.Log(message);
                }
            }
            else
            {
                UnityEngine.Debug.Log(message);
            }
        }

        /// <summary>
        /// Log debug info
        /// </summary>
        /// <param name="message">message</param>
        /// <param name="context">context</param>
        internal static void Log(object message, Object context)
        {
            if (!logEnabled)
                return;

            if (!logStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);
                    UnityEngine.Debug.Log(message, context);
                    Application.SetStackTraceLogType(LogType.Log, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.Log(message, context);
                }
            }
            else
            {
                UnityEngine.Debug.Log(message, context);
            }
        }
#else
        internal static void LogWarning(object message)
        {

        }

        internal static void LogWarning(object message, Object context)
        {

        }

        internal static void LogError(object message)
        {

        }

        internal static void LogError(object message, Object context)
        {

        }

        internal static void Log(object message)
        {

        }

        internal static void Log(object message, Object context)
        {

        }
#endif
    }

    class Debug : D { }
}
