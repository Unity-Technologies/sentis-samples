using UnityEngine;

namespace HuggingfaceHub.Utilities
{
    public interface ILogger
    {
        public void LogDebug(string message) => Debug.Log(message);
        public void LogInformation(string message) => Debug.Log(message);
        public void LogWarning(string message) => Debug.LogWarning(message);
        public void LogError(string message) => Debug.LogError(message);
        public void LogException(System.Exception ex) => Debug.LogException(ex);
    }

    class Logging : ILogger
    {
        public void LogDebug(string message) => Debug.Log(message);
        public void LogInformation(string message) => Debug.Log(message);
        public void LogWarning(string message) => Debug.LogWarning(message);
        public void LogError(string message) => Debug.LogError(message);
        public void LogException(System.Exception ex) => Debug.LogException(ex);
    }
}
