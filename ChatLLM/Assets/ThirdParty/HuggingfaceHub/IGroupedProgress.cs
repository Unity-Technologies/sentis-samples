using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HuggingfaceHub
{
    /// <summary>
    /// Progress callback when downloading multiple files
    /// </summary>
    public interface IGroupedProgress
    {
        /// <summary>
        /// Report the progress.
        /// </summary>
        /// <param name="filename"></param>
        /// <param name="progress">A value from 0 to 100.</param>
        void Report(string filename, int progress);
    }
}
