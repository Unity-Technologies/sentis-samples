using System;
using System.Collections.Generic;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents a dictionary of partial tensors during partial tensor inference.
    /// </summary>
    class PartialInferenceContext
    {
        Dictionary<string, PartialTensor> m_PartialTensors;

        /// <summary>
        /// Instantiates and returns an empty partial inference context.
        /// </summary>
        public PartialInferenceContext()
        {
            m_PartialTensors = new Dictionary<string, PartialTensor>();
        }

        /// <summary>
        /// Dictionary of partial tensors indexed by name.
        /// </summary>
        public Dictionary<string, PartialTensor> PartialTensors => m_PartialTensors;

        /// <summary>
        /// Add partial tensor with a given name to context.
        /// </summary>
        public void AddPartialTensor(string name, PartialTensor partialTensor)
        {
            if (m_PartialTensors.TryGetValue(name, out var prevTensor))
                partialTensor = PartialTensor.MaxDefinedPartialTensor(partialTensor, prevTensor);
            m_PartialTensors[name] = partialTensor;
        }

        /// <summary>
        /// Returns array of partial tensors from array of names.
        /// </summary>
        public PartialTensor[] GetPartialTensors(string[] names)
        {
            var partialTensors = new PartialTensor[names.Length];

            for (var i = 0; i < partialTensors.Length; i++)
            {
                partialTensors[i] = GetPartialTensor(names[i]);
            }

            return partialTensors;
        }

        /// <summary>
        /// Returns partial tensor from name.
        /// </summary>
        public PartialTensor GetPartialTensor(string name)
        {
            return string.IsNullOrEmpty(name) ? null : m_PartialTensors[name];
        }
    }
}
