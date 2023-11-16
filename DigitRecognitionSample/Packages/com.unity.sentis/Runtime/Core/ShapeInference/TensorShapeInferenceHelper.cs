using System;
using UnityEngine;

namespace Unity.Sentis
{
    static class TensorShapeHelper
    {
        public static TensorShape ConcatShape(Tensor[] tensors, int axis)
        {
            TensorShape output = tensors[0].shape;

            for (int i = 1; i < tensors.Length; ++i)
            {
                TensorShape shape = tensors[i].shape;
                output = output.Concat(shape, axis);
            }

            return output;
        }

        public static TensorShape BroadcastShape(Tensor[] tensors)
        {
            TensorShape output = tensors[0].shape;
            for (int i = 1; i < tensors.Length; ++i)
            {
                TensorShape shape = tensors[i].shape;
                output = output.Broadcast(shape);
            }

            return output;
        }

        public static TensorShape BroadcastShape(Tensor a, Tensor b)
        {
            return a.shape.Broadcast(b.shape);
        }
    }
}

