using System;

namespace Unity.Sentis {

static class OpsUtils
{
    public static unsafe void PinTensorShapeStrides(TensorShape X, int* shape, int* strides)
    {
        shape[0] = Math.Max(1, X.UnsafeGet(0));
        shape[1] = Math.Max(1, X.UnsafeGet(1));
        shape[2] = Math.Max(1, X.UnsafeGet(2));
        shape[3] = Math.Max(1, X.UnsafeGet(3));
        shape[4] = Math.Max(1, X.UnsafeGet(4));
        shape[5] = Math.Max(1, X.UnsafeGet(5));
        shape[6] = Math.Max(1, X.UnsafeGet(6));
        shape[7] = Math.Max(1, X.UnsafeGet(7));
        strides[7] = 1;
        strides[6] = shape[7];
        strides[5] = strides[6] * shape[6];
        strides[4] = strides[5] * shape[5];
        strides[3] = strides[4] * shape[4];
        strides[2] = strides[3] * shape[3];
        strides[1] = strides[2] * shape[2];
        strides[0] = strides[1] * shape[1];
    }

    public static unsafe void PinMatMulTensorShapeStrides(TensorShape X, TensorShape Y, TensorShape O, int* shapeA, int* stridesA, int* shapeB, int* stridesB, int* shapeO, int* stridesO)
    {
        shapeA[0] = Math.Max(1, X.UnsafeGet(0));
        shapeA[1] = Math.Max(1, X.UnsafeGet(1));
        shapeA[2] = Math.Max(1, X.UnsafeGet(2));
        shapeA[3] = Math.Max(1, X.UnsafeGet(3));
        shapeA[4] = Math.Max(1, X.UnsafeGet(4));
        shapeA[5] = Math.Max(1, X.UnsafeGet(5));
        stridesA[5] = 1;
        stridesA[4] = Math.Max(X.UnsafeGet(5), 1);
        stridesA[3] = stridesA[4] * Math.Max(X.UnsafeGet(4), 1);
        stridesA[2] = stridesA[3] * Math.Max(X.UnsafeGet(3), 1);
        stridesA[1] = stridesA[2] * Math.Max(X.UnsafeGet(2), 1);
        stridesA[0] = stridesA[1] * Math.Max(X.UnsafeGet(1), 1);

        shapeB[0] = Math.Max(1, Y.UnsafeGet(0));
        shapeB[1] = Math.Max(1, Y.UnsafeGet(1));
        shapeB[2] = Math.Max(1, Y.UnsafeGet(2));
        shapeB[3] = Math.Max(1, Y.UnsafeGet(3));
        shapeB[4] = Math.Max(1, Y.UnsafeGet(4));
        shapeB[5] = Math.Max(1, Y.UnsafeGet(5));
        stridesB[5] = 1;
        stridesB[4] = shapeB[5];
        stridesB[3] = stridesB[4] * shapeB[4];
        stridesB[2] = stridesB[3] * shapeB[3];
        stridesB[1] = stridesB[2] * shapeB[2];
        stridesB[0] = stridesB[1] * shapeB[1];

        shapeO[0] = Math.Max(1, O.UnsafeGet(0));
        shapeO[1] = Math.Max(1, O.UnsafeGet(1));
        shapeO[2] = Math.Max(1, O.UnsafeGet(2));
        shapeO[3] = Math.Max(1, O.UnsafeGet(3));
        shapeO[4] = Math.Max(1, O.UnsafeGet(4));
        shapeO[5] = Math.Max(1, O.UnsafeGet(5));
        stridesO[5] = 1;
        stridesO[4] = shapeO[5];
        stridesO[3] = stridesO[4] * shapeO[4];
        stridesO[2] = stridesO[3] * shapeO[3];
        stridesO[1] = stridesO[2] * shapeO[2];
        stridesO[0] = stridesO[1] * shapeO[1];
    }

    public static void GetScaleAndBias(float inputShape, float outputShape, float inputScale, Layers.CoordTransformMode coordTransformMode, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, out float outputScale, out float outputBias)
    {
        if (coordTransformMode == Layers.CoordTransformMode.HalfPixel ||
            coordTransformMode == Layers.CoordTransformMode.PytorchHalfPixel)
            outputBias = 0.5f / inputScale - 0.5f;
        else
            outputBias = 0;

        if (coordTransformMode == Layers.CoordTransformMode.AlignCorners)
            outputScale = (inputShape - 1.0f) / (outputShape - 1.0f);
        else
            outputScale = 1 / inputScale;

        if ((coordTransformMode == Layers.CoordTransformMode.AlignCorners ||
             coordTransformMode == Layers.CoordTransformMode.PytorchHalfPixel) &&
            outputShape <= 1)
        {
            outputScale = 0;
            outputBias = 0;
        }

        if (interpolationMode == Layers.InterpolationMode.Nearest)
        {
            if (nearestMode == Layers.NearestMode.RoundPreferCeil)
                outputBias += 0.5f;
            else if (nearestMode == Layers.NearestMode.RoundPreferFloor)
                outputBias -= 0.5f;
        }
    }
}
} // namespace Unity.Sentis
