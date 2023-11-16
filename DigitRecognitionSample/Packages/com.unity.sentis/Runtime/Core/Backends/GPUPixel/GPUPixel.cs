using UnityEngine;
using System.Runtime.CompilerServices;
using UnityEngine.Assertions;
using System;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis
{
    readonly struct PixelFunc
    {
        readonly Material m_Material;

        public PixelFunc(string name)
        {
            m_Material = PixelShaderSingleton.Instance.FindMaterial(name);
        }

        public void SetBool(int nameID, bool value)
        {
            m_Material.SetInt(nameID, value ? 1 : 0);
        }

        public void SetInt(int nameID, int value)
        {
            m_Material.SetInteger(nameID, value);
        }

        public void SetFloatArray(int nameID, float[] values)
        {
            m_Material.SetFloatArray(nameID, values);
        }

        public void SetFloat(int nameID, float value)
        {
            m_Material.SetFloat(nameID, value);
        }

        public void SetVector(int nameID, Vector4 value)
        {
            m_Material.SetVector(nameID, value);
        }

        public void SetTexture(int nameID, Texture value)
        {
            m_Material.SetTexture(nameID, value);
        }

        public void EnableKeyword(string keyword)
        {
            m_Material.EnableKeyword(keyword);
        }

        internal void SetTensor(TensorProperties tensorProperties, TextureTensorData pinX)
        {
            m_Material.SetTexture(tensorProperties.k_ID_Ptr, pinX.bufferAsTexture);
            m_Material.SetInteger(tensorProperties.k_ID_WidthMask, pinX.widthMask);
            m_Material.SetInteger(tensorProperties.k_ID_WidthShift, pinX.widthShift);
        }

        internal void SetTensorBlockStride(TensorProperties tensorProperties, TextureTensorData pinX)
        {
            m_Material.SetInteger(tensorProperties.k_ID_StrideAxis, pinX.strideAxis);
            m_Material.SetInteger(tensorProperties.k_ID_DimAxis, pinX.dimAxis);
            m_Material.SetInteger(tensorProperties.k_ID_DimBlocked, pinX.dimAxisDiv4);
        }

        public void Dispatch(TextureTensorData pinO)
        {
            m_Material.SetInteger(k_TensorPropertiesO.k_ID_WidthShift, pinO.widthShift);
            m_Material.SetInteger(k_ID_LengthO, pinO.shape.length);
            Graphics.Blit(null, pinO.bufferAsTexture, m_Material);
        }

        public void Dispatch(RenderTexture renderTexture)
        {
            Graphics.Blit(null, renderTexture, m_Material);
        }
    }

    static class PixelShaderHelper
    {
        static readonly float[] k_ScratchPadFloat8 = new float[8];

        public static unsafe void SetInt8(this PixelFunc func, int nameID, int* ptr)
        {
            for (var i = 0; i < 8; i++)
                k_ScratchPadFloat8[i] = ptr[i];

            func.SetFloatArray(nameID, k_ScratchPadFloat8);
        }

        public static void SetShapeStrides(this PixelFunc func, TensorProperties properties, TensorShape shape)
        {
            unsafe
            {
                var pShape = stackalloc int[TensorShape.maxRank];
                var pStrides = stackalloc int[TensorShape.maxRank];
                OpsUtils.PinTensorShapeStrides(shape, pShape, pStrides);
                func.SetInt8(properties.k_ID_Shape, pShape);
                func.SetInt8(properties.k_ID_Strides, pStrides);
            }
            func.SetInt(properties.k_ID_Rank, shape.rank);
        }

        public static void SetShape(this PixelFunc func, int nameID, TensorShape shape)
        {
            for (var i = 0; i < 8; i++)
            {
                k_ScratchPadFloat8[i] = i < shape.rank ? shape[-1 - i] : 1;
            }

            func.SetFloatArray(nameID, k_ScratchPadFloat8);
        }

        public static void SetStrides(this PixelFunc func, int nameID, TensorShape shape)
        {
            var stride = 1;
            var rank = shape.rank;
            for (var i = 0; i < rank; i++)
            {
                var dim = shape[rank - 1 - i];
                k_ScratchPadFloat8[i] = dim == 1 ? 0 : stride;
                stride *= dim;
            }

            Array.Clear(k_ScratchPadFloat8, rank, 8 - rank);

            func.SetFloatArray(nameID, k_ScratchPadFloat8);
        }
    }

    /// <summary>
    /// Represents a GPUPixel backend ops.
    /// </summary>
    public class GPUPixelBackend : CPUBackend
    {
        /// <summary>
        /// Initializes and returns an instance of `GPUPixelBackend`.
        /// </summary>
        /// <param name="allocator">The allocator to use when allocating tensors.</param>
        public GPUPixelBackend(ITensorAllocator allocator = null)
            : base(allocator) { }

        /// <inheritdoc/>
        public override DeviceType deviceType => DeviceType.CPU;

        /// <summary>
        /// Pins the tensor as `TextureTensorData` on any axis (choose last).
        /// </summary>
        static TextureTensorData PinBlockAny(Tensor X, bool clearOnInit = true)
        {
            if (X.tensorOnDevice is TextureTensorData textureTensorData)
                return textureTensorData;
            return TextureTensorData.Pin(X, X.shape.rank - 1, clearOnInit);
        }

        /// <summary>
        /// Pins the tensor as TextureTensorData on any axis except `nonBlockAxis`. (Choose last unless avoid, else one before last.)
        /// </summary>
        static TextureTensorData PinBlockOther(Tensor X, int nonBlockAxis, bool clearOnInit = true)
        {
            if (X.tensorOnDevice is TextureTensorData textureTensorData)
                if (textureTensorData.blockAxis != nonBlockAxis)
                    return textureTensorData;
            var axis = nonBlockAxis == X.shape.rank - 1 ? X.shape.rank - 2 : X.shape.rank - 1;
            return TextureTensorData.Pin(X, axis, clearOnInit);
        }

        /// <summary>
        /// Pins the tensor X blocking along the same axis as a given other TextureTensorData
        /// This can be used to block an output tensor along the same axis as an input tensor for an op
        /// </summary>
        static TextureTensorData PinAsSame(Tensor X, TextureTensorData other, bool clearOnInit = true)
        {
            return TextureTensorData.Pin(X, X.shape.rank - other.shape.rank + other.blockAxis, clearOnInit);
        }

        /// <summary>
        /// Pin tensors A and B along the same axis, the blocking for A takes priority in case neither tensor is pinned or
        /// both tensors are pinned
        /// </summary>
        static void PinBothSame(Tensor A, Tensor B)
        {
            var pinA = A.tensorOnDevice as TextureTensorData;
            var pinB = B.tensorOnDevice as TextureTensorData;
            if (pinA == null == pinB is null)
                pinA = PinBlockAny(A);
            else if (pinB != null)
                pinA = PinAsSame(A, pinB);
            PinAsSame(B, pinA);
        }

        /// <inheritdoc/>
        public override void Cast(Tensor X, Tensor O)
        {
            if (X.dataType == O.dataType)
            {
                MemCopy(X, O);
                return;
            }

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/Cast");
            func.EnableKeyword(X.dataType == DataType.Int ? "IntToFloat" : "FloatToInt");
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void MemClear(Tensor O)
        {
            if (O.dataType == DataType.Float)
                MemSet(O as TensorFloat, 0f);
            else
                MemSet(O as TensorInt, 0);
        }

        /// <inheritdoc/>
        public override void MemSet(TensorFloat O, float value)
        {
            var func = new PixelFunc("Hidden/Sentis/ConstantOfShape");
            var pinO = PinBlockAny(O, false);
            func.EnableKeyword("Float");
            func.SetFloat(k_ID_memValueFloat, value);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void MemSet(TensorInt O, int value)
        {
            var func = new PixelFunc("Hidden/Sentis/ConstantOfShape");
            var pinO = PinBlockAny(O, false);
            func.EnableKeyword("Int");
            func.SetInt(k_ID_memValueInt, value);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void MatMul(TensorFloat X, TensorFloat Y, TensorFloat O)
        {
            var xShape = X.shape.rank == 1 ? new TensorShape(1, X.shape[0]) : X.shape;
            var yShape = Y.shape.rank == 1 ? new TensorShape(Y.shape[0], 1) : Y.shape;
            var oShape = X.shape.rank > 1 && Y.shape.rank > 1 ? O.shape : xShape.MatMul(yShape);

            var func = new PixelFunc("Hidden/Sentis/MatMul");

            var pinO = TextureTensorData.Pin(O, Y.shape.rank == 1 ? -1 : O.shape.rank - 1, clearOnInit: false);
            var pinA = TextureTensorData.Pin(X, X.shape.rank - 1);
            var pinB = TextureTensorData.Pin(Y, Y.shape.rank == 1 ? -1 : Y.shape.rank - 1);
            if (xShape != pinA.shape)
                pinA.SetShape(xShape, xShape.rank - 1);
            if (yShape != pinB.shape)
                pinB.SetShape(yShape, yShape.rank - 1);
            if (oShape != pinO.shape)
                pinO.SetShape(oShape, oShape.rank - 1);

            func.SetTensor(k_TensorPropertiesA, pinA);
            func.SetTensor(k_TensorPropertiesB, pinB);

            func.SetShape(k_ID_DimO, pinO.blockedShape);
            func.SetStrides(k_TensorPropertiesA.k_ID_Strides, pinA.blockedShape);
            func.SetStrides(k_TensorPropertiesB.k_ID_Strides, pinB.blockedShape);
            func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.dimAxis);
            func.SetInt(k_ID_Kdiv4, pinA.dimAxisDiv4);

            func.Dispatch(pinO);

            if (X.shape != pinA.shape)
                pinA.SetShape(X.shape, X.shape.rank - 1);
            if (Y.shape != pinB.shape)
                pinB.SetShape(Y.shape, Y.shape.rank == 1 ? -1 : Y.shape.rank - 1);
            if (O.shape != pinO.shape)
                pinO.SetShape(O.shape, Y.shape.rank == 1 ? -1 : O.shape.rank - 1);
        }

        /// <inheritdoc/>
        public override void MatMul2D(TensorFloat X, TensorFloat Y, TensorFloat O, bool xTranspose, bool yTranspose)
        {
            var func = new PixelFunc("Hidden/Sentis/Gemm");

            var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);
            var pinX = TextureTensorData.Pin(X, xTranspose ? 0 : 1);
            var pinW = TextureTensorData.Pin(Y, yTranspose ? 0 : 1);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesW, pinW);

            if (xTranspose)
                func.EnableKeyword("TRANSPOSE_X");
            if (yTranspose)
                func.EnableKeyword("TRANSPOSE_W");
            func.SetInt(k_ID_M, pinO.blockedShape[0]);
            func.SetInt(k_ID_K, pinX.dimAxis);
            func.SetInt(k_ID_Kdiv4, pinX.dimAxisDiv4);
            func.SetInt(k_ID_Ndiv4, pinO.dimAxisDiv4);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Dense(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Layers.FusableActivation fusedActivation)
        {
            var func = new PixelFunc("Hidden/Sentis/Dense");

            var pinO = TextureTensorData.Pin(O, O.shape.rank - 1, clearOnInit: false);
            var pinX = TextureTensorData.Pin(X, X.shape.rank - 1);
            var pinW = TextureTensorData.Pin(W, W.shape.rank - 1);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesW, pinW);

            if (B != null)
            {
                var pinB = TextureTensorData.Pin(B, B.shape.rank - 1);
                func.SetTensor(k_TensorPropertiesB, pinB);
                func.EnableKeyword("Dense");
            }

            func.SetInt(k_TensorPropertiesO.k_ID_DimBlocked, pinO.dimAxisDiv4);
            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.dimAxis);
            func.SetInt(k_TensorPropertiesX.k_ID_DimBlocked, pinX.dimAxisDiv4);
            func.SetInt(k_TensorPropertiesW.k_ID_DimBlocked, pinW.dimAxisDiv4);

            if (fusedActivation == Layers.FusableActivation.Relu)
                func.EnableKeyword("Relu");

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Conv(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
        {
            if (X.shape.rank > 5)
            {
                base.Conv(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
                return;
            }

            var isDepthwise = K.shape[0] == groups && K.shape[1] == 1;

            var pinX = TextureTensorData.Pin(X, 1);
            var pinK = TextureTensorData.Pin(K, isDepthwise ? 0 : 1);
            var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

            var numSpatialDims = X.shape.rank - 2;

            PixelFunc func;

            if (isDepthwise)
            {
                func = new PixelFunc("Hidden/Sentis/DepthwiseConv");
            }
            else if (groups > 1)
            {
                func = new PixelFunc("Hidden/Sentis/GroupedConv");
                func.SetInt(k_ID_X_channels, pinX.shape[1]);
                func.SetInt(k_ID_O_channels, pinO.shape[1]);
                func.SetInt(k_ID_K_channelsDivGroupDiv4, pinK.dimAxisDiv4);
            }
            else
            {
                func = new PixelFunc("Hidden/Sentis/Conv");
                func.SetInt(k_ID_X_channels, pinX.shape[1]);
            }

            if (numSpatialDims == 1)
                func.EnableKeyword("CONV1D");
            else if (numSpatialDims == 2)
                func.EnableKeyword("CONV2D");
            else
                func.EnableKeyword("CONV3D");

            func.SetInt(k_ID_O_width, pinO.shape[-1]);
            func.SetInt(k_ID_X_width, pinX.shape[-1]);
            func.SetInt(k_ID_K_width, pinK.shape[-1]);
            func.SetInt(k_ID_StrideX, strides[numSpatialDims - 1]);
            func.SetInt(k_ID_PadX, pads[numSpatialDims - 1]);
            func.SetInt(k_ID_DilationX, dilations[numSpatialDims - 1]);
            func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);
            func.SetTensor(k_TensorPropertiesK, pinK);
            if (B != null)
            {
                func.EnableKeyword("USEBIAS");
                var pinB = TextureTensorData.Pin(B, 0);
                func.SetTensor(k_TensorPropertiesB, pinB);
            }
            func.SetInt(k_ID_Groups, groups);

            if (numSpatialDims > 1)
            {
                func.SetInt(k_ID_O_height, pinO.shape[-2]);
                func.SetInt(k_ID_X_height, pinX.shape[-2]);
                func.SetInt(k_ID_K_height, pinK.shape[-2]);
                func.SetInt(k_ID_StrideY, strides[numSpatialDims - 2]);
                func.SetInt(k_ID_PadY, pads[numSpatialDims - 2]);
                func.SetInt(k_ID_DilationY, dilations[numSpatialDims - 2]);
            }

            if (numSpatialDims > 2)
            {
                func.SetInt(k_ID_O_depth, pinO.shape[-3]);
                func.SetInt(k_ID_X_depth, pinX.shape[-3]);
                func.SetInt(k_ID_K_depth, pinK.shape[-3]);
                func.SetInt(k_ID_StrideZ, strides[numSpatialDims - 3]);
                func.SetInt(k_ID_PadZ, pads[numSpatialDims - 3]);
                func.SetInt(k_ID_DilationZ, dilations[numSpatialDims - 3]);
            }

            if (fusedActivation == Layers.FusableActivation.Relu)
                func.EnableKeyword("RELU");

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void ConvTranspose(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
        {
            if (X.shape.rank > 5)
            {
                base.ConvTranspose(X, W, B, O, strides, pads, outputPadding, fusedActivation);
                return;
            }

            var func = new PixelFunc("Hidden/Sentis/ConvTranspose");

            var pinX = TextureTensorData.Pin(X, 1);
            var pinK = TextureTensorData.Pin(W, 0);
            if (B != null)
            {
                var pinB = TextureTensorData.Pin(B, 0);
                func.SetTensor(k_TensorPropertiesB, pinB);
                func.EnableKeyword("USEBIAS");
            }
            var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

            var numSpatialDims = X.shape.rank - 2;

            if (numSpatialDims == 1)
                func.EnableKeyword("CONVTRANSPOSE1D");
            else if (numSpatialDims == 2)
                func.EnableKeyword("CONVTRANSPOSE2D");
            else
                func.EnableKeyword("CONVTRANSPOSE3D");

            func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_X_channels, pinX.dimAxis);
            func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);
            func.SetTensor(k_TensorPropertiesK, pinK);
            func.SetInt(k_ID_K_mDivGroup, pinK.shape[1]);

            func.SetInt(k_ID_K_width, pinK.shape[-1]);
            func.SetInt(k_ID_O_width, pinO.shape[-1]);
            func.SetInt(k_ID_X_width, pinX.shape[-1]);
            func.SetInt(k_ID_PadX, W.shape[-1] - pads[numSpatialDims - 1] - 1);
            func.SetInt(k_ID_StrideX, strides[numSpatialDims - 1]);

            if (numSpatialDims > 1)
            {
                func.SetInt(k_ID_K_height, pinK.shape[-2]);
                func.SetInt(k_ID_X_height, pinX.shape[-2]);
                func.SetInt(k_ID_O_height, pinO.shape[-2]);
                func.SetInt(k_ID_StrideY, strides[numSpatialDims - 2]);
                func.SetInt(k_ID_PadY, W.shape[-2] - pads[numSpatialDims - 2] - 1);
            }

            if (numSpatialDims > 2)
            {
                func.SetInt(k_ID_K_depth, pinK.shape[-3]);
                func.SetInt(k_ID_X_depth, pinX.shape[-3]);
                func.SetInt(k_ID_O_depth, pinO.shape[-3]);
                func.SetInt(k_ID_StrideZ, strides[numSpatialDims - 3]);
                func.SetInt(k_ID_PadZ, W.shape[-3] - pads[numSpatialDims - 3] - 1);
            }

            if (fusedActivation == Layers.FusableActivation.Relu)
                func.EnableKeyword("RELU");

            func.Dispatch(pinO);
        }

        void Activation(TensorFloat X, TensorFloat O, string kernelName, float alpha = 0f, float beta = 0f)
        {
            var func = new PixelFunc("Hidden/Sentis/Activation");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, clearOnInit: false);

            func.SetFloat(k_ID_Alpha, alpha);
            func.SetFloat(k_ID_Beta, beta);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

            func.EnableKeyword(kernelName);

            func.Dispatch(pinO);
        }

        void Activation(TensorInt X, TensorInt O, string kernelName)
        {
            var func = new PixelFunc("Hidden/Sentis/ActivationInt");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, clearOnInit: false);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

            func.EnableKeyword(kernelName);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Relu(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Relu");
        }

        /// <inheritdoc/>
        public override void Relu6(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Relu6");
        }

        /// <inheritdoc/>
        public override void LeakyRelu(TensorFloat X, TensorFloat O, float alpha)
        {
            Logger.AssertIsTrue(alpha <= 1, "LeakyRelu.ValueError: alpha is supposed to be <= 1, got {0}", alpha);
            Activation(X, O, "LeakyRelu", alpha);
        }

        /// <inheritdoc/>
        public override void Tanh(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Tanh");
        }

        /// <inheritdoc/>
        public override void Softplus(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Softplus");
        }

        /// <inheritdoc/>
        public override void Sigmoid(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Sigmoid");
        }

        /// <inheritdoc/>
        public override void HardSigmoid(TensorFloat X, TensorFloat O, float alpha, float beta)
        {
            Activation(X, O, "HardSigmoid", alpha, beta);
        }

        /// <inheritdoc/>
        public override void Elu(TensorFloat X, TensorFloat O, float alpha)
        {
            Activation(X, O, "Elu", alpha);
        }

        /// <inheritdoc/>
        public override void Gelu(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Gelu");
        }

        /// <inheritdoc/>
        public override void Shrink(TensorFloat X, TensorFloat O, float bias, float lambd)
        {
            Activation(X, O, "Shrink", bias, lambd);
        }

        /// <inheritdoc/>
        public override void Selu(TensorFloat X, TensorFloat O, float alpha, float gamma)
        {
            Activation(X, O, "Selu", alpha, gamma);
        }

        /// <inheritdoc/>
        public override void Swish(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Swish");
        }

        /// <inheritdoc/>
        public override void Abs(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Abs");
        }

        /// <inheritdoc/>
        public override void Abs(TensorInt X, TensorInt O)
        {
            Activation(X, O, "Abs");
        }

        /// <inheritdoc/>
        public override void Neg(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Neg");
        }

        /// <inheritdoc/>
        public override void Neg(TensorInt X, TensorInt O)
        {
            Activation(X, O, "Neg");
        }

        /// <inheritdoc/>
        public override void Not(TensorInt X, TensorInt O)
        {
            Activation(X, O, "Not");
        }

        /// <inheritdoc/>
        public override void Ceil(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Ceil");
        }

        /// <inheritdoc/>
        public override void Clip(TensorFloat X, TensorFloat O, float min, float max)
        {
            Activation(X, O, "Clip", min, max);
        }

        /// <inheritdoc/>
        public override void Floor(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Floor");
        }

        /// <inheritdoc/>
        public override void Round(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Round");
        }

        /// <inheritdoc/>
        public override void Reciprocal(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Reciprocal");
        }

        /// <inheritdoc/>
        public override void Square(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Square");
        }

        /// <inheritdoc/>
        public override void Exp(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Exp");
        }

        /// <inheritdoc/>
        public override void Log(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Log");
        }

        /// <inheritdoc/>
        public override void Sqrt(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Sqrt");
        }

        /// <inheritdoc/>
        public override void Celu(TensorFloat X, TensorFloat O, float alpha)
        {
            Activation(X, O, "Celu", alpha);
        }

        /// <inheritdoc/>
        public override void HardSwish(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "HardSwish");
        }

        /// <inheritdoc/>
        public override void Softsign(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Softsign");
        }

        /// <inheritdoc/>
        public override void ScalarMad(TensorFloat X, TensorFloat O, float s, float b)
        {
            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/ScalarMad");

            func.SetTensor(k_TensorPropertiesX, pinX);

            func.SetFloat(k_ID_s, s);
            func.SetFloat(k_ID_b, b);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Sign(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Sign");
        }

        /// <inheritdoc/>
        public override void Sign(TensorInt X, TensorInt O)
        {
            Activation(X, O, "Sign");
        }

        /// <inheritdoc/>
        public override void ThresholdedRelu(TensorFloat X, TensorFloat O, float alpha)
        {
            Activation(X, O, "ThresholdedRelu", alpha);
        }

        /// <inheritdoc/>
        public override void PRelu(TensorFloat X, TensorFloat S, TensorFloat O)
        {
            Broadcast(X, S, O, "PRelu");
        }

        /// <inheritdoc/>
        public override void And(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "And");
        }

        /// <inheritdoc/>
        public override void Equal(TensorFloat A, TensorFloat B, TensorInt O)
        {
            Broadcast(A, B, O, "Equal");
        }

        /// <inheritdoc/>
        public override void Equal(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "EqualInt");
        }

        /// <inheritdoc/>
        public override void Greater(TensorFloat A, TensorFloat B, TensorInt O)
        {
            Broadcast(A, B, O, "Greater");
        }

        /// <inheritdoc/>
        public override void Greater(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "GreaterInt");
        }

        /// <inheritdoc/>
        public override void GreaterOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
        {
            Broadcast(A, B, O, "GreaterOrEqual");
        }

        /// <inheritdoc/>
        public override void GreaterOrEqual(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "GreaterOrEqualInt");
        }

        /// <inheritdoc/>
        public override void Less(TensorFloat A, TensorFloat B, TensorInt O)
        {
            Broadcast(A, B, O, "Less");
        }

        /// <inheritdoc/>
        public override void Less(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "LessInt");
        }

        /// <inheritdoc/>
        public override void LessOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
        {
            Broadcast(A, B, O, "LessOrEqual");
        }

        /// <inheritdoc/>
        public override void LessOrEqual(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "LessOrEqualInt");
        }

        /// <inheritdoc/>
        public override void Or(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "Or");
        }

        /// <inheritdoc/>
        public override void Xor(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "Xor");
        }

        /// <inheritdoc/>
        public override void Where(TensorInt C, Tensor A, Tensor B, Tensor O)
        {
            PinBothSame(A, B);
            PinBothSame(A, C);
            var pinX = C.tensorOnDevice as TextureTensorData;
            var pinA = A.tensorOnDevice as TextureTensorData;
            var pinB = B.tensorOnDevice as TextureTensorData;
            var pinO = PinAsSame(O, pinA, false);

            var func = new PixelFunc("Hidden/Sentis/Where");
            func.EnableKeyword(A.dataType == DataType.Int ? "WhereInt" : "WhereFloat");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesA, pinA);
            func.SetTensor(k_TensorPropertiesB, pinB);

            func.SetShape(k_ID_DimO, pinO.blockedShape);
            func.SetStrides(k_TensorPropertiesX.k_ID_Strides, pinX.blockedShape);
            func.SetStrides(k_TensorPropertiesA.k_ID_Strides, pinA.blockedShape);
            func.SetStrides(k_TensorPropertiesB.k_ID_Strides, pinB.blockedShape);

            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.dimAxis);
            func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.dimAxis);
            func.SetInt(k_TensorPropertiesB.k_ID_DimAxis, pinB.dimAxis);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void IsInf(TensorFloat X, TensorInt O, bool detectNegative, bool detectPositive)
        {
            var func = new PixelFunc("Hidden/Sentis/IsInfNaN");
            func.EnableKeyword("IsInf");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, clearOnInit: false);

            func.SetInt(k_ID_detectNegative, detectNegative ? 1 : 0);
            func.SetInt(k_ID_detectPositive, detectPositive ? 1 : 0);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void IsNaN(TensorFloat X, TensorInt O)
        {
            var func = new PixelFunc("Hidden/Sentis/IsInfNaN");
            func.EnableKeyword("IsNaN");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, clearOnInit: false);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Acos(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Acos");
        }

        /// <inheritdoc/>
        public override void Acosh(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Acosh");
        }

        /// <inheritdoc/>
        public override void Asin(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Asin");
        }

        /// <inheritdoc/>
        public override void Asinh(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Asinh");
        }

        /// <inheritdoc/>
        public override void Atan(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Atan");
        }

        /// <inheritdoc/>
        public override void Atanh(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Atanh");
        }

        /// <inheritdoc/>
        public override void Cos(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Cos");
        }

        /// <inheritdoc/>
        public override void Cosh(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Cosh");
        }

        /// <inheritdoc/>
        public override void Sin(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Sin");
        }

        /// <inheritdoc/>
        public override void Sinh(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Sinh");
        }

        /// <inheritdoc/>
        public override void Tan(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Tan");
        }

        /// <inheritdoc/>
        public override void Erf(TensorFloat X, TensorFloat O)
        {
            Activation(X, O, "Erf");
        }

        /// <inheritdoc/>
        public override void Add(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            Broadcast(A, B, O, "Add");
        }

        /// <inheritdoc/>
        public override void Add(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "AddInt");
        }

        /// <inheritdoc/>
        public override void Sub(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            Broadcast(A, B, O, "Sub");
        }

        /// <inheritdoc/>
        public override void Sub(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "SubInt");
        }

        /// <inheritdoc/>
        public override void Div(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            Broadcast(A, B, O, "Div");
        }

        /// <inheritdoc/>
        public override void Div(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "DivInt");
        }

        /// <inheritdoc/>
        public override void Pow(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            Broadcast(A, B, O, "Pow");
        }

        /// <inheritdoc/>
        public override void Pow(TensorFloat A, TensorInt B, TensorFloat O)
        {
            Broadcast(A, B, O, "PowInt");
        }

        /// <inheritdoc/>
        public override void FMod(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            Broadcast(A, B, O, "FMod");
        }

        /// <inheritdoc/>
        public override void FMod(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "FModInt");
        }

        /// <inheritdoc/>
        public override void Mod(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "ModInt");
        }

        /// <inheritdoc/>
        public override void Mul(TensorFloat A, TensorFloat B, TensorFloat O)
        {
            Broadcast(A, B, O, "Mul");
        }

        /// <inheritdoc/>
        public override void Mul(TensorInt A, TensorInt B, TensorInt O)
        {
            Broadcast(A, B, O, "MulInt");
        }

        /// <inheritdoc/>
        public override void Sum(TensorFloat[] inputs, TensorFloat O)
        {
            Broadcast(inputs, O, "Add");
        }
        /// <inheritdoc/>
        public override void Min(TensorFloat[] inputs, TensorFloat O)
        {
            Broadcast(inputs, O, "Min");
        }

        /// <inheritdoc/>
        public override void Min(TensorInt[] inputs, TensorInt O)
        {
            Broadcast(inputs, O, "MinInt");
        }

        /// <inheritdoc/>
        public override void Max(TensorFloat[] inputs, TensorFloat O)
        {
            Broadcast(inputs, O, "Max");
        }

        /// <inheritdoc/>
        public override void Max(TensorInt[] inputs, TensorInt O)
        {
            Broadcast(inputs, O, "MaxInt");
        }

        /// <inheritdoc/>
        public override void Mean(TensorFloat[] inputs, TensorFloat O)
        {
            Broadcast(inputs, O, "Mean");
        }

        /// <inheritdoc/>
        public override void Expand(Tensor X, Tensor O)
        {
            var func = new PixelFunc("Hidden/Sentis/Expand");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            func.SetTensor(k_TensorPropertiesX, pinX);

            func.SetShape(k_ID_DimO, pinO.blockedShape);
            func.SetStrides(k_TensorPropertiesX.k_ID_Strides, pinX.blockedShape);

            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.dimAxis);
            func.Dispatch(pinO);
        }

        void Broadcast(Tensor[] inputs, Tensor O, string kernelName)
        {
            var curX = inputs[0];
            var normalization = 1.0f / inputs.Length;
            for (var t = 1; t < inputs.Length; t++)
            {
                var nextX = inputs[t];
                var oTmp = t == inputs.Length - 1 ? O : NewTensor(TensorShapeHelper.BroadcastShape(curX, nextX), O.dataType, AllocScope.InternalToLayer);
                Broadcast(curX, inputs[t], oTmp, kernelName, t == 1 ? normalization : 1.0f, normalization);
                curX = oTmp;
            }
        }

        void Broadcast(Tensor A, Tensor B, Tensor O, string kernelName, float normalizationX = 0, float normalizationY = 0)
        {
            var isALarger = A.shape.length > B.shape.length;
            PinBothSame(isALarger ? A : B, isALarger ? B : A);
            var pinA = A.tensorOnDevice as TextureTensorData;
            var pinB = B.tensorOnDevice as TextureTensorData;
            var pinO = PinAsSame(O, pinA, false);

            var func = new PixelFunc("Hidden/Sentis/Broadcast");
            func.EnableKeyword(kernelName);

            func.SetTensor(k_TensorPropertiesA, pinA);
            func.SetTensor(k_TensorPropertiesB, pinB);

            func.SetShape(k_ID_DimO, pinO.blockedShape);
            func.SetStrides(k_TensorPropertiesA.k_ID_Strides, pinA.blockedShape);
            func.SetStrides(k_TensorPropertiesB.k_ID_Strides, pinB.blockedShape);

            func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.dimAxis);
            func.SetInt(k_TensorPropertiesB.k_ID_DimAxis, pinB.dimAxis);

            func.SetFloat(k_ID_alpha, normalizationX);
            func.SetFloat(k_ID_beta, normalizationY);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Concat(Tensor[] inputs, Tensor O, int axis)
        {
            axis = O.shape.Axis(axis);

            var oShape = O.shape;
            oShape[axis] = 0;
            TextureTensorData pinA = null;
            TextureTensorData pinB = null;

            var func = new PixelFunc("Hidden/Sentis/Concat");
            if (O.dataType == DataType.Int)
                func.EnableKeyword("INT");
            var strideAxis = O.shape.Strides(axis);
            func.SetInt(k_ID_StrideAxis, strideAxis);
            foreach (var tensor in inputs)
            {
                if (tensor.shape.length == 0)
                    continue;
                if (pinA == null)
                {
                    pinA = PinBlockAny(tensor);
                    func.SetInt(k_ID_StrideAxis, pinA.blockedShape.Strides(axis));
                    if (axis != pinA.blockAxis)
                        func.EnableKeyword("BLOCKWISE");
                    oShape[axis] += pinA.shape[axis];
                    continue;
                }

                pinB = PinAsSame(tensor, pinA);
                oShape[axis] += pinB.shape[axis];
                var pinO = PinAsSame(oShape == O.shape ? O : NewTensor(oShape, O.dataType, AllocScope.InternalToLayer), pinA, false);

                func.SetTensor(k_TensorPropertiesA, pinA);
                func.SetTensor(k_TensorPropertiesB, pinB);
                func.SetInt(k_ID_ConcatLengthA, pinA.shape[axis]);

                if (axis == pinO.blockAxis)
                    func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
                else
                    func.SetInt(k_TensorPropertiesO.k_ID_DimAxis, pinO.blockedShape[axis]);
                func.SetInt(k_TensorPropertiesA.k_ID_DimAxis, pinA.blockedShape[axis]);
                func.SetInt(k_TensorPropertiesB.k_ID_DimAxis, pinB.blockedShape[axis]);
                func.Dispatch(pinO);
                pinA = pinO;
            }

            if (pinB == null)
            {
                func = new PixelFunc("Hidden/Sentis/Copy");
                if (O.dataType == DataType.Int)
                    func.EnableKeyword("INT");
                func.SetTensor(k_TensorPropertiesX, pinA);
                func.Dispatch(PinAsSame(O, pinA, false));
            }
        }

        unsafe void Slice(Tensor X, Tensor O, int* startsLocal, int* stepsLocal)
        {
            if (!(X.tensorOnDevice is TextureTensorData))
            {
                // find axis that isn't sliced along
                for (var axis = X.shape.rank - 1; axis >= 0; axis--)
                {
                    if (X.shape[axis] == O.shape[axis] && startsLocal[axis] == 1 && stepsLocal[axis] == 1)
                    {
                        TextureTensorData.Pin(X, axis);
                        break;
                    }
                }
            }

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/Slice");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");

            func.SetTensor(k_TensorPropertiesX, pinX);

            TensorShape xShape;

            if (pinX.dimAxis == pinO.dimAxis && startsLocal[pinX.blockAxis] == 1 && stepsLocal[pinX.blockAxis] == 1)
            {
                func.EnableKeyword("BLOCKWISE");
                func.SetShape(k_ID_DimO, pinO.blockedShape);
                xShape = pinX.blockedShape;
            }
            else
            {
                func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
                func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
                func.SetShape(k_ID_DimO, pinO.shape);
                xShape = pinX.shape;
            }

            var offsetX = 0;
            var strideX = 1;
            var stridesX = stackalloc int[8];
            for (var i = 0; i < pinX.shape.rank; i++)
            {
                var axis = pinO.shape.rank - 1 - i;
                offsetX += startsLocal[axis] * strideX;
                stridesX[i] = strideX * stepsLocal[axis];
                strideX *= xShape[axis];
            }

            func.SetInt8(k_TensorPropertiesX.k_ID_Strides, stridesX);
            func.SetInt(k_ID_OffsetX, offsetX);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Slice(Tensor X, Tensor O, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
        {
            unsafe
            {
                var startsLocal = stackalloc int[TensorShape.maxRank];
                var stepsLocal = stackalloc int[TensorShape.maxRank];

                for (var i = 0; i < 8; i++)
                {
                    stepsLocal[i] = 1;
                }

                for (var i = 0; i < starts.Length; i++)
                {
                    var axis = axes == null ? i : X.shape.Axis(axes[i]);
                    var step = steps != null ? steps[i] : 1;
                    var dim = X.shape[axis];

                    var clampAdjustDirection = step < 0 ? -1 : 0;

                    var start = starts[i];
                    start = start < 0 ? dim + start : start;
                    start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

                    startsLocal[axis] = start;
                    stepsLocal[axis] = step;
                }

                Slice(X, O, startsLocal, stepsLocal);
            }
        }

        void SoftmaxActivation(Tensor X, TensorFloat O, int reduceAxis, string endKernelName)
        {
            //Allocate temp tensors
            var reduceOpShape = X.shape.Reduce(reduceAxis);
            var B = NewTempTensorFloat(reduceOpShape);
            var S = NewTempTensorFloat(reduceOpShape);

            reduceAxis = X.shape.Axis(reduceAxis);

            var pinX = PinBlockOther(X, nonBlockAxis: reduceAxis);
            var pinO = PinAsSame(O, pinX, false);
            var pinB = PinAsSame(B, pinX, false);
            var pinS = PinAsSame(S, pinX, false);

            var dimAxis = pinX.blockedShape[reduceAxis];
            var strideAxis = pinX.blockedShape.Strides(reduceAxis);

            // x_max = X.max(axis=1)
            {
                var func = new PixelFunc("Hidden/Sentis/Reduce");
                func.EnableKeyword("ReduceMax");
                func.SetTensor(k_TensorPropertiesX, pinX);
                func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, strideAxis);
                func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, dimAxis);
                func.Dispatch(pinB);
            }
            // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
            {
                var func = new PixelFunc("Hidden/Sentis/ReduceExpBias");
                func.SetTensor(k_TensorPropertiesX, pinX);
                func.SetTensor(k_TensorPropertiesB, pinB);
                func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, strideAxis);
                func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, dimAxis);
                func.Dispatch(pinS);
            }
            {
                var func = new PixelFunc("Hidden/Sentis/Softmax");
                func.EnableKeyword(endKernelName);
                func.SetTensor(k_TensorPropertiesX, pinX);
                func.SetTensor(k_TensorPropertiesB, pinB);
                func.SetTensor(k_TensorPropertiesS, pinS);
                func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, strideAxis);
                func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, dimAxis);
                func.Dispatch(pinO);
            }
        }

        /// <inheritdoc/>
        public override void Softmax(TensorFloat X, TensorFloat O, int axis)
        {
            SoftmaxActivation(X, O, axis, "SOFTMAXEND");
        }

        /// <inheritdoc/>
        public override void LogSoftmax(TensorFloat X, TensorFloat O, int axis)
        {
            SoftmaxActivation(X, O, axis, "LOGSOFTMAXEND");
        }

        /// <inheritdoc/>
        public override void Hardmax(TensorFloat X, TensorFloat O, int axis)
        {
            axis = X.shape.Axis(axis);

            // Allocate temp tensors
            var reduceOpShape = X.shape.Reduce(axis);
            var argMax = NewTempTensorInt(reduceOpShape);

            // argmax
            ReduceIndices(X, argMax, "ArgMax", axis, false);

            // one hot from argmax
            var pinArgMax = PinBlockAny(argMax);
            var pinO = PinAsSame(O, pinArgMax);

            var func = new PixelFunc("Hidden/Sentis/HardmaxEnd");

            func.SetTensor(k_TensorPropertiesX, pinArgMax);
            func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
            func.SetInt(k_TensorPropertiesO.k_ID_DimAxis, pinO.blockedShape[axis]);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void OneHot(TensorInt X, TensorInt O, int axis, int depth, int offValue, int onValue)
        {
            axis = O.shape.Axis(axis);

            var pinX = PinBlockAny(X);
            var pinO = TextureTensorData.Pin(O, axis > pinX.blockAxis ? pinX.blockAxis : pinX.blockAxis + 1, false);

            var func = new PixelFunc("Hidden/Sentis/OneHot");
            func.EnableKeyword("OneHotInt");
            func.SetInt(k_ID_offValueInt, offValue);
            func.SetInt(k_ID_onValueInt, onValue);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
            func.SetInt(k_TensorPropertiesO.k_ID_DimAxis, pinO.blockedShape[axis]);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void OneHot(TensorInt X, TensorFloat O, int axis, int depth, float offValue, float onValue)
        {
            axis = O.shape.Axis(axis);

            var pinX = PinBlockAny(X);
            var pinO = TextureTensorData.Pin(O, axis > pinX.blockAxis ? pinX.blockAxis : pinX.blockAxis + 1, false);

            var func = new PixelFunc("Hidden/Sentis/OneHot");
            func.SetFloat(k_ID_offValue, offValue);
            func.SetFloat(k_ID_onValue, onValue);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
            func.SetInt(k_TensorPropertiesO.k_ID_DimAxis, pinO.blockedShape[axis]);

            func.Dispatch(pinO);
        }

        void Reduce(Tensor X, Tensor O, int reduceAxis, string kernelName)
        {
            reduceAxis = X.shape.Axis(reduceAxis);

            var pinX = PinBlockOther(X, nonBlockAxis: reduceAxis);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/Reduce");
            func.EnableKeyword(kernelName);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, pinX.blockedShape.Strides(reduceAxis));
            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.blockedShape[reduceAxis]);
            func.SetFloat(k_ID_Normalization, 1.0f / X.shape[reduceAxis]);
            func.Dispatch(pinO);
        }

        void Reduce(Tensor X, Tensor O, ReadOnlySpan<int> axes, bool keepdim, string fullKernelName, string startKernelName = null, string middleKernelName = null, string endKernelName = null)
        {
            var Oout = keepdim ? O : NewTensor(X.shape.Reduce(axes, true), X.dataType, AllocScope.InternalToLayer);

            startKernelName ??= fullKernelName;
            middleKernelName ??= fullKernelName;
            endKernelName ??= fullKernelName;

            var allAxes = (axes == null) || (axes.Length == 0);
            var axesDim = allAxes ? X.shape.rank : axes.Length;
            var shapeXReduced = X.shape;
            var isInitial = true;

            for (var i = 0; i < axesDim - 1; i++)
            {
                var axis = allAxes ? i : X.shape.Axis(axes[i]);
                Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation which has no identity.");

                shapeXReduced[axis] = 1;
                var Otmp = NewTensor(shapeXReduced, O.dataType, AllocScope.InternalToLayer);
                Reduce(X, Otmp, axis, isInitial ? startKernelName : middleKernelName);

                X = Otmp;

                isInitial = false;
            }

            {
                var axis = allAxes ? axesDim - 1 : X.shape.Axis(axes[axesDim - 1]);
                Reduce(X, Oout, axis, isInitial ? fullKernelName : endKernelName);
            }

            if (!keepdim)
                Reshape(Oout, O);
        }

        /// <inheritdoc/>
        public override void ReduceMax(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceMax");
        }

        /// <inheritdoc/>
        public override void ReduceMax(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceMaxInt");
        }

        /// <inheritdoc/>
        public override void ReduceMean(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceMean");
        }

        /// <inheritdoc/>
        public override void ReduceMin(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceMin");
        }

        /// <inheritdoc/>
        public override void ReduceMin(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceMinInt");
        }

        /// <inheritdoc/>
        public override void ReduceProd(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceProd");
        }

        /// <inheritdoc/>
        public override void ReduceProd(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceProdInt");
        }

        /// <inheritdoc/>
        public override void ReduceSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceSum");
        }

        /// <inheritdoc/>
        public override void ReduceSum(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceSumInt");
        }

        /// <inheritdoc/>
        public override void ReduceL1(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceL1", "ReduceL1", "ReduceSum", "ReduceSum");
        }

        /// <inheritdoc/>
        public override void ReduceL1(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceL1Int", "ReduceL1Int", "ReduceSumInt", "ReduceSumInt");
        }

        /// <inheritdoc/>
        public override void ReduceL2(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceL2", "ReduceSumSquare", "ReduceSum", "ReduceSqrt");
        }

        /// <inheritdoc/>
        public override void ReduceLogSum(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceLogSum", "ReduceSum", "ReduceSum", "ReduceLogSum");
        }

        /// <inheritdoc/>
        public override void ReduceLogSumExp(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceLogSumExp");
        }

        /// <inheritdoc/>
        public override void ReduceSumSquare(TensorFloat X, TensorFloat O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceSumSquare", "ReduceSumSquare", "ReduceSum", "ReduceSum");
        }

        /// <inheritdoc/>
        public override void ReduceSumSquare(TensorInt X, TensorInt O, ReadOnlySpan<int> axes, bool keepdim)
        {
            Reduce(X, O, axes, keepdim, "ReduceSumSquareInt", "ReduceSumSquareInt", "ReduceSumInt", "ReduceSumInt");
        }

        void ReduceIndices(Tensor X, Tensor O, string kernelName, int axis, bool selectLastIndex)
        {
            axis = X.shape.Axis(axis);
            var pinX = PinBlockOther(X, nonBlockAxis: axis);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/ReduceIndices");
            func.EnableKeyword(kernelName);
            if (X.dataType == DataType.Int)
                func.EnableKeyword("X_INT");
            func.EnableKeyword(selectLastIndex ? "Last" : "First");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_TensorPropertiesX.k_ID_StrideAxis, pinX.blockedShape.Strides(axis));
            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.blockedShape[axis]);
            func.Dispatch(pinO);
        }

        void ReduceIndices(Tensor X, TensorInt O, string kernelName, int axis, bool keepdim, bool selectLastIndex)
        {
            var Otmp = keepdim ? O : NewOutputTensorInt(X.shape.Reduce(axis, true));

            ReduceIndices(X, Otmp, kernelName, axis, selectLastIndex);

            if (!keepdim)
                Reshape(Otmp, O);
        }

        /// <inheritdoc/>
        public override void ArgMax(TensorFloat X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
        {
            ReduceIndices(X, O, "ArgMax", axis, keepdim, selectLastIndex);
        }

        /// <inheritdoc/>
        public override void ArgMax(TensorInt X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
        {
            ReduceIndices(X, O, "ArgMax", axis, keepdim, selectLastIndex);
        }

        /// <inheritdoc/>
        public override void ArgMin(TensorFloat X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
        {
            ReduceIndices(X, O, "ArgMin", axis, keepdim, selectLastIndex);
        }

        /// <inheritdoc/>
        public override void ArgMin(TensorInt X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
        {
            ReduceIndices(X, O, "ArgMin", axis, keepdim, selectLastIndex);
        }

        /// <inheritdoc/>
        public override void Gather(Tensor X, TensorInt indices, Tensor O, int axis)
        {
            var pinX = PinBlockAny(X);
            var pinB = PinBlockAny(indices);
            var pinO = PinBlockAny(O, false);

            var func = new PixelFunc("Hidden/Sentis/Gather");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("GatherInt");
            func.SetInt(k_ID_endLength, X.shape.Strides(axis));
            func.SetInt(k_ID_indicesLength, indices.shape.length);
            func.SetInt(k_ID_axisDim, X.shape[axis]);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void GatherElements(Tensor X, TensorInt indices, Tensor O, int axis)
        {
            var pinX = PinBlockAny(X);
            var pinB = PinBlockAny(indices);
            var pinO = PinBlockAny(O, false);

            var func = new PixelFunc("Hidden/Sentis/GatherElements");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("GatherInt");
            func.SetInt(k_ID_endLength, pinO.shape.Strides(axis));
            func.SetInt(k_ID_endLengthX, pinX.shape.Strides(axis));
            func.SetInt(k_ID_axisDim, pinO.shape[axis]);
            func.SetInt(k_ID_axisDimX, pinX.shape[axis]);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void GatherND(Tensor X, TensorInt indices, Tensor O, int batchDims)
        {
            var pinX = PinBlockAny(X);
            var pinB = PinBlockAny(indices);
            var pinO = PinBlockAny(O, false);

            var func = new PixelFunc("Hidden/Sentis/GatherND");
            func.SetInt(k_ID_iStart, TensorShape.maxRank - pinO.shape.rank);
            func.SetInt(k_ID_iEndIndices, TensorShape.maxRank - pinO.shape.rank + pinB.shape.rank - 1);
            func.SetInt(k_ID_iEndX, TensorShape.maxRank - pinO.shape.rank + batchDims);
            func.SetInt(k_ID_iEndMin, TensorShape.maxRank - pinO.shape.rank + Math.Min(batchDims, pinB.shape.rank - 1));
            func.SetInt(k_ID_iStartB, TensorShape.maxRank - pinX.shape.rank + batchDims);
            func.SetInt(k_ID_iEndB, TensorShape.maxRank - pinX.shape.rank + batchDims + pinB.shape[-1]);
            func.SetShapeStrides(k_TensorPropertiesX, pinX.shape);
            func.SetShapeStrides(k_TensorPropertiesO, pinO.shape);
            func.SetShapeStrides(k_TensorPropertiesB, pinB.shape);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void ScatterElements(Tensor X, TensorInt indices, Tensor updates, Tensor O, int axis, Layers.ScatterReductionMode reduction)
        {
            axis = X.shape.Axis(axis);
            var pinX = PinBlockOther(X, axis);
            var pinB = PinAsSame(indices, pinX);
            var pinW = PinAsSame(updates, pinX);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/ScatterElements");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("ScatterInt");
            switch (reduction)
            {
                case Layers.ScatterReductionMode.None:
                    func.EnableKeyword("ReduceNone");
                    break;
                case Layers.ScatterReductionMode.Add:
                    func.EnableKeyword("ReduceAdd");
                    break;
                case Layers.ScatterReductionMode.Mul:
                    func.EnableKeyword("ReduceMul");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(reduction), reduction, null);
            }
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
            func.SetTensor(k_TensorPropertiesW, pinW);
            func.SetTensorBlockStride(k_TensorPropertiesW, pinW);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.SetInt(k_ID_DimAxis, pinO.blockedShape[axis]);
            func.SetInt(k_ID_NumIndices, pinB.blockedShape[axis]);
            func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
            func.Dispatch(pinO);
        }

        void ScatterND(Tensor X, TensorInt indices, Tensor updates, Tensor O, Layers.ScatterReductionMode reduction)
        {
            var func = new PixelFunc("Hidden/Sentis/ScatterND");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("ScatterInt");

            var K = indices.shape[-1];
            var pinX = PinBlockAny(X);
            if (pinX.dimAxis >= X.shape.rank - K)
                pinX = TextureTensorData.Pin(X, X.shape.rank - K - 1);
            var pinB = TextureTensorData.Pin(indices, indices.shape.rank - 1);
            var pinW = PinAsSame(updates, pinX);
            var pinO = PinAsSame(O, pinX, false);

            switch (reduction)
            {
                case Layers.ScatterReductionMode.None:
                    func.EnableKeyword("ReduceNone");
                    break;
                case Layers.ScatterReductionMode.Add:
                    func.EnableKeyword("ReduceAdd");
                    break;
                case Layers.ScatterReductionMode.Mul:
                    func.EnableKeyword("ReduceMul");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(reduction), reduction, null);
            }
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
            func.SetTensor(k_TensorPropertiesW, pinW);
            func.SetTensorBlockStride(k_TensorPropertiesW, pinW);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

            var Kdiv4 = pinB.blockedShape[-1];
            if (Kdiv4 > 1)
                func.EnableKeyword("K_LARGE");
            func.SetShape(k_TensorPropertiesX.k_ID_Shape, X.shape);
            func.SetInt(k_ID_SliceLength, pinX.blockedShape.Length(K));
            func.SetInt(k_ID_NumIndices, pinB.blockedShape.length / Kdiv4);
            func.SetInt(k_ID_K, K);
            func.SetInt(k_ID_Kdiv4, Kdiv4);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, TensorFloat O, Layers.ScatterReductionMode reduction)
        {
            ScatterND(X, indices, updates, O, reduction);
        }

        /// <inheritdoc/>
        public override void ScatterND(TensorInt X, TensorInt indices, TensorInt updates, TensorInt O, Layers.ScatterReductionMode reduction)
        {
            ScatterND(X, indices, updates, O, reduction);
        }

        /// <inheritdoc/>
        public override void Transpose(Tensor X, Tensor O)
        {
            var pinX = PinBlockAny(X);
            var oAxis = pinX.blockAxis < 0 ? -1 : X.shape.rank - 1 - pinX.blockAxis;
            var pinO = TextureTensorData.Pin(O, oAxis, clearOnInit: false);

            var func = new PixelFunc("Hidden/Sentis/Transpose");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");

            func.SetTensor(k_TensorPropertiesX, pinX);

            var rank = pinX.shape.rank;
            unsafe
            {
                var permutedStridesX = stackalloc int[TensorShape.maxRank];
                var strideX = 1;
                for (var i = 0; i < rank; i++)
                {
                    var dim = pinX.blockedShape[-1 - i];
                    permutedStridesX[rank - 1 - i] = dim > 1 ? strideX : 0;
                    strideX *= dim;
                }

                func.SetInt8(k_TensorPropertiesX.k_ID_Strides, permutedStridesX);
            }

            func.SetShape(k_ID_DimO, pinO.blockedShape);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Transpose(Tensor X, Tensor O, int[] permutations)
        {
            var pinX = PinBlockAny(X);
            var oAxis = pinX.blockAxis;
            for (var i = 0; i < permutations.Length; i++)
            {
                if (permutations[i] == pinX.blockAxis)
                {
                    oAxis = i;
                    break;
                }
            }

            // pin O so that the transposed blocked axis matches
            var pinO = TextureTensorData.Pin(O, oAxis, clearOnInit: false);

            var func = new PixelFunc("Hidden/Sentis/Transpose");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");

            func.SetTensor(k_TensorPropertiesX, pinX);

            var rank = pinX.shape.rank;
            unsafe
            {
                var stridesX = stackalloc int[TensorShape.maxRank];
                var strideX = 1;
                for (var i = 0; i < rank; i++)
                {
                    var dim = pinX.blockedShape[-1 - i];
                    stridesX[i] = dim > 1 ? strideX : 0;
                    strideX *= dim;
                }

                var permutedStridesX = stackalloc int[TensorShape.maxRank];
                for (var i = 0; i < rank; i++)
                {
                    permutedStridesX[i] = stridesX[rank - 1 - permutations[rank - 1 - i]];
                }

                func.SetInt8(k_TensorPropertiesX.k_ID_Strides, permutedStridesX);
            }

            func.SetShape(k_ID_DimO, pinO.blockedShape);

            func.Dispatch(pinO);
        }

        void GlobalPool(TensorFloat X, TensorFloat O, string kernelName)
        {
            var pinX = TextureTensorData.Pin(X, 1);
            var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

            var func = new PixelFunc("Hidden/Sentis/GlobalPool");
            func.EnableKeyword(kernelName);

            func.SetTensor(k_TensorPropertiesX, pinX);
            var spatialSize = X.shape.Strides(1);
            func.SetInt(k_ID_SpatialSizeX, spatialSize);
            func.SetInt(k_ID_DimAxis, pinX.blockedShape[1]);
            func.SetFloat(k_ID_Normalization, 1.0f / spatialSize);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void GlobalAveragePool(TensorFloat X, TensorFloat O)
        {
            GlobalPool(X, O, "AVGPOOL");
        }

        /// <inheritdoc/>
        public override void GlobalMaxPool(TensorFloat X, TensorFloat O)
        {
            GlobalPool(X, O, "MAXPOOL");
        }

        void LocalPool(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad, string kernelName)
        {
            var pinX = TextureTensorData.Pin(X, 1);
            var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

            var numSpatialDims = X.shape.rank - 2;

            var func = new PixelFunc("Hidden/Sentis/LocalPool");
            func.EnableKeyword(numSpatialDims == 2 ? "POOL2D" : "POOL1D");
            func.EnableKeyword(kernelName);

            func.SetInt(k_ID_O_width, pinO.shape[-1]);
            func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_X_width, pinX.shape[-1]);
            func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

            func.SetInt(k_ID_StrideX, stride[numSpatialDims - 1]);
            func.SetInt(k_ID_PadX, pad[numSpatialDims - 1]);
            func.SetInt(k_ID_PoolX, pool[numSpatialDims - 1]);

            if (numSpatialDims > 1)
            {
                func.SetInt(k_ID_StrideY, stride[numSpatialDims - 2]);
                func.SetInt(k_ID_PadY, pad[numSpatialDims - 2]);
                func.SetInt(k_ID_PoolY, pool[numSpatialDims - 2]);
                func.SetInt(k_ID_X_height, pinX.shape[-2]);
                func.SetInt(k_ID_O_height, pinO.shape[-2]);
            }

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void MaxPool(TensorFloat X, TensorFloat O, int[] kernelShape, int[] strides, int[] pads)
        {
            if (X.shape.rank > 4)
                base.MaxPool(X, O, kernelShape, strides, pads);
            else
                LocalPool(X, O, kernelShape, strides, pads, "MAXPOOL");
        }

        /// <inheritdoc/>
        public override void AveragePool(TensorFloat X, TensorFloat O, int[] kernelShape, int[] strides, int[] pads)
        {
            if (X.shape.rank > 4)
                base.AveragePool(X, O, kernelShape, strides, pads);
            else
                LocalPool(X, O, kernelShape, strides, pads, "AVGPOOL");
        }

        /// <inheritdoc/>
        public override void DepthToSpace(TensorFloat X, TensorFloat O, int blocksize, Layers.DepthToSpaceMode mode)
        {
            var func = new PixelFunc("Hidden/Sentis/DepthToSpace");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            func.SetInt(k_ID_O_width, pinO.shape[3]);
            func.SetInt(k_ID_O_height, pinO.shape[2]);
            func.SetInt(k_ID_O_channels, pinO.shape[1]);

            func.SetTensor(k_TensorPropertiesX, pinX);

            func.SetInt(k_ID_X_width, pinX.shape[3]);
            func.SetInt(k_ID_X_height, pinX.shape[2]);
            func.SetInt(k_ID_X_channels, pinX.shape[1]);

            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);

            func.SetInt(k_ID_BlockSize, blocksize);

            if (mode == Layers.DepthToSpaceMode.ColumnRowDepth)
                func.EnableKeyword("COLUMNROWDEPTH");
            else if (mode == Layers.DepthToSpaceMode.DepthColumnRow)
                func.EnableKeyword("DEPTHCOLUMNROW");

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void SpaceToDepth(TensorFloat X, TensorFloat O, int blocksize)
        {
            var func = new PixelFunc("Hidden/Sentis/SpaceToDepth");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            func.SetInt(k_ID_O_width, pinO.shape[3]);
            func.SetInt(k_ID_O_height, pinO.shape[2]);
            func.SetInt(k_ID_O_channels, pinO.shape[1]);

            func.SetTensor(k_TensorPropertiesX, pinX);

            func.SetInt(k_ID_X_width, pinX.shape[3]);
            func.SetInt(k_ID_X_height, pinX.shape[2]);
            func.SetInt(k_ID_X_channels, pinX.shape[1]);

            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.SetTensorBlockStride(k_TensorPropertiesX, pinX);

            func.SetInt(k_ID_BlockSize, blocksize);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Split(Tensor X, Tensor O, int axis, int start)
        {
            axis = X.shape.Axis(axis);

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/Split");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_StrideAxis, pinO.blockedShape.Strides(axis));
            func.SetInt(k_TensorPropertiesX.k_ID_DimAxis, pinX.blockedShape[axis]);
            func.SetInt(k_ID_SplitStart, start);
            func.SetInt(k_ID_SplitLength, pinO.blockedShape[axis]);
            if (pinX.blockAxis != axis)
                func.EnableKeyword("BLOCKWISE");

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Tile(Tensor X, Tensor O, ReadOnlySpan<int> repeats)
        {
            var pinX = X.tensorOnDevice as TextureTensorData;
            var xRank = X.shape.rank;
            var blockAxis = pinX?.blockAxis ?? 0;

            if (pinX == null || (blockAxis >= 0 && repeats[blockAxis] > 1))
            {
                // repin X again if repeat on blocked axis
                blockAxis = xRank - 1;
                for (; blockAxis >= 0; blockAxis--)
                {
                    if (X.shape[blockAxis] > 1 && repeats[blockAxis] == 1)
                        break;
                }

                pinX = TextureTensorData.Pin(X, blockAxis);
            }

            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/Tile");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetShape(k_ID_DimO, pinO.blockedShape);
            func.SetShape(k_ID_DimX, pinX.blockedShape);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Pad(TensorFloat X, TensorFloat O, ReadOnlySpan<int> pad, Layers.PadMode padMode, float constant)
        {
            var pinX = X.tensorOnDevice as TextureTensorData;
            var xRank = X.shape.rank;
            var blockAxis = pinX?.blockAxis ?? 0;

            if (pinX == null || (blockAxis >= 0 && pad[blockAxis] + pad[blockAxis + xRank] > 0))
            {
                // repin X again if pad on blocked axis
                blockAxis = xRank - 1;
                for (; blockAxis >= 0; blockAxis--)
                {
                    if (X.shape[blockAxis] > 1 && pad[blockAxis] + pad[blockAxis + xRank] == 0)
                        break;
                }

                pinX = TextureTensorData.Pin(X, blockAxis);
            }

            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/Pad");

            switch (padMode)
            {
                case Layers.PadMode.Constant:
                    func.EnableKeyword("CONSTANT");
                    break;
                case Layers.PadMode.Reflect:
                    func.EnableKeyword("REFLECT");
                    break;
                case Layers.PadMode.Edge:
                    func.EnableKeyword("EDGE");
                    break;
                case Layers.PadMode.Symmetric:
                    func.EnableKeyword("SYMMETRIC");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(padMode), padMode, null);
            }

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_MaxBlockIndexX, pinX.blockedShape.length - 1);

            unsafe
            {
                var padArray = stackalloc int[8];
                for (var i = 0; i < xRank; i++)
                {
                    padArray[i] = pad[xRank - 1 - i];
                }

                func.SetInt8(k_ID_Pad, padArray);
            }

            func.SetShape(k_ID_DimO, pinO.blockedShape);
            func.SetShape(k_ID_DimX, pinX.blockedShape);
            func.SetStrides(k_TensorPropertiesX.k_ID_Strides, pinX.blockedShape);

            func.SetFloat(k_ID_Beta, constant);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Resize(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
        {
            if (X.shape.rank > 5 || scale[0] != 1f || scale[1] != 1f)
            {
                base.Resize(X, O, scale, interpolationMode, nearestMode, coordTransformMode);
                return;
            }

            var numSpatialDims = X.shape.rank - 2;

            var pinX = TextureTensorData.Pin(X, 1);
            var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);

            var func = new PixelFunc("Hidden/Sentis/Upsample");
            switch (numSpatialDims)
            {
                case 1:
                    func.EnableKeyword("UPSAMPLE1D");
                    break;
                case 2:
                    func.EnableKeyword("UPSAMPLE2D");
                    break;
                case 3:
                    func.EnableKeyword("UPSAMPLE3D");
                    break;
            }

            var scaleXY = Vector4.zero;
            var biasXY = Vector4.zero;
            for (var i = 0; i < numSpatialDims; i++)
            {
                OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
                scaleXY[i] = outputScale;
                biasXY[i] = outputBias;
            }
            func.SetVector(k_ID_Scale, scaleXY);
            func.SetVector(k_ID_Bias, biasXY);

            if (interpolationMode == Layers.InterpolationMode.Nearest)
            {
                switch (nearestMode)
                {
                    case Layers.NearestMode.RoundPreferFloor:
                    case Layers.NearestMode.Ceil:
                        func.EnableKeyword("NEAREST_CEIL");
                        break;
                    case Layers.NearestMode.RoundPreferCeil:
                    case Layers.NearestMode.Floor:
                        func.EnableKeyword("NEAREST_FLOOR");
                        break;
                    default:
                        throw new NotImplementedException();
                }
            }
            else //if (interpolationMode == Layers.InterpolationMode.Linear)
            {
                func.EnableKeyword("LINEAR");
            }

            func.SetInt(k_ID_O_width, pinO.shape[-1]);
            func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_X_width, pinX.shape[-1]);
            func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

            if (numSpatialDims > 1)
            {
                func.SetInt(k_ID_O_height, pinO.shape[-2]);
                func.SetInt(k_ID_X_height, pinX.shape[-2]);
            }
            if (numSpatialDims > 2)
            {
                func.SetInt(k_ID_O_depth, pinO.shape[-3]);
                func.SetInt(k_ID_X_depth, pinX.shape[-3]);
            }

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O)
        {
            var func = new PixelFunc("Hidden/Sentis/ScaleBias");

            var pinX = X.tensorOnDevice as TextureTensorData;
            pinX ??= TextureTensorData.Pin(X, X.shape.rank - 2);
            var pinS = TextureTensorData.Pin(S, 0);
            var pinB = TextureTensorData.Pin(B, 0);
            var pinO = PinAsSame(O, pinX, false);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesS, pinS);
            func.SetTensor(k_TensorPropertiesB, pinB);
            if (pinX.blockAxis == 1)
            {
                func.EnableKeyword("BLOCK_C");
                func.SetInt(k_ID_StrideAxis, pinO.strideAxis);
                func.SetInt(k_TensorPropertiesO.k_ID_DimBlocked, pinO.dimAxisDiv4);
            }
            else
            {
                func.SetInt(k_ID_StrideC, pinO.blockedShape.Strides(1));
                func.SetInt(k_ID_DimC, pinO.blockedShape[1]);
            }

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void BatchNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat mean, TensorFloat variance, TensorFloat O, float epsilon)
        {
            var func = new PixelFunc("Hidden/Sentis/BatchNormalization");

            var pinX = TextureTensorData.Pin(X, X.shape.rank == 1 ? -1 : 1);
            var pinS = TextureTensorData.Pin(S, 0);
            var pinB = TextureTensorData.Pin(B, 0);
            var pinM = TextureTensorData.Pin(mean, 0);
            var pinV = TextureTensorData.Pin(variance, 0);
            var pinO = PinAsSame(O, pinX, false);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesS, pinS);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensor(k_TensorPropertiesM, pinM);
            func.SetTensor(k_TensorPropertiesV, pinV);

            func.SetInt(k_ID_O_channels, pinO.dimAxis);
            func.SetInt(k_ID_O_width, pinO.strideAxis);
            func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);

            func.SetFloat(k_ID_epsilon, epsilon);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O, float epsilon)
        {
            var spatialSize = X.shape.Strides(1);
            var pinX = TextureTensorData.Pin(X, 1);
            var pooledShape = ShapeInference.GlobalPool(X.shape);
            var A = NewTempTensorFloat(pooledShape);
            var K = NewTempTensorFloat(pooledShape);
            var pinA = PinAsSame(A, pinX, false);
            var pinK = PinAsSame(K, pinX, false);

            {
                var func = new PixelFunc("Hidden/Sentis/GlobalPool");
                func.EnableKeyword("AVGPOOL");

                func.SetTensor(k_TensorPropertiesX, pinX);
                func.SetInt(k_ID_SpatialSizeX, spatialSize);
                func.SetInt(k_ID_DimAxis, pinX.blockedShape[1]);
                func.SetFloat(k_ID_Normalization, 1.0f / spatialSize);

                func.Dispatch(pinA);
            }

            {
                var func = new PixelFunc("Hidden/Sentis/GlobalPool");
                func.EnableKeyword("AVGSQUAREPOOL");

                func.SetTensor(k_TensorPropertiesX, pinX);
                func.SetInt(k_ID_SpatialSizeX, spatialSize);
                func.SetInt(k_ID_DimAxis, pinX.blockedShape[1]);
                func.SetFloat(k_ID_Normalization, 1.0f / spatialSize);

                func.Dispatch(pinK);
            }

            {
                var pinO = TextureTensorData.Pin(O, 1, clearOnInit: false);
                var pinS = TextureTensorData.Pin(S, 0);
                var pinB = TextureTensorData.Pin(B, 0);
                var func = new PixelFunc("Hidden/Sentis/InstanceNormalizationTail");

                func.SetTensor(k_TensorPropertiesX, pinX);
                func.SetTensor(k_TensorPropertiesS, pinS);
                func.SetTensor(k_TensorPropertiesA, pinA);
                func.SetTensor(k_TensorPropertiesB, pinB);
                func.SetTensor(k_TensorPropertiesK, pinK);
                func.SetInt(k_ID_StrideAxis, spatialSize);
                func.SetInt(k_ID_O_channelsDiv4, pinO.blockedShape[1]);
                func.SetFloat(k_ID_epsilon, epsilon);

                func.Dispatch(pinO);
            }
        }

        /// <inheritdoc/>
        public override void LayerNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O, float epsilon)
        {
            var axis = X.shape.Axis(-1);
            var reducedShape = X.shape.Reduce(axis);
            var A = NewTempTensorFloat(reducedShape);
            var K = NewTempTensorFloat(reducedShape);

            Reduce(X, A, axis, "ReduceMean");
            Reduce(X, K, axis, "ReduceMeanSquare");

            var pinX = PinBlockAny(X);
            var pinA = PinAsSame(A, pinX);
            var pinK = PinAsSame(K, pinX);
            var pinS = TextureTensorData.Pin(S, -1);
            var pinB = TextureTensorData.Pin(B, -1);
            var pinO = PinAsSame(O, pinX, clearOnInit: false);
            var func = new PixelFunc("Hidden/Sentis/LayerNormalizationTail");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesS, pinS);
            func.SetTensor(k_TensorPropertiesA, pinA);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensor(k_TensorPropertiesK, pinK);
            func.SetInt(k_ID_reduceLength, pinO.shape[-1]);
            func.SetFloat(k_ID_epsilon, epsilon);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void RoiAlign(TensorFloat X, TensorFloat rois, TensorInt indices, TensorFloat O, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
        {
            var pinX = TextureTensorData.Pin(X, 1);
            var pinB = PinBlockAny(indices);
            var pinS = TextureTensorData.Pin(rois, 1);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/RoiAlign");
            func.EnableKeyword(mode == Layers.RoiPoolingMode.Avg ? "RoiAlignAvg" : "RoiAlignMax");

            func.SetFloat(k_ID_spatialScale, spatialScale);
            func.SetInt(k_ID_numRois, rois.shape[0]);
            func.SetFloat(k_ID_normalizeOHeight, 1.0f / outputHeight);
            func.SetFloat(k_ID_normalizeOWidth, 1.0f / outputWidth);
            func.SetInt(k_ID_samplingRatio, samplingRatio);

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensor(k_TensorPropertiesB, pinB);
            func.SetTensorBlockStride(k_TensorPropertiesB, pinB);
            func.SetTensor(k_TensorPropertiesS, pinS);

            func.SetInt(k_ID_O_width, pinO.shape[-1]);
            func.SetInt(k_ID_O_height, pinO.shape[-2]);
            func.SetInt(k_ID_O_channelsDiv4, pinO.dimAxisDiv4);
            func.SetInt(k_ID_X_width, pinX.shape[-1]);
            func.SetInt(k_ID_X_height, pinX.shape[-2]);
            func.SetInt(k_ID_X_channelsDiv4, pinX.dimAxisDiv4);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void RandomUniform(TensorFloat O, float low, float high, float? seed)
        {
            var pinO = PinBlockAny(O, false);

            var func = new PixelFunc("Hidden/Sentis/Random");
            func.EnableKeyword("RandomUniform");
            func.SetInt(k_ID_seed, (int)Random.GetOpSeed(seed));
            func.SetFloat(k_ID_low, low);
            func.SetFloat(k_ID_high, high);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void RandomNormal(TensorFloat O, float mean, float scale, float? seed)
        {
            var pinO = PinBlockAny(O, false);

            var func = new PixelFunc("Hidden/Sentis/Random");
            func.EnableKeyword("RandomNormal");
            func.SetInt(k_ID_seed, (int)Random.GetOpSeed(seed));
            func.SetFloat(k_ID_mean, mean);
            func.SetFloat(k_ID_scale, scale);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Bernoulli(TensorFloat X, Tensor O, float? seed)
        {
            var func = new PixelFunc("Hidden/Sentis/Random");
            func.EnableKeyword(O.dataType == DataType.Int ? "BernoulliInt" : "Bernoulli");

            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, clearOnInit: false);

            func.SetInt(k_ID_seed, (int)Random.GetOpSeed(seed));
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);

            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Range(TensorFloat O, float start, float delta)
        {
            var pinO = PinBlockAny(O, clearOnInit: false);

            var func = new PixelFunc("Hidden/Sentis/Range");
            func.SetFloat(k_ID_rangeStartFloat, start);
            func.SetFloat(k_ID_rangeDeltaFloat, delta);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Range(TensorInt O, int start, int delta)
        {
            var pinO = PinBlockAny(O, clearOnInit: false);

            var func = new PixelFunc("Hidden/Sentis/Range");
            func.EnableKeyword("INT");
            func.SetInt(k_ID_rangeStartInt, start);
            func.SetInt(k_ID_rangeDeltaInt, delta);
            func.Dispatch(pinO);
        }

        void Trilu(Tensor X, Tensor O, int k, bool upper)
        {
            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/Trilu");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
            func.SetInt(k_ID_width, X.shape[-1]);
            func.SetInt(k_ID_height, X.shape[-2]);
            func.SetInt(k_ID_direction, upper ? 1 : -1);
            func.SetInt(k_ID_offset, k);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Tril(Tensor X, Tensor O, int k)
        {
            Trilu(X, O, k, false);
        }

        /// <inheritdoc/>
        public override void Triu(Tensor X, Tensor O, int k)
        {
            Trilu(X, O, k, true);
        }

        void CumSum(Tensor X, Tensor O, int axis, bool reverse, bool exclusive)
        {
            axis = X.shape.Axis(axis);

            var pinX = PinBlockOther(X, nonBlockAxis: axis);
            var pinO = PinAsSame(O, pinX, false);

            var func = new PixelFunc("Hidden/Sentis/CumSum");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");
            func.EnableKeyword(reverse ? "REVERSE" : "FORWARD");
            func.EnableKeyword(exclusive ? "EXCLUSIVE" : "INCLUSIVE");

            func.SetTensor(k_TensorPropertiesX, pinX);
            func.SetInt(k_ID_StrideAxis, pinX.blockedShape.Strides(axis));
            func.SetInt(k_ID_DimAxis, pinX.blockedShape[axis]);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void CumSum(TensorFloat X, TensorFloat O, int axis, bool reverse, bool exclusive)
        {
            CumSum(X, O, axis, reverse, exclusive);
        }

        /// <inheritdoc/>
        public override void CumSum(TensorInt X, TensorInt O, int axis, bool reverse, bool exclusive)
        {
            CumSum(X, O, axis, reverse, exclusive);
        }

        /// <inheritdoc/>
        public override Tensor ShallowReshape(Tensor X, TensorShape shape, AllocScope allocScope)
        {
            var O = NewTensor(shape, X.dataType, allocScope);
            if (O.shape.HasZeroDims())
                return O;
            Reshape(X, O);
            return O;
        }

        /// <inheritdoc/>
        public override void MemCopy(Tensor X, Tensor O)
        {
            var pinX = PinBlockAny(X);
            var pinO = PinAsSame(O, pinX, clearOnInit: false);
            var func = new PixelFunc("Hidden/Sentis/Copy");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");
            func.SetTensor(k_TensorPropertiesX, pinX);
            func.Dispatch(pinO);
        }

        /// <inheritdoc/>
        public override void Reshape(Tensor X, Tensor O)
        {
            TextureTensorData pinO;

            if (X.tensorOnDevice is TextureTensorData pinX)
            {
                // try and pin O in a layout that can be read in float4 blocks from x
                var blockAxis = O.shape.rank - 1;
                var strideO = 1;
                for (; blockAxis >= 0; blockAxis--)
                {
                    if (strideO >= pinX.strideAxis)
                        break;
                    strideO *= O.shape[blockAxis];
                }

                pinO = TextureTensorData.Pin(O, blockAxis, false);
            }
            else
            {
                pinX = PinBlockAny(X);
                pinO = PinAsSame(O, pinX, false);
            }

            var func = new PixelFunc("Hidden/Sentis/Reshape");
            if (X.dataType == DataType.Int)
                func.EnableKeyword("INT");
            func.SetTensor(k_TensorPropertiesX, pinX);
            if (pinX.strideAxis == pinO.strideAxis && (pinX.dimAxis == pinO.dimAxis) || (pinX.dimAxis % 4 == 0 && pinO.dimAxis % 4 == 0))
            {
                func.EnableKeyword("BLOCKWISE");
            }
            else
            {
                func.SetTensorBlockStride(k_TensorPropertiesO, pinO);
                func.SetTensorBlockStride(k_TensorPropertiesX, pinX);
            }

            func.Dispatch(pinO);
        }
    }
} // namespace Unity.Sentis
