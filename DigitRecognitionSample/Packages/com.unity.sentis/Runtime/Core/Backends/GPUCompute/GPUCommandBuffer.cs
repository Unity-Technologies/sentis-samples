// This is auto-generated -- do not modify directly

using UnityEngine.Rendering;
using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using static Unity.Sentis.ComputeTensorData;
using static Unity.Sentis.ShaderPropertyID;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.Sentis {

/// <summary>
/// Represents a GPUCompute backend ops.
/// </summary>
public partial class GPUCommandBufferBackend : CPUBackend
{
    /// <summary>
    /// The command buffer to use for scheduling.
    /// </summary>
    public CommandBuffer cb;

    /// <summary>
    /// Initializes and returns an instance of `GPUComputeOps`.
    /// </summary>
    /// <param name="allocator">The allocator to use when allocating tensors.</param>
    /// <param name="cb">The command buffer to use for scheduling.</param>
    public GPUCommandBufferBackend(CommandBuffer cb, ITensorAllocator allocator = null) : base(allocator) { this.cb = cb; }

    /// <summary>
    /// Initializes and returns an instance of `GPUComputeOps`.
    /// </summary>
    /// <param name="allocator">The allocator to use when allocating tensors.</param>
    public GPUCommandBufferBackend(ITensorAllocator allocator = null)
        : this(new CommandBuffer(), allocator) { }

    /// <inheritdoc/>
    public override Tensor NewTensor(TensorShape shape, DataType dataType, AllocScope scope)
    {
        return m_Allocator.Alloc(shape, dataType, DeviceType.GPU, scope);
    }

    /// <inheritdoc/>
    public override DeviceType deviceType => DeviceType.CPU;

    /// <inheritdoc/>
    public override void MatMul2D(TensorFloat X, TensorFloat Y, TensorFloat O, bool xTranspose, bool yTranspose)
    {
        Gemm(X, Y, O, O.shape[0], xTranspose ? X.shape[0] : X.shape[1], O.shape[1], xTranspose, yTranspose);
    }

    /// <inheritdoc/>
    public override void MatMul(TensorFloat X, TensorFloat Y, TensorFloat O)
    {
        var xShape = X.shape.rank == 1 ? new TensorShape(1, X.shape[0]) : X.shape;
        var yShape = Y.shape.rank == 1 ? new TensorShape(Y.shape[0], 1) : Y.shape;
        var oShape = X.shape.rank > 1 && Y.shape.rank > 1 ? O.shape : xShape.MatMul(yShape);

        var M = xShape[-2];
        var K = xShape[-1];
        var N = yShape[-1];
        var batch = oShape.Length(0, -2);

        if (batch == 1)
        {
            Gemm(X, Y, O, M, K, N);
            return;
        }

        if (xShape.Length(0, -2) == batch && yShape.Length(0, -2) == batch)
        {
            BatchedGemm(X, Y, O, batch, M, K, N);
            return;
        }

        var fn = ComputeFuncSingleton.Instance.Get("MatMul");

        unsafe
        {
            var shapeA = stackalloc int[6];
            var stridesA = stackalloc int[6];
            var shapeB = stackalloc int[6];
            var stridesB = stackalloc int[6];
            var shapeO = stackalloc int[6];
            var stridesO = stackalloc int[6];
            OpsUtils.PinMatMulTensorShapeStrides(xShape, yShape, oShape, shapeA, stridesA, shapeB, stridesB, shapeO, stridesO);

            cb.SetInt6(fn, k_ID_shapeA, shapeA);
            cb.SetInt6(fn, k_ID_stridesA, stridesA);
            cb.SetInt6(fn, k_ID_shapeB, shapeB);
            cb.SetInt6(fn, k_ID_stridesB, stridesB);
            cb.SetInt6(fn, k_ID_shapeO, shapeO);
            cb.SetInt6(fn, k_ID_stridesO, stridesO);
        }

        cb.SetInt(fn, k_ID_AM, M);
        cb.SetInt(fn, k_ID_AN, K);
        cb.SetInt(fn, k_ID_BM, K);
        cb.SetInt(fn, k_ID_BN, N);
        cb.SetInt(fn, k_ID_CB, batch);
        cb.SetInt(fn, k_ID_CM, M);
        cb.SetInt(fn, k_ID_CN, N);
        cb.SetInt(fn, k_ID_rank, oShape.rank);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Y));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.Dispatch(fn, batch, M, N);
    }

    void BatchedGemm(TensorFloat X, TensorFloat Y, TensorFloat O, int batch, int M, int K, int N)
    {
        string kernel;

        if (N % 64 == 0 && K % 16 == 0)
            kernel = "GemmBatched_T16x16_R4x4";
        else
            kernel = "GemmBatched_T8x8_R4x4";

        var fn = ComputeFuncSingleton.Instance.Get(kernel);

        cb.SetInt(fn, k_ID_maxXIndex, X.shape.length - 1);
        cb.SetInt(fn, k_ID_maxWIndex, Y.shape.length - 1);
        cb.SetInt(fn, k_ID_X_width, K);
        cb.SetInt(fn, k_ID_W_width, N);
        cb.SetInt(fn, k_ID_O_width, N);
        cb.SetInt(fn, k_ID_O_height, M);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(Y));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, ComputeHelper.IDivC(N, 4), ComputeHelper.IDivC(M, 4), batch);
    }

    /// <inheritdoc/>
    public override void Dense(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Layers.FusableActivation fusedActivation)
    {
        var Otmp = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(O.shape) : O;
        var M = Otmp.shape.Length(0, -1);
        var K = X.shape[-1];
        var N = Otmp.shape[-1];
        if (B != null)
            Gemm(X, W, B, Otmp, M, K, N);
        else
            Gemm(X, W, Otmp, M, K, N);

        if (fusedActivation != Layers.FusableActivation.None)
            ApplyFusedActivation(Otmp, O, fusedActivation);
    }

    void Trilu(Tensor X, Tensor O, int k, string kernel)
    {
        // Warning, for some reason shared mem implementation on intel gpu is x2 faster than regular one
        ComputeFunc fn = ComputeFuncSingleton.Instance.Get(kernel);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_X_width, X.shape[-1]);
        cb.SetInt(fn, k_ID_X_height, X.shape[-2]);
        cb.SetInt(fn, k_ID_X_length, X.shape.length);
        cb.SetInt(fn, k_ID_diagonalK, k);

        cb.Dispatch(fn, ComputeHelper.IDivC(X.shape.length, 4), 1, 1);
    }

    /// <inheritdoc/>
    public override void Tril(Tensor X, Tensor O, int k)
    {
        Trilu(X, O, k, "Tril");
    }

    /// <inheritdoc/>
    public override void Triu(Tensor X, Tensor O, int k)
    {
        Trilu(X, O, k, "Triu");
    }

    void ApplyFusedActivation(TensorFloat X, TensorFloat O, Layers.FusableActivation fusedActivation)
    {
        switch (fusedActivation)
        {
            case Layers.FusableActivation.None:
                return;
            case Layers.FusableActivation.Relu:
                Relu(X, O);
                return;
            default:
                throw new NotImplementedException();
        }
    }

    /// <inheritdoc/>
    public override void Conv(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
    {
        if (X.shape.rank > 5)
        {
            base.Conv(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
            return;
        }

        if (X.shape.rank == 4 && K.shape[0] == groups && K.shape[1] == 1)
        {
            DepthwiseConv2D(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
            return;
        }

        if (groups != 1)
        {
            GroupedConv(X, K, B, O, groups, strides, pads, dilations, fusedActivation);
            return;
        }

        if (ComputeInfo.IsMobileGPU())
        {
            ConvMobile(X, K, B, O, strides, pads, dilations, fusedActivation);
            return;
        }

        int workItemsX, workItemsY, workItemsZ;

        ComputeFunc fn;
        if (X.shape.rank == 5)
        {
            var n = O.shape[0];
            var k = O.shape[1];
            var d = O.shape[2];
            var h = O.shape[3];
            var w = O.shape[4];

            fn = K.shape.Length(2) == 1 ? ComputeFuncSingleton.Instance.Get("Conv3D_1x1_T16x16_R4x4") : ComputeFuncSingleton.Instance.Get("Conv3D_T16x16_R4x4");
            cb.SetInt(fn, k_ID_O_depth, O.shape[2]);
            cb.SetInt(fn, k_ID_O_height, O.shape[3]);
            cb.SetInt(fn, k_ID_O_width, O.shape[4]);
            cb.SetInt(fn, k_ID_X_depth, X.shape[2]);
            cb.SetInt(fn, k_ID_X_height, X.shape[3]);
            cb.SetInt(fn, k_ID_X_width, X.shape[4]);
            cb.SetInt(fn, k_ID_K_depth, K.shape[2]);
            cb.SetInt(fn, k_ID_K_height, K.shape[3]);
            cb.SetInt(fn, k_ID_K_width, K.shape[4]);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));
            if (B != null)
            {
                cb.EnableKeyword(fn, "USEBIAS");
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            }
            else
            {
                cb.DisableKeyword(fn, "USEBIAS");
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
            cb.SetInt(fn, k_ID_O_batch, O.shape[0]); cb.SetInt(fn, k_ID_O_channels, O.shape[1]);
            cb.SetInt(fn, k_ID_X_channels, X.shape[1]);
            cb.SetInt4(fn, k_ID__Stride, strides);
            cb.SetInt4(fn, k_ID__Pad, pads);
            cb.SetInt4(fn, k_ID__Dilation, dilations);
            workItemsX = ComputeHelper.IDivC(k, 4);
            workItemsY = ComputeHelper.IDivC(d * h * w, 4);
            workItemsZ = n;
        }
        // TODO multiplte dispatch + reduce for thin conv
        else if (X.shape.rank == 4)
        {
            var n = O.shape[0];
            var k = O.shape[1];
            var h = O.shape[2];
            var w = O.shape[3];

            workItemsX = ComputeHelper.IDivC(h * w, 4);
            workItemsY = ComputeHelper.IDivC(k, 8);
            workItemsZ = n;

            fn = K.shape.Length(2) == 1 ? ComputeFuncSingleton.Instance.Get("Conv2D_1x1") : ComputeFuncSingleton.Instance.Get("Conv2D_KxK");
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(K));
            if (B != null)
            {
                cb.EnableKeyword(fn, "USEBIAS");
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            }
            else
            {
                cb.DisableKeyword(fn, "USEBIAS");
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
            cb.SetInt(fn, k_ID_inputChannels, X.shape[1]);
            cb.SetInt(fn, k_ID_inputHeight, X.shape[2]);
            cb.SetInt(fn, k_ID_inputWidth, X.shape[3]);
            cb.SetInt(fn, k_ID_kernelHeight, K.shape[2]);
            cb.SetInt(fn, k_ID_kernelWidth, K.shape[3]);
            cb.SetInt(fn, k_ID_outputChannels, O.shape[1]);
            cb.SetInt(fn, k_ID_outputHeight, O.shape[2]);
            cb.SetInt(fn, k_ID_outputWidth, O.shape[3]);
            cb.SetInt(fn, k_ID_strideHeight, strides[0]);
            cb.SetInt(fn, k_ID_strideWidth, strides[1]);
            cb.SetInt(fn, k_ID_padHeight, pads[0]);
            cb.SetInt(fn, k_ID_padWidth, pads[1]);
            cb.SetInt(fn, k_ID_dilationHeight, dilations != null ? dilations[0] : 1);
            cb.SetInt(fn, k_ID_dilationWidth, dilations != null ? dilations[1] : 1);
            cb.SetInt(fn, k_ID_inputChannelsSize, X.shape[1] * X.shape[2] * X.shape[3]);
            cb.SetInt(fn, k_ID_outputChannelsSize, O.shape[1] * O.shape[2] * O.shape[3]);
            cb.SetInt(fn, k_ID_kernelChannelSize, K.shape[1] * K.shape[2] * K.shape[3]);
            cb.SetInt(fn, k_ID_inputSize, X.shape[2] * X.shape[3]);
            cb.SetInt(fn, k_ID_outputSize, O.shape[2] * O.shape[3]);
        }
        else //if (X.shape.rank == 3)
        {
            var n = O.shape[0];
            var k = O.shape[1];
            var h = O.shape[2];

            workItemsX = ComputeHelper.IDivC(h, 4);
            workItemsY = ComputeHelper.IDivC(k, 8);
            workItemsZ = n;

            fn = K.shape.Length(2) == 1 ? ComputeFuncSingleton.Instance.Get("Conv1D_1x1") : ComputeFuncSingleton.Instance.Get("Conv1D_KxK");
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(K));
            if (B != null)
            {
                cb.EnableKeyword(fn, "USEBIAS");
                cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            }
            else
            {
                cb.DisableKeyword(fn, "USEBIAS");
            }
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
            cb.SetInt(fn, k_ID_inputChannels, X.shape[1]);
            cb.SetInt(fn, k_ID_inputHeight, X.shape[2]);
            cb.SetInt(fn, k_ID_kernelHeight, K.shape[2]);
            cb.SetInt(fn, k_ID_outputChannels, O.shape[1]);
            cb.SetInt(fn, k_ID_outputHeight, O.shape[2]);
            cb.SetInt(fn, k_ID_strideHeight, strides[0]);
            cb.SetInt(fn, k_ID_padHeight, pads[0]);
            cb.SetInt(fn, k_ID_dilationHeight, dilations[0]);
            cb.SetInt(fn, k_ID_inputChannelsSize, X.shape[1] * X.shape[2]);
            cb.SetInt(fn, k_ID_outputChannelsSize, O.shape[1] * O.shape[2]);
            cb.SetInt(fn, k_ID_kernelChannelSize, K.shape[1] * K.shape[2]);
            cb.SetInt(fn, k_ID_inputSize, X.shape[2]);
            cb.SetInt(fn, k_ID_outputSize, O.shape[2]);
        }

        cb.SetFloat(fn, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

        cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
    }

    void ConvMobile(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
    {
        int workItemsX, workItemsY, workItemsZ;

        ComputeFunc fn;
        // TODO regular conv faster for small spatial/channels size, figure good rule of thumb
        // TODO see when to call T8x8
        if (X.shape.rank == 5)
        {
            var n = O.shape[0];
            var k = O.shape[1];
            var d = O.shape[2];
            var h = O.shape[3];
            var w = O.shape[4];

            fn = ComputeFuncSingleton.Instance.Get("Conv3D_T16x16_R4x4");
            if (K.shape.Length(2) == 1)
                fn = ComputeFuncSingleton.Instance.Get("Conv3D_1x1_T16x16_R4x4");
            cb.SetInt(fn, k_ID_O_depth, O.shape[2]); cb.SetInt(fn, k_ID_O_height, O.shape[3]); cb.SetInt(fn, k_ID_O_width, O.shape[4]);
            cb.SetInt(fn, k_ID_X_depth, X.shape[2]); cb.SetInt(fn, k_ID_X_height, X.shape[3]); cb.SetInt(fn, k_ID_X_width, X.shape[4]);
            cb.SetInt(fn, k_ID_K_depth, K.shape[2]); cb.SetInt(fn, k_ID_K_height, K.shape[3]); cb.SetInt(fn, k_ID_K_width, K.shape[4]);
            workItemsX = ComputeHelper.IDivC(k, 4); workItemsY = ComputeHelper.IDivC(d * h * w, 4); workItemsZ = n;
        }
        else if (X.shape.rank == 4)
        {
            var n = O.shape[0];
            var k = O.shape[1];
            var h = O.shape[2];
            var w = O.shape[3];

            fn = ComputeFuncSingleton.Instance.Get("Conv2D_T16x16_R4x4");
            if (K.shape.Length(2) == 1)
                fn = ComputeFuncSingleton.Instance.Get("Conv2D_1x1_T16x16_R4x4");
            cb.SetInt(fn, k_ID_O_height, O.shape[2]); cb.SetInt(fn, k_ID_O_width, O.shape[3]);
            cb.SetInt(fn, k_ID_X_height, X.shape[2]); cb.SetInt(fn, k_ID_X_width, X.shape[3]);
            cb.SetInt(fn, k_ID_K_height, K.shape[2]); cb.SetInt(fn, k_ID_K_width, K.shape[3]);
            workItemsX = ComputeHelper.IDivC(k, 4); workItemsY = ComputeHelper.IDivC(h * w, 4); workItemsZ = n;
        }
        else //if (X.shape.rank == 3)
        {
            var n = O.shape[0];
            var k = O.shape[1];
            var w = O.shape[2];

            fn = ComputeFuncSingleton.Instance.Get("Conv1D_T16x16_R4x4");
            if (K.shape.Length(2) == 1)
                fn = ComputeFuncSingleton.Instance.Get("Conv1D_1x1_T16x16_R4x4");
            cb.SetInt(fn, k_ID_O_width, O.shape[2]);
            cb.SetInt(fn, k_ID_X_width, X.shape[2]);
            cb.SetInt(fn, k_ID_K_width, K.shape[2]);
            workItemsX = ComputeHelper.IDivC(k, 4);
            workItemsY = ComputeHelper.IDivC(w, 4);
            workItemsZ = n;
        }

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));
        if (B != null)
        {
            cb.EnableKeyword(fn, "USEBIAS");
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
        }
        else
        {
            cb.DisableKeyword(fn, "USEBIAS");
        }
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_O_batch, O.shape[0]);
        cb.SetInt(fn, k_ID_O_channels, O.shape[1]);
        cb.SetInt(fn, k_ID_X_channels, X.shape[1]);
        cb.SetInt4(fn, k_ID__Stride, strides);
        cb.SetInt4(fn, k_ID__Pad, pads);
        cb.SetInt4(fn, k_ID__Dilation, dilations);

        cb.SetFloat(fn, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

        cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
    }

    /// <inheritdoc/>
    public override void ConvTranspose(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Span<int> strides, Span<int> pads, Span<int> outputPadding, Layers.FusableActivation fusedActivation)
    {
        if (X.shape.rank > 5)
        {
            base.ConvTranspose(X, W, B, O, strides, pads, outputPadding, fusedActivation);
            return;
        }

        if (ComputeInfo.IsMobileGPU() || X.shape.rank > 4)
        {
            ConvTransposeMobile(X, W, B, O, strides, pads, outputPadding, fusedActivation);
            return;
        }

        ComputeFunc fn;

        var numSpatialDims = X.shape.rank - 2;

        if (numSpatialDims == 1)
            fn = ComputeFuncSingleton.Instance.Get("ConvTranspose1D_KxK");
        else
            fn = ComputeFuncSingleton.Instance.Get("ConvTranspose2D_KxK");

        var workItemsX = ComputeHelper.IDivC(O.shape.Length(2), 4);
        var workItemsY = ComputeHelper.IDivC(O.shape[1], 8);
        var workItemsZ = O.shape[0];

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(W));
        if (B != null)
        {
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.EnableKeyword(fn, "USEBIAS");
        }
        else
        {
            cb.DisableKeyword(fn, "USEBIAS");
        }
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_inputChannels, X.shape[1]);
        cb.SetInt(fn, k_ID_outputChannels, O.shape[1]);
        cb.SetInt(fn, k_ID_dilationHeight, 1);
        cb.SetInt(fn, k_ID_dilationWidth, 1);

        var kernelSize = W.shape.Length(2);
        var inputSize = X.shape.Length(2);
        var outputSize = O.shape.Length(2);
        cb.SetInt(fn, k_ID_kernelSize, kernelSize);
        cb.SetInt(fn, k_ID_inputSize, inputSize);
        cb.SetInt(fn, k_ID_outputSize, outputSize);
        cb.SetInt(fn, k_ID_inputChannelsSize, X.shape[1] * inputSize);
        cb.SetInt(fn, k_ID_outputChannelsSize, O.shape[1] * outputSize);
        cb.SetInt(fn, k_ID_kernelChannelSize, W.shape[0] * kernelSize);
        cb.SetInt(fn, k_ID_inputWidth, X.shape[-1]);
        cb.SetInt(fn, k_ID_kernelWidth, W.shape[-1]);
        cb.SetInt(fn, k_ID_outputWidth, O.shape[-1]);
        cb.SetInt(fn, k_ID_padWidth, W.shape[-1] - pads[numSpatialDims - 1] - 1);
        cb.SetInt(fn, k_ID_strideWidth, strides[numSpatialDims - 1]);
        if (numSpatialDims > 1)
        {
            cb.SetInt(fn, k_ID_inputHeight, X.shape[-2]);
            cb.SetInt(fn, k_ID_kernelHeight, W.shape[-2]);
            cb.SetInt(fn, k_ID_outputHeight, O.shape[-2]);
            cb.SetInt(fn, k_ID_padHeight, W.shape[-2] - pads[numSpatialDims - 2] - 1);
            cb.SetInt(fn, k_ID_strideHeight, strides[numSpatialDims - 2]);
        }

        cb.SetFloat(fn, k_ID__MinValue, fusedActivation == Layers.FusableActivation.Relu ? 0.0f : float.MinValue);

        cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
    }

    void ConvTransposeMobile(TensorFloat X, TensorFloat W, TensorFloat B, TensorFloat O, Span<int> stride, Span<int> pad, Span<int> outputAdjustment, Layers.FusableActivation fusedActivation)
    {
        ComputeFunc fn;

        var numSpatialDims = X.shape.rank - 2;

        if (numSpatialDims == 1)
            fn = ComputeFuncSingleton.Instance.Get("ConvTranspose1D_T16x16_R4x4");
        else if (numSpatialDims == 2)
            fn = ComputeFuncSingleton.Instance.Get("ConvTranspose2D_T16x16_R4x4");
        else
            fn = ComputeFuncSingleton.Instance.Get("ConvTranspose3D_T16x16_R4x4");

        cb.SetInt(fn, k_ID_O_channels, O.shape[1]);
        cb.SetInt(fn, k_ID_X_channels, X.shape[1]);
        cb.SetInt(fn, k_ID_maxXIndex, X.shape.length - 1);
        cb.SetInt(fn, k_ID_maxKIndex, W.shape.length - 1);
        cb.SetInt4(fn, k_ID__Pad, pad);
        cb.SetInt4(fn, k_ID__Stride, stride);

        cb.SetInt(fn, k_ID_O_width, O.shape[-1]);
        cb.SetInt(fn, k_ID_X_width, X.shape[-1]);
        cb.SetInt(fn, k_ID_K_width, W.shape[-1]);

        if (numSpatialDims > 1)
        {
            cb.SetInt(fn, k_ID_O_height, O.shape[-2]);
            cb.SetInt(fn, k_ID_X_height, X.shape[-2]);
            cb.SetInt(fn, k_ID_K_height, W.shape[-2]);
        }

        if (numSpatialDims > 2)
        {
            cb.SetInt(fn, k_ID_O_depth, O.shape[-3]);
            cb.SetInt(fn, k_ID_X_depth, X.shape[-3]);
            cb.SetInt(fn, k_ID_K_depth, W.shape[-3]);
        }

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(W));
        if (B != null)
        {
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.SetInt(fn, k_ID_maxBIndex, B.shape.length - 1);
            cb.EnableKeyword(fn, "USEBIAS");
        }
        else
        {
            cb.DisableKeyword(fn, "USEBIAS");
        }
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        var workItemsX = ComputeHelper.IDivC(O.shape[1], 4);
        var workItemsY = ComputeHelper.IDivC(O.shape.Length(2), 4);
        var workItemsZ = O.shape[0];
        if (fusedActivation == Layers.FusableActivation.Relu)
            cb.SetFloat(fn, k_ID__MinValue, 0.0f);
        else
            cb.SetFloat(fn, k_ID__MinValue, float.MinValue);

        cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
    }

    /// <inheritdoc/>
    public override void Resize(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.InterpolationMode interpolationMode, Layers.NearestMode nearestMode, Layers.CoordTransformMode coordTransformMode)
    {
        if (X.shape.rank > 5 || scale[0] != 1f || scale[1] != 1f)
        {
            base.Resize(X, O, scale, interpolationMode, nearestMode, coordTransformMode);
            return;
        }

        switch (X.shape.rank)
        {
            case 3:
                Upsample1D(X, O, scale, nearestMode, interpolationMode, coordTransformMode);
                break;
            case 4:
                Upsample2D(X, O, scale, nearestMode, interpolationMode, coordTransformMode);
                break;
            case 5:
                Upsample3D(X, O, scale, nearestMode, interpolationMode, coordTransformMode);
                break;
        }
    }

    void Upsample1D(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
    {
        Vector4 scaleXY = Vector4.zero;
        Vector4 biasXY = Vector4.zero;
        for (int i = 0; i < 1; i++)
        {
            OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
            scaleXY[i] = outputScale;
            biasXY[i] = outputBias;
        }

        ComputeFunc fn;
        if (interpolationMode == Layers.InterpolationMode.Nearest)
        {
            string kernelName;
            switch (nearestMode)
            {
                case Layers.NearestMode.RoundPreferFloor:
                case Layers.NearestMode.Ceil:
                    kernelName = "Upsample1D_Nearest_Ceil";
                    break;
                case Layers.NearestMode.RoundPreferCeil:
                case Layers.NearestMode.Floor:
                    kernelName = "Upsample1D_Nearest_Floor";
                    break;
                default:
                    throw new NotImplementedException();
            }
            fn = ComputeFuncSingleton.Instance.Get(kernelName);
        }
        else //if (interpolationMode == Layers.InterpolationMode.Linear)
        {
            fn = ComputeFuncSingleton.Instance.Get("Upsample1D_Linear_None");
        }

        cb.SetVector(fn, k_ID_scale, scaleXY);
        cb.SetVector(fn, k_ID_bias, biasXY);
        cb.SetInt(fn, k_ID_inWidth, X.shape[2]);
        cb.SetInt(fn, k_ID_outWidth, O.shape[2]);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2], 1);
    }

    void Upsample2D(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
    {
        Vector4 scaleXY = Vector4.zero;
        Vector4 biasXY = Vector4.zero;
        for (int i = 0; i < 2; i++)
        {
            OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
            scaleXY[i] = outputScale;
            biasXY[i] = outputBias;
        }

        ComputeFunc fn;
        if (interpolationMode == Layers.InterpolationMode.Nearest)
        {
            string kernelName;
            switch (nearestMode)
            {
                case Layers.NearestMode.RoundPreferFloor:
                case Layers.NearestMode.Ceil:
                    kernelName = "Upsample2D_Nearest_Ceil";
                    break;
                case Layers.NearestMode.RoundPreferCeil:
                case Layers.NearestMode.Floor:
                    kernelName = "Upsample2D_Nearest_Floor";
                    break;
                default:
                    throw new NotImplementedException();
            }
            fn = ComputeFuncSingleton.Instance.Get(kernelName);
        }
        else //if (interpolationMode == Layers.InterpolationMode.Linear)
        {
            fn = ComputeFuncSingleton.Instance.Get("Upsample2D_Linear_None");
        }

        cb.SetVector(fn, k_ID_scale, scaleXY);
        cb.SetVector(fn, k_ID_bias, biasXY);
        cb.SetInt(fn, k_ID_inHeight, X.shape[2]);
        cb.SetInt(fn, k_ID_inWidth, X.shape[3]);
        cb.SetInt(fn, k_ID_outHeight, O.shape[2]);
        cb.SetInt(fn, k_ID_outWidth, O.shape[3]);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2], O.shape[3]);
    }

    void Upsample3D(TensorFloat X, TensorFloat O, ReadOnlySpan<float> scale, Layers.NearestMode nearestMode, Layers.InterpolationMode interpolationMode, Layers.CoordTransformMode coordTransformMode)
    {
        Vector4 scaleXYD = Vector4.zero;
        Vector4 biasXYD = Vector4.zero;
        for (int i = 0; i < 3; i++)
        {
            OpsUtils.GetScaleAndBias(X.shape[2 + i], O.shape[2 + i], scale[2 + i], coordTransformMode, interpolationMode, nearestMode, out float outputScale, out float outputBias);
            scaleXYD[i] = outputScale;
            biasXYD[i] = outputBias;
        }

        ComputeFunc fn;
        if (interpolationMode == Layers.InterpolationMode.Nearest)
        {
            string kernelName;
            switch (nearestMode)
            {
                case Layers.NearestMode.RoundPreferFloor:
                case Layers.NearestMode.Ceil:
                    kernelName = "Upsample3D_Nearest_Ceil";
                    break;
                case Layers.NearestMode.RoundPreferCeil:
                case Layers.NearestMode.Floor:
                    kernelName = "Upsample3D_Nearest_Floor";
                    break;
                default:
                    throw new NotImplementedException();
            }
            fn = ComputeFuncSingleton.Instance.Get(kernelName);
        }
        else //if (interpolationMode == Layers.InterpolationMode.Linear)
        {
            fn = ComputeFuncSingleton.Instance.Get("Upsample3D_Linear_None");
        }

        cb.SetVector(fn, k_ID_scale, scaleXYD);
        cb.SetVector(fn, k_ID_bias, biasXYD);
        cb.SetInt(fn, k_ID_inDepth, X.shape[2]);
        cb.SetInt(fn, k_ID_inHeight, X.shape[3]);
        cb.SetInt(fn, k_ID_inWidth, X.shape[4]);
        cb.SetInt(fn, k_ID_outBatch, O.shape[0]);
        cb.SetInt(fn, k_ID_outChannels, O.shape[1]);
        cb.SetInt(fn, k_ID_outDepth, O.shape[2]);
        cb.SetInt(fn, k_ID_outHeight, O.shape[3]);
        cb.SetInt(fn, k_ID_outWidth, O.shape[4]);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape[2], O.shape[3], O.shape[4]);
    }

    /// <inheritdoc/>
    public override void DepthToSpace(TensorFloat X, TensorFloat O, int blocksize, Layers.DepthToSpaceMode mode)
    {
        var fn = ComputeFuncSingleton.Instance.Get(mode == Layers.DepthToSpaceMode.DepthColumnRow ? "DepthToSpaceDepthColumnRow" : "DepthToSpaceColumnRowDepth");
        cb.SetInt(fn, k_ID_blocksize, blocksize);
        cb.SetInt(fn, k_ID_inputChannels, X.shape[1]);
        cb.SetInt(fn, k_ID_inputHeight, X.shape[2]);
        cb.SetInt(fn, k_ID_inputWidth, X.shape[3]);
        cb.SetInt(fn, k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
        cb.SetInt(fn, k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
        cb.SetInt(fn, k_ID_outputChannels, O.shape[1]);
        cb.SetInt(fn, k_ID_outputHeight, O.shape[2]);
        cb.SetInt(fn, k_ID_outputWidth, O.shape[3]);
        cb.SetInt(fn, k_ID_outputSpatialSize, O.shape[2] * O.shape[3]);
        cb.SetInt(fn, k_ID_outputBatch, O.shape[0]);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
    }

    /// <inheritdoc/>
    public override void SpaceToDepth(TensorFloat X, TensorFloat O, int blocksize)
    {
        var fn = ComputeFuncSingleton.Instance.Get("SpaceToDepth");
        cb.SetInt(fn, k_ID_blocksize, blocksize);
        cb.SetInt(fn, k_ID_inputChannels, X.shape[1]);
        cb.SetInt(fn, k_ID_inputHeight, X.shape[2]);
        cb.SetInt(fn, k_ID_inputWidth, X.shape[3]);
        cb.SetInt(fn, k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
        cb.SetInt(fn, k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
        cb.SetInt(fn, k_ID_outputChannels, O.shape[1]);
        cb.SetInt(fn, k_ID_outputHeight, O.shape[2]);
        cb.SetInt(fn, k_ID_outputWidth, O.shape[3]);
        cb.SetInt(fn, k_ID_outputSpatialSize, O.shape[2] * O.shape[3]);
        cb.SetInt(fn, k_ID_outputBatch, O.shape[0]);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
    }

    void LocalPool1D(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad, string kernelName)
    {
        var fn = ComputeFuncSingleton.Instance.Get(kernelName);
        cb.SetInt(fn, k_ID_stride, stride[0]);
        cb.SetInt(fn, k_ID_pad, pad[0]);
        cb.SetInt(fn, k_ID_inHeight, X.shape[2]);
        cb.SetInt(fn, k_ID_pool, pool[0]);
        cb.SetInt(fn, k_ID_outHeight, O.shape[2]);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    void LocalPool2D(TensorFloat X, TensorFloat O, int[] pool, int[] stride, int[] pad, string kernelName)
    {
        var fn = ComputeFuncSingleton.Instance.Get(kernelName);
        cb.SetInt(fn, k_ID_strideX, stride[1]);
        cb.SetInt(fn, k_ID_strideY, stride[0]);
        cb.SetInt(fn, k_ID_padX, pad[1]);
        cb.SetInt(fn, k_ID_padY, pad[0]);

        cb.SetInt(fn, k_ID_inHeight, X.shape[2]);
        cb.SetInt(fn, k_ID_inWidth, X.shape[3]);

        cb.SetInt(fn, k_ID_poolX, pool[1]);
        cb.SetInt(fn, k_ID_poolY, pool[0]);

        cb.SetInt(fn, k_ID_outHeight, O.shape[2]);
        cb.SetInt(fn, k_ID_outWidth, O.shape[3]);

        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void MaxPool(TensorFloat X, TensorFloat O, int[] kernelShape, int[] strides, int[] pads)
    {
        switch (X.shape.rank)
        {
            case 3:
                LocalPool1D(X, O, kernelShape, strides, pads, "MaxPool1D");
                return;
            case 4:
                LocalPool2D(X, O, kernelShape, strides, pads, "MaxPool2D");
                return;
            default:
                base.MaxPool(X, O, kernelShape, strides, pads);
                return;
        }
    }

    /// <inheritdoc/>
    public override void AveragePool(TensorFloat X, TensorFloat O, int[] kernelShape, int[] strides, int[] pads)
    {
        switch (X.shape.rank)
        {
            case 3:
                LocalPool1D(X, O, kernelShape, strides, pads, "AveragePool1D");
                return;
            case 4:
                LocalPool2D(X, O, kernelShape, strides, pads, "AveragePool2D");
                return;
            default:
                base.AveragePool(X, O, kernelShape, strides, pads);
                return;
        }
    }

    void Reduce(Tensor X, Tensor O, int outerLength, int reduceLength, int innerLength, string localKernel, string globalKernel, string fallbackKernel)
    {
        Reduce(X, null, O, outerLength, reduceLength, innerLength, localKernel, globalKernel, fallbackKernel);
    }

    void Reduce(Tensor X, Tensor Xmax, Tensor O, int outerLength, int reduceLength, int innerLength, string localKernel, string globalKernel, string fallbackKernel)
    {
        if (innerLength > (int)ComputeFunc.SafeDispatchLimit || outerLength > (int)ComputeFunc.SafeDispatchLimit)
        {
            var fnUnrolled = ComputeFuncSingleton.Instance.Get(fallbackKernel);
            cb.SetInt(fnUnrolled, k_ID_ReducedDim, reduceLength);
            cb.SetInt(fnUnrolled, k_ID_InnerDim, innerLength);
            cb.SetFloat(fnUnrolled, k_ID_Normalization, 1.0f / reduceLength);

            if (Xmax != null)
                cb.ScheduleXBO(fnUnrolled, Pin(X), Pin(Xmax), Pin(O, clearOnInit: false), outerLength * innerLength);
            else
                cb.ScheduleXO(fnUnrolled, Pin(X), Pin(O, clearOnInit: false), outerLength * innerLength);
            return;
        }

        int localReduceLength = reduceLength;
        bool isFirstDispatch = true;

        const int kernelReductionThreadCount = 64 * 4;

        // downsample with pyramid approach
        while (localReduceLength > kernelReductionThreadCount)
        {
            int spatialLengthO = ComputeHelper.IDivC(localReduceLength, kernelReductionThreadCount);
            var Otemp = NewTempTensorFloat(new TensorShape(outerLength * spatialLengthO * innerLength));

            var fnPool = ComputeFuncSingleton.Instance.Get(localKernel);
            cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
            if (Xmax != null)
                cb.SetTensorAsBuffer(fnPool, k_ID_Bptr, Pin(Xmax));
            cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp, clearOnInit: false));
            cb.SetInt(fnPool, k_ID_ReducedDim, localReduceLength);
            cb.SetInt(fnPool, k_ID_InnerDim, innerLength);
            cb.SetInt(fnPool, k_ID_SpatialDimsO, spatialLengthO);
            cb.SetInt(fnPool, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

            cb.Dispatch(fnPool, outerLength, ComputeHelper.IDivC(localReduceLength, 4), innerLength);

            X = Otemp;
            localReduceLength = spatialLengthO;
            isFirstDispatch = false;
        }

        var fn = ComputeFuncSingleton.Instance.Get(globalKernel);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        if (Xmax != null)
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(Xmax));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_ReducedDim, localReduceLength);
        cb.SetInt(fn, k_ID_InnerDim, innerLength);
        cb.SetInt(fn, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);
        cb.SetFloat(fn, k_ID_Normalization, 1.0f / reduceLength);

        cb.Dispatch(fn, outerLength, 1, innerLength);
    }

    void GlobalPool(TensorFloat X, TensorFloat O, string localKernel, string globalKernel)
    {
        int globalSpatialDims = X.shape.Length(2);
        int globalNonSpatialLength = X.shape[0] * X.shape[1];

        int localSpatialLength = globalSpatialDims;

        var Oshape = new TensorShape(X.shape[0], X.shape[1], localSpatialLength);

        // downsample with pyramid approach
        while (localSpatialLength > 64 * 4)
        {
            int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
            Oshape[2] = spatialLengthO;
            var Otemp = NewTempTensorFloat(Oshape);

            var fnPool = ComputeFuncSingleton.Instance.Get(localKernel);
            cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp, clearOnInit: false));
            cb.SetInt(fnPool, k_ID_SpatialDims, localSpatialLength);
            cb.SetInt(fnPool, k_ID_SpatialDimsO, spatialLengthO);

            cb.Dispatch(fnPool, globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

            X = Otemp;
            localSpatialLength = spatialLengthO;
        }

        var fn = ComputeFuncSingleton.Instance.Get(globalKernel);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_SpatialDims, localSpatialLength);
        cb.SetInt(fn, k_ID_GlobalSpatialDims, globalSpatialDims);

        cb.Dispatch(fn, globalNonSpatialLength, 1, 1);
    }

    /// <inheritdoc/>
    public override void GlobalMaxPool(TensorFloat X, TensorFloat O)
    {
        GlobalPool(X, O, "MaxPoolReduce", "GlobalMaxPool");
    }

    /// <inheritdoc/>
    public override void GlobalAveragePool(TensorFloat X, TensorFloat O)
    {
        GlobalPool(X, O, "AveragePoolReduce", "GlobalAveragePool");
    }

    /// <inheritdoc/>
    public override void GlobalAverageVariancePool(TensorFloat X, TensorFloat O, int axis)
    {
        int globalNonSpatialLength = X.shape.Length(0, axis);
        int globalSpatialDims = X.shape.length / globalNonSpatialLength;

        int localSpatialLength = globalSpatialDims;

        var Oshape = new TensorShape(globalNonSpatialLength, localSpatialLength);

        TensorFloat X2 = X; // save a X^2 and do it in the first dispatch
        bool isFirstDispatch = true;

        // downsample with pyramid approach
        while (localSpatialLength > 64 * 4)
        {
            int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
            Oshape[-1] = spatialLengthO;
            var Otemp = NewTempTensorFloat(Oshape);
            var O2temp = NewTempTensorFloat(Oshape);

            var fnPool = ComputeFuncSingleton.Instance.Get("AverageVariancePoolReduce");
            cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fnPool, k_ID_X2ptr, Pin(X2));
            cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp, clearOnInit: false));
            cb.SetTensorAsBuffer(fnPool, k_ID_O2ptr, Pin(O2temp, clearOnInit: false));
            cb.SetInt(fnPool, k_ID_SpatialDims, localSpatialLength);
            cb.SetInt(fnPool, k_ID_SpatialDimsO, spatialLengthO);
            cb.SetInt(fnPool, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

            cb.Dispatch(fnPool, globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

            X = Otemp;
            X2 = O2temp;
            localSpatialLength = spatialLengthO;
            isFirstDispatch = false;
        }

        var fn = ComputeFuncSingleton.Instance.Get("GlobalAverageVariancePool");
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_X2ptr, Pin(X2));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_SpatialDims, localSpatialLength);
        cb.SetInt(fn, k_ID_GlobalSpatialDims, globalSpatialDims);
        cb.SetInt(fn, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

        cb.Dispatch(fn, globalNonSpatialLength, 1, 1);
    }

    void GroupedConv(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int groups, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
    {
        var Otmp = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(O.shape) : O;

        int outputGroupedChannels = Otmp.shape[1] / groups;

        ComputeFunc fn;

        if (X.shape.rank == 5)
        {
            fn = ComputeFuncSingleton.Instance.Get(outputGroupedChannels < 64 ? "GroupedConv3D" : "GroupedConv3D_GroupLower64");
            cb.SetInt(fn, k_ID_O_depth, Otmp.shape[2]);
            cb.SetInt(fn, k_ID_O_height, Otmp.shape[3]);
            cb.SetInt(fn, k_ID_O_width, Otmp.shape[4]);
            cb.SetInt(fn, k_ID_X_depth, X.shape[2]);
            cb.SetInt(fn, k_ID_X_height, X.shape[3]);
            cb.SetInt(fn, k_ID_X_width, X.shape[4]);
            cb.SetInt(fn, k_ID_K_depth, K.shape[2]);
            cb.SetInt(fn, k_ID_K_height, K.shape[3]);
            cb.SetInt(fn, k_ID_K_width, K.shape[4]);
        }
        else if (X.shape.rank == 4)
        {
            fn = ComputeFuncSingleton.Instance.Get(outputGroupedChannels < 64 ? "GroupedConv2D" : "GroupedConv2D_GroupLower64");
            cb.SetInt(fn, k_ID_O_height, Otmp.shape[2]);
            cb.SetInt(fn, k_ID_O_width, Otmp.shape[3]);
            cb.SetInt(fn, k_ID_X_height, X.shape[2]);
            cb.SetInt(fn, k_ID_X_width, X.shape[3]);
            cb.SetInt(fn, k_ID_K_height, K.shape[2]);
            cb.SetInt(fn, k_ID_K_width, K.shape[3]);
        }
        else
        {
            fn = ComputeFuncSingleton.Instance.Get(outputGroupedChannels < 64 ? "GroupedConv1D" : "GroupedConv1D_GroupLower64");
            cb.SetInt(fn, k_ID_O_width, Otmp.shape[2]);
            cb.SetInt(fn, k_ID_X_width, X.shape[2]);
            cb.SetInt(fn, k_ID_K_width, K.shape[2]);
        }

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));
        if (B != null)
        {
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
            cb.EnableKeyword(fn, "USEBIAS");
        }
        else
        {
            cb.DisableKeyword(fn, "USEBIAS");
        }
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(Otmp, clearOnInit: false));
        cb.SetInt(fn, k_ID_O_channels, O.shape[1]);
        cb.SetInt(fn, k_ID_X_channels, X.shape[1]);
        cb.SetInt4(fn, k_ID__Stride, strides);
        cb.SetInt4(fn, k_ID__Pad, pads);
        cb.SetInt4(fn, k_ID__Dilation, dilations);
        cb.SetInt(fn, k_ID__Groups, groups);
        cb.SetInt(fn, k_ID_strideX, X.shape.Length(2));
        cb.SetInt(fn, k_ID_strideO, Otmp.shape.Length(2));
        cb.SetInt(fn, k_ID_strideK, K.shape.Length(2));
        cb.SetInt(fn, k_ID_inputGroupedChannels, X.shape[1] / groups);
        cb.SetInt(fn, k_ID_outputGroupedChannels, Otmp.shape[1] / groups);

        cb.Dispatch(fn, ComputeHelper.IDivC(Otmp.shape[1], 4), ComputeHelper.IDivC(Otmp.shape.Length(2), 4), Otmp.shape[0]);

        if (fusedActivation != Layers.FusableActivation.None)
            ApplyFusedActivation(Otmp, O, fusedActivation);
    }

    void DepthwiseConv2D(TensorFloat X, TensorFloat K, TensorFloat B, TensorFloat O, int group, Span<int> strides, Span<int> pads, Span<int> dilations, Layers.FusableActivation fusedActivation)
    {
        var Otmp = (fusedActivation != Layers.FusableActivation.None) ? NewTempTensorFloat(O.shape) : O;

        ComputeFunc fn;
        int workItemsX, workItemsY, workItemsZ;

        if (K.shape[2] == 3 && K.shape[3] == 3 && strides[0] == 1 && strides[1] == 1 && dilations[0] == 1 && dilations[1] == 1)
        {
            var KWE = NewTempTensorFloat(new TensorShape(Otmp.shape[1], 4, 4));

            ComputeFunc fnKE = ComputeFuncSingleton.Instance.Get("KernelWinoExpand");
            cb.SetTensorAsBuffer(fnKE, k_ID_Kptr, Pin(K));
            cb.SetTensorAsBuffer(fnKE, k_ID_Optr, Pin(KWE, clearOnInit: false));
            cb.SetInt(fnKE, k_ID_O_channels, O.shape[1]);
            cb.Dispatch(fnKE, O.shape[1], 1, 1);

            fn = ComputeFuncSingleton.Instance.Get("DepthwiseConv2DWinograd");

            cb.SetTensorAsBuffer(fn, k_ID_KWEptr, Pin(KWE));

            workItemsX = ComputeHelper.IDivC(Otmp.shape[3], 2);
            workItemsY = ComputeHelper.IDivC(Otmp.shape[2], 2);
            workItemsZ = Otmp.shape[0] * Otmp.shape[1];
        }
        else
        {
            fn = ComputeFuncSingleton.Instance.Get("DepthwiseConv2DDirect");

            cb.SetTensorAsBuffer(fn, k_ID_Kptr, Pin(K));

            cb.SetInt(fn, k_ID_K_heightDiv4, ComputeHelper.IDivC(K.shape[2], 4));
            cb.SetInt(fn, k_ID_K_widthDiv4, ComputeHelper.IDivC(K.shape[3], 4));
            cb.SetInt(fn, k_ID_K_height, K.shape[2]);
            cb.SetInt(fn, k_ID_K_width, K.shape[3]);
            cb.SetInt(fn, k_ID_StrideK, K.shape[2] * K.shape[3]);

            workItemsX = Otmp.shape[3];
            workItemsY = Otmp.shape[2];
            workItemsZ = Otmp.shape[0] * Otmp.shape[1];
        }

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        if (B != null)
        {
            cb.EnableKeyword(fn, "USEBIAS");
            cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
        }
        else
        {
            cb.DisableKeyword(fn, "USEBIAS");
        }
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(Otmp, clearOnInit: false));
        cb.SetInt(fn, k_ID_X_channels, X.shape[1]);
        cb.SetInt(fn, k_ID_X_height, X.shape[2]);
        cb.SetInt(fn, k_ID_X_width, X.shape[3]);
        cb.SetInt(fn, k_ID_O_batch, O.shape[0]);
        cb.SetInt(fn, k_ID_O_channels, O.shape[1]);
        cb.SetInt(fn, k_ID_O_height, O.shape[2]);
        cb.SetInt(fn, k_ID_O_width, O.shape[3]);
        cb.SetInt4(fn, k_ID_Stride, strides);
        cb.SetInt4(fn, k_ID_Pad, pads);
        cb.SetInt4(fn, k_ID_Dilation, dilations);
        cb.SetInt(fn, k_ID_StrideX, X.shape[2] * X.shape[3]);
        cb.SetInt(fn, k_ID_MaxLengthX, X.shape.length - 1);
        cb.SetInt(fn, k_ID_MaxLengthK, K.shape.length - 1);
        cb.SetInt(fn, k_ID_StrideO, Otmp.shape[2] * Otmp.shape[3]);
        cb.SetInt(fn, k_ID_StrideFeaturesO, Otmp.shape[0] * Otmp.shape[1]);

        cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);

        if (fusedActivation != Layers.FusableActivation.None)
            ApplyFusedActivation(Otmp, O, fusedActivation);
    }

    /// <inheritdoc/>
    public override void ScaleBias(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O)
    {
        int batch = X.shape[0];
        int channels = X.shape[1];
        int spatialDims = X.shape.Length(2);

        var fn = ComputeFuncSingleton.Instance.Get("ScaleBias");

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_LengthO, O.shape.length);
        cb.SetInt(fn, k_ID_batch, batch);
        cb.SetInt(fn, k_ID_channels, channels);
        cb.SetInt(fn, k_ID_spatialDims, spatialDims);
        cb.Dispatch(fn, spatialDims, ComputeHelper.IDivC(channels, 4), batch);
    }

    /// <inheritdoc/>
    public override void InstanceNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O, float epsilon)
    {
        var reduceOpShape = ShapeInference.GlobalAverageVariancePool(X.shape);
        var meanVariance = NewTempTensorFloat(reduceOpShape);
        GlobalAverageVariancePool(X, meanVariance, 2);

        var fn = ComputeFuncSingleton.Instance.Get("InstanceNormalizationTail");

        cb.SetInt(fn, k_ID_channels, X.shape[1]);
        cb.SetInt(fn, k_ID_spatialDims, X.shape.length / (X.shape[0] * X.shape[1]));
        cb.SetFloat(fn, k_ID_epsilon, epsilon);

        cb.ScheduleXSBWO(fn, Pin(X), Pin(S), Pin(B), Pin(meanVariance), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void LayerNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat O, float epsilon)
    {
        int axis = X.shape.Axis(-1);

        var reducedShape = X.shape.Reduce(axis);
        reducedShape[axis] = 2;

        int axisDim = X.shape[axis];
        int outerLength = X.shape.Length(0, -1);

        var meanVariance = NewTempTensorFloat(reducedShape);
        GlobalAverageVariancePool(X, meanVariance, -1);

        var fn = ComputeFuncSingleton.Instance.Get("LayerNormalizationTail");
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(meanVariance));
        cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_axisDim, axisDim);
        cb.SetInt(fn, k_ID_outerLength, outerLength);
        cb.SetFloat(fn, k_ID_epsilon, epsilon);
        cb.Dispatch(fn, axisDim, outerLength, 1);
    }

    /// <inheritdoc/>
    public override void BatchNormalization(TensorFloat X, TensorFloat S, TensorFloat B, TensorFloat mean, TensorFloat variance, TensorFloat O, float epsilon)
    {
        var batch = X.shape[0];
        var channels = X.shape.rank == 1 ? 1 : X.shape[1];
        var spatialDims = X.shape.Length(2);

        var fn = ComputeFuncSingleton.Instance.Get("BatchNormalization");

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(mean));
        cb.SetTensorAsBuffer(fn, k_ID_Zptr, Pin(variance));
        cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(S));
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_LengthO, O.shape.length);
        cb.SetInt(fn, k_ID_batch, batch);
        cb.SetInt(fn, k_ID_channels, channels);
        cb.SetInt(fn, k_ID_spatialDims, spatialDims);
        cb.SetFloat(fn, k_ID_epsilon, epsilon);
        cb.Dispatch(fn, spatialDims, ComputeHelper.IDivC(channels, 4), batch);
    }

    /// <inheritdoc/>
    public override void Range(TensorFloat O, float start, float delta)
    {
        var fn = ComputeFuncSingleton.Instance.Get("RangeFloat");
        cb.SetFloat(fn, k_ID_rangeStartFloat, start);
        cb.SetFloat(fn, k_ID_rangeDeltaFloat, delta);
        cb.SetInt(fn, k_ID_O_length, O.shape.length);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.Dispatch(fn, ComputeHelper.IDivC(O.shape.length, 4), 1, 1);
    }

    /// <inheritdoc/>
    public override void Range(TensorInt O, int start, int delta)
    {
        var fn = ComputeFuncSingleton.Instance.Get("RangeInt");
        cb.SetInt(fn, k_ID_rangeStartInt, start);
        cb.SetInt(fn, k_ID_rangeDeltaInt, delta);
        cb.SetInt(fn, k_ID_O_length, O.shape.length);
        cb.SetTensorAsBuffer(fn, k_ID_OIntptr, Pin(O, clearOnInit: false));
        cb.Dispatch(fn, ComputeHelper.IDivC(O.shape.length, 4), 1, 1);
    }

    /// <inheritdoc/>
    public override void Relu(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Relu");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void PRelu(TensorFloat X, TensorFloat S, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("PRelu");
        cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeS, k_ID_stridesS, S.shape);
        cb.SetInt(fn, k_ID_rank, O.shape.rank);

        cb.ScheduleXBO(fn, Pin(X), Pin(S), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Relu6(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Relu6");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void LeakyRelu(TensorFloat X, TensorFloat O, float alpha)
    {
        var fn = ComputeFuncSingleton.Instance.Get("LeakyRelu");
        cb.SetFloat(fn, k_ID_alpha, alpha);
        cb.SetFloat(fn, k_ID_f1, 0.5f * (1f + alpha));
        cb.SetFloat(fn, k_ID_f2, 0.5f * (1f - alpha));
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Tanh(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Tanh");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Softplus(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Softplus");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Sigmoid(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Sigmoid");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void HardSigmoid(TensorFloat X, TensorFloat O, float alpha, float beta)
    {
        var fn = ComputeFuncSingleton.Instance.Get("HardSigmoid");
        cb.SetFloat(fn, k_ID_alpha, alpha);
        cb.SetFloat(fn, k_ID_beta, beta);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Elu(TensorFloat X, TensorFloat O, float alpha)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Elu");
        cb.SetFloat(fn, k_ID_alpha, alpha);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Gelu(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Gelu");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Selu(TensorFloat X, TensorFloat O, float alpha, float gamma)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Selu");
        cb.SetFloat(fn, k_ID_alpha, alpha);
        cb.SetFloat(fn, k_ID_gamma, gamma);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Swish(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Swish");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Abs(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("AbsFloat");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Abs(TensorInt X, TensorInt O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("AbsInt");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Neg(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("NegFloat");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Neg(TensorInt X, TensorInt O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("NegInt");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Ceil(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Ceil");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Clip(TensorFloat X, TensorFloat O, float min, float max)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Clip");
        cb.SetFloat(fn, k_ID_minV, min);
        cb.SetFloat(fn, k_ID_maxV, max);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Floor(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Floor");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Round(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Round");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Reciprocal(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Reciprocal");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Square(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Square");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Exp(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Exp");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Log(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Log");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Sqrt(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Sqrt");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Acos(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Acos");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Acosh(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Acosh");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Asin(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Asin");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Asinh(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Asinh");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Atan(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Atan");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Atanh(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Atanh");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Cos(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Cos");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Cosh(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Cosh");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Sin(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Sin");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Sinh(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Sinh");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Tan(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Tan");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Erf(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Erf");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Celu(TensorFloat X, TensorFloat O, float alpha)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Celu");
        cb.SetFloat(fn, k_ID_alpha, alpha);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Shrink(TensorFloat X, TensorFloat O, float bias, float lambd)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Shrink");
        cb.SetFloat(fn, k_ID_bias, bias);
        cb.SetFloat(fn, k_ID_lambd, lambd);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Softsign(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Softsign");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void ThresholdedRelu(TensorFloat X, TensorFloat O, float alpha)
    {
        var fn = ComputeFuncSingleton.Instance.Get("ThresholdedRelu");
        cb.SetFloat(fn, k_ID_alpha, alpha);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Softmax(TensorFloat X, TensorFloat O, int axis)
    {
        // Allocate temp tensors
        int innerLength = X.shape.Strides(axis);
        int outerLength = X.shape.Length(0, axis);
        int reduceLength = X.shape[axis];

        var Xmax = NewTempTensorFloat(new TensorShape(outerLength * innerLength));
        var XexpSums = NewTempTensorFloat(Xmax.shape);

        // x_max = X.max(axis=1)
        // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
        Reduce(X, Xmax, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
        Reduce(X, Xmax, XexpSums, outerLength, reduceLength, innerLength, "ReduceSumExpFloat", "GlobalReduceSumExpFloat", "UnrolledReduceSumExpFloat");

        // exp(x[n,c] - x_max[n]) / e_x_sum[n]
        var fn = ComputeFuncSingleton.Instance.Get("SoftmaxEnd");
        cb.SetInt(fn, k_ID_innerLength, innerLength);
        cb.SetInt(fn, k_ID_reduceLength, reduceLength);
        cb.ScheduleXSBO(fn, Pin(X), Pin(XexpSums), Pin(Xmax), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void LogSoftmax(TensorFloat X, TensorFloat O, int axis)
    {
        // Allocate temp tensors
        int innerLength = X.shape.Strides(axis);
        int outerLength = X.shape.Length(0, axis);
        int reduceLength = X.shape[axis];

        var Xmax = NewTempTensorFloat(new TensorShape(outerLength * innerLength));
        var XexpSums = NewTempTensorFloat(Xmax.shape);

        // x_max = X.max(axis=1)
        // logexp_sum = log(Sum[exp(x[:,c] - x_max[:]), c]) - x_max[:]
        Reduce(X, Xmax, outerLength, reduceLength, innerLength, "ReduceMaxFloat", "GlobalReduceMaxFloat", "UnrolledReduceMaxFloat");
        Reduce(X, Xmax, XexpSums, outerLength, reduceLength, innerLength, "ReduceLogSumExpFloat", "GlobalReduceLogSumExpFloat", "UnrolledReduceLogSumExpFloat");

        // x[n,c] - logexp_sum
        var fn = ComputeFuncSingleton.Instance.Get("LogSoftmaxEnd");
        cb.SetInt(fn, k_ID_innerLength, innerLength);
        cb.SetInt(fn, k_ID_reduceLength, reduceLength);
        cb.ScheduleXBO(fn, Pin(X), Pin(XexpSums), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Hardmax(TensorFloat X, TensorFloat O, int axis)
    {
        //Allocate temp tensors
        var reduceOpShape = X.shape.Reduce(axis);
        var argMax = NewTempTensorFloat(reduceOpShape);

        int offsetReduce = X.shape.Strides(axis);

        // argmax
        {
            var fn = ComputeFuncSingleton.Instance.Get("ArgMaxFloatFirst");
            cb.SetInt(fn, k_ID_innerLength, offsetReduce);
            cb.SetInt(fn, k_ID_reduceLength, X.shape[axis]);
            cb.ScheduleXO(fn, Pin(X), Pin(argMax, clearOnInit: false), reduceOpShape.length);
        }
        // one hot from argmax
        {
            var fn = ComputeFuncSingleton.Instance.Get("HardmaxEnd");
            cb.SetInt(fn, k_ID_innerLength, offsetReduce);
            cb.SetInt(fn, k_ID_reduceLength, X.shape[axis]);
            cb.ScheduleXBO(fn, Pin(X), Pin(argMax), Pin(O, clearOnInit: false), O.shape.length);
        }
    }

    /// <inheritdoc/>
    public override void CumSum(TensorFloat X, TensorFloat O, int axis, bool reverse, bool exclusive)
    {
        var reduceOpShape = X.shape.Reduce(axis);
        var offsetReduce = X.shape.Strides(axis);

        var fn = ComputeFuncSingleton.Instance.Get(reverse ? (exclusive ? "CumSumFloatReverseExclusive" : "CumSumFloatReverseInclusive") : (exclusive ? "CumSumFloatForwardExclusive" : "CumSumFloatForwardInclusive"));
        cb.SetInt(fn, k_ID_innerLength, offsetReduce);
        cb.SetInt(fn, k_ID_reduceLength, X.shape[axis]);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), reduceOpShape.length);
    }

    /// <inheritdoc/>
    public override void CumSum(TensorInt X, TensorInt O, int axis, bool reverse, bool exclusive)
    {
        var reduceOpShape = X.shape.Reduce(axis);
        var offsetReduce = X.shape.Strides(axis);

        var fn = ComputeFuncSingleton.Instance.Get(reverse ? (exclusive ? "CumSumIntReverseExclusive" : "CumSumIntReverseInclusive") : (exclusive ? "CumSumIntForwardExclusive" : "CumSumIntForwardInclusive"));
        cb.SetInt(fn, k_ID_innerLength, offsetReduce);
        cb.SetInt(fn, k_ID_reduceLength, X.shape[axis]);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), reduceOpShape.length);
    }

    /// <inheritdoc/>
    public override void Einsum(TensorFloat[] inputTensors, TensorFloat O, TensorIndex[] operandIndices, TensorIndex outputIndices, TensorIndex sumIndices, TensorShape sumShape)
    {
        switch (inputTensors.Length)
        {
            case 1:
            {
                var fn = ComputeFuncSingleton.Instance.Get("EinsumOne");

                unsafe
                {
                    var outStridesA = stackalloc int[TensorShape.maxRank];
                    var sumStridesA = stackalloc int[TensorShape.maxRank];
                    EinsumHelper.PinOperandStrides(inputTensors[0].shape, operandIndices[0], outputIndices, sumIndices, outStridesA, sumStridesA);
                    cb.SetInt8(fn, k_ID_outStridesA, outStridesA);
                    cb.SetInt8(fn, k_ID_sumStridesA, sumStridesA);

                    cb.SetTensorShapeStrides(fn, k_ID_outLengths, k_ID_outStrides, O.shape);
                    cb.SetTensorShapeStrides(fn, k_ID_sumLengths, k_ID_sumStrides, sumShape);
                }

                cb.SetInt(fn, k_ID_sumSize, sumShape.length);
                cb.SetInt(fn, k_ID_sumRank, sumShape.rank);
                cb.SetInt(fn, k_ID_outRank, O.shape.rank);

                cb.ScheduleXO(fn, Pin(inputTensors[0]), Pin(O, clearOnInit: false), O.shape.length);
                return;
            }
            case 2:
            {
                var fn = ComputeFuncSingleton.Instance.Get("EinsumTwo");

                unsafe
                {
                    var outStridesA = stackalloc int[TensorShape.maxRank];
                    var sumStridesA = stackalloc int[TensorShape.maxRank];
                    EinsumHelper.PinOperandStrides(inputTensors[0].shape, operandIndices[0], outputIndices, sumIndices, outStridesA, sumStridesA);
                    cb.SetInt8(fn, k_ID_outStridesA, outStridesA);
                    cb.SetInt8(fn, k_ID_sumStridesA, sumStridesA);

                    var outStridesB = stackalloc int[TensorShape.maxRank];
                    var sumStridesB = stackalloc int[TensorShape.maxRank];
                    EinsumHelper.PinOperandStrides(inputTensors[1].shape, operandIndices[1], outputIndices, sumIndices, outStridesB, sumStridesB);
                    cb.SetInt8(fn, k_ID_outStridesB, outStridesB);
                    cb.SetInt8(fn, k_ID_sumStridesB, sumStridesB);

                    cb.SetTensorShapeStrides(fn, k_ID_outLengths, k_ID_outStrides, O.shape);
                    cb.SetTensorShapeStrides(fn, k_ID_sumLengths, k_ID_sumStrides, sumShape);
                }

                cb.SetInt(fn, k_ID_sumSize, sumShape.length);
                cb.SetInt(fn, k_ID_sumRank, sumShape.rank);
                cb.SetInt(fn, k_ID_outRank, O.shape.rank);

                cb.ScheduleXBO(fn, Pin(inputTensors[0]), Pin(inputTensors[1]), Pin(O, clearOnInit: false), O.shape.length);
                return;
            }
        }
    }

    /// <inheritdoc/>
    public override void Concat(Tensor[] inputs, Tensor O, int axis)
    {
        unsafe
        {
            // product of all tensor dimensions starting from axis
            var copyBlockLengths = stackalloc int[inputs.Length];
            var copyBlockLengthsAcum = stackalloc int[inputs.Length];
            int copyBlockLengthsSum = 0;
            for (int i = 0; i < inputs.Length; ++i)
            {
                copyBlockLengthsAcum[i] = copyBlockLengthsSum;
                copyBlockLengths[i] = inputs[i].shape.Length(axis);
                copyBlockLengthsSum += copyBlockLengths[i];
            }

            // copy tensor data interleaved into O
            int takes = O.shape.Length(0, axis);
            for (int i = 0; i < inputs.Length; ++i)
            {
                if (inputs[i].shape.HasZeroDims())
                    continue;

                MemCopyStride(inputs[i], O, copyBlockLengths[i], copyBlockLengthsSum, copyBlockLengths[i], takes, 0, copyBlockLengthsAcum[i]);
            }
        }
    }

    /// <inheritdoc/>
    public override void Slice(Tensor X, Tensor O, ReadOnlySpan<int> starts, ReadOnlySpan<int> axes, ReadOnlySpan<int> steps)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Slice");
        unsafe
        {
            cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
            cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
            var pStarts = stackalloc int[8] { 0, 0, 0, 0, 0, 0, 0, 0 };
            var pSteps = stackalloc int[8] { 1, 1, 1, 1, 1, 1, 1, 1 };

            for (int i = 0; i < starts.Length; i++)
            {
                int axis = axes != null ? X.shape.Axis(axes[i]) : i;
                int start = Math.Min(starts[i], X.shape[axis] - 1);
                start = start < 0 ? X.shape[axis] + start : start;
                int step = steps != null ? steps[i] : 1;
                pStarts[(TensorShape.maxRank - X.shape.rank) + axis] = start;
                pSteps[(TensorShape.maxRank - X.shape.rank) + axis] = step;
            }
            cb.SetInt8(fn, k_ID_starts, pStarts);
            cb.SetInt8(fn, k_ID_steps, pSteps);
        }
        cb.SetInt(fn, k_ID_rank, O.shape.rank);

        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Split(Tensor X, Tensor O, int axis, int start)
    {
        axis = X.shape.Axis(axis);

        var fn = ComputeFuncSingleton.Instance.Get("Split");
        cb.SetInt(fn, k_ID_start, start);
        cb.SetInt(fn, k_ID_lengthO, O.shape.length);
        cb.SetInt(fn, k_ID_strideLower, O.shape.Strides(axis));
        int strideUpperX = axis == 0 ? X.shape.length : X.shape.Strides(axis - 1);
        int strideUpperO = axis == 0 ? O.shape.length : O.shape.Strides(axis - 1);
        cb.SetInt(fn, k_ID_strideUpperX, strideUpperX);
        cb.SetInt(fn, k_ID_strideUpperO, strideUpperO);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape.length, 1, 1);
    }

    /// <inheritdoc/>
    public override void Pad(TensorFloat X, TensorFloat O, ReadOnlySpan<int> pad, Layers.PadMode padMode, float constant)
    {
        string padKernel;
        switch (padMode)
        {
            case Layers.PadMode.Constant:
                padKernel = "PadBorderND";
                break;
            case Layers.PadMode.Reflect:
                padKernel = "PadReflectND";
                break;
            case Layers.PadMode.Edge:
                padKernel = "PadEdgeND";
                break;
            case Layers.PadMode.Symmetric:
                padKernel = "PadSymmetricND";
                break;
            default:
                throw new NotImplementedException();
        }

        var fn = ComputeFuncSingleton.Instance.Get(padKernel);

        cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
        cb.SetInt16(fn, k_ID_pad, pad);
        cb.SetInt(fn, k_ID_rank, X.shape.rank);
        if (padMode == Layers.PadMode.Constant)
            cb.SetFloat(fn, k_ID_Beta, constant);

        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Transpose(Tensor X, Tensor O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Transpose");
        unsafe
        {
            cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);

            int* permutations = stackalloc int[TensorShape.maxRank];
            for (int i = 0; i < X.shape.rank; i++)
                permutations[i] = (X.shape.rank - 1) - i;
            cb.SetInt8(fn, k_ID_permutations, permutations);
        }
        cb.SetInt(fn, k_ID_rank, X.shape.rank);

        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), X.shape.length);
    }

    bool IsTranspose2D(TensorShape X, int[] permutations, out int Height, out int Width)
    {
        Height = 1; Width = 1;
        if (X.length == 1)
            return true;

        var O = X.Transpose(permutations);
        var squeezedO = O.Squeeze();

        unsafe
        {
            var remapDim = stackalloc int[X.rank];
            int index = 0;
            for (int i = 0; i < X.rank; i++)
            {
                if (X[i] == 1)
                    continue;
                remapDim[i] = index;
                index++;
            }

            var squeezedPermutations = stackalloc int[squeezedO.rank];
            index = 0;
            for (int i = 0; i < permutations.Length; i++)
            {
                if (O[i] == 1)
                    continue;
                squeezedPermutations[index] = remapDim[permutations[i]];
                index++;
            }

            int widthO = 1;
            int heightO = squeezedO[0];
            int prevDim = squeezedPermutations[0];
            bool switchedDim = false;

            for (int i = 1; i < squeezedO.rank; i++)
            {
                var dim = squeezedPermutations[i];
                if (!switchedDim && ((prevDim + 1) == dim))
                {
                    prevDim = dim;
                    heightO *= squeezedO[i];
                }
                else if (switchedDim && ((prevDim + 1) != dim))
                {
                    return false;
                }
                else
                {
                    switchedDim = true;
                    prevDim = dim;
                    widthO *= squeezedO[i];
                }
            }

            Height = widthO; Width = heightO;

            return true;
        }
    }

    /// <inheritdoc/>
    public override void Transpose(Tensor X, Tensor O, int[] permutations)
    {
        bool is2DTranspose = IsTranspose2D(X.shape, permutations, out int equivalentXH, out int equivalentXW);

        if (is2DTranspose)
        {
            var fn = ComputeFuncSingleton.Instance.Get("Transpose2D");
            cb.SetInt(fn, k_ID_X_width, equivalentXW);
            cb.SetInt(fn, k_ID_X_height, equivalentXH);

            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

            cb.Dispatch(fn, equivalentXW, equivalentXH, 1);
        }
        else
        {

            var fn = ComputeFuncSingleton.Instance.Get("Transpose");
            unsafe
            {
                cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
                cb.SetInt8(fn, k_ID_permutations, permutations);
            }
            cb.SetInt(fn, k_ID_rank, X.shape.rank);

            cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), X.shape.length);
        }
    }

    void ArgMaxTail(TensorFloat X, TensorInt O, int axis)
    {
        int globalNonSpatialLength = X.shape.Length(0, axis);
        int globalSpatialDims = X.shape.length / globalNonSpatialLength;

        int localSpatialLength = globalSpatialDims;

        var Oshape = new TensorShape(globalNonSpatialLength, localSpatialLength);

        TensorInt Xindices = NewTempTensorInt(X.shape); // save max(X)
        bool isFirstDispatch = true;

        // downsample with pyramid approach
        while (localSpatialLength > 64 * 4)
        {
            int spatialLengthO = ComputeHelper.IDivC(localSpatialLength, 64 * 4);
            Oshape[-1] = spatialLengthO;
            var Otemp = NewTempTensorFloat(Oshape);
            var Oindicestemp = NewTempTensorInt(Oshape);

            var fnPool = ComputeFuncSingleton.Instance.Get("ArgMaxReduce");
            cb.SetTensorAsBuffer(fnPool, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fnPool, k_ID_XIndices, Pin(Xindices));
            cb.SetTensorAsBuffer(fnPool, k_ID_Optr, Pin(Otemp, clearOnInit: false));
            cb.SetTensorAsBuffer(fnPool, k_ID_OIndices, Pin(Oindicestemp, clearOnInit: false));
            cb.SetInt(fnPool, k_ID_SpatialDims, localSpatialLength);
            cb.SetInt(fnPool, k_ID_SpatialDimsO, spatialLengthO);
            cb.SetInt(fnPool, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

            cb.Dispatch(fnPool, globalNonSpatialLength, ComputeHelper.IDivC(localSpatialLength, 4), 1);

            X = Otemp;
            Xindices = Oindicestemp;
            localSpatialLength = spatialLengthO;
            isFirstDispatch = false;
        }

        var fn = ComputeFuncSingleton.Instance.Get("GlobalArgMaxReduce");
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_XIndices, Pin(Xindices));
        cb.SetTensorAsBuffer(fn, k_ID_OIndices, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_SpatialDims, localSpatialLength);
        cb.SetInt(fn, k_ID_IsFirstDispatch, isFirstDispatch ? 1 : 0);

        cb.Dispatch(fn, globalNonSpatialLength, 1, 1);
    }

    /// <inheritdoc/>
    public override void ArgMax(TensorFloat X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        int dimAxis = X.shape[axis];
        Assert.AreNotEqual(0, dimAxis, "ValueError: zero-size array to reduction operation maximum which has no identity.");

        if (!selectLastIndex && (dimAxis == X.shape.Length(axis)))
        {
            ArgMaxTail(X, O, axis);
            return;
        }

        var fn = ComputeFuncSingleton.Instance.Get(selectLastIndex ? "ArgMaxFloatLast" : "ArgMaxFloatFirst");
        cb.SetInt(fn, k_ID_innerLength, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_reduceLength, dimAxis);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void ArgMax(TensorInt X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation maximum which has no identity.");

        var fn = ComputeFuncSingleton.Instance.Get(selectLastIndex ? "ArgMaxIntLast" : "ArgMaxIntFirst");
        cb.SetInt(fn, k_ID_innerLength, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_reduceLength, X.shape[axis]);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void ArgMin(TensorFloat X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        Assert.AreNotEqual(0, X.shape[axis], "ValueError: zero-size array to reduction operation minimum which has no identity.");

        var fn = ComputeFuncSingleton.Instance.Get(selectLastIndex ? "ArgMinFloatLast" : "ArgMinFloatFirst");
        cb.SetInt(fn, k_ID_innerLength, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_reduceLength, X.shape[axis]);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void ArgMin(TensorInt X, TensorInt O, int axis, bool keepdim, bool selectLastIndex)
    {
        var fn = ComputeFuncSingleton.Instance.Get(selectLastIndex ? "ArgMinIntLast" : "ArgMinIntFirst");
        cb.SetInt(fn, k_ID_innerLength, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_reduceLength, X.shape[axis]);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    void Compare(Tensor A, Tensor B, TensorInt O, string kernel)
    {
        var fn = ComputeFuncSingleton.Instance.Get(kernel);
        cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
        cb.SetInt(fn, k_ID_rank, O.shape.rank);

        cb.ScheduleXBO(fn, Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Greater(TensorFloat A, TensorFloat B, TensorInt O)
    {
        Compare(A, B, O, "Greater");
    }

    /// <inheritdoc/>
    public override void Greater(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "GreaterInt");
    }

    /// <inheritdoc/>
    public override void GreaterOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
    {
        Compare(A, B, O, "GreaterOrEqual");
    }

    /// <inheritdoc/>
    public override void GreaterOrEqual(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "GreaterOrEqualInt");
    }

    /// <inheritdoc/>
    public override void Less(TensorFloat A, TensorFloat B, TensorInt O)
    {
        Compare(A, B, O, "Less");
    }

    /// <inheritdoc/>
    public override void Less(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "LessInt");
    }

    /// <inheritdoc/>
    public override void LessOrEqual(TensorFloat A, TensorFloat B, TensorInt O)
    {
        Compare(A, B, O, "LessOrEqual");
    }

    /// <inheritdoc/>
    public override void LessOrEqual(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "LessOrEqualInt");
    }

    /// <inheritdoc/>
    public override void Equal(TensorFloat A, TensorFloat B, TensorInt O)
    {
        Compare(A, B, O, "Equal");
    }

    /// <inheritdoc/>
    public override void Equal(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "EqualInt");
    }

    /// <inheritdoc/>
    public override void Or(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "Or");
    }

    /// <inheritdoc/>
    public override void And(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "And");
    }

    /// <inheritdoc/>
    public override void Xor(TensorInt A, TensorInt B, TensorInt O)
    {
        Compare(A, B, O, "Xor");
    }

    /// <inheritdoc/>
    public override void Not(TensorInt X, TensorInt O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Not");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void HardSwish(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("HardSwish");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Sign(TensorFloat X, TensorFloat O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("SignFloat");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Sign(TensorInt X, TensorInt O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("SignInt");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void IsInf(TensorFloat X, TensorInt O, bool detectNegative, bool detectPositive)
    {
        var fn = ComputeFuncSingleton.Instance.Get("IsInf");
        cb.SetBool(fn, k_ID_detectNegative, detectNegative);
        cb.SetBool(fn, k_ID_detectPositive, detectPositive);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void IsNaN(TensorFloat X, TensorInt O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("IsNaN");
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Where(TensorInt C, Tensor A, Tensor B, Tensor O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Where");
        cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeC, k_ID_stridesC, C.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeA, k_ID_stridesA, A.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeB, k_ID_stridesB, B.shape);
        cb.SetInt(fn, k_ID_rank, O.shape.rank);
        cb.ScheduleXSBO(fn, Pin(C), Pin(A), Pin(B), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Tile(Tensor X, Tensor O, ReadOnlySpan<int> repeats)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Tile");
        cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
        cb.SetInt(fn, k_ID_rank, O.shape.rank);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void MemClear(Tensor O)
    {
        var length = O.shape.length;
        var numWords = ComputeHelper.IDivC(length, 4);
        var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeFunc.SafeDispatchLimit * 32 * 8);
        var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

        var fn = ComputeFuncSingleton.Instance.Get("MemSet");
        cb.SetFloat(fn, k_ID_memValueFloat, 0);
        cb.SetInt(fn, k_ID_offsetO, 0);
        cb.SetInt(fn, k_ID_count, length);
        cb.SetInt(fn, k_ID_O_width, wordsWidth * 4);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, wordsWidth, wordsHeight, 1);
    }

    /// <inheritdoc/>
    public override void MemSet(TensorFloat O, float value)
    {
        var length = O.shape.length;
        var numWords = ComputeHelper.IDivC(length, 4);
        var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeFunc.SafeDispatchLimit * 32 * 8);
        var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

        var fn = ComputeFuncSingleton.Instance.Get("MemSet");
        cb.SetFloat(fn, k_ID_memValueFloat, value);
        cb.SetInt(fn, k_ID_offsetO, 0);
        cb.SetInt(fn, k_ID_count, length);
        cb.SetInt(fn, k_ID_O_width, wordsWidth * 4);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, wordsWidth, wordsHeight, 1);
    }

    /// <inheritdoc/>
    public override void MemSet(TensorInt O, int value)
    {
        var length = O.shape.length;
        var numWords = ComputeHelper.IDivC(length, 4);
        var wordsHeight = ComputeHelper.IDivC(numWords, (int)ComputeFunc.SafeDispatchLimit * 32 * 8);
        var wordsWidth = ComputeHelper.IDivC(numWords, wordsHeight);

        var fn = ComputeFuncSingleton.Instance.Get("MemSet");
        cb.SetFloat(fn, k_ID_memValueFloat, math.asfloat(value));
        cb.SetInt(fn, k_ID_offsetO, 0);
        cb.SetInt(fn, k_ID_count, length);
        cb.SetInt(fn, k_ID_O_width, wordsWidth * 4);
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, wordsWidth, wordsHeight, 1);
    }

    /// <inheritdoc/>
    public override void Expand(Tensor X, Tensor O)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Expand");
        cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
        cb.SetInt(fn, k_ID_rank, O.shape.rank);
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void CompressWithIndices(Tensor X, TensorInt indices, Tensor O, int numIndices, int axis)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Gather");
        cb.SetInt(fn, k_ID_endLength, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_indicesLength, numIndices);
        cb.SetInt(fn, k_ID_axisDim, X.shape[axis]);
        cb.ScheduleXBO(fn, Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Gather(Tensor X, TensorInt indices, Tensor O, int axis)
    {
        var fn = ComputeFuncSingleton.Instance.Get("Gather");
        cb.SetInt(fn, k_ID_endLength, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_indicesLength, indices.shape.length);
        cb.SetInt(fn, k_ID_axisDim, X.shape[axis]);
        cb.ScheduleXBO(fn, Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void GatherElements(Tensor X, TensorInt indices, Tensor O, int axis)
    {
        var fn = ComputeFuncSingleton.Instance.Get("GatherElements");
        cb.SetInt(fn, k_ID_endLength, O.shape.Strides(axis));
        cb.SetInt(fn, k_ID_endLengthX, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_axisDim, O.shape[axis]);
        cb.SetInt(fn, k_ID_axisDimX, X.shape[axis]);
        cb.ScheduleXBO(fn, Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void GatherND(Tensor X, TensorInt indices, Tensor O, int batchDims)
    {
        var fn = ComputeFuncSingleton.Instance.Get("GatherND");
        cb.SetInt(fn, k_ID_rankX, X.shape.rank);
        cb.SetInt(fn, k_ID_rankO, O.shape.rank);
        cb.SetInt(fn, k_ID_rankIndices, indices.shape.rank);
        cb.SetInt(fn, k_ID_iStart, TensorShape.maxRank - O.shape.rank);
        cb.SetInt(fn, k_ID_iEndIndices, TensorShape.maxRank - O.shape.rank + indices.shape.rank - 1);
        cb.SetInt(fn, k_ID_iEndX, TensorShape.maxRank - O.shape.rank + batchDims);
        cb.SetInt(fn, k_ID_iEndMin, TensorShape.maxRank - O.shape.rank + Math.Min(batchDims, indices.shape.rank - 1));
        cb.SetInt(fn, k_ID_iStartB, TensorShape.maxRank - X.shape.rank + batchDims);
        cb.SetInt(fn, k_ID_iEndB, TensorShape.maxRank - X.shape.rank + batchDims + indices.shape[-1]);
        cb.SetTensorShapeStrides(fn, k_ID_shapeO, k_ID_stridesO, O.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeX, k_ID_stridesX, X.shape);
        cb.SetTensorShapeStrides(fn, k_ID_shapeIndices, k_ID_stridesIndices, indices.shape);
        cb.ScheduleXBO(fn, Pin(X), Pin(indices), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void ScatterElements(Tensor X, TensorInt indices, Tensor updates, Tensor O, int axis, Layers.ScatterReductionMode reduction)
    {
        // TODO: The ONNX definition for ScatterElements allows duplicate indices when using the
        // reduction modes, but allowing this introduces race conditions for updating the output
        // tensor. As the current use cases for ScatterElements do not use reductions, fallback
        // to the single-threaded burst cpu implementation.
        if (reduction != Layers.ScatterReductionMode.None)
        {
            base.ScatterElements(X, indices, updates, O, axis, reduction);
            return;
        }
        MemCopy(X, O);

        var fn = ComputeFuncSingleton.Instance.Get("ScatterElements");
        cb.SetInt(fn, k_ID_endLength, X.shape.Strides(axis));
        cb.SetInt(fn, k_ID_axisDim, X.shape[axis]);
        cb.SetInt(fn, k_ID_axisDimIndices, indices.shape[axis]);
        cb.SetInt(fn, k_ID_reduction, (int)reduction);
        cb.ScheduleXBO(fn, Pin(updates), Pin(indices), Pin(O, clearOnInit: false), indices.shape.length);
    }

    /// <inheritdoc/>
    public override void ScatterND(TensorFloat X, TensorInt indices, TensorFloat updates, TensorFloat O, Layers.ScatterReductionMode reduction)
    {
        MemCopy(X, O);
        int indexRemapDim = indices.shape[-1];
        int indicesLength = indices.shape.Length(0, -1);
        int updatesLength = updates.shape.length / indicesLength;

        var fn = ComputeFuncSingleton.Instance.Get("ScatterNDFloat");
        cb.SetInt(fn, k_ID_updatesLength, updatesLength);
        cb.SetInt(fn, k_ID_indicesLength, indicesLength);
        cb.SetInt(fn, k_ID_indexRemapDim, indexRemapDim);
        cb.SetInt(fn, k_ID_reduction, (int)reduction);
        unsafe
        {
            var trailing = stackalloc int[8];
            int trailingDim = 1;
            for (int j = (indexRemapDim - 1); j >= 0; j--)
            {
                trailing[j] = trailingDim;
                trailingDim *= X.shape[j];
            }
            cb.SetInt8(fn, k_ID_trailing, trailing);
        }
        cb.SetTensorAsBuffer(fn, k_ID_Iptr, Pin(indices));
        cb.SetTensorAsBuffer(fn, k_ID_Uptr, Pin(updates));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.Dispatch(fn, updatesLength, indicesLength, 1);
    }

    /// <inheritdoc/>
    public override void ScatterND(TensorInt X, TensorInt indices, TensorInt updates, TensorInt O, Layers.ScatterReductionMode reduction)
    {
        MemCopy(X, O);

        int indexRemapDim = indices.shape[-1];
        int indicesLength = indices.shape.Length(0, -1);
        int updatesLength = updates.shape.length / indicesLength;

        var fn = ComputeFuncSingleton.Instance.Get("ScatterNDInt");
        cb.SetInt(fn, k_ID_updatesLength, updatesLength);
        cb.SetInt(fn, k_ID_indicesLength, indicesLength);
        cb.SetInt(fn, k_ID_indexRemapDim, indexRemapDim);
        cb.SetInt(fn, k_ID_reduction, (int)reduction);
        unsafe
        {
            var trailing = stackalloc int[8];
            int trailingDim = 1;
            for (int j = indexRemapDim - 1; j >= 0; j--)
            {
                trailing[j] = trailingDim;
                trailingDim *= X.shape[j];
            }
            cb.SetInt8(fn, k_ID_trailing, trailing);
        }
        cb.SetTensorAsBuffer(fn, k_ID_Iptr, Pin(indices));
        cb.SetTensorAsBuffer(fn, k_ID_UIntptr, Pin(updates));
        cb.SetTensorAsBuffer(fn, k_ID_OIntptr, Pin(O, clearOnInit: false));
        cb.Dispatch(fn, updatesLength, indicesLength, 1);
    }

    /// <inheritdoc/>
    public override void OneHot(TensorInt X, TensorInt O, int axis, int depth, int offValue, int onValue)
    {
        axis = O.shape.Axis(axis);

        var fn = ComputeFuncSingleton.Instance.Get("OneHot");
        cb.SetInt(fn, k_ID_depth, depth);
        cb.SetInt(fn, k_ID_offValue, offValue);
        cb.SetInt(fn, k_ID_onValue, onValue);
        cb.SetInt(fn, k_ID_rankO, O.shape.rank);

        cb.SetInt(fn, k_ID_stridesToAxis, O.shape.Strides(axis));
        cb.SetInt(fn, k_ID_axisDim, O.shape[axis]);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape.length, 1, 1);
    }

    /// <inheritdoc/>
    public override void OneHot(TensorInt X, TensorFloat O, int axis, int depth, float offValue, float onValue)
    {
        axis = O.shape.Axis(axis);

        var fn = new ComputeFunc("OneHot");
        cb.SetInt(fn, k_ID_depth, depth);
        cb.SetInt(fn, k_ID_offValue, math.asint(offValue));
        cb.SetInt(fn, k_ID_onValue, math.asint(onValue));
        cb.SetInt(fn, k_ID_rankO, O.shape.rank);

        cb.SetInt(fn, k_ID_stridesToAxis, O.shape.Strides(axis));
        cb.SetInt(fn, k_ID_axisDim, O.shape[axis]);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape.length, 1, 1);
    }

    /// <inheritdoc/>
    public override void TopK(TensorFloat X, TensorFloat values, TensorInt indices, int k, int axis, bool largest)
    {
        int reduceLength = X.shape[axis];
        int innerLength = X.shape.Strides(axis);
        int outerLength = X.shape.length / (reduceLength * innerLength);

        var fn = ComputeFuncSingleton.Instance.Get(largest ? "TopKLargest" : "TopKSmallest");
        cb.SetInt(fn, k_ID_innerLength, innerLength);
        cb.SetInt(fn, k_ID_outerLength, outerLength);
        cb.SetInt(fn, k_ID_reduceLength, reduceLength);
        cb.SetInt(fn, k_ID_maxK, k);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Valuesptr, Pin(values, clearOnInit: false));
        cb.SetTensorAsBuffer(fn, k_ID_Indicesptr, Pin(indices, clearOnInit: false));
        cb.Dispatch(fn, innerLength, outerLength, 1);
    }

    /// <inheritdoc/>
    public override void RoiAlign(TensorFloat X, TensorFloat rois, TensorInt indices, TensorFloat O, Layers.RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    {
        var fn = ComputeFuncSingleton.Instance.Get(mode == Layers.RoiPoolingMode.Avg ? "RoiAlignAvg" : "RoiAlignMax");
        cb.SetInt(fn, k_ID_numRois, rois.shape[0]);
        cb.SetInt(fn, k_ID_inputChannels, X.shape[1]);
        cb.SetInt(fn, k_ID_inputHeight, X.shape[2]);
        cb.SetInt(fn, k_ID_inputWidth, X.shape[3]);
        cb.SetInt(fn, k_ID_inputSpatialSize, X.shape[2] * X.shape[3]);
        cb.SetInt(fn, k_ID_inputBatchOffset, X.shape[1] * X.shape[2] * X.shape[3]);
        cb.SetInt(fn, k_ID_outputHeight, outputHeight);
        cb.SetInt(fn, k_ID_outputWidth, outputWidth);
        cb.SetInt(fn, k_ID_outputSpatialSize, outputHeight * outputWidth);
        cb.SetFloat(fn, k_ID_normalizeOHeight, 1.0f / outputHeight);
        cb.SetFloat(fn, k_ID_normalizeOWidth, 1.0f / outputWidth);
        cb.SetInt(fn, k_ID_samplingRatio, samplingRatio);
        cb.SetFloat(fn, k_ID_spatialScale, spatialScale);

        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Sptr, Pin(rois));
        cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(indices));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape[0] * O.shape[1], O.shape[2] * O.shape[3], 1);
    }

    /// <inheritdoc/>
    public override void RandomNormal(TensorFloat O, float mean, float scale, float? seed)
    {
        var fn = ComputeFuncSingleton.Instance.Get("RandomNormal");
        cb.SetInt(fn, k_ID_lengthO, O.shape.length);
        cb.SetInt(fn, k_ID_seed, (int)Random.GetOpSeed(seed));
        cb.SetFloat(fn, k_ID_mean, mean);
        cb.SetFloat(fn, k_ID_scale, scale);

        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape.length, 1, 1);
    }

    /// <inheritdoc/>
    public override void RandomUniform(TensorFloat O, float low, float high, float? seed)
    {
        var fn = ComputeFuncSingleton.Instance.Get("RandomUniform");
        cb.SetInt(fn, k_ID_lengthO, O.shape.length);
        cb.SetInt(fn, k_ID_seed, (int)Random.GetOpSeed(seed));
        cb.SetFloat(fn, k_ID_low, low);
        cb.SetFloat(fn, k_ID_high, high);

        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

        cb.Dispatch(fn, O.shape.length, 1, 1);
    }

    /// <inheritdoc/>
    public override void Bernoulli(TensorFloat X, Tensor O, float? seed)
    {
        var fn = ComputeFuncSingleton.Instance.Get(O.dataType == DataType.Float ? "BernoulliFloat" : "BernoulliInt");
        cb.SetInt(fn, k_ID_lengthO, O.shape.length);
        cb.SetInt(fn, k_ID_seed, (int)Random.GetOpSeed(seed));
        cb.ScheduleXO(fn, Pin(X), Pin(O, clearOnInit: false), O.shape.length);
    }

    /// <inheritdoc/>
    public override void Cast(Tensor X, Tensor O)
    {
        if (X.dataType == O.dataType)
        {
            MemCopy(X, O);
            return;
        }

        ComputeFunc fn;
        if (O.dataType == DataType.Float)
        {
            fn = ComputeFuncSingleton.Instance.Get("CastToFloat");
            cb.SetTensorAsBuffer(fn, k_ID_XIntptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        }
        else
        {
            fn = ComputeFuncSingleton.Instance.Get("CastToInt");
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_OIntptr, Pin(O, clearOnInit: false));
        }
        cb.SetInt(fn, k_ID_X_length, X.shape.length);

        cb.Dispatch(fn, ComputeHelper.IDivC(X.shape.length, 4), 1, 1);
    }

    /// <inheritdoc/>
    public override void MemCopy(Tensor X, Tensor O)
    {
        var length = O.shape.length;
        var fn = ComputeFuncSingleton.Instance.Get("MemCopy");
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_offsetX, 0);
        cb.SetInt(fn, k_ID_offsetO, 0);
        cb.SetInt(fn, k_ID_count, length);
        cb.Dispatch(fn, ComputeHelper.IDivC(length, 4), 1, 1);
    }

    /// <inheritdoc/>
    public override void MemCopyStride(Tensor X, Tensor O, int strideX, int strideO, int length, int count, int offsetX, int offsetO)
    {
        if (length == 0 || count == 0)
            return;
        Logger.AssertIsTrue(length > 0, "MemCopy.InputError: copy stride length must be greater than 0");
        Logger.AssertIsTrue(count > 0, "MemCopy.InputError: copy stride count must be greater than 0");
        Logger.AssertIsTrue(offsetX >= 0, "MemCopy.BoundsError: copy stride out of bounds for tensor X");
        Logger.AssertIsTrue(offsetX + (count - 1) * strideX + length <= X.shape.length, "MemCopy.BoundsError: copy stride out of bounds for tensor X");
        Logger.AssertIsTrue(offsetO >= 0, "MemCopy.BoundsError: copy stride out of bounds for tensor O");
        Logger.AssertIsTrue(offsetO + (count - 1) * strideO + length <= O.shape.length, "MemCopy.BoundsError: copy stride out of bounds for tensor O");
        var fn = ComputeFuncSingleton.Instance.Get("MemCopyStride");
        var copyLength = count * length;
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_strideX, strideX);
        cb.SetInt(fn, k_ID_strideO, strideO);
        cb.SetInt(fn, k_ID_offsetX, offsetX);
        cb.SetInt(fn, k_ID_offsetO, offsetO);
        cb.SetInt(fn, k_ID_elementSize, length);
        cb.SetInt(fn, k_ID_count, copyLength);
        cb.Dispatch(fn, ComputeHelper.IDivC(copyLength, 4), 1, 1);
    }

    void Gemm(TensorFloat X, TensorFloat Y, TensorFloat B, TensorFloat O, int M, int K, int N)
    {
        int workItemsX, workItemsY;
        string kernel;
        if (M == 1)
        {
            kernel = "Dense_V_L1Cached64";
            workItemsX = ComputeHelper.IDivC(N, 4);
            workItemsY = 1;
        }
        else if (N % 64 == 0 && K % 16 == 0)
        {
            kernel = "Dense_T16x16_R4x4";
            workItemsX = ComputeHelper.IDivC(N, 4);
            workItemsY = ComputeHelper.IDivC(M, 4);
        }
        else
        {
            kernel = "Dense_T8x8_R4x4";
            workItemsX = ComputeHelper.IDivC(N, 4);
            workItemsY = ComputeHelper.IDivC(M, 4);
        }

        var fn = ComputeFuncSingleton.Instance.Get(kernel);

        cb.SetInt(fn, k_ID_X_width, K);
        cb.SetInt(fn, k_ID_W_width, N);
        cb.SetInt(fn, k_ID_O_height, M);
        cb.SetInt(fn, k_ID_O_width, N);
        cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
        cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(Y));
        cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));
        cb.SetInt(fn, k_ID_maxXIndex, M * K - 1);
        cb.SetInt(fn, k_ID_maxWIndex, K * N - 1);

        cb.SetTensorAsBuffer(fn, k_ID_Bptr, Pin(B));
        cb.SetInt(fn, k_ID_maxBIndex, N - 1);

        cb.Dispatch(fn, workItemsX, workItemsY, 1);
    }

    void Gemm(TensorFloat X, TensorFloat Y, TensorFloat O, int M, int K, int N, bool transposeA = false, bool transposeB = false)
    {
        if (transposeA || transposeB)
        {
            string kernel;

            if (transposeA)
                kernel = transposeB ? "GemmT_XT_WT_T8x8_R4x4" : "GemmT_XT_T8x8_R4x4";
            else
                kernel = "GemmT_WT_T8x8_R4x4";

            var fn = ComputeFuncSingleton.Instance.Get(kernel);

            cb.SetInt(fn, k_ID_M, M);
            cb.SetInt(fn, k_ID_N, N);
            cb.SetInt(fn, k_ID_K, K);
            cb.SetInt(fn, k_ID_maxXIndex, M * K - 1);
            cb.SetInt(fn, k_ID_maxWIndex, K * N - 1);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(Y));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

            cb.Dispatch(fn, ComputeHelper.IDivC(N, 4), ComputeHelper.IDivC(M, 4), 1);
        }
        else
        {
            int workItemsX, workItemsY, workItemsZ;
            string kernel;

            if (M == 1)
            {
                kernel = "Gemm_V_L1Cached64";
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = 1;
                workItemsZ = 1;
            }
            else if (N % 64 == 0 && K % 16 == 0)
            {
                kernel = "Gemm_T16x16_R4x4";
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
                workItemsZ = 1;
            }
            else
            {
                kernel = "Gemm_T8x8_R4x4";
                workItemsX = ComputeHelper.IDivC(N, 4);
                workItemsY = ComputeHelper.IDivC(M, 4);
                workItemsZ = 1;
            }

            var fn = ComputeFuncSingleton.Instance.Get(kernel);

            cb.SetInt(fn, k_ID_X_width, K);
            cb.SetInt(fn, k_ID_W_width, N);
            cb.SetInt(fn, k_ID_O_width, N);
            cb.SetInt(fn, k_ID_O_height, M);
            cb.SetInt(fn, k_ID_maxXIndex, M * K - 1);
            cb.SetInt(fn, k_ID_maxWIndex, K * N - 1);
            cb.SetTensorAsBuffer(fn, k_ID_Xptr, Pin(X));
            cb.SetTensorAsBuffer(fn, k_ID_Wptr, Pin(Y));
            cb.SetTensorAsBuffer(fn, k_ID_Optr, Pin(O, clearOnInit: false));

            cb.Dispatch(fn, workItemsX, workItemsY, workItemsZ);
        }
    }

    /// <inheritdoc/>
    protected override void SinglePassLSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat P, TensorFloat Y, TensorFloat Y_h, TensorFloat Y_c, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, bool isReverse, int dirIndex, Layers.RnnLayout layout)
    {
        var numDirections = B.shape[0];
        var inputSize = X.shape[2];
        var hiddenSize = R.shape[2];

        var seqLength = X.shape[0];
        var batchSize = X.shape[1];

        var xStrideSeq = batchSize * 4 * hiddenSize;
        var xStrideBatch = 4 * hiddenSize;

        var yStrideDir = batchSize * hiddenSize;
        var yStrideSeq = numDirections * batchSize * hiddenSize;
        var yStrideBatch = hiddenSize;

        if (layout == Layers.RnnLayout.BatchFirst)
        {
            seqLength = X.shape[1];
            batchSize = X.shape[0];

            xStrideSeq = 4 * hiddenSize;
            xStrideBatch = seqLength * 4 * hiddenSize;

            yStrideDir = hiddenSize;
            yStrideSeq = numDirections * hiddenSize;
            yStrideBatch = seqLength * numDirections * hiddenSize;
        }

        var HtxRT = NewTempTensorFloat(new TensorShape(batchSize * 4 * hiddenSize));
        var XsixWT = NewTempTensorFloat(new TensorShape(seqLength * batchSize * 4 * hiddenSize));

        Gemm(X, W, XsixWT, seqLength * batchSize, inputSize, 4 * hiddenSize, transposeB: true);

        var endFn = ComputeFuncSingleton.Instance.Get("LSTMEnd");
        cb.SetInt(endFn, k_ID_hiddenSize, hiddenSize);
        cb.SetInt(endFn, k_ID_batchSize, batchSize);
        cb.SetInt(endFn, k_ID_xStride, xStrideBatch);
        cb.SetInt(endFn, k_ID_yStride, yStrideBatch);
        cb.SetBool(endFn, k_ID_inputForget, inputForget);
        cb.SetFloat(endFn, k_ID_clipValue, clip);
        cb.SetInt(endFn, k_ID_fActivation, (int)activations[3 * dirIndex + 0]);
        cb.SetFloat(endFn, k_ID_fAlpha, activationAlpha[3 * dirIndex + 0]);
        cb.SetFloat(endFn, k_ID_fBeta, activationAlpha[3 * dirIndex + 0]);
        cb.SetInt(endFn, k_ID_gActivation, (int)activations[3 * dirIndex + 1]);
        cb.SetFloat(endFn, k_ID_gAlpha, activationAlpha[3 * dirIndex + 1]);
        cb.SetFloat(endFn, k_ID_gBeta, activationAlpha[3 * dirIndex + 1]);
        cb.SetInt(endFn, k_ID_hActivation, (int)activations[3 * dirIndex + 2]);
        cb.SetFloat(endFn, k_ID_hAlpha, activationAlpha[3 * dirIndex + 2]);
        cb.SetFloat(endFn, k_ID_hBeta, activationAlpha[3 * dirIndex + 2]);
        cb.SetTensorAsBuffer(endFn, k_ID_Yptr, Pin(Y, clearOnInit: false));
        cb.SetTensorAsBuffer(endFn, k_ID_YHptr, Pin(Y_h, clearOnInit: false));
        cb.SetTensorAsBuffer(endFn, k_ID_YCptr, Pin(Y_c, clearOnInit: false));
        cb.SetTensorAsBuffer(endFn, k_ID_Bptr, Pin(B));
        cb.SetInt(endFn, k_ID_bOffset, dirIndex * 8 * hiddenSize);
        cb.SetTensorAsBuffer(endFn, k_ID_Pptr, Pin(P));
        cb.SetInt(endFn, k_ID_pOffset, dirIndex * 3 * hiddenSize);
        cb.SetTensorAsBuffer(endFn, k_ID_XsixWTptr, Pin(XsixWT, clearOnInit: false));
        cb.SetTensorAsBuffer(endFn, k_ID_HtxRTptr, Pin(HtxRT, clearOnInit: false));
        cb.SetTensorAsBuffer(endFn, k_ID_SequenceLensptr, Pin(sequenceLens));

        for (var i = 0; i < seqLength; i++)
        {
            var seqIndex = isReverse ? seqLength - 1 - i : i;

            Gemm(Y_h, R, HtxRT, batchSize, hiddenSize, 4 * hiddenSize, transposeB: true);

            cb.SetInt(endFn, k_ID_seqIndex, seqIndex);
            cb.SetInt(endFn, k_ID_yOffset, dirIndex * yStrideDir + seqIndex * yStrideSeq);
            cb.SetInt(endFn, k_ID_xOffset, seqIndex * xStrideSeq);
            cb.Dispatch(endFn, batchSize, hiddenSize, 1);
        }
    }

    /// <summary>
    /// Sets final output tensor for W, R, initialH and initialC from provided input tensors
    /// if no input is provided the tensor is cleared to 0 as a default
    /// otherwise if the input tensor can be used directly in the calculation this will early out
    /// </summary>
    void SetRnnInput(TensorFloat X, TensorFloat O, int index, int count, int length, int strideX)
    {
        if (X == O)
            return;
        if (X == null)
            MemClear(O);
        else
            MemCopyStride(X, O, strideX, length, length, count, index * length, 0);
    }

    /// <summary>
    /// Sets intermediate input tensors for Y_h and Y_c from intermediate output tensor
    /// if the calculation is single direction and sequenceFirst layout then the output
    /// tensor will be used directly and this command early outs
    /// </summary>
    void SetRnnOutput(TensorFloat X, TensorFloat O, int index, int count, int length, int strideO)
    {
        if (X == O)
            return;
        MemCopyStride(X, O, length, strideO, length, count, 0, index * length);
    }

    /// <inheritdoc/>
    public override void LSTM(TensorFloat X, TensorFloat W, TensorFloat R, TensorFloat B, TensorInt sequenceLens, TensorFloat initialH, TensorFloat initialC, TensorFloat P, TensorFloat Y, TensorFloat Yh, TensorFloat Yc, Layers.RnnDirection direction, Layers.RnnActivation[] activations, float[] activationAlpha, float[] activationBeta, bool inputForget, float clip, Layers.RnnLayout layout)
    {
        var seqLength = X.shape[layout == Layers.RnnLayout.SequenceFirst ? 0 : 1];
        var batchSize = X.shape[layout == Layers.RnnLayout.SequenceFirst ? 1 : 0];
        var inputSize = X.shape[2];
        var hiddenSize = R.shape[2];
        var numDirections = W.shape[0];

        var W1 = numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, 4 * hiddenSize, inputSize)) : W;
        var R1 = numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, 4 * hiddenSize, hiddenSize)) : R;

        var Bi = B;
        if (Bi == null)
        {
            Bi = NewTempTensorFloat(new TensorShape(numDirections, 8 * hiddenSize));
            MemClear(Bi);
        }
        var sequenceLensi = sequenceLens;
        if (sequenceLensi == null)
        {
            sequenceLensi = NewTempTensorInt(new TensorShape(batchSize));
            MemSet(sequenceLensi, math.asint(seqLength));
        }
        var Pi = P;
        if (Pi == null)
        {
            Pi = NewTempTensorFloat(new TensorShape(numDirections, 3 * hiddenSize));
            MemClear(Pi);
        }

        var Y_h1 = layout == Layers.RnnLayout.SequenceFirst ? (numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, batchSize, hiddenSize)) : Yh) : NewTempTensorFloat(new TensorShape(batchSize, 1, hiddenSize));
        var Y_c1 = layout == Layers.RnnLayout.SequenceFirst ? (numDirections == 2 ? NewTempTensorFloat(new TensorShape(1, batchSize, hiddenSize)) : Yc) : NewTempTensorFloat(new TensorShape(batchSize, 1, hiddenSize));

        var Y_hcLower = layout == Layers.RnnLayout.SequenceFirst ? batchSize * hiddenSize : hiddenSize;
        var Y_hcUpper = layout == Layers.RnnLayout.SequenceFirst ? 1 : batchSize;

        for (var i = 0; i < numDirections; i++)
        {
            SetRnnInput(W, W1, i, 1, 4 * hiddenSize * inputSize, 0);
            SetRnnInput(R, R1, i, 1, 4 * hiddenSize * hiddenSize, 0);
            SetRnnInput(initialH, Y_h1, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
            SetRnnInput(initialC, Y_c1, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
            var isReverse = direction == Layers.RnnDirection.Reverse || (direction == Layers.RnnDirection.Bidirectional && i == 1);
            SinglePassLSTM(X, W1, R1, Bi, sequenceLensi, Pi, Y, Y_h1, Y_c1, activations, activationAlpha, activationBeta, inputForget, clip, isReverse, i, layout);
            SetRnnOutput(Y_h1, Yh, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
            SetRnnOutput(Y_c1, Yc, i, Y_hcUpper, Y_hcLower, numDirections * Y_hcLower);
        }
    }

    /// <inheritdoc/>
    public override void Reshape(Tensor X, Tensor O)
    {
        MemCopy(X, O);
    }

    /// <inheritdoc/>
    public override Tensor PinToDevice(Tensor X, bool clearOnInit = true)
    {
        Pin(X, clearOnInit);
        return X;
    }
}
} // namespace Unity.Sentis
