using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using static Unity.Sentis.ShaderPropertyID;

namespace Unity.Sentis
{
    /// <summary>
    /// Provides methods for converting between textures and tensors.
    /// </summary>
    public static class TextureConverter
    {
        /// <summary>
        /// Converts a texture to a `TensorFloat`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="width">The width of the output tensor. If the value is -1, Sentis uses the texture to infer the width.</param>
        /// <param name="height">The height of the output tensor. If the value is -1, Sentis uses the texture to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output tensor. If the value is -1, Sentis uses the texture to infer the number of channels.</param>
        /// <returns>The converted tensor.</returns>
        public static TensorFloat ToTensor(Texture texture, int width = -1, int height = -1, int channels = -1)
        {
            return ToTensor(texture, new TextureTransform().SetDimensions(width, height, channels));
        }

        /// <summary>
        /// Converts a texture to a `TensorFloat`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The converted tensor.</returns>
        public static TensorFloat ToTensor(Texture texture, TextureTransform transform)
        {
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            transform.InferDimensions(texture.width, texture.height, textureChannels);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var shape = transform.GetTensorShape();

            var O = new TensorFloat(shape);
            ToTensor(texture, O, transform);
            return O;
        }

        /// <summary>
        /// Converts a texture to a `TensorFloat`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="tensor">The output tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void ToTensor(Texture texture, TensorFloat tensor, TextureTransform transform)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "ToTensor.ValueError: tensor must have rank 4 but has rank {0}", tensor.shape.rank);
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            transform.SetDimensions(tensor.shape[transform.tensorLayoutAxisW], tensor.shape[transform.tensorLayoutAxisH], tensor.shape[transform.tensorLayoutAxisC]);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var isExact = texture.height == transform.height && texture.width == transform.width;
            transform.InferChannelSettings(textureChannels);

            if (SystemInfo.supportsComputeShaders && ComputeInfo.supportsCompute)
            {
                // GPUCompute

                var tensorData = ComputeTensorData.Pin(tensor, false);

                var fn = ComputeFuncSingleton.Instance.Get(isExact ? "TextureToTensorExact" : "TextureToTensorLinear");

                fn.SetTexture(k_ID_X_tex2D, texture);
                fn.SetTensorAsBuffer(k_ID_Optr, tensorData);

                fn.SetInt(k_ID_O_width, transform.width);
                fn.SetInt(k_ID_O_height, transform.height);
                fn.SetInt(k_ID_O_channels, transform.channels);
                fn.SetInt(k_ID_O_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
                fn.SetInt(k_ID_O_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
                fn.SetInt(k_ID_O_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
                fn.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
                fn.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                fn.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                fn.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                fn.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);

                fn.Dispatch(transform.height, transform.width, 1);
            }
            else
            {
                // PixelShader

                var pinO = TextureTensorData.Pin(tensor, transform.tensorLayoutAxisC, false);

                var func = new PixelFunc("Hidden/Sentis/TextureConversion/TextureToTensor");
                func.EnableKeyword(isExact ? "EXACT" : "LINEAR");

                func.SetTexture(k_ID_Xptr, texture);
                func.SetInt(k_ID_StrideWidthO, pinO.blockedShape.Strides(transform.tensorLayoutAxisW));
                func.SetInt(k_ID_StrideWidthO, pinO.blockedShape.Strides(transform.tensorLayoutAxisW));
                func.SetInt(k_ID_StrideHeightO, pinO.blockedShape.Strides(transform.tensorLayoutAxisH));
                func.SetInt(k_ID_WidthO, transform.width);
                func.SetInt(k_ID_HeightO, transform.height);
                func.SetInt(k_ID_Channels, transform.channels);
                func.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
                func.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                func.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                func.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                func.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);

                func.Dispatch(pinO);
            }
        }

        /// <summary>
        /// Appends the conversion of a texture to a `TensorFloat`to a CommandBuffer`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="width">The width of the output tensor. If the value is -1, Sentis uses the texture to infer the width.</param>
        /// <param name="height">The height of the output tensor. If the value is -1, Sentis uses the texture to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output tensor. If the value is -1, Sentis uses the texture to infer the number of channels.</param>
        /// <returns>The converted tensor.</returns>
        public static TensorFloat ToTensor(this CommandBuffer cb, Texture texture, int width = -1, int height = -1, int channels = -1)
        {
            return cb.ToTensor(texture, new TextureTransform().SetDimensions(width, height, channels));
        }

        /// <summary>
        /// Appends the conversion of a texture to a `TensorFloat`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The converted tensor.</returns>
        public static TensorFloat ToTensor(this CommandBuffer cb, Texture texture, TextureTransform transform)
        {
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            transform.InferDimensions(texture.width, texture.height, textureChannels);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var shape = transform.GetTensorShape();

            var O = new TensorFloat(shape);
            cb.ToTensor(texture, O, transform);
            return O;
        }

        /// <summary>
        /// Appends the conversion of a texture to a `TensorFloat`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="tensor">The output tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void ToTensor(this CommandBuffer cb, Texture texture, TensorFloat tensor, TextureTransform transform)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "ToTensor.ValueError: tensor must have rank 4 but has rank {0}", tensor.shape.rank);
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            transform.SetDimensions(tensor.shape[transform.tensorLayoutAxisW], tensor.shape[transform.tensorLayoutAxisH], tensor.shape[transform.tensorLayoutAxisC]);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var isExact = texture.height == transform.height && texture.width == transform.width;
            transform.InferChannelSettings(textureChannels);

            var fn = ComputeFuncSingleton.Instance.Get(isExact ? "TextureToTensorExact" : "TextureToTensorLinear");

            cb.SetTexture(fn, k_ID_X_tex2D, texture);
            cb.SetTensorAsBuffer(fn, k_ID_Optr, ComputeTensorData.Pin(tensor, false));

            cb.SetInt(fn, k_ID_O_width, transform.width);
            cb.SetInt(fn, k_ID_O_height, transform.height);
            cb.SetInt(fn, k_ID_O_channels, transform.channels);
            cb.SetInt(fn, k_ID_O_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
            cb.SetInt(fn, k_ID_O_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
            cb.SetInt(fn, k_ID_O_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
            cb.SetInt(fn, k_ID_CoordOrigin, (int)transform.coordOrigin);
            cb.SetInt(fn, k_ID_ChannelSwizzleR, transform.channelSwizzleR);
            cb.SetInt(fn, k_ID_ChannelSwizzleG, transform.channelSwizzleG);
            cb.SetInt(fn, k_ID_ChannelSwizzleB, transform.channelSwizzleB);
            cb.SetInt(fn, k_ID_ChannelSwizzleA, transform.channelSwizzleA);

            cb.Dispatch(fn, transform.height, transform.width, 1);
        }

        /// <summary>
        /// Appends the conversion of a RenderTargetIdentifier to a `TensorFloat`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="rte">The RenderTargetIdentifier to convert.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The converted tensor.</returns>
        public static TensorFloat ToTensor(this CommandBuffer cb, RenderTargetIdentifier rte, TextureTransform transform = default)
        {
            var textureChannels = 4;
            transform.InferDimensions(Screen.width, Screen.height, textureChannels);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var shape = transform.GetTensorShape();

            var O = new TensorFloat(shape);
            cb.ToTensor(rte, O, transform);
            return O;
        }

        /// <summary>
        /// Appends the conversion of a RenderTargetIdentifier to a `TensorFloat`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="rte">The RenderTargetIdentifier to convert.</param>
        /// <param name="tensor">The output tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void ToTensor(this CommandBuffer cb, RenderTargetIdentifier rte, TensorFloat tensor, TextureTransform transform = default)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "ToTensor.ValueError: tensor must have rank 4 but has rank {0}", tensor.shape.rank);
            var textureChannels = 4;
            transform.SetDimensions(tensor.shape[transform.tensorLayoutAxisW], tensor.shape[transform.tensorLayoutAxisH], tensor.shape[transform.tensorLayoutAxisC]);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var isExact = Screen.height == transform.height && Screen.width == transform.width;
            transform.InferChannelSettings(textureChannels);

            var fn = ComputeFuncSingleton.Instance.Get(isExact ? "TextureToTensorExact" : "TextureToTensorLinear");

            cb.SetTexture(fn, k_ID_X_tex2D, rte);
            cb.SetTensorAsBuffer(fn, k_ID_Optr, ComputeTensorData.Pin(tensor));

            cb.SetInt(fn, k_ID_O_width, transform.width);
            cb.SetInt(fn, k_ID_O_height, transform.height);
            cb.SetInt(fn, k_ID_O_channels, transform.channels);
            cb.SetInt(fn, k_ID_O_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
            cb.SetInt(fn, k_ID_O_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
            cb.SetInt(fn, k_ID_O_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
            cb.SetInt(fn, k_ID_CoordOrigin, (int)transform.coordOrigin);
            cb.SetInt(fn, k_ID_ChannelSwizzleR, transform.channelSwizzleR);
            cb.SetInt(fn, k_ID_ChannelSwizzleG, transform.channelSwizzleG);
            cb.SetInt(fn, k_ID_ChannelSwizzleB, transform.channelSwizzleB);
            cb.SetInt(fn, k_ID_ChannelSwizzleA, transform.channelSwizzleA);

            cb.Dispatch(fn, transform.height, transform.width, 1);
        }

        /// <summary>
        /// Returns the default `RenderTextureFormat` with the provided number of `channels`.
        /// </summary>
        /// <param name="channels">The number of channels.</param>
        /// <returns>The render texture format.</returns>
        static RenderTextureFormat GetDefaultRenderTextureFormatFromComponentCount(int channels)
        {
            Logger.AssertIsTrue(channels <= 4 && channels > 0, "GetDefaultRenderTextureFormatFromComponentCount.ValueError: cannot convert channel count {0} to texture format", channels);
            return channels switch
            {
                1 => RenderTextureFormat.RFloat,
                2 => RenderTextureFormat.RGFloat,
                3 => RenderTextureFormat.RGB111110Float,
                _ => RenderTextureFormat.ARGBFloat
            };
        }

        /// <summary>
        /// Write the float data in a tensor to a render texture. Sentis only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the tensor don't match the width and height of the render texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="renderTexture">The render texture to write to. If the value is `null`, Sentis blits the tensor data to the screen.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToTexture(TensorFloat tensor, RenderTexture renderTexture, TextureTransform transform = default)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "BlitTensorToTexture.RankError: tensor rank should be equal to 4, got {0}.", tensor.shape.rank);
            if (renderTexture != null)
                Logger.AssertIsTrue(renderTexture.dimension == TextureDimension.Tex2D, "BlitTensorToTexture.ValueError: renderTexture must have dimension Tex2D, got {0}.", renderTexture.dimension);

            var renderWidth = renderTexture != null ? renderTexture.width : Screen.width;
            var renderHeight = renderTexture != null ? renderTexture.height : Screen.height;

            var tensorWidth = tensor.shape[transform.tensorLayoutAxisW];
            var tensorHeight = tensor.shape[transform.tensorLayoutAxisH];
            var tensorChannels = tensor.shape[transform.tensorLayoutAxisC];

            var isExact = renderWidth == tensorWidth && renderHeight == tensorHeight;
            transform.InferChannelSettings(tensorChannels);

            if (SystemInfo.supportsComputeShaders && ComputeInfo.supportsCompute)
            {
                if (renderTexture != null)
                {
                    if (!renderTexture.enableRandomWrite || !renderTexture.IsCreated())
                    {
                        renderTexture.Release();
                        renderTexture.enableRandomWrite = true;
                        renderTexture.Create();
                    }

                    var fn = ComputeFuncSingleton.Instance.Get(isExact ? "TensorToTextureExact" : "TensorToTextureLinear");

                    fn.SetTensorAsBuffer(k_ID_Xptr, ComputeTensorData.Pin(tensor));
                    fn.SetTexture(k_ID_O_tex2D, renderTexture);

                    fn.SetInt(k_ID_X_channels, tensorChannels);
                    fn.SetInt(k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
                    fn.SetInt(k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
                    fn.SetInt(k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
                    fn.SetInt(k_ID_O_width, renderWidth);
                    fn.SetInt(k_ID_O_height, renderHeight);
                    fn.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
                    fn.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                    fn.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                    fn.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                    fn.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                    fn.SetVector(k_ID_ChannelScale, transform.channelScale);
                    fn.SetVector(k_ID_ChannelBias, transform.channelBias);

                    if (!isExact)
                    {
                        fn.SetInt(k_ID_X_width, tensorWidth);
                        fn.SetInt(k_ID_X_height, tensorHeight);
                    }

                    fn.Dispatch(renderHeight, renderWidth, 1);
                }
                else
                {
                    var material = PixelShaderSingleton.Instance.FindMaterial("Hidden/Sentis/TextureConversion/ComputeBufferToTexture");
                    material.EnableKeyword(isExact ? "EXACT" : "LINEAR");

                    material.SetBuffer(k_ID_Xptr, ComputeTensorData.Pin(tensor).buffer);

                    material.SetInt(k_ID_X_channels, tensorChannels);
                    material.SetInt(k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
                    material.SetInt(k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
                    material.SetInt(k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
                    material.SetInt(k_ID_O_width, renderWidth);
                    material.SetInt(k_ID_O_height, renderHeight);
                    material.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
                    material.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                    material.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                    material.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                    material.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                    material.SetVector(k_ID_ChannelScale, transform.channelScale);
                    material.SetVector(k_ID_ChannelBias, transform.channelBias);

                    if (!isExact)
                    {
                        material.SetInt(k_ID_X_width, tensorWidth);
                        material.SetInt(k_ID_X_height, tensorHeight);
                    }

                    Graphics.Blit(null, renderTexture, material);
                }
            }
            else
            {
                var func = new PixelFunc("Hidden/Sentis/TextureConversion/TensorToTexture");
                func.EnableKeyword(isExact ? "EXACT" : "LINEAR");

                var pinX = TextureTensorData.Pin(tensor, transform.tensorLayoutAxisC);
                func.SetTensor(k_TensorPropertiesX, pinX);

                switch (transform)
                {
                    case { channelSwizzleR: 0, channelSwizzleG: 1, channelSwizzleB: 2, channelSwizzleA: 3 }:
                        func.EnableKeyword("RGBA");
                        break;
                    case { channelSwizzleB: 0, channelSwizzleG: 1, channelSwizzleR: 2, channelSwizzleA: 3 }:
                        func.EnableKeyword("BGRA");
                        break;
                    default:
                        func.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                        func.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                        func.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                        func.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                        break;
                }

                func.SetInt(k_ID_Stride0X, pinX.blockedShape.Strides(transform.tensorLayoutAxisW));
                func.SetInt(k_ID_Stride1X, pinX.blockedShape.Strides(transform.tensorLayoutAxisH));

                func.SetInt(k_ID_Channels, tensorChannels);
                func.SetInt(k_ID_WidthX, tensorWidth);
                func.SetInt(k_ID_HeightX, tensorHeight);
                func.SetInt(k_ID_WidthO, renderWidth);
                func.SetInt(k_ID_HeightO, renderHeight);
                func.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
                func.SetVector(k_ID_ChannelScale, transform.channelScale);
                func.SetVector(k_ID_ChannelBias, transform.channelBias);

                func.Dispatch(renderTexture);
            }
        }

        /// <summary>
        /// Appends the write the float data in a tensor to a render texture in a CommandBuffer. Sentis only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the tensor don't match the width and height of the render texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="renderTexture">The render texture to write to. If the value is `null`, Sentis blits the tensor data to the screen.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToTexture(this CommandBuffer cb, TensorFloat tensor, RenderTexture renderTexture, TextureTransform transform = default)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "BlitTensorToTexture.RankError: tensor rank should be equal to 4, got {0}.", tensor.shape.rank);
            if (renderTexture != null)
                Logger.AssertIsTrue(renderTexture.dimension == TextureDimension.Tex2D, "BlitTensorToTexture.ValueError: renderTexture must have dimension Tex2D, got {0}.", renderTexture.dimension);

            var renderWidth = renderTexture != null ? renderTexture.width : Screen.width;
            var renderHeight = renderTexture != null ? renderTexture.height : Screen.height;

            var tensorWidth = tensor.shape[transform.tensorLayoutAxisW];
            var tensorHeight = tensor.shape[transform.tensorLayoutAxisH];
            var tensorChannels = tensor.shape[transform.tensorLayoutAxisC];

            var isExact = renderWidth == tensorWidth && renderHeight == tensorHeight;
            transform.InferChannelSettings(tensorChannels);

            if (renderTexture != null)
            {
                if (!renderTexture.enableRandomWrite || !renderTexture.IsCreated())
                {
                    renderTexture.Release();
                    renderTexture.enableRandomWrite = true;
                    renderTexture.Create();
                }

                var fn = ComputeFuncSingleton.Instance.Get(isExact ? "TensorToTextureExact" : "TensorToTextureLinear");

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, ComputeTensorData.Pin(tensor));
                cb.SetTexture(fn, k_ID_O_tex2D, renderTexture);

                cb.SetInt(fn, k_ID_X_channels, tensorChannels);
                cb.SetInt(fn, k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
                cb.SetInt(fn, k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
                cb.SetInt(fn, k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
                cb.SetInt(fn, k_ID_O_width, renderWidth);
                cb.SetInt(fn, k_ID_O_height, renderHeight);
                cb.SetInt(fn, k_ID_CoordOrigin, (int)transform.coordOrigin);
                cb.SetInt(fn, k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                cb.SetInt(fn, k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                cb.SetInt(fn, k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                cb.SetInt(fn, k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                cb.SetVector(fn, k_ID_ChannelScale, transform.channelScale);
                cb.SetVector(fn, k_ID_ChannelBias, transform.channelBias);

                if (!isExact)
                {
                    cb.SetInt(fn, k_ID_X_width, tensorWidth);
                    cb.SetInt(fn, k_ID_X_height, tensorHeight);
                }

                cb.Dispatch(fn, renderHeight, renderWidth, 1);
            }
            else
            {
                var material = PixelShaderSingleton.Instance.FindMaterial("Hidden/Sentis/TextureConversion/ComputeBufferToTexture");
                material.EnableKeyword(isExact ? "EXACT" : "LINEAR");

                material.SetBuffer(k_ID_Xptr, ComputeTensorData.Pin(tensor).buffer);

                material.SetInt(k_ID_X_channels, tensorChannels);
                material.SetInt(k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
                material.SetInt(k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
                material.SetInt(k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
                material.SetInt(k_ID_O_width, renderWidth);
                material.SetInt(k_ID_O_height, renderHeight);
                material.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
                material.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                material.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                material.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                material.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                material.SetVector(k_ID_ChannelScale, transform.channelScale);
                material.SetVector(k_ID_ChannelBias, transform.channelBias);

                if (!isExact)
                {
                    material.SetInt(k_ID_X_width, tensorWidth);
                    material.SetInt(k_ID_X_height, tensorHeight);
                }

                cb.Blit(null, renderTexture, material);
            }
        }

        /// <summary>
        /// Write the float data in a tensor to the frame buffer. Sentis only writes batch == 0 to the frame buffer.
        ///
        /// If the width and height of the tensor don't match the width and height of the frame buffer, Sentis applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToScreen(TensorFloat tensor, TextureTransform transform = default)
        {
            RenderToTexture(tensor, null, transform);
        }

        /// <summary>
        ///  Appends the write the float data in a tensor to the frame buffer texture in a CommandBuffer. Sentis only writes batch == 0 to the frame buffer.
        ///
        /// If the width and height of the tensor don't match the width and height of the frame buffer, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToScreen(this CommandBuffer cb, TensorFloat tensor, TextureTransform transform = default)
        {
            cb.RenderToTexture(tensor, null, transform);
        }

        /// <summary>
        ///  Appends the write the float data in a tensor to the render target identifier in a CommandBuffer. Sentis only writes batch == 0 to the frame buffer.
        ///
        /// If the width and height of the tensor don't match the width and height of the frame buffer, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="rte">The render target identifier to write to.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToScreen(this CommandBuffer cb, TensorFloat tensor, RenderTargetIdentifier rte, TextureTransform transform = default)
        {
            var renderWidth = Screen.width;
            var renderHeight = Screen.height;

            var tensorWidth = tensor.shape[transform.tensorLayoutAxisW];
            var tensorHeight = tensor.shape[transform.tensorLayoutAxisH];
            var tensorChannels = tensor.shape[transform.tensorLayoutAxisC];

            var isExact = renderWidth == tensorWidth && renderHeight == tensorHeight;
            transform.InferChannelSettings(tensorChannels);

            var material =
                PixelShaderSingleton.Instance.FindMaterial("Hidden/Sentis/TextureConversion/ComputeBufferToTexture");
            material.EnableKeyword(isExact ? "EXACT" : "LINEAR");

            material.SetBuffer(k_ID_Xptr, ComputeTensorData.Pin(tensor).buffer);

            material.SetInt(k_ID_X_channels, tensorChannels);
            material.SetInt(k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
            material.SetInt(k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
            material.SetInt(k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
            material.SetInt(k_ID_O_width, renderWidth);
            material.SetInt(k_ID_O_height, renderHeight);
            material.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
            material.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
            material.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
            material.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
            material.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
            material.SetVector(k_ID_ChannelScale, transform.channelScale);
            material.SetVector(k_ID_ChannelBias, transform.channelBias);

            if (!isExact)
            {
                material.SetInt(k_ID_X_width, tensorWidth);
                material.SetInt(k_ID_X_height, tensorHeight);
            }

            cb.Blit(null, rte, material);
        }

        /// <summary>
        /// Converts the data in a tensor to a render texture. Sentis only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the render texture don't match the width and height of the tensor, Sentis applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="width">The width of the output render texture. If the value is -1, Sentis uses the tensor to infer the width.</param>
        /// <param name="height">The height of the output render texture. If the value is -1, Sentis uses the tensor to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output render texture. If the value is -1, Sentis uses the tensor to infer the number of channels.</param>
        /// <param name="broadcastChannels">When the value is `true`, Sentis broadcasts the tensor values to additional channels in the render texture. For example, a tensor with a single channel R maps to (R, R, R, R) if `channels` is 4. When the value is `false`, Sentis applies a (0, 0, 0, 1) color mask to additional channels in the render texture. For example, a tensor with a single channel R becomes (R, 0, 0, 1) if `channels` is 4.</param>
        /// <returns>The created render texture.</returns>
        public static RenderTexture ToTexture(TensorFloat tensor, int width = -1, int height = -1, int channels = -1, bool broadcastChannels = false)
        {
            return ToTexture(tensor, new TextureTransform().SetDimensions(width, height, channels).SetBroadcastChannels(broadcastChannels));
        }

        /// <summary>
        /// Appends the convertion of the data in a tensor to a render texture to a CommandBuffer. Sentis only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the render texture don't match the width and height of the tensor, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="width">The width of the output render texture. If the value is -1, Sentis uses the tensor to infer the width.</param>
        /// <param name="height">The height of the output render texture. If the value is -1, Sentis uses the tensor to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output render texture. If the value is -1, Sentis uses the tensor to infer the number of channels.</param>
        /// <param name="broadcastChannels">When the value is `true`, Sentis broadcasts the tensor values to additional channels in the render texture. For example, a tensor with a single channel R maps to (R, R, R, R) if `channels` is 4. When the value is `false`, Sentis applies a (0, 0, 0, 1) color mask to additional channels in the render texture. For example, a tensor with a single channel R becomes (R, 0, 0, 1) if `channels` is 4.</param>
        /// <returns>The created render texture.</returns>
        public static RenderTexture ToTexture(this CommandBuffer cb, TensorFloat tensor, int width = -1, int height = -1, int channels = -1, bool broadcastChannels = false)
        {
            return cb.ToTexture(tensor, new TextureTransform().SetDimensions(width, height, channels).SetBroadcastChannels(broadcastChannels));
        }

        /// <summary>
        /// Converts the data in a tensor to a render texture. Sentis only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The created render texture.</returns>
        public static RenderTexture ToTexture(TensorFloat tensor, TextureTransform transform)
        {
            var width = transform.width == -1 ? tensor.shape[transform.tensorLayoutAxisW] : transform.width;
            var height = transform.height == -1 ? tensor.shape[transform.tensorLayoutAxisH] : transform.height;
            Logger.AssertIsTrue(width <= SystemInfo.maxTextureSize, "TensorToTexture:InputError width can be at most SystemInfo.maxTextureSize, got {0}", width);
            Logger.AssertIsTrue(height <= SystemInfo.maxTextureSize, "TensorToTexture:InputError height can be at most SystemInfo.maxTextureSize, got {0}", height);
            var channels = transform.channels == -1 ? tensor.shape[transform.tensorLayoutAxisC] : transform.channels;

            var renderTexture = new RenderTexture(width, height, 0, GetDefaultRenderTextureFormatFromComponentCount(channels));
            RenderToTexture(tensor, renderTexture, transform);

            return renderTexture;
        }

        /// <summary>
        /// Appends the conversion of the data in a tensor to a render texture in a CommandBuffer. Sentis only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Sentis applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The created render texture.</returns>
        public static RenderTexture ToTexture(this CommandBuffer cb, TensorFloat tensor, TextureTransform transform)
        {
            var width = transform.width == -1 ? tensor.shape[transform.tensorLayoutAxisW] : transform.width;
            var height = transform.height == -1 ? tensor.shape[transform.tensorLayoutAxisH] : transform.height;
            Logger.AssertIsTrue(width <= SystemInfo.maxTextureSize, "TensorToTexture:InputError width can be at most SystemInfo.maxTextureSize, got {0}", width);
            Logger.AssertIsTrue(height <= SystemInfo.maxTextureSize, "TensorToTexture:InputError height can be at most SystemInfo.maxTextureSize, got {0}", height);
            var channels = transform.channels == -1 ? tensor.shape[transform.tensorLayoutAxisC] : transform.channels;

            var renderTexture = new RenderTexture(width, height, 0, GetDefaultRenderTextureFormatFromComponentCount(channels));
            cb.RenderToTexture(tensor, renderTexture, transform);

            return renderTexture;
        }
    }
}
