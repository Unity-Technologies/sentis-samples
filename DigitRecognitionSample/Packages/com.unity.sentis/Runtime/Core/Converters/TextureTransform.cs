using System;
using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Options for the dimension layout in a texture tensor.
    /// </summary>
    public enum TensorLayout
    {
        /// <summary>
        /// A tensor layout with rank 4 and dimensions batch, channels, height, width (NCHW).
        /// </summary>
        NCHW = 0,

        /// <summary>
        /// A tensor layout with rank 4 and dimensions batch, height, width, channels (NHWC).
        /// </summary>
        NHWC = 1,
    }

    /// <summary>
    /// Options for the position of the origin (0, 0) in the x and y dimensions of a texture tensor.
    /// </summary>
    public enum CoordOrigin
    {
        /// <summary>
        /// Use the top-left of the texture as position (0, 0) in the data.
        /// </summary>
        TopLeft = 0,

        /// <summary>
        /// Use the bottom-left of the texture as position (0, 0) in the data.
        /// </summary>
        BottomLeft = 1,
    }

    /// <summary>
    /// Options for the order of the color channels in a texture tensor.
    /// </summary>
    public enum ChannelSwizzle
    {
        /// <summary>
        /// RGBA color channel order. This is the default.
        /// </summary>
        RGBA = 0,

        /// <summary>
        /// BGRA color channel order.
        /// </summary>
        BGRA = 1,
    }

    /// <summary>
    /// Set the position of each color channel in a texture tensor.
    /// </summary>
    public enum Channel
    {
        /// <summary>
        /// The position of the red channel.
        /// </summary>
        R = 0,

        /// <summary>
        /// The position of the green channel.
        /// </summary>
        G = 1,

        /// <summary>
        /// The position of the blue channel.
        /// </summary>
        B = 2,

        /// <summary>
        /// The position of the alpha channel.
        /// </summary>
        A = 3,
    }

    /// <summary>
    /// Represents settings for converting between textures and tensors.
    ///
    /// Create an instance of `TextureTransform` using the constructor, then use the `TextureTransform` object as a parameter in `TextureConverter` methods.
    ///
    /// For example: `TextureTransform settings = new TextureTransform().SetDimensions(256, 256, 4).SetTensorLayout(TensorLayout.NHWC);`
    /// </summary>
    public unsafe struct TextureTransform
    {
        bool m_IsSetDimensions;
        int m_Width;
        int m_Height;
        int m_Channels;

        internal bool isSetDimensions => m_IsSetDimensions;
        internal int width => m_IsSetDimensions ? m_Width : -1;
        internal int height => m_IsSetDimensions ? m_Height : -1;
        internal int channels => m_IsSetDimensions ? m_Channels : -1;

        bool m_IsSetTensorLayout;
        int m_TensorLayoutAxisN;
        int m_TensorLayoutAxisC;
        int m_TensorLayoutAxisH;
        int m_TensorLayoutAxisW;

        internal int tensorLayoutAxisN => m_IsSetTensorLayout ? m_TensorLayoutAxisN : 0;
        internal int tensorLayoutAxisC => m_IsSetTensorLayout ? m_TensorLayoutAxisC : 1;
        internal int tensorLayoutAxisH => m_IsSetTensorLayout ? m_TensorLayoutAxisH : 2;
        internal int tensorLayoutAxisW => m_IsSetTensorLayout ? m_TensorLayoutAxisW : 3;

        bool m_BroadcastChannels;

        fixed bool m_IsChannelSwizzleSet[4];
        fixed int m_ChannelSwizzle[4];
        fixed bool m_IsChannelMaskSet[4];
        fixed float m_ChannelMask[4];
        fixed float m_ChannelColor[4];

        internal Vector4 channelScale => new Vector4(1 - m_ChannelMask[0], 1 - m_ChannelMask[1], 1 - m_ChannelMask[2], 1 - m_ChannelMask[3]);
        internal Vector4 channelBias => new Vector4(m_ChannelMask[0] * m_ChannelColor[0], m_ChannelMask[1] * m_ChannelColor[1], m_ChannelMask[2] * m_ChannelColor[2], m_ChannelMask[3] * m_ChannelColor[3]);
        internal int channelSwizzleR => m_ChannelSwizzle[0];
        internal int channelSwizzleG => m_ChannelSwizzle[1];
        internal int channelSwizzleB => m_ChannelSwizzle[2];
        internal int channelSwizzleA => m_ChannelSwizzle[3];

        internal void InferChannelSettings(int numTensorChannels)
        {
            for (var i = 0; i < 4; i++)
            {
                if (!m_IsChannelSwizzleSet[i])
                    m_ChannelSwizzle[i] = i;
                if (m_ChannelSwizzle[i] >= numTensorChannels)
                {
                    m_ChannelSwizzle[i] = numTensorChannels - 1;
                    if (!m_IsChannelMaskSet[i])
                    {
                        m_ChannelMask[i] = m_BroadcastChannels ? 0 : 1;
                        m_ChannelColor[i] = i == 3 ? 1f : 0f;
                    }
                }
            }
        }

        /// <summary>
        /// Sets the default behaviour when the output texture has more channels than the input tensor.
        ///
        /// When `broadcastChannels` is `true`, Sentis broadcasts the tensor values to additional channels in the render texture. For example, a tensor with a single channel R maps to (R, R, R, R) if the number of channels is 4.
        ///
        /// When `broadcastChannels` is `false`, Sentis applies a (0, 0, 0, 1) color mask to additional channels in the render texture. For example, a tensor with a single channel R becomes (R, 0, 0, 1) if the number of channels is 4.
        /// </summary>
        /// <param name="broadcastChannels">Whether to broadcast the input channels across output channels.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetBroadcastChannels(bool broadcastChannels)
        {
            m_BroadcastChannels = broadcastChannels;
            return this;
        }

        /// <summary>
        /// Sets a specific texture channel, for example `Channel.R`, to a specific position in the tensor.
        ///
        /// A color mask for tensor to texture conversions might override this setting.
        /// </summary>
        /// <param name="c">The color channel to set.</param>
        /// <param name="swizzle">The index in the channel tensor axis.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetChannelSwizzle(Channel c, int swizzle)
        {
            m_ChannelSwizzle[(int)c] = swizzle;
            m_IsChannelSwizzleSet[(int)c] = true;
            return this;
        }

        /// <summary>
        /// Sets a specific texture channel, for example `Channel.R`, to a specific `color`. The channel ignores input tensor values.
        /// </summary>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        /// <param name="c">The color channel to set.</param>
        /// <param name="mask">When the value is `false`, Sentis ignores `color` and uses input tensor values.</param>
        /// <param name="color">The color value to use.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetChannelColorMask(Channel c, bool mask, float color)
        {
            m_ChannelMask[(int)c] = mask ? 1 : 0;
            m_ChannelColor[(int)c] = color;
            m_IsChannelMaskSet[(int)c] = true;
            return this;
        }

        /// <summary>
        /// Sets which channels in the tensor map to which RGBA channels in the texture, using four channel position values.
        ///
        /// A color mask for tensor to texture conversions might override this setting.
        /// </summary>
        /// <param name="channelSwizzleR">Index in tensor channel axis for red texture channel.</param>
        /// <param name="channelSwizzleG">Index in tensor channel axis for green texture channel.</param>
        /// <param name="channelSwizzleB">Index in tensor channel axis for blue texture channel.</param>
        /// <param name="channelSwizzleA">Index in tensor channel axis for alpha texture channel.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetChannelSwizzle(int channelSwizzleR = 0, int channelSwizzleG = 1, int channelSwizzleB = 2, int channelSwizzleA = 3)
        {
            SetChannelSwizzle(Channel.R, channelSwizzleR);
            SetChannelSwizzle(Channel.G, channelSwizzleG);
            SetChannelSwizzle(Channel.B, channelSwizzleB);
            SetChannelSwizzle(Channel.A, channelSwizzleA);
            return this;
        }

        /// <summary>
        /// Sets which channels in the tensor map to which RGBA channels in the texture, using a `ChannelSwizzle` enum value.
        ///
        /// A color mask for tensor to texture conversions might override this setting.
        /// </summary>
        /// <param name="channelSwizzle">The channel swizzle enum to use.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if unknown `ChannelSwizzle` is used.</exception>
        public TextureTransform SetChannelSwizzle(ChannelSwizzle channelSwizzle)
        {
            return channelSwizzle switch
            {
                ChannelSwizzle.RGBA => SetChannelSwizzle(0, 1, 2, 3),
                ChannelSwizzle.BGRA => SetChannelSwizzle(2, 1, 0, 3),
                _ => throw new ArgumentOutOfRangeException(nameof(channelSwizzle), channelSwizzle, null)
            };
        }

        /// <summary>
        /// Sets which channels in the output texture ignore input tensor values and write a specific `color` instead.
        ///
        /// The method returns a `TextureTransform` that you can use to chain other methods.
        /// </summary>
        /// <param name="maskR">The mask value for the red channel.</param>
        /// <param name="maskG">The mask value for the green channel.</param>
        /// <param name="maskB">The mask value for the blue channel.</param>
        /// <param name="maskA">The mask value for the alpha channel.</param>
        /// <param name="color">The color value to use when masking.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetChannelColorMask(bool maskR, bool maskG, bool maskB, bool maskA, Color color)
        {
            SetChannelColorMask(Channel.R, maskR, color.r);
            SetChannelColorMask(Channel.G, maskG, color.g);
            SetChannelColorMask(Channel.B, maskB, color.b);
            SetChannelColorMask(Channel.A, maskA, color.a);
            return this;
        }

        internal CoordOrigin coordOrigin { get; private set; }

        /// <summary>
        /// Sets the dimensions of the output texture or tensor. The default value is -1, which means Sentis infers the dimensions from the input texture or tensor.
        ///
        /// If the width and height of the input don't match the width and height of the output, Sentis applies linear resampling.
        ///
        /// If you use `SetDimensions` in a blit to an existing texture, Sentis ignores `width`, `height`, and `channels`.
        ///
        /// The method returns a `TextureTransform` that you can use to chain other methods.
        /// </summary>
        /// <param name="width">The width to use for the output.</param>
        /// <param name="height">The height to use for the output.</param>
        /// <param name="channels">The channel count to use for the output.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetDimensions(int width = -1, int height = -1, int channels = -1)
        {
            Logger.AssertIsTrue(width == -1 || 0 < width, "TextureTransform.SetDimensions:InputError width must be -1 or greater than 0");
            Logger.AssertIsTrue(height == -1 || 0 < height, "TextureTransform.SetDimensions:InputError height must be -1 or greater than 0");
            Logger.AssertIsTrue(channels == -1 || (0 < channels && channels <= 4), "TextureTransform.SetDimensions:InputError channels must be -1 or greater than 0 and at most 4");
            m_Width = width;
            m_Height = height;
            m_Channels = channels;
            m_IsSetDimensions = true;
            return this;
        }

        /// <summary>
        /// Sets the layout of the input tensor with four int values.
        /// </summary>
        /// <param name="tensorLayoutAxisN">The axis in the tensor for the batch size of the texture.</param>
        /// <param name="tensorLayoutAxisC">The axis in the tensor for the channel count of the texture.</param>
        /// <param name="tensorLayoutAxisH">The axis in the tensor for the height of the texture.</param>
        /// <param name="tensorLayoutAxisW">The axis in the tensor for the width of the texture.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetTensorLayout(int tensorLayoutAxisN, int tensorLayoutAxisC, int tensorLayoutAxisH, int tensorLayoutAxisW)
        {
            m_TensorLayoutAxisN = tensorLayoutAxisN;
            m_TensorLayoutAxisC = tensorLayoutAxisC;
            m_TensorLayoutAxisH = tensorLayoutAxisH;
            m_TensorLayoutAxisW = tensorLayoutAxisW;
            m_IsSetTensorLayout = true;
            return this;
        }

        /// <summary>
        /// Sets the layout of the input tensor with a `TensorLayout` object, for example `TensorLayout.NHWC`. The default is `TensorLayout.NCHW`.
        /// </summary>
        /// <param name="tensorLayout"></param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if unknown `TensorLayout` is used.</exception>
        public TextureTransform SetTensorLayout(TensorLayout tensorLayout)
        {
            return tensorLayout switch
            {
                TensorLayout.NCHW => SetTensorLayout(0, 1, 2, 3),
                TensorLayout.NHWC => SetTensorLayout(0, 3, 1, 2),
                _ => throw new ArgumentOutOfRangeException(nameof(tensorLayout), tensorLayout, null)
            };
        }

        /// <summary>
        /// Sets the position of the origin (0, 0) in the tensor.
        /// </summary>
        /// <param name="coordOrigin">The position of the texture origin in the tensor.</param>
        /// <returns>`TextureTransform` that you can use to chain other methods.</returns>
        public TextureTransform SetCoordOrigin(CoordOrigin coordOrigin)
        {
            this.coordOrigin = coordOrigin;
            return this;
        }

        /// <summary>
        /// Infer the dimensions in case they are -1
        /// </summary>
        internal void InferDimensions(int width, int height, int channels)
        {
            m_Width = !m_IsSetDimensions || m_Width == -1 ? width : m_Width;
            m_Height = !m_IsSetDimensions || m_Height == -1 ? height : m_Height;
            m_Channels = !m_IsSetDimensions || m_Channels == -1 ? channels : m_Channels;
            m_IsSetDimensions = true;
        }

        /// <summary>
        /// Shape of the output tensor from the conversion, this is inferred from the dimensions and tensor layout
        /// </summary>
        internal TensorShape GetTensorShape()
        {
            var tensorShape = TensorShape.Ones(rank: 4);
            tensorShape[tensorLayoutAxisC] = channels;
            tensorShape[tensorLayoutAxisH] = height;
            tensorShape[tensorLayoutAxisW] = width;
            return tensorShape;
        }
    }
}
