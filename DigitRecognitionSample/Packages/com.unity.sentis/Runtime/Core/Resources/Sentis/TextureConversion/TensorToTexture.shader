Shader "Hidden/Sentis/TextureConversion/TensorToTexture"
{
    Properties
    {
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma multi_compile EXACT LINEAR
            #pragma multi_compile _ RGBA BGRA

            #pragma vertex vert
            #pragma fragment frag

            #include "../PixelShaders/CommonVertexShader.cginc"
            #include "../PixelShaders/CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);

            uint WidthO, HeightO;
            uint WidthX, HeightX;
            uint Stride1X, Stride0X;
            uint Channels;
            uint CoordOrigin;
            uint ChannelSwizzleR;
            uint ChannelSwizzleG;
            uint ChannelSwizzleB;
            uint ChannelSwizzleA;
            float4 ChannelScale;
            float4 ChannelBias;

            float4 SampleTextureColor(uint x, uint y)
            {
                uint blockIndexX = Stride0X * x + Stride1X * y;
                float4 colorX = SampleBlockX(blockIndexX);
                float4 colorO = 0;
            #ifdef RGBA
                if (Channels >= 1)
                    colorO.x = colorX.x;
                if (Channels >= 2)
                    colorO.y = colorX.y;
                if (Channels >= 3)
                    colorO.z = colorX.z;
                if (Channels >= 4)
                    colorO.w = colorX.w;
            #elif BGRA
                if (Channels >= 3)
                    colorO.x = colorX.z;
                if (Channels >= 2)
                    colorO.y = colorX.y;
                if (Channels >= 1)
                    colorO.z = colorX.x;
                if (Channels >= 4)
                    colorO.w = colorX.w;
            #else // Use swizzles
                if (Channels >= ChannelSwizzleR)
                    colorO.x = colorX[ChannelSwizzleR];
                if (Channels >= ChannelSwizzleG)
                    colorO.y = colorX[ChannelSwizzleG];
                if (Channels >= ChannelSwizzleB)
                    colorO.z = colorX[ChannelSwizzleB];
                if (Channels >= ChannelSwizzleA)
                    colorO.w = colorX[ChannelSwizzleA];
            #endif
                return ChannelScale * colorO + ChannelBias;
            }

            float4 ComputeColor(uint2 posO)
            {
                uint2 SizeO = uint2(WidthO, HeightO);

                if(CoordOrigin == 0) // CoordOrigin.TopLeft
                    posO.y = SizeO.y - 1 - posO.y;

            #ifdef EXACT
                float4 c = SampleTextureColor(posO.x, posO.y);
            #else
                uint2 SizeX = uint2(WidthX, HeightX);

                float2 p = (posO + 0.5f) / SizeO * SizeX - 0.5f;
                uint2 p_floor = floor(p);
                float2 p_frac = p - p_floor;

                uint2 clampMax = SizeX - 1;
                uint2 p_lower = clamp(p_floor, uint2(0, 0), clampMax);
                uint2 p_upper = clamp(p_floor + 1, uint2(0, 0), clampMax);

                // Bilinear filter
                float4 c0 = (1 - p_frac.y) * SampleTextureColor(p_lower.x, p_lower.y) + p_frac.y * SampleTextureColor(p_lower.x, p_upper.y);
                float4 c1 = (1 - p_frac.y) * SampleTextureColor(p_upper.x, p_lower.y) + p_frac.y * SampleTextureColor(p_upper.x, p_upper.y);

                float4 c = (1 - p_frac.x) * c0 + p_frac.x * c1;
            #endif
                return c;
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint2 O_pos = (uint2)(screenPos.xy - 0.5f);
                return ComputeColor(O_pos);
            }
            ENDCG
        }
    }
}
