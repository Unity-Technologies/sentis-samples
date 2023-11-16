Shader "Hidden/Sentis/TextureConversion/TextureToTensor"
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

            #ifdef EXACT
            Texture2D<float4> Xptr;
            #else
            sampler2D Xptr;
            #endif

            uint WidthShiftO;
            uint StrideWidthO, StrideHeightO;
            uint WidthO, HeightO;
            uint Channels;
            uint CoordOrigin;
            uint ChannelSwizzleR;
            uint ChannelSwizzleG;
            uint ChannelSwizzleB;
            uint ChannelSwizzleA;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint2 tid = (uint2)(screenPos.xy - 0.5f);
                uint pixelO = (tid.y << WidthShiftO) + tid.x;
                uint2 posO = uint2((pixelO / StrideWidthO) % WidthO, (pixelO / StrideHeightO) % HeightO);

                if(CoordOrigin == 0) // CoordOrigin.TopLeft
                    posO.y = HeightO - 1 - posO.y;

                #ifdef EXACT
                float4 colorX = Xptr.Load(uint3(posO.x, posO.y, 0));
                #else
                    float2 uv = ((float2)posO + 0.5f) / uint2(WidthO, HeightO);
                    float4 colorX = tex2D(Xptr, uv);
                #endif

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
                    if (Channels >= 1)
                        colorO.x = colorX.z;
                    if (Channels >= 2)
                        colorO.y = colorX.y;
                    if (Channels >= 3)
                        colorO.z = colorX.x;
                    if (Channels >= 4)
                        colorO.w = colorX.w;
                #else // Use swizzles
                    if (Channels >= 1)
                        colorO.x = colorX[ChannelSwizzleR];
                    if (Channels >= 2)
                        colorO.y = colorX[ChannelSwizzleG];
                    if (Channels >= 3)
                        colorO.z = colorX[ChannelSwizzleB];
                    if (Channels >= 4)
                        colorO.w = colorX[ChannelSwizzleA];
                #endif

                return colorO;
            }
            ENDCG
        }
    }
}
