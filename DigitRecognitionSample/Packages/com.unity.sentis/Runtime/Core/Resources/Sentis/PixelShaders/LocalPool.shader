Shader "Hidden/Sentis/LocalPool"
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
            #pragma multi_compile POOL1D POOL2D
            #pragma multi_compile MAXPOOL AVGPOOL

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #define FLT_MIN -3.402823466e+38F

            DECLARE_TENSOR(X, float);

            uint O_width, O_height, O_channelsDiv4;
            uint X_width, X_height, X_channelsDiv4;

            int StrideY, StrideX, PadY, PadX, PoolY, PoolX;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint n = blockIndexO;
                uint w = n % O_width;
                n /= O_width;
                #if defined(POOL2D)
                uint h = n % O_height;
                n /= O_height;
                #endif
                uint cDiv4 = n % O_channelsDiv4;
                n /= O_channelsDiv4;

                uint4 indexX = X_width * (cDiv4 + X_channelsDiv4 * n);

                float counter = 0.0f;
                float4 accVal = 0.0f;
                #ifdef MAXPOOL
                accVal = FLT_MIN;
                #endif
                #if defined(POOL2D)
                indexX *= X_height;
                for (int dy = 0; dy < PoolY; ++dy)
                {
                uint oy = (h * StrideY + dy) - PadY;
                if (oy >= X_height) continue;
                indexX[1] = indexX[2] + oy * X_width;
                #endif
                for (int dx = 0; dx < PoolX; ++dx)
                {
                    uint ox = (w * StrideX + dx) - PadX;
                    if (ox >= X_width) continue;
                    float4 v = SampleBlockX(indexX[1] + ox);
                    #ifdef MAXPOOL
                    accVal = max(accVal, v);
                    #endif
                    #ifdef AVGPOOL
                    accVal += v;
                    #endif
                    counter += 1.0f;
                }
                #if defined(POOL2D)
                }
                #endif
                #ifdef AVGPOOL
                accVal /= counter;
                #endif

                return accVal;
            }
            ENDCG
        }
    }
}
