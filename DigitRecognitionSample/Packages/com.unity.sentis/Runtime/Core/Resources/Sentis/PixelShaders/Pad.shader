Shader "Hidden/Sentis/Pad"
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
            #pragma multi_compile CONSTANT REFLECT EDGE SYMMETRIC WRAP

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);

            uint DimO[8];
            int Pad[8];
            int DimX[8];
            uint StridesX[8];
            uint MaxBlockIndexX;

            float Beta;

            inline uint IndexX(int indexX, int dimX)
            {
                #if defined(REFLECT)
                int underlap = max(0, -indexX);
                int overlap = max(0, indexX - (dimX - 1));
                indexX = indexX + 2 * underlap - 2 * overlap;
                #elif defined(SYMMETRIC)
                float underlap = max(0, -indexX - 0.5f);
                float overlap = max(0, indexX - (dimX - 0.5f));
                indexX = round(indexX + 2 * underlap - 2 * overlap);
                #elif defined(WRAP)
                indexX = (indexX % dimX + dimX) % dimX;
                #endif
                return clamp(indexX, 0, dimX - 1);
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint blockIndexX = 0;
                float4 v = 0;
                bool isUseConstant = false;
                uint n = blockIndexO;
                [unroll]
                for (uint j = 0; j < 8; j++)
                {
                    int index = (int)(n % DimO[j]) - Pad[j];
                    n /= DimO[j];
                    #ifdef CONSTANT
                    blockIndexX += index * StridesX[j];
                    isUseConstant = isUseConstant || index < 0 || index >= DimX[j];
                    #else
                    blockIndexX += IndexX(index, DimX[j]) * StridesX[j];
                    #endif
                }

                #ifdef CONSTANT
                blockIndexX = clamp(blockIndexX, 0, MaxBlockIndexX);
                v = isUseConstant ? Beta : SampleBlockX(blockIndexX);
                #else
                v = SampleBlockX(blockIndexX);
                #endif

                return v;
            }
            ENDCG
        }
    }
}
