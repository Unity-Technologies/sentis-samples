Shader "Hidden/Sentis/ReduceIndices"
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
            #pragma multi_compile ArgMin ArgMax
            #pragma multi_compile First Last
            #pragma multi_compile _ X_INT

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #ifdef X_INT
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif


            uint StrideAxisX, DimAxisX;

            #define FLT_MAX asfloat(0x7F7FFFFF) //  3.402823466 E + 38
            #define FLT_MIN asfloat(0xFF7FFFFF) // -3.402823466 E + 38
            #define INT_MAX 2147483647
            #define INT_MIN -2147483648

            int4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint2 lowerUpper = Unravel(uint1(StrideAxisX), blockIndexO);

                DTYPE4 acc4;

                #ifdef ArgMin
                #ifdef X_INT
                acc4 = INT_MAX;
                #else
                acc4 = FLT_MAX;
                #endif
                #endif
                #ifdef ArgMax
                #ifdef X_INT
                acc4 = INT_MIN;
                #else
                acc4 = FLT_MIN;
                #endif
                #endif
                int4 accIdx4 = 0;
                uint blockIndexX = Ravel(uint1(StrideAxisX * DimAxisX), lowerUpper);
                for (int j = 0; j < (int)DimAxisX; j++)
                {
                    DTYPE4 v = SampleBlockX(blockIndexX);
                    bool4 c4;
                    #ifdef ArgMax
                    #ifdef First
                    c4 = v > acc4;
                    #else
                    c4 = v >= acc4;
                    #endif
                    #endif
                    #ifdef ArgMin
                    #ifdef First
                    c4 = v < acc4;
                    #else
                    c4 = v <= acc4;
                    #endif
                    #endif
                    accIdx4 = c4 ? j : accIdx4;
                    acc4 = c4 ? v : acc4;
                    blockIndexX += StrideAxisX;
                }

                return accIdx4;
            }
            ENDCG
        }
    }
}
