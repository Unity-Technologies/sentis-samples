Shader "Hidden/Sentis/CumSum"
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
            #pragma multi_compile _ INT
            #pragma multi_compile FORWARD REVERSE
            #pragma multi_compile INCLUSIVE EXCLUSIVE

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #ifdef INT
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            uint StrideAxis, DimAxis;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint3 lowerAxisUpper = Unravel(uint2(StrideAxis, DimAxis), blockIndexO);
                DTYPE4 acc4 = 0;

                int start, count, delta;
                #ifdef FORWARD
                start = lowerAxisUpper[0] + StrideAxis * (DimAxis * lowerAxisUpper[2]);
                delta = StrideAxis;
                #ifdef INCLUSIVE
                count = lowerAxisUpper[1] + 1;
                #endif
                #ifdef EXCLUSIVE
                count = lowerAxisUpper[1];
                #endif
                #endif
                #ifdef REVERSE
                start = lowerAxisUpper[0] + StrideAxis * ((DimAxis - 1) + DimAxis * lowerAxisUpper[2]);
                delta = -(int)StrideAxis;
                #ifdef INCLUSIVE
                count = DimAxis - lowerAxisUpper[1];
                #endif
                #ifdef EXCLUSIVE
                count = DimAxis - lowerAxisUpper[1] - 1;
                #endif
                #endif

                uint blockIndexX = start;
                for (int j = 0; j < count; j++)
                {
                    DTYPE4 v = SampleBlockX(blockIndexX);
                    acc4 += v;
                    blockIndexX += delta;
                }
                return acc4;
            }
            ENDCG
        }
    }
}
