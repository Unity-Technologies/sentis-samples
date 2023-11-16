Shader "Hidden/Sentis/ReduceExpBias"
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

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, float);
            DECLARE_TENSOR(B, float);

            uint StrideAxisX, DimAxisX;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint2 lowerUpper = Unravel(uint1(StrideAxisX), blockIndexO);
                float4 acc4 = 0;
                float4 bias = SampleBlockB(blockIndexO);
                uint blockIndexXMin = Ravel(uint1(StrideAxisX * DimAxisX), lowerUpper);
                uint blockIndexXMax = blockIndexXMin + StrideAxisX * DimAxisX;
                for (uint blockIndexX = blockIndexXMin; blockIndexX < blockIndexXMax; blockIndexX += StrideAxisX)
                {
                    float4 v = SampleBlockX(blockIndexX);
                    acc4 += exp(v - bias);
                }
                return acc4;
            }
            ENDCG
        }
    }
}
