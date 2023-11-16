Shader "Hidden/Sentis/Trilu"
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

            uint width, height;
            int direction, offset;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndex = GetBlockIndexO(screenPos);
                uint upper = blockIndex;
                uint lower = upper % StrideAxisO;
                upper /= StrideAxisO;
                uint axis = upper % DimBlockedO;
                upper /= DimBlockedO;
                uint4 index4 = lower + StrideAxisO * ((axis << 2) + uint4(0, 1, 2, 3) + DimAxisO * upper);
                uint4 n4 = index4;
                int4 w4 = n4 % width;
                n4 /= width;
                int4 h4 = n4 % height;
                bool4 mask4 = direction * (w4 - h4) >= direction * offset;
                DTYPE4 v = SampleBlockX(blockIndex);
                return mask4 * v;
            }
            ENDCG
        }
    }
}
