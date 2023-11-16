Shader "Hidden/Sentis/OneHot"
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
            #pragma multi_compile _ OneHotInt
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, int);
            #ifdef OneHotInt
            int onValueInt, offValueInt;
            #define DTYPE4 int4
            #else
            float onValue, offValue;
            #define DTYPE4 float4
            #endif


            uint StrideAxis, DimAxisO;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint3 lowerAxisUpper = Unravel(uint2(StrideAxis, DimAxisO), blockIndexO);
                int4 indices = SampleBlockX(Ravel(uint1(StrideAxis), lowerAxisUpper.xz));
                bool4 mask4 = (indices == (int4)lowerAxisUpper.y) || ((indices + (int)DimAxisO) == (int4)lowerAxisUpper.y);
                #ifdef OneHotInt
                return mask4 ? onValueInt : offValueInt;
                #else
                return mask4 ? onValue : offValue;
                #endif
            }
            ENDCG
        }
    }
}
