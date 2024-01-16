Shader "Unlit/RainbowScan"
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

            #include "UnityCG.cginc"

            struct v2f
            {
            };

            v2f vert(float4 vertex : POSITION, out float4 outpos : SV_POSITION)
            {
                v2f o;
                outpos = UnityObjectToClipPos(vertex);
                return o;
            }

            Texture2D<float4> WebCamTex;
            Texture2D<float4> DepthTex;
            Texture2D<float4> ColorRampTex;

            SamplerState LinearClampSampler;

            float4 ScreenCamResolution;
            int DepthOnly;

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                #if defined (SHADER_API_MOBILE)
                float4 rgba = WebCamTex.SampleLevel(LinearClampSampler, float2((screenPos.y / ScreenCamResolution.x), (screenPos.x / ScreenCamResolution.y)), 0);
                float depth = DepthTex.SampleLevel(LinearClampSampler, float2((screenPos.y / ScreenCamResolution.x), 1 - (screenPos.x / ScreenCamResolution.y)), 0);
                #else
                float4 rgba = WebCamTex.SampleLevel(LinearClampSampler, float2((screenPos.x / ScreenCamResolution.y), 1 - (screenPos.y / ScreenCamResolution.x)), 0);
                float depth = DepthTex.SampleLevel(LinearClampSampler, float2((screenPos.x / ScreenCamResolution.y), (screenPos.y / ScreenCamResolution.x)), 0);
                #endif

                float t = (sin(20*_Time)+0.5);
                depth = clamp(depth*1024, 0, 1023);
                float4 col = ColorRampTex.Load(uint3(depth,0,0));

                float4 outcol = depth <= 1023 * t ? rgba : col;
                float fringe = abs(depth - 1023 * t);
                return fringe < 20 ? float4(1,1,1,0) : outcol;
            }
            ENDCG
        }
    }
}
