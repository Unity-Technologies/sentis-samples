Shader "Unlit/Star"
{
    Properties
    {
        [HDR] _EmissionColor("Emission Color", Color) = (0,0,0)
        _MainTex("Texture", 2D) = "white" {}
    }
        SubShader
    {
        Tags {
        "Queue" = "Transparent+110"
        }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            float4 Color;
            StructuredBuffer<float> Positions;
            int Index;
            float4 _EmissionColor;
            float MassRatio;

            Texture2D<float4> _MainTex;

            v2f vert(appdata v)
            {
                v2f o;
                float4x4 viewMatrix = unity_ObjectToWorld;
                viewMatrix[0][3] = Positions[3 * Index + 0];
                viewMatrix[1][3] = Positions[3 * Index + 1];
                viewMatrix[2][3] = Positions[3 * Index + 2];

                float3 pos = v.vertex;
                o.vertex = mul(UNITY_MATRIX_VP, mul(viewMatrix, float4(pos, 1.0)));
                o.uv = v.uv;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float4 starRGB = _MainTex.Load(uint3(MassRatio*4096, 0, 0));
                return float4(starRGB.rgb * _EmissionColor, 1);
            }
            ENDCG
        }
    }
}
