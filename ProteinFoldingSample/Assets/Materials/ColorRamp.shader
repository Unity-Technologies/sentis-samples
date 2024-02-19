Shader "Unlit/ColorRamp"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _ColorRampTex("ColorRamp", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

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
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST; 
            sampler2D _ColorRampTex;
            float4 _ColorRampTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed scale = tex2D(_MainTex, i.uv).r;
                scale = clamp(scale, 0.1, 0.9);
                fixed4 colour = tex2D(_ColorRampTex, scale.rr).rgba;
                colour.a = 1;
                return colour;
            }
            ENDCG
        }
    }
}
