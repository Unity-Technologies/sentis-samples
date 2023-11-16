Shader "Hidden/Sentis/Activation"
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
            #pragma multi_compile Relu Selu Abs Neg Ceil Floor Round Reciprocal Swish Tanh Softplus Sigmoid HardSigmoid Relu6 Elu LeakyRelu Exp Log Sqrt Acos Acosh Asin Asinh Atan Atanh Cos Cosh Sin Sinh Tan Pow Clip Erf Sign Square Celu HardSwish Softsign ThresholdedRelu Gelu Shrink

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR_BLOCK_STRIDE_O;

            float Alpha;
            float Beta;

            DECLARE_TENSOR(X, float);

            float4 erf(float4 v)
            {
                // Abramowitz/Stegun approximations
                // erf(x) = -erf(-x)
                float4 x = abs(v);

                float p = 0.3275911f;
                float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
                float a4 = -1.453152027f; float a5 = 1.061405429f;

                float4 t = 1.0f / (1.0f + p * x);
                float4 t2 = t * t;
                float4 t3 = t2 * t;
                float4 t4 = t3 * t;
                float4 t5 = t4 * t;

                return sign(v)*(1.0f - (a1*t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5)*exp(-x * x));
            }

            float4 gelu(float4 v)
            {
                return (erf(v / 1.41421356237f) + 1) * v * 0.5f;
            }

            float selu(float v, float alpha, float gamma)
            {
                if (v <= 0.0f)
                    v = gamma * (alpha * exp(v) - alpha);
                else
                    v = gamma * v;

                return v;
            }

            float4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                int3 lowerAxisUpper = UnravelO(blockIndexO);
                uint4 unblocked4 = UnblockAxis(lowerAxisUpper[1]);
                uint4 index4 = lowerAxisUpper[0] + StrideAxisO * (unblocked4 + DimAxisO * lowerAxisUpper[2]);
                bool4 mask4 = (index4 < LengthO && unblocked4 < DimAxisO) ? 1 : 0;
                float4 v = SampleBlockX(blockIndexO);
                #ifdef Relu
                    v = 0.5f * (v + abs(v));
                #endif
                #ifdef Selu
                    v.x = selu(v.x, Alpha, Beta);
                    v.y = selu(v.y, Alpha, Beta);
                    v.z = selu(v.z, Alpha, Beta);
                    v.w = selu(v.w, Alpha, Beta);
                #endif
                #ifdef Abs
                    v = abs(v);
                #endif
                #ifdef Neg
                    v = -v;
                #endif
                #ifdef Ceil
                    v = ceil(v);
                #endif
                #ifdef Floor
                    v = floor(v);
                #endif
                #ifdef Round
                    v = round(v);
                #endif
                #ifdef Reciprocal
                    v.x = 1.0 / v.x;
                    v.y = 1.0 / v.y;
                    v.z = 1.0 / v.z;
                    v.w = 1.0 / v.w;
                #endif
                #ifdef Swish
                    v = v / (1.0 + exp(-v));
                #endif
                #ifdef Tanh
                    v = tanh(clamp(v,-16.0f,16.0f)); // clamp to avoid NaNs for large values.
                #endif
                #ifdef Softplus
                    v = log(exp(v) + 1.0);
                #endif
                #ifdef Sigmoid
                    v = 1.0 / (1.0 + exp(-v));
                #endif
                #ifdef HardSigmoid
                    v = max(0.0, min(1.0, Alpha * v + Beta));
                #endif
                #ifdef Relu6
                    v = min(max(0.0, v), 6.0f);
                #endif
                #ifdef Elu
                    if (v.x <= 0.0)
                        v.x = Alpha * (exp(v.x) - 1.0);
                    if (v.y <= 0.0)
                        v.y = Alpha * (exp(v.y) - 1.0);
                    if (v.z <= 0.0)
                        v.z = Alpha * (exp(v.z) - 1.0);
                    if (v.w <= 0.0)
                        v.w = Alpha * (exp(v.w) - 1.0);
                #endif
                #ifdef LeakyRelu
                    v = max(v, Alpha * v);
                #endif
                #ifdef Exp
                    v = exp(v);
                #endif
                #ifdef Log
                    v = log(v);
                #endif
                #ifdef Sqrt
                    v = sqrt(v);
                #endif
                #ifdef Acos
                    v = acos(v);
                #endif
                #ifdef Acosh
                    v = log(v + sqrt(v * v - 1.0));
                #endif
                #ifdef Asin
                    v = asin(v);
                #endif
                #ifdef Asinh
                    v = log(v + sqrt(v*v + 1.0));
                #endif
                #ifdef Atan
                    v = atan(v);
                #endif
                #ifdef Atanh
                    v = 0.5f * log((1.0 + v) / (1.0 - v));
                #endif
                #ifdef Cos
                    v = cos(v);
                #endif
                #ifdef Cosh
                    v = 0.5f * (exp(v) + exp(-v));
                #endif
                #ifdef Sin
                    v = sin(v);
                #endif
                #ifdef Sinh
                    v = 0.5f * (exp(v) - exp(-v));
                #endif
                #ifdef Tan
                    v = tan(v);
                #endif
                #ifdef Pow
                    v = pow(v, Alpha);
                #endif
                #ifdef Clip
                    v = clamp(v, Alpha, Beta);
                #endif
                #ifdef Erf
                    v = erf(v);
                #endif
                #ifdef Sign
                    v = sign(v);
                #endif
                #ifdef Square
                    v = v * v;
                #endif
                #ifdef Celu
                    v = max(0.0f, v) + min(0.0f, Alpha * (exp(v / Alpha) - 1.0f));
                #endif
                #ifdef HardSwish
                    v = v * max(0, min(1, 0.16666667f * v + 0.5f));
                #endif
                #ifdef Softsign
                    v = v / (1.0f + abs(v));
                #endif
                #ifdef ThresholdedRelu
                    if (v.x <= Alpha)
                        v.x = 0.0f;
                    if (v.y <= Alpha)
                        v.y = 0.0f;
                    if (v.z <= Alpha)
                        v.z = 0.0f;
                    if (v.w <= Alpha)
                        v.w = 0.0f;
                #endif
                #ifdef Gelu
                    v = gelu(v);
                #endif
                #ifdef Shrink
                    float4 vOut = 0;
                    vOut = v < -Beta ? v + Alpha : vOut;
                    vOut = v > Beta ? v - Alpha : vOut;
                    v = vOut;
                #endif

                if (!mask4.x)
                    v.x = 0;
                if (!mask4.y)
                    v.y = 0;
                if (!mask4.z)
                    v.z = 0;
                if (!mask4.w)
                    v.w = 0;

                return v;
            }
            ENDCG
        }
    }
}
