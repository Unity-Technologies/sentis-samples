Shader "Hidden/Sentis/Reduce"
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
            #pragma multi_compile ReduceMin ReduceMax ReduceSum ReduceSumSquare ReduceMean ReduceMeanSquare ReduceProd ReduceL1 ReduceL2 ReduceSqrt ReduceLogSum ReduceLogSumExp ReduceMinInt ReduceMaxInt ReduceSumInt ReduceSumSquareInt ReduceProdInt ReduceL1Int

            #pragma vertex vert
            #pragma fragment frag


            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            #if defined(ReduceMinInt) | (ReduceMaxInt) | (ReduceSumInt) | (ReduceSumSquareInt) | (ReduceProdInt) | (ReduceL1Int)
            #define DTYPE4 int4
            DECLARE_TENSOR(X, int);
            #else
            #define DTYPE4 float4
            DECLARE_TENSOR(X, float);
            #endif

            uint StrideAxisX, DimAxisX;

            float Normalization;

            #if defined(ReduceLogSumExp)
            float4 maxVal;
            #endif

            #define FLT_MAX asfloat(0x7F7FFFFF) //  3.402823466 E + 38
            #define FLT_MIN asfloat(0xFF7FFFFF) // -3.402823466 E + 38
            #define INT_MAX 0x7FFFFFFF //  2147483647
            #define INT_MIN 0x80000000 // –2147483648

            inline DTYPE4 Default4()
            {
                #if defined(ReduceMin)
                return FLT_MAX;
                #elif defined(ReduceMax)
                return FLT_MIN;
                #elif defined(ReduceMinInt)
                return INT_MAX;
                #elif defined(ReduceMaxInt)
                return INT_MIN;
                #elif defined(ReduceProd) | defined(ReduceProdInt)
                return 1;
                #else
                return 0;
                #endif
            }

            inline DTYPE4 Initialize4(DTYPE4 v)
            {
                #if defined(ReduceSumSquare) | defined(ReduceMeanSquare) | defined(ReduceL2) | defined(ReduceSumSquareInt)
                return v * v;
                #elif defined(ReduceL1) | defined(ReduceL1Int)
                return abs(v);
                #elif defined(ReduceLogSumExp)
                return exp(v - maxVal);
                #else
                return v;
                #endif
            }

            inline DTYPE4 Reduce4(DTYPE4 acc, DTYPE4 v)
            {
                #if defined(ReduceMin) | defined(ReduceMinInt)
                return min(acc, v);
                #elif defined(ReduceMax) | defined(ReduceMaxInt)
                return max(acc, v);
                #elif defined(ReduceProd) | defined(ReduceProdInt)
                return acc * v;
                #else
                return acc + v;
                #endif
            }

            inline DTYPE4 Finalize4(DTYPE4 acc)
            {
                #if defined(ReduceMean) | defined(ReduceMeanSquare)
                return Normalization * acc;
                #elif defined(ReduceSqrt) | defined(ReduceL2)
                return sqrt(acc);
                #elif defined(ReduceLogSum)
                float4 u = log(acc);
                bool4 accNaN = acc <= 0.0f;
                u.x = accNaN.x ? 0.0f : u.x;
                u.y = accNaN.y ? 0.0f : u.y;
                u.z = accNaN.z ? 0.0f : u.z;
                u.w = accNaN.w ? 0.0f : u.w;
                return u;
                #elif defined(ReduceLogSumExp)
                return log(acc) + maxVal;
                #else
                return acc;
                #endif
            }

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint2 lowerUpper = Unravel(uint1(StrideAxisX), blockIndexO);

                DTYPE4 acc4 = Default4();
                uint blockIndexXMin = Ravel(uint1(StrideAxisX * DimAxisX), lowerUpper);
                uint blockIndexXMax = blockIndexXMin + StrideAxisX * DimAxisX;
                uint blockIndexX;
                #if defined(ReduceLogSumExp)
                maxVal = FLT_MIN;
                for (blockIndexX = blockIndexXMin; blockIndexX < blockIndexXMax; blockIndexX += StrideAxisX)
                {
                    float4 v = SampleBlockX(blockIndexX);
                    maxVal = max(maxVal, v);
                }
                #endif
                for (blockIndexX = blockIndexXMin; blockIndexX < blockIndexXMax; blockIndexX += StrideAxisX)
                {
                    DTYPE4 v = SampleBlockX(blockIndexX);
                    v = Initialize4(v);
                    acc4 = Reduce4(acc4, v);
                }
                acc4 = Finalize4(acc4);

                return acc4;
            }
            ENDCG
        }
    }
}
