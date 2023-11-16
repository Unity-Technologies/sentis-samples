StructuredBuffer<float> Xptr;

uint X_channels;
uint X_strideW;
uint X_strideH;
uint X_strideC;
uint O_width;
uint O_height;
uint CoordOrigin;
uint ChannelSwizzleR;
uint ChannelSwizzleG;
uint ChannelSwizzleB;
uint ChannelSwizzleA;
float4 ChannelScale;
float4 ChannelBias;

#define COORDORIGIN_TOPLEFT 0

#ifdef LINEAR
uint X_width;
uint X_height;
#endif

float4 SampleTensorColor(uint x, uint y)
{
    float4 color = float4(
        Xptr[x * X_strideW + y * X_strideH + ChannelSwizzleR * X_strideC],
        Xptr[x * X_strideW + y * X_strideH + ChannelSwizzleG * X_strideC],
        Xptr[x * X_strideW + y * X_strideH + ChannelSwizzleB * X_strideC],
        Xptr[x * X_strideW + y * X_strideH + ChannelSwizzleA * X_strideC]
    );
    return ChannelScale * color + ChannelBias;
}

float4 ComputeColor(uint2 posO)
{
    uint2 O_size = uint2(O_width, O_height);

    if(CoordOrigin == COORDORIGIN_TOPLEFT) // CoordOrigin.TopLeft
        posO.y = O_size.y - 1 - posO.y;

#ifdef EXACT
    float4 c = SampleTensorColor(posO.x, posO.y);
#else
    uint2 X_size = uint2(X_width, X_height);

    float2 p = (posO + 0.5f) / O_size * X_size - 0.5f;
    uint2 p_floor = floor(p);
    float2 p_frac = p - p_floor;

    uint2 clampMax = X_size - 1;
    uint2 p_lower = clamp(p_floor, uint2(0, 0), clampMax);
    uint2 p_upper = clamp(p_floor + 1, uint2(0, 0), clampMax);

    // Bilinear filter
    float4 c0 = (1 - p_frac.y) * SampleTensorColor(p_lower.x, p_lower.y) + p_frac.y * SampleTensorColor(p_lower.x, p_upper.y);
    float4 c1 = (1 - p_frac.y) * SampleTensorColor(p_upper.x, p_lower.y) + p_frac.y * SampleTensorColor(p_upper.x, p_upper.y);

    float4 c = (1 - p_frac.x) * c0 + p_frac.x * c1;
#endif
    return c;
}
