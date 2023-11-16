uint WidthShiftO, LengthO;

#define DECLARE_TENSOR(X, DTYPE) Texture2D<DTYPE##4> X##ptr; \
uint WidthShift##X, WidthMask##X; \
DTYPE##4 SampleBlock##X(uint blockIndex) { \
return SampleBlock(X##ptr, WidthMask##X, WidthShift##X, blockIndex); \
} \
DTYPE##4 SampleElements##X(uint4 blockIndex4, uint4 c4) { \
return SampleElements(X##ptr, WidthMask##X, WidthShift##X, blockIndex4, c4); \
} \

#define DECLARE_TENSOR_BLOCK_STRIDE_O uint StrideAxisO, DimAxisO, DimBlockedO; \
uint3 UnravelO(uint blockIndex) { \
uint3 index; \
index[0] = blockIndex % StrideAxisO; \
blockIndex /= StrideAxisO; \
index[1] = blockIndex % DimBlockedO; \
blockIndex /= DimBlockedO; \
index[2] = blockIndex; \
return index; \
} \
uint4 GetIndexO(UNITY_VPOS_TYPE screenPos) { \
uint blockIndex = GetBlockIndexO(screenPos); \
uint lower = blockIndex % StrideAxisO; \
blockIndex /= StrideAxisO; \
uint axis = blockIndex % DimBlockedO; \
blockIndex /= DimBlockedO; \
uint upper = blockIndex; \
return lower + StrideAxisO * ((axis << 2) + uint4(0, 1, 2, 3) + DimAxisO * upper); \
} \

#define DECLARE_TENSOR_BLOCK_STRIDE(X) uint StrideAxis##X, DimAxis##X, DimBlocked##X; \
float4 SampleElements##X(uint4 index) { \
uint4 lower = index % StrideAxis##X; \
index /= StrideAxis##X; \
uint4 axis = index % DimAxis##X; \
index /= DimAxis##X; \
uint4 upper = index; \
uint4 axisDiv4 = axis >> 2; \
uint4 axisMod4 = axis & 3; \
uint4 blockIndex4 = lower + StrideAxis##X * (axisDiv4 + DimBlocked##X * upper); \
return SampleElements(X##ptr, WidthMask##X, WidthShift##X, blockIndex4, axisMod4); \
} \
float4 SampleBlock##X(uint lower, uint axis, uint upper) { \
uint4 axisDiv4 = axis >> 2; \
uint4 axisMod4 = axis & 3; \
uint4 blockIndex4 = lower + StrideAxis##X * (axisDiv4 + DimBlocked##X * upper); \
return SampleElements(X##ptr, WidthMask##X, WidthShift##X, blockIndex4, axisMod4); \
} \
float4 SampleElement##X(uint index) { \
uint lower = index % StrideAxis##X; \
index /= StrideAxis##X; \
uint axis = index % DimAxis##X; \
index /= DimAxis##X; \
uint upper = index; \
uint axisDiv4 = axis >> 2; \
uint axisMod4 = axis & 3; \
uint blockIndex = lower + StrideAxis##X * (axisDiv4 + DimBlocked##X * upper); \
return SampleElement(X##ptr, WidthMask##X, WidthShift##X, blockIndex, axisMod4); \
} \

inline float4 SampleBlock(Texture2D ptr, uint widthMask, uint widthShift, uint blockIndex)
{
    return ptr.Load(uint3(blockIndex & widthMask, blockIndex >> widthShift, 0));
}

inline float SampleElement(Texture2D ptr, uint widthMask, uint widthShift, uint blockIndex, uint c)
{
    uint x = blockIndex & widthMask;
    uint y = blockIndex >> widthShift;
    return ptr.Load(uint3(x, y, 0))[c];
}

inline float4 SampleElements(Texture2D ptr, uint widthMask, uint widthShift, uint4 blockIndex4, uint4 c4)
{
    float4 v = 0;
    uint4 x4 = blockIndex4 & widthMask;
    uint4 y4 = blockIndex4 >> widthShift;
    v.x = ptr.Load(uint3(x4.x, y4.x, 0))[c4.x];
    v.y = ptr.Load(uint3(x4.y, y4.y, 0))[c4.y];
    v.z = ptr.Load(uint3(x4.z, y4.z, 0))[c4.z];
    v.w = ptr.Load(uint3(x4.w, y4.w, 0))[c4.w];
    return v;
}

inline int4 SampleBlock(Texture2D<int4> ptr, uint widthMask, uint widthShift, uint blockIndex)
{
    return ptr.Load(uint3(blockIndex & widthMask, blockIndex >> widthShift, 0));
}

inline int SampleElement(Texture2D<int4> ptr, uint widthMask, uint widthShift, uint blockIndex, uint c)
{
    uint x = blockIndex & widthMask;
    uint y = blockIndex >> widthShift;
    return ptr.Load(uint3(x, y, 0))[c];
}

inline int4 SampleElements(Texture2D<int4> ptr, uint widthMask, uint widthShift, uint4 blockIndex4, uint4 c4)
{
    int4 v = 0;
    uint4 x4 = blockIndex4 & widthMask;
    uint4 y4 = blockIndex4 >> widthShift;
    v.x = ptr.Load(uint3(x4.x, y4.x, 0))[c4.x];
    v.y = ptr.Load(uint3(x4.y, y4.y, 0))[c4.y];
    v.z = ptr.Load(uint3(x4.z, y4.z, 0))[c4.z];
    v.w = ptr.Load(uint3(x4.w, y4.w, 0))[c4.w];
    return v;
}

inline int GetBlockIndexO(UNITY_VPOS_TYPE screenPos)
{
    uint2 tid = (uint2)(screenPos.xy - 0.5f);
    return (tid.y << WidthShiftO) + tid.x;
}

inline uint Ravel(uint1 shape, uint2 index)
{
    return index[0] + shape[0] * index[1];
}

inline uint Ravel(uint2 shape, uint3 index)
{
    return index[0] + shape[0] * (index[1] + shape[1] * index[2]);
}

inline uint Ravel(uint3 shape, uint4 index)
{
    return index[0] + shape[0] * (index[1] + shape[1] * (index[2] + shape[2] * index[3]));
}

inline uint2 Unravel(uint1 shape, uint i)
{
    uint2 index;
    index[0] = i % shape[0];
    i /= shape[0];
    index[1] = i;
    return index;
}

inline uint3 Unravel(uint2 shape, uint i)
{
    uint3 index;
    index[0] = i % shape[0];
    i /= shape[0];
    index[1] = i % shape[1];
    i /= shape[1];
    index[2] = i;
    return index;
}

inline uint4 Unravel(uint3 shape, uint i)
{
    uint4 index;
    index[0] = i % shape[0];
    i /= shape[0];
    index[1] = i % shape[1];
    i /= shape[1];
    index[2] = i % shape[2];
    i /= shape[2];
    index[3] = i;
    return index;
}

inline uint4 UnblockAxis(uint axisDiv4)
{
    return (axisDiv4 << 2) + uint4(0, 1, 2, 3);
}
