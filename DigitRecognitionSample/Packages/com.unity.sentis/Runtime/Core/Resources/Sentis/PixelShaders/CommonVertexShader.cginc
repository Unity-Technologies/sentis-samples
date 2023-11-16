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
