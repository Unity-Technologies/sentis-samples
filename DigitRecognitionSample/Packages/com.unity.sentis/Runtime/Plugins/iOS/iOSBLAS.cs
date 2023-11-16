#if UNITY_IOS
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Scripting;

[assembly: AlwaysLinkAssembly]

namespace Unity.Sentis
{
    [Preserve]
    class iOSBLAS : BLASPlugin
    {
        [DllImport("__Internal")]
        static extern unsafe void ios_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
            int M, int N, int K, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc);

        public bool IsCurrentPlatformSupported()
        {
            return Application.platform == RuntimePlatform.IPhonePlayer;
        }

        public unsafe void SGEMM(int M, int N, int K, float* A, int lda, float* B, int ldb,
            float* C, int ldc, float beta, bool transposeA, bool transposeB)
        {
            ios_sgemm(CBLAS_ORDER.CblasRowMajor,
                transposeA ? CBLAS_TRANSPOSE.CblasTrans : CBLAS_TRANSPOSE.CblasNoTrans,
                transposeB ? CBLAS_TRANSPOSE.CblasTrans : CBLAS_TRANSPOSE.CblasNoTrans,
                M, N, K, 1.0f, A, lda, B, ldb, beta, C, ldc);
        }

        internal enum CBLAS_ORDER
        {
            CblasRowMajor=101,
            CblasColMajor=102
        };

        internal enum CBLAS_TRANSPOSE
        {
            CblasNoTrans=111,
            CblasTrans=112,
            CblasConjTrans=113,
            AtlasConj=114
        };
    }
}
#endif
