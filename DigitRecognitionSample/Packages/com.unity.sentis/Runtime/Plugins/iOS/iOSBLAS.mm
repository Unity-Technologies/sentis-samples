#import <Accelerate/Accelerate.h>

extern "C"
{
    void ios_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, int M, int N, int K,
                   float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
    {
        cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}
