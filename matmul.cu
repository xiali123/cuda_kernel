#include <cuda_runtime.h>
#include <iostream>

using namespace std;


//最原始的矩阵乘法，一个线程计算出一个Cij元素+循环展开
template<typename T>
__global__ void matmul_v0(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int idxA = threadIdx.x + blockIdx.x * blockDim.x;
    const int idxB = threadIdx.y + blockIdx.y * blockDim.y;

    idxA_offset = idxA * K;
    idxB_offset = idxB;
    if(idxA < M && idxB < N)
    {
        #pragma unroll
        for(int i = 0; i < K; i++)
            tmp += A[idxA_offset+i]*B[i*N+idxB_offset];

        C[idxA*N + idxB] = tmp;
    }
}

//使用共享内存，减少访存次数+循环展开
template<typename T, int BLOCK_DIM, int BK>
__global__ void matmul_v2(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ T SA[BLOCK_DIM][BLOCK_DIM];
    __shared__ T SB[BLOCK_DIM][BLOCK_DIM];
    const int width = (K+BK-1) / BK;
    T tmp = 0.0f;

    for(int ph = 0; ph < width; ph++)
    {
        if(row < M && threadIdx.x+ph*BK < K) SA[threadIdx.y][threadIdx.x] = A[row*K+threadIdx.x+ph*BK];
        else SA[threadIdx.y][threadIdx.x] = 0.0f;
        
        if(col < N && threadIdx.y+ph*BK < K) SB[threadIdx.y][threadIdx.x] = B[(threadIdx.y+ph*BK)*N+col];
        else SB[threadIdx.y][threadIdx.x] = 0.0f;

        #pragma unroll
        for(int i = 0; i < BLOCK_DIM; i++)
            tmp += SA[threadIdx.y][i] * SB[i][threadIdx.x];

        __syncthread();
    }

    if(row < M && col < N)
        C[row*N+col] = tmp;
}

//使用共享内存，减少访存次数+优化bank冲突+循环展开
template<typename T, int BLOCK_DIM, int BK>
__global__ void matmul_v3(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ T SA[BLOCK_DIM][BLOCK_DIM+1];
    __shared__ T SB[BLOCK_DIM][BLOCK_DIM+1];
    const int width = (K+BK-1) / BK;
    T tmp = 0.0f;

    for(int ph = 0; ph < width; ph++)
    {
        if(row < M && threadIdx.x+ph*BK < K) SA[threadIdx.y][threadIdx.x] = A[row*K+threadIdx.x+ph*BK];
        else SA[threadIdx.y][threadIdx.x] = 0.0f;
        
        if(col < N && threadIdx.y+ph*BK < K) SB[threadIdx.y][threadIdx.x] = B[(threadIdx.y+ph*BK)*N+col];
        else SB[threadIdx.y][threadIdx.x] = 0.0f;

        #pragma unroll
        for(int i = 0; i < BLOCK_DIM; i++)
            tmp += SA[threadIdx.y][i] * SB[i][threadIdx.x];

        __syncthread();
    }

    if(row < M && col < N)
        C[row*N+col] = tmp;
}


//使用共享内存，减少访存次数+寄存器buffer+一个线程计算多个结果+循环展开
template<typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v3(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int col = TN * (threadIdx.x + blockIdx.x * blockDim.x);
    const int row = TM * (threadIdx.y + blockIdx.y * blockDim.y);

    __shared__ T SA[BM*BK];
    __shared__ T SB[BK*BN];
    const int width = (K+BK-1) / BK;
    T tmp[TM][TN] = {0.0f};

    for(int ph = 0; ph < width; ph++)
    {
        for(int i = 0; i < TM; i++)
        {
            for(int j = 0; j < BK; j++)
            {
                if(row+threadIdx.y*TM+i < M && j+ph*BK < N) SA[(threadIdx.y*TM+i)*BK+j] = A[(row+threadIdx.y*TM+i)*K+j+ph*BK];
                else SA[(threadIdx.y*TM+i)*BK+j] = 0.0f;
            }
        }
        __syncthread();
        for(int i = 0; i < TN; i++)
        {
            for(int j = 0; j < BK; j++)
            {
                if(j < K && j+ph*BK < N) SB[j*BN+threadIdx.x*TN+i] = A[(j+ph*BK)*K+threadIdx.x*TN+i];
                else SB[j*BN+threadIdx.x*TN+i] = 0.0f;
            }
        }
        __syncthread();
        for(int i = 0; i < TM; i++)
        {
            for(int j = 0; j < TN; j++)
            {
                #pragma unroll
                for(int k = 0; k < BK; k++)
                    tmp[i][j] += SA[(threadIdx.y*TM+i)*BK+k] * SB[j*BN+threadIdx.x*TN+k];
            }
        }
        __syncthread();
    }

    #pragma unroll
    for(int i = 0; i < TM; i++)
    {
        #pragma unroll
        for(int j = 0; j < TN; j++)
        {
            if(row+i < M && col+j < N)
                C[(row+i)*N+col+j] = tmp[i][j];
        }
    }
}

//使用共享内存，减少访存次数+寄存器buffer+一个线程计算多个结果+优化线程利用率+循环展开
template<typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v4(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int col = TN * (blockIdx.x * blockDim.x);
    const int row = TM * (blockIdx.y * blockDim.y);

    __shared__ T SA[BM*BK];
    __shared__ T SB[BK*BN];

    int tmp[TM*TN] = {0.0f};
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;

    const int A_warp_id = tid / BK;
    const int A_lain_id = tid % BK;
    const int B_warp_id = tid / BN;
    const int B_lain_id = tid % BN;
    const int width = (K + BK - 1) / BK;

    for(int ph = 0; ph < width; ph++)
    {
        if(row+A_warp_id < M && A_lain_id+ph*BK < N) SA[A_warp_id*BK+A_lain_id] = A[(row+A_warp_id)*K+A_lain_id+ph*BK];
        else SA[A_warp_id*BK+A_lain_id] = 0.0f;

        if(row+A_warp_id < M && A_lain_id+ph*BK < N) SB[B_warp_id*BN+B_lain_id] = B[(B_warp_id+ph*width)*N+B_lain_id+col];
        else SB[B_warp_id*BN+B_lain_id] = 0.0f;

        for(int i = 0; i < TM; i++)
        {
            for(int j = 0; j < TN; j++)
            {
                int SA_pos = threadIdx.y * TM + i;
                int SB_pos = threadIdx.x * TN + j;
                #pragma unroll
                for(int k = 0; k < BK; k++)
                    tmp[i*TN+j] += SA[SA_pos*BK+k] * SB[k*BN+SB_pos];
            }
        }

        __syncthread();
    }
    
    #pragma unroll
    for(int i = 0; i < TM; i++)
    {
        #pragma unroll
        for(int j = 0; j < TN; j++)
        {
            const int row_pos = row+threadIdx.y * TM + i;
            const int col_pos = col+threadIdx.x * TN + j;
            if(row_pos < M && col_pos < N)
                C[row_pos*K+col_pos] = tmp[i*TN+j];
        }
    }
}


//使用共享内存，减少访存次数+寄存器buffer+一个线程计算多个结果+优化线程利用率+向量化访存+循环展开
template<typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v5(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int row = TM * (blockIdx.y * blockDim.y);
    const int col = TN * (blockIdx.x * blockDim.x);

    __shared__ int SA[BM * BK];
    __shared__ int SB[BK * BN];

    T tmp[TM*TN] = {0.0f};
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int width = (K+BK-1) / BK;
    const int A_warp_id = tid / (BK/4);
    const int A_lain_id = tid % (BK/4);
    const int B_warp_id = tid / (BN/4);
    const int B_lain_id = tid % (BN/4);

    for(int ph = 0; ph < width; ph++)
    {
        (float4 &)SA[A_warp_id*BK+4*A_lain_id] = (float4 &)A[(row+A_warp_id)*K+4*A_lain_id+ph*BK];
        (float4 &)SA[B_warp_id*BN+4*B_lain_id] = (float4 &)B[(ph*BK+B_warp_id)*N+4*B_lain_id+col];

        #pragma unroll
        for(int id = 0; id < 4; id++)
        {
            if(row+A_warp_id >= M || ph*BK+4*A_lain_id+id >= K) 
            {
                SA[A_warp_id*BK+4*A_lain_id+id] = 0.0f;
            }

            if(ph*BK+B_warp_id >= M || 4*B_lain_id+col >= N)
            {
                SA[B_warp_id*BN+4*B_lain_id+id] = 0.0f;
            } 
        }

        __syncthread();

        for(int i = 0; i < TM; i++)
        {
            for(int j = 0; j < TN; j++)
            {
                int SA_pos = threadIdx.y * TM + i;
                int SB_pos = threadIdx.x * TN + j;
                #pragma unroll
                for(int k = 0; k < BK; k++)
                    tmp[i*TN+j] += SA[SA_pos*BK+k] * SB[k*BN+SB_pos];
            }
        }
    }

    #pragma unroll
    for(int i = 0; i < TM; i++)
    {
        #pragma unroll
        for(int j = 0; j < TN; j++)
        {
            int c_row = row + threadIdx.y * TM + i;
            int c_col = col + threadIdx.x * TN + j;
            if(c_row < M && c_col < N)
                C[c_row*N+c_col] = tmp[i*TN+j];
        }
    }
}

//使用共享内存，减少访存次数+寄存器buffer+一个线程计算多个结果+优化线程利用率+向量化访存+减少边界访存+循环展开
template<typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v6(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int row = TM * (blockIdx.y * blockDim.y);
    const int col = TN * (blockIdx.x * blockDim.x);

    __shared__ int SA[BM * BK];
    __shared__ int SB[BK * BN];

    T tmp[TM*TN] = {0.0f};
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int width = (K+BK-1) / BK;
    const int A_warp_id = tid / (BK/4);
    const int A_lain_id = tid % (BK/4);
    const int B_warp_id = tid / (BN/4);
    const int B_lain_id = tid % (BN/4);
    float a[4];
    for(int ph = 0; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)A[(row+A_warp_id)*K+4*A_lain_id+ph*BK];
        #pragma unroll
        for(int id = 0; id < 4; id++)
        {
            if(row+A_warp_id >= M || ph*BK+4*A_lain_id+id >= K) SA[A_warp_id*BK+4*A_lain_id+id] = 0.0f;
            else SA[A_warp_id*BK+4*A_lain_id+id] = a[i];
        }

        (float4 &)a[0] = (float4 &)B[(ph*BK+B_warp_id)*N+4*B_lain_id+col];
        #pragma unroll
        for(int id = 0; id < 4; id++)
        {
            if(ph*BK+B_warp_id >= M || 4*B_lain_id+col >= N) SB[B_warp_id*BN+4*B_lain_id+id] = 0.0f;
            else SB[B_warp_id*BN+4*B_lain_id+id] = a[i];
        }

        __syncthread();

        for(int i = 0; i < TM; i++)
        {
            for(int j = 0; j < TN; j++)
            {
                int SA_pos = threadIdx.y * TM + i;
                int SB_pos = threadIdx.x * TN + j;
                #pragma unroll
                for(int k = 0; k < BK; k++)
                    tmp[i*TN+j] += SA[SA_pos*BK+k] * SB[k*BN+SB_pos];
            }
        }
    }

    #pragma unroll
    for(int i = 0; i < TM; i++)
    {
        #pragma unroll
        for(int j = 0; j < TN; j++)
        {
            int c_row = row + threadIdx.y * TM + i;
            int c_col = col + threadIdx.x * TN + j;
            if(c_row < M && c_col < N)
                C[c_row*N+c_col] = tmp[i*TN+j];
        }
    }
}

//使用共享内存，减少访存次数+寄存器buffer+一个线程计算多个结果+优化线程利用率+向量化访存+减少边界访存+循环展开+寄存器buffer
template<typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v7(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int row = TM * (blockIdx.y * blockDim.y);
    const int col = TN * (blockIdx.x * blockDim.x);
    
    __shared__ T SA[BM * BK];
    __shared__ T SB[BK * BN];

    const int width = (K + BK - 1) / BK;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int A_warp_id = tid / (BK/4);
    const int A_lain_id = tid % (BK/4);
    const int B_warp_id = tid / (BN/4);
    const int B_warp_id = tid % (BN/4);
    T tmp[TM * TN] = {0.0f};
    float a[4];
    float comm_a[TM];
    float comm_b[TN];

    for(int ph = 0; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)A[(row+A_warp_id)*K+4*A_lain_id+ph*BK];
        for(int id = 0; id < 4; id++)
        {
            if(row+A_warp_id >= M || ph*BK+4*A_lain_id+id >= N) SA[(A_lain_id*4+id)*BM+A_wapr_id] = 0.0f;
            else SA[(A_lain_id*4+id)*BM+A_wapr_id] = a[id];
        }

        (float4 &)a[0] = (float4 &)B[(ph*BK+B_warp_id)*N+4*B_lain_id+col];
        for(int id = 0; id < 4; id++)
        {
            if(ph*BK+B_warp_id >= M || 4*B_lain_id+col+id >= N) SB[B_warp_id*BN+4*B_lain_id+id] = 0.0f;
            else SB[B_warp_id*BN+4*B_lain_id+id] = = a[id];
        }
        __syncthread();
        
        #pragma unroll
        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int i = 0; i < TM; i += 4)
                (float4 &)comm_a[i] = (float4 &)SA[k*BM+threadIdx.y*TM+i*4];
            
            #pragma unroll
            for(int i = 0; i < TN; i += 4)
                (float4 &)comm_b[i] = (float4 &)SB[k*BN+threadIdx.x*TN+i*4];
            
            #pragma unroll
            for(int i = 0; i < TM; i++)
            {
                #pragma unroll
                for(int j = 0; j < TN; j++)
                    tmp[i*TN+j] += comm_a[i] * comm_b[j];
            }
        }
        __syncthread();
    }

    #pragma unroll
    for(int i = 0; i < TM; i++)
    {
        #pragma unroll
        for(int j = 0; j < TN; j++)
        {
            int row_pos = row + threadIdx.y * TM + i;
            int col_pos = col + threadIdx.x * TN + j;
            if(row_pos < M && col_pos < N)
                C[row_pos*K+col_pos] = tmp[i*TN+j];
        }
    }
}


//使用共享内存，减少访存次数+寄存器buffer+一个线程计算多个结果+优化线程利用率+向量化访存+减少边界访存+循环展开+寄存器buffer+共享内存双缓冲
template<typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v8(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int row = TM * (threadIdx.y * blockDim.y);
    const int col = TN * (threadIdx.x * blockDim.x);

    __shared__ T SA[BM*BK*2];
    __shared__ T SB[BK*BN*2];

    const int tid = threadIdx.x+threadIdx.y*blockDim.x;
    const int width = (K+BK-1)/BK;
    
    const int A_warp_id = tid / (BK/4);
    const int A_lain_id = tid % (BK/4);
    const int B_warp_id = tid / (BN/4);
    const int B_warp_id = tid % (BN/4);

    float a[4];
    float comm_a[TM];
    float comm_b[TN];
    T tmp[TM*TN] = {0.0f};


    (float4 &)a[0] = (float4 &)A[(row+A_warp_id)*K+A_lain_id*4];
    #pragma unroll
    for(int id = 0; id < 4; id++)
    {
        if(row+A_warp_id < M && A_lain_id+id < N) SA[(A_lain_id*4+id)*BM+A_warp_id] = a[id];
        else SA[(A_lain_id*4+id)*BM+A_warp_id] = 0.0f;
    }

    (float4 &)a[0] = (float4 &)B[(B_warp_id)*N+B_lain_id*4+col];
    #pragma unroll
    for(int id = 0; id < 4; id++)
    {
        if(B_warp_id < K && B_lain_id+col+id < N) SB[B_warp_id*BN+B_lain_id*4+id] = a[id];
        else SB[B_warp_id*BN+B_lain_id*4+id] = 0.0f;
    }    
    

    for(int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)A[(row+A_warp_id)*K+A_lain_id*4+ph*BK];
        #pragma unroll
        for(int id = 0; id < 4; id++)
        {
            if(row+A_warp_id < M && A_lain_id+id < N) SA[(A_lain_id*4+id)*BM+A_warp_id+ph%2*BM*BK] = a[id];
            else SA[(A_lain_id*4+id)*BM+A_warp_id+ph%2*BM*BK] = 0.0f;
        }
    
        (float4 &)a[0] = (float4 &)B[(B_warp_id+ph*BK)*N+B_lain_id*4+col];
        #pragma unroll
        for(int id = 0; id < 4; id++)
        {
            if(B_warp_id < K && B_lain_id+col+id < N) SB[B_warp_id*BN+B_lain_id*4+id+ph%2*BM*BK] = a[id];
            else SB[B_warp_id*BN+B_lain_id*4+id+ph%2*BM*BK] = 0.0f;
        }    

        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int i = 0; i < TM; i += 4)
                (float4 &)comm_a[i] = (float4 &)SA[k*BM+threadIdx.y*TM+4*i+(ph-1)%2*BM*BK];
            
            #pragma unroll
            for(int i = 0; i < TN; i += 4)
                (float4 &)comm_b[i] = (float4 &)SB[k*BN+threadIdx.x*TN+4*i+(ph-1)%2*BK*BN]; 
            
            #pragma unroll
            for(int i = 0; i < TM; i++)
            {
                #pragma unroll
                for(int j = 0; j < TN; j++)
                {
                    tmp[i*TN+j] += comm_a[i] * comm_b[j];
                }
            }
        }

        __syncthread();
    }

    ph = width;
    for(int k = 0; k < BK; k++)
    {
        #pragma unroll
        for(int i = 0; i < TM; i += 4)
            (float4 &)comm_a[i] = (float4 &)SA[k*BM+threadIdx.y*TM+4*i+(ph-1)%2*BM*BK];
        
        #pragma unroll
        for(int i = 0; i < TN; i += 4)
            (float4 &)comm_b[i] = (float4 &)SB[k*BN+threadIdx.x*TN+4*i+(ph-1)%2*BK*BN]; 
        
        #pragma unroll
        for(int i = 0; i < TM; i++)
        {
            #pragma unroll
            for(int j = 0; j < TN; j++)
            {
                tmp[i*TN+j] += comm_a[i] * comm_b[j];
            }
        }
    }
    __syncthread();

    for(int i = 0; i < TM; i++)
    {
        for(int j = 0; j < TN; j++)
        {
            int row_pos = row + threadIdx.y * TM + i;
            int col_pos = col + threadIdx.x * TN + j;
            if(row_pos < M && col_pos < N)
                C[row_pos*N+col_pos] = tmp[i*TN+j];
        }
    }
}


//使用共享内存，减少访存次数+寄存器buffer+一个线程计算多个结果+优化线程利用率+向量化访存+减少边界访存+循环展开+寄存器buffer+共享内存双缓冲优化
template<typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_v9(const T *A, const T *B, T *C, const int M, const int K, const int N)
{
    const int row = TM * (threadIdx.y * blockDim.y);
    const int col = TN * (threadIdx.x * blockDim.x);
    __shared__ T SA[BM*BK*2];
    __shared__ T SB[BK*BN*2];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int width = (K + BK - 1) / BK;
    const int A_warp_id = tid / (BK/4);
    const int A_lain_id = tid % (BK/4);
    const int B_warp_id = tid / (BK/4);
    const int B_warp_id = tid % (BK/4);
    float b[4];
    float a[4];
    float comm_a[TM];
    float comm_b[TN];
    T tmp[TM*TN] = {0.0f};

    int ph = 0;
    (float4 &)a[0] = (float4 &)A[(row+A_warp_id)*K+A_lain_id*4];
    SA[(A_lain_id*4+0)*BM+A_warp_id+ph%2*BM*BK] = a[0];
    SA[(A_lain_id*4+1)*BM+A_warp_id+ph%2*BM*BK] = a[1];
    SA[(A_lain_id*4+2)*BM+A_warp_id+ph%2*BM*BK] = a[2];
    SA[(A_lain_id*4+3)*BM+A_warp_id+ph%2*BM*BK] = a[3];

    (float4 &)b[0] = (float4 &)B[(B_warp_id)*N+B_lain_id*4+col];
    (float4 &)SB[B_warp_id*BN+B_lain_id*4+ph%2*BN*BK] = (float4 &)b[0];
    __syncthread();

    for(int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)A[(row+A_warp_id)*K+A_lain_id*4+ph*BK];
        (float4 &)b[0] = (float4 &)B[(B_warp_id+ph*BK)*N+B_lain_id*4+col];
        for(int k = 0; k < BK; k++)
        {
            #pragma unroll
            for(int i = 0; i < TM; i += 4)
                (float4 &)comm_a[i] = (float4 &)SA[k*BM+threadIdx.y*TM+4*i+(ph-1)%2*BM*BK];
            
            #pragma unroll
            for(int i = 0; i < TN; i += 4)
                (float4 &)comm_b[i] = (float4 &)SB[k*BN+threadIdx.x*TN+4*i+(ph-1)%2*BK*BN]; 
            
            #pragma unroll
            for(int i = 0; i < TM; i++)
            {
                #pragma unroll
                for(int j = 0; j < TN; j++)
                    tmp[i*TN+j] += comm_a[i] * comm_b[j];
            }
        }

        SA[(A_lain_id*4+0)*BM+A_warp_id+ph%2*BM*BK] = a[0];
        SA[(A_lain_id*4+1)*BM+A_warp_id+ph%2*BM*BK] = a[1];
        SA[(A_lain_id*4+2)*BM+A_warp_id+ph%2*BM*BK] = a[2];
        SA[(A_lain_id*4+3)*BM+A_warp_id+ph%2*BM*BK] = a[3];
        (float4 &)SB[B_warp_id*BN+B_lain_id*4+ph%2*BN*BK] = (float4 &)b[0];
        __syncthread();
    }

    ph = width;
    for(int k = 0; k < BK; k++)
    {
        #pragma unroll
        for(int i = 0; i < TM; i += 4)
            (float4 &)comm_a[i] = (float4 &)SA[k*BM+threadIdx.y*TM+4*i+(ph-1)%2*BM*BK];
        
        #pragma unroll
        for(int i = 0; i < TN; i += 4)
            (float4 &)comm_b[i] = (float4 &)SB[k*BN+threadIdx.x*TN+4*i+(ph-1)%2*BK*BN]; 
        
        #pragma unroll
        for(int i = 0; i < TM; i++)
        {
            #pragma unroll
            for(int j = 0; j < TN; j++)
                tmp[i*TN+j] += comm_a[i] * comm_b[j];
        }
    }
    __syncthread();

    for(int i = 0; i < TM; i++)
    {
        for(int j = 0; j < TN; j++)
        {
            int row_pos = row + threadIdx.y * TM + i;
            int col_pos = col + threadIdx.x * TN + j;
            if(row_pos < M && col_pos < N)
                C[row_pos*N+col_pos] = tmp[i*TN+j];
        }
    }
}



int main()
{
    return 0;
}
