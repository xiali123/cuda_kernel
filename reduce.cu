#include <cuda_runtime.h>
#include <iostream>

using namespace std;

//普通版本，合并访存
template<typename T, int block_size>
__global__ void reduce_v1(const int rows, const int cols, T *input, T *output)
{
    const int blk = blockIdx.x;
    const int thd = threadIdx.x;
    __shared__ T shm[block_size];
    
    for(int cur_row = blk; cur_row < rows; cur_row += gridDim.x)
    {
        const int row_offset = cur_row * cols;
        T *row_ptr = input + row_offset;
        shm[thd] = static_cast<T>(0);
        for(int i = thd; i < cols; i += block_size)
            shm[thd] += row_ptr[i]; 
        
        for(int offset = block_size/2; offset > 0; offset /= 2)
        {
            if(thd < offset)
                shm[thd] += shm[thd+offset];
            __syncthread();
        }
        if(thd == 0) output[cur_row] = shm[0];
    }
}

//普通版本，合并访存+循环展开
template<typename T, int block_size>
__global__ void reduce_v2(const int rows, const int cols, T *input, T *output)
{
    const int blk = blockIdx.x;
    const int thd = threadIdx.x;
    __shared__ T shm[block_size];
    
    for(int cur_row = blk; cur_row < rows; cur_row += gridDim.x)
    {
        const int row_offset = cur_row * cols;
        T *row_ptr = input + row_offset;
        shm[thd] = static_cast<T>(0);
        #pragma unroll
        for(int i = thd; i < cols; i += block_size)
            shm[thd] += row_ptr[i]; 
        
        #pragma unroll
        for(int offset = block_size/2; offset > 0; offset /= 2)
        {
            if(thd < offset)
                shm[thd] += shm[thd+offset];
            __syncthread();
        }
        if(thd == 0) output[cur_row] = shm[0];
    }
}

//普通版本，合并访存+循环展开+向量化
template<typename T, int block_size>
__global__ void reduce_v3(const int rows, const int cols, T *input, T *output)
{
    const int blk = blockIdx.x;
    const int thd = threadIdx.x;
    __shared__ T shm[block_size];
    float4 *shm_ptr = reinterpret_cast<float4*>(shm);
    for(int cur_row = blk; cur_row < rows; cur_row += gridDim.x)
    {
        const int row_offset = cur_row * cols;
        T *row_ptr = input + row_offset;
        float4 *row_vector_ptr = reinterpret_cast<float4*>(row_ptr);
        shm_ptr[thd] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for(int i = thd; i < cols/4; i += block_size)
        {
            shm_ptr[thd].x += row_vector_ptr[i].x; 
            shm_ptr[thd].y += row_vector_ptr[i].y; 
            shm_ptr[thd].w += row_vector_ptr[i].w; 
            shm_ptr[thd].z += row_vector_ptr[i].z; 
        }
        
        #pragma unroll
        for(int i = thd+cols/4*4; i < cols; i += block_size)
        {
            shm[thd] += row_ptr[i]; 
        }
        
        #pragma unroll
        for(int offset = block_size/2; offset > 0; offset /= 2)
        {
            if(thd < offset)
            {
                shm_ptr[thd].x += shm_ptr[thd+offset].x; 
                shm_ptr[thd].y += shm_ptr[thd+offset].y; 
                shm_ptr[thd].w += shm_ptr[thd+offset].w; 
                shm_ptr[thd].z += shm_ptr[thd+offset].z;                
            }
                
            __syncthread();
        }
        if(thd == 0) output[cur_row] = shm_ptr[0].x + shm_ptr[0].y + shm_ptr[0].w + shm_ptr[0].z;
    }
}

//普通版本，合并访存+循环展开+shuffle
template<typename T, int block_size, int warps_per_block>
__global__ void reduce_v4(const int rows, const int cols, T *input, T *output)
{
    const int blk = blockIdx.x;
    const int thd = threadIdx.x;
    const int warp_id = thd / kWarpSize;
    const int lain_id = thd % kWarpSize;
    __shared__ T shm[warps_per_block];
    
    for(int cur_row = blk; cur_row < rows; cur_row += gridDim.x)
    {
        const int row_offset = cur_row * cols;
        T *row_ptr = input + row_offset;
        shm[thd] = static_cast<T>(0);
        #pragma unroll
        for(int i = thd; i < cols; i += block_size)
            shm[thd] += row_ptr[i]; 
        
        T sum_thread = shm[thd];

        sum_thread += __shfl_xor_sync(0xFFFFFFFF, sum_thread, 16);
        sum_thread += __shfl_xor_sync(0xFFFFFFFF, sum_thread, 8);
        sum_thread += __shfl_xor_sync(0xFFFFFFFF, sum_thread, 4);
        sum_thread += __shfl_xor_sync(0xFFFFFFFF, sum_thread, 2);
        sum_thread += __shfl_xor_sync(0xFFFFFFFF, sum_thread, 1);

        if(lain_id == 0) shm[warp_id] = sum_thread;

        T sum_block = static_cast<T>(0);
        if(thd == 0)
        {
            #pragma unroll
            for(int i = 0; i < warps_per_block; i++)
                sum_block += shm[i];

            output[cur_row] = sum_block;
        }
    }
}


//普通版本，合并访存+循环展开+shuffle+向量化
template<typename T, int block_size, int warps_per_block>
__global__ void reduce_v5(const int rows, const int cols, T *input, T *output)
{
    const int thd = threadIdx.x;
    const int warp_id = thd / kWarpSize;
    const int lain_id = thd % kWarpSize;
    __shared__ T shm[warps_per_block];
    float4 * const shm_ptr = reinterpret_cast<float4 * const>(shm);
    for(int cur_row = blockIdx.x; cur_row < rows; cur_row += gridDim.x)
    {
        const int row_offset = cur_row * cols;
        T *row_ptr = input + row_offset;
        const float4 *row_vec_ptr = reinterpret_cast<const float4*>(row_ptr);
        shm_ptr[thd] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for(int i = thd; i < cols/4; i += block_size)
        {
            shm_ptr[thd].x += row_vec_ptr[i].x; 
            shm_ptr[thd].y += row_vec_ptr[i].y; 
            shm_ptr[thd].w += row_vec_ptr[i].w; 
            shm_ptr[thd].z += row_vec_ptr[i].z; 
        }   
        #pragma unroll
        for(int i = thd+cols/4*4; i < cols; i += block_size)
            shm[thd] += row_ptr[i];

        float4 sum_thread = shm_ptr[thd];
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            float4 tmp_val = __shfl_xor_sync(0xFFFFFFFF, sum_thread, offset);
            sum_thread.x += tmp_val.x;
            sum_thread.y += tmp_val.y;
            sum_thread.z += tmp_val.z;
            sum_thread.w += tmp_val.w;    
        }
        if(lain_id == 0) shm_ptr[warp_id] = sum_thread;
        float4 sum_block = {0.0f, 0.0f, 0.0f, 0.0f};
        if(thd == 0)
        {
            #pragma unroll
            for(int i = 0; i < warps_per_block; i++)
            {
                sum_block.x += shm_ptr[i].x;
                sum_block.y += shm_ptr[i].y;
                sum_block.w += shm_ptr[i].w;
                sum_block.z += shm_ptr[i].z;
            } 
            output[cur_row] = sum_block.x + sum_block.y + sum_block.w + sum_block.z;
        }
    }
}

template<T>
struct SumOp
{
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

template<typname T, typname U, int page_size>
struct MultiFetch
{
    __device__ void operator()(T *dst, U *src)
    {
        #pragma unroll
        for(int i = 0; i < page_size; i++)
            dst[i] = static_cast<T>(src[i]);
    }
};

template<template<typename> typename ReduceOp, typename T>
__inline__ __device__ T ReduceWarpAll(T val)
{
    for(int offset = kWarpSize/2; offset > 0; offset /= 2)
    {
        val = ReduceOp<T>(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}


//循环展开+shuffle+buffer
template<typename T, int block_size, int warps_per_block, int cols_per_thread, int page_size>
__global__ void reduce_v6(const int rows, const int cols, T *input, T *output)
{
    assert(cols_per_thread % page_size == 0);
    assert(cols % page_size == 0)
    const int page_num = cols_per_thread / page_size;
    const int lain_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int blk = blockIdx.x;
    T buff[cols_per_thread];
    __shared__ T shm[warps_per_block];

    for(int cur_row = blk; cur_row < rows; cur_row += gridDim.x)
    {
        T *row_ptr = input + cur_row * N;
        T sum_thread = static_cast<T>(0);
        #pragma unroll
        for(int page_id = 0; page_id < page_num; page_id++)
        {
            const int cur_col = (page_id*kWarpSize + lain_id) * page_size;
            if(cur_col < cols)
            {
                MultiFetch<T, T, page_size>()(buff+page_id*page_size, row_ptr+cur_col);
                #pragma unroll
                for(int i = 0; i < page_size; i++) sum_thread += buff[i];
            }
        }

        const T sum_max = ReduceWarpAll<SumOp, T>(sum_thread);
        if(lain_id == 0) shm[warp_id] = sum_max;
        __syncthread();

        T sum_block = static_cast<T>(0);
        if(warp_id*kWarpSize+lain_id == 0) 
        {
            #pragma unroll
            for(int i = 0; i < warps_per_block; i++)
                sum_block += shm[i];

            output[cur_row] = sum_block;
        }
    }
}


template<typname T, typname U, int page_size>
struct MultiFetchVector
{
    __device__ void operator()(T *dst, U *src)
    {
        #pragma unroll
        for(int i = 0; i < page_size; i++)
        {
            dst[i].x = static_cast<T>(src[i].x);
            dst[i].y = static_cast<T>(src[i].y);
            dst[i].z = static_cast<T>(src[i].z);
            dst[i].w = static_cast<T>(src[i].w);
        }
    }
};

template<T>
struct SumOpVector
{
    __device__ T operator()(T a, T b) 
    {
        T tmp_va;
        tmp_val.x = a.x + b.x;
        tmp_val.y = a.y + b.y;
        tmp_val.w = a.w + b.w;
        tmp_val.z = a.z + b.z;
        return tmp_val;
    }
}

template<template<typename> typename ReduceOp, typename T>
__inline__ __device__ T ReduceWarpAllVector(T val)
{
    for(int offset = kWarpSize/2; offset > 0; offset /= 2)
    {
        T tmp_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = ReduceOp<T>(val, tmp_val);
    }
    return val;
}

//循环展开+shuffle+buffer+向量化
template<typename T, int block_size, int warps_per_block, int cols_per_thread, int page_size>
__global__ void reduce_v7(const int rows, const int cols, T *input, T *output)
{
    assert(cols_per_thread % page_size == 0);
    assert(cols % page_size == 0)
    const int page_num = cols_per_thread / page_size;
    const int lain_id = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int blk = blockIdx.x;
    float4 buff[cols_per_thread];
    __shared__ float4 shm[warps_per_block];
    
    const int vec_cols = (cols+4-1)/4;
    for(int cur_row = blk; cur_row < rows; cur_row += gridDim.x)
    {
        T *row_ptr = input + cur_row * N;
        float4 * row_vec_ptr = reinterpret_cast<float4 *const>(row_ptr);

        float4 sum_thread = {0.0, 0.0, 0.0, 0.0};
        #pragma unroll
        for(int page_id = 0; page_id < page_num; page_id++)
        {
            const int cur_col = (page_id*kWarpSize + lain_id) * page_size;
            if(cur_col < vec_cols)
            {
                MultiFetchVector<float4, float4, page_size>()(buff+page_id*page_size, row_vec_ptr+cur_col);
                #pragma unroll
                for(int i = 0; i < page_size; i++) 
                {
                    sum_thread.x += buff[i].x;
                    sum_thread.y += buff[i].y;
                    sum_thread.z += buff[i].z;
                    sum_thread.w += buff[i].w;
                }
            }
        }

        const float4 sum_max = ReduceWarpAllVector<SumOp, float4>(sum_thread);
        if(lain_id == 0) shm[warp_id] = sum_max;
        __syncthread();

        float4 sum_block = static_cast<float4>(0);
        if(warp_id*kWarpSize+lain_id == 0) 
        {
            #pragma unroll
            for(int i = 0; i < warps_per_block; i++)
            {
                sum_block.x += shm[i].x;
                sum_block.y += shm[i].y;
                sum_block.z += shm[i].z;
                sum_block.w += shm[i].w;
            }

            output[cur_row] = (sum_block.x + sum_block.y) + (sum_block.w + sum_block.z);
        }
    }
}


int main()
{

}
