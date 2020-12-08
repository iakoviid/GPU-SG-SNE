#include "sort.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#define MAX_BLOCK_SZ 128

__global__ void gpu_radix_sort_local(uint64_t* d_out_sorted,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    uint64_t* d_in,
    unsigned int d_in_len,
    unsigned int max_elems_per_block,
    unsigned int split,uint32_t* iPerm_out,uint32_t* iPerm_in,double* Y_out,double* Y_in,int d
  )
{
    // need shared memory array for:
    // - block's share of the input data (local sort will be put here too)
    // - mask outputs
    // - scanned mask outputs
    // - merged scaned mask outputs ("local prefix sum")
    // - local sums of scanned mask outputs
    // - scanned local sums of scanned mask outputs

    // for all radix combinations:
    //  build mask output for current radix combination
    //  scan mask ouput
    //  store needed value from current prefix sum array to merged prefix sum array
    //  store total sum of mask output (obtained from scan) to global block sum array
    // calculate local sorted address from local prefix sum and scanned mask output's total sums
    // shuffle input block according to calculated local sorted addresses
    // shuffle local prefix sums according to calculated local sorted addresses
    // copy locally sorted array back to global memory
    // copy local prefix sum array back to global memory

    extern __shared__ unsigned int shmem[];
    uint64_t* s_data = (uint64_t *)shmem;
    // s_mask_out[] will be scanned in place
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int* s_mask_out = (uint32_t *)&s_data[max_elems_per_block];
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[split];

    unsigned int thid = threadIdx.x;

    // Copy block's portion of global input data to shared memory
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;
    if (cpy_idx < d_in_len)
        s_data[thid] = d_in[cpy_idx];
    else
        s_data[thid] = 0;

    __syncthreads();

    // To extract the correct 2 bits, we first shift the number
    //  to the right until the correct 2 bits are in the 2 LSBs,
    //  then mask on the number with 11 (3) to remove the bits
    //  on the left
    uint64_t t_data = s_data[thid];
    unsigned int t_2bit_extract =(uint32_t) (t_data >> input_shift_width) & (split-1);

    for (unsigned int i = 0; i < split; ++i)
    {
        // Zero out s_mask_out
        s_mask_out[thid] = 0;
        if (thid == 0)
            s_mask_out[s_mask_out_len - 1] = 0;

        __syncthreads();

        // build bit mask output
        bool val_equals_i = false;
        if (cpy_idx < d_in_len)
        {
            val_equals_i = t_2bit_extract == i;
            s_mask_out[thid] = val_equals_i;
        }
        __syncthreads();

        // Scan mask outputs (Hillis-Steele)
        int partner = 0;
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int) log2f(max_elems_per_block);
        for (unsigned int d = 0; d < max_steps; d++) {
            partner = thid - (1 << d);
            if (partner >= 0) {
                sum = s_mask_out[thid] + s_mask_out[partner];
            }
            else {
                sum = s_mask_out[thid];
            }
            __syncthreads();
            s_mask_out[thid] = sum;
            __syncthreads();
        }

        // Shift elements to produce the same effect as exclusive scan
        unsigned int cpy_val = 0;
        cpy_val = s_mask_out[thid];
        __syncthreads();
        s_mask_out[thid + 1] = cpy_val;
        __syncthreads();

        if (thid == 0)
        {
            // Zero out first element to produce the same effect as exclusive scan
            s_mask_out[0] = 0;
            unsigned int total_sum = s_mask_out[s_mask_out_len - 1];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();

        if (val_equals_i && (cpy_idx < d_in_len))
        {
            s_merged_scan_mask_out[thid] = s_mask_out[thid];
        }

        __syncthreads();
    }

    // Scan mask output sums
    // Just do a naive scan since the array is really small
    if (thid == 0)
    {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < split; ++i)
        {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thid];
        unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[t_2bit_extract];

        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        s_data[new_pos] = t_data;
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;

        __syncthreads();

        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thid];
        d_out_sorted[cpy_idx] = s_data[thid];

        uint32_t g_new_pos=max_elems_per_block * blockIdx.x+new_pos;
        iPerm_out[g_new_pos]=iPerm_in[cpy_idx];
        for(int dim=0;dim<d;dim++){
          Y_out[g_new_pos+dim*d_in_len]=Y_in[cpy_idx];
        }

    }
}

__global__ void gpu_glbl_shuffle(uint64_t* d_out,
    uint64_t* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block,
    unsigned int split,uint32_t * iPerm_out,uint32_t *iPerm_in,double* Y_out,double* Y_in,int d
  )
{
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_out

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
    {
        uint64_t t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract =(uint32_t) (t_data >> input_shift_width) & (split-1);
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
            + t_prefix_sum;
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
        iPerm_out[data_glbl_pos]=iPerm_in[cpy_idx];
        for(int dim=0;dim<d;dim++){
          Y_out[data_glbl_pos+dim*d_in_len]=Y_in[cpy_idx];
        }
    }
}
__global__ void orderCheck(uint64_t* d,uint32_t n,uint32_t shift_width,uint32_t split,int* order){

    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID<n-1){
    uint32_t t_bit_extractcur =(uint32_t) (d[TID] >> shift_width) & (split-1);
    uint32_t t_bit_extractnext =(uint32_t) (d[TID+1] >> shift_width) & (split-1);
    if(t_bit_extractnext<t_bit_extractcur){
      order[blockIdx.x]=1;

    }
  }else{return ;}


}

__global__ void orderCheck3(uint64_t* d,uint32_t n,uint32_t shift_width,uint32_t split,int* order){

    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID<n-1){
    uint32_t t_bit_extractcur =(uint32_t) (d[TID] >> shift_width) & (split-1);
    uint32_t t_bit_extractnext =(uint32_t) (d[TID+1] >> shift_width) & (split-1);
    if(t_bit_extractnext<t_bit_extractcur){
      order[0]=1;

    }
  }else{return ;}


}
__global__ void orderCheck2(uint64_t* d,uint32_t n,uint32_t shift_width,uint32_t split,int* order){
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;

    if(threadIdx.x==0){
    order[TID]=0;}
    if (TID<n-1){
    uint32_t t_bit_extractcur =(uint32_t) (d[TID] >> shift_width) & (split-1);
    uint32_t t_bit_extractnext =(uint32_t) (d[TID+1] >> shift_width) & (split-1);
    if(t_bit_extractnext<t_bit_extractcur){
      order[TID]=1;

    }
  }else{return ;}


}
// An attempt at the gpu radix sort variant described in this paper:
// https://vgc.poly.edu/~csilva/papers/cgf.pdf
void radix_sort(uint64_t* const d_out,
    uint64_t* const d_in,
    unsigned int d_in_len,uint32_t bitStride,uint32_t total,uint32_t* iPerm_out,uint32_t* iPerm_in,double* Y_out,double* Y_in,int d)

{
    unsigned int split=1<<bitStride;
    unsigned int block_sz = MAX_BLOCK_SZ;
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = d_in_len / max_elems_per_block; //length /blocks
    // Take advantage of the fact that integer division drops the decimals
    if (d_in_len % max_elems_per_block != 0) //ciel
        grid_sz += 1;
    //prefix sum for all the array
    unsigned int* d_prefix_sums;
    unsigned int d_prefix_sums_len = d_in_len;
    checkCudaErrors(cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_prefix_sums_len));
    checkCudaErrors(cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len));

    unsigned int* d_block_sums;
    unsigned int d_block_sums_len = split * grid_sz; // split-way split
    checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len));
    checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));

    unsigned int* d_scan_block_sums;
    checkCudaErrors(cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len));
    checkCudaErrors(cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len));

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    unsigned int s_data_len = max_elems_per_block;
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
    unsigned int s_mask_out_sums_len = split; // 4-way split
    unsigned int s_scan_mask_out_sums_len = split;
    unsigned int shmem_sz = s_data_len*sizeof(uint64_t)+
                            +( s_mask_out_len
                            + s_merged_scan_mask_out_len
                            + s_mask_out_sums_len
                            + s_scan_mask_out_sums_len)
                            * sizeof(unsigned int);

    int* order;
    int* order_h=new int[1];

    checkCudaErrors(cudaMalloc(&order, grid_sz*sizeof(int) ));
    checkCudaErrors(cudaMemset(order, 0, grid_sz*sizeof( int) ));

    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= total; shift_width += bitStride)
    {


        orderCheck<<<grid_sz,block_sz>>>(d_in,d_in_len,shift_width,split,order);
        int outOfOrder=thrust::reduce(thrust::device, order, order + grid_sz, 0);
        //checkCudaErrors(cudaMemcpy(order_h, order, sizeof(bool) , cudaMemcpyDeviceToHost));

        if(outOfOrder>0){

        //cudaDeviceSynchronize();

        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out,
                                                                d_prefix_sums,
                                                                d_block_sums,
                                                                shift_width,
                                                                d_in,
                                                                d_in_len,
                                                                max_elems_per_block,split,iPerm_out,iPerm_in, Y_out, Y_in,d);

        //unsigned int* h_test = new unsigned int[d_in_len];
        //checkCudaErrors(cudaMemcpy(h_test, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToHost));
        //for (unsigned int i = 0; i < d_in_len; ++i)
        //    std::cout << h_test[i] << " ";
        //std::cout << std::endl;
        //delete[] h_test;

        // scan global block sum array
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

        // scatter/shuffle block-wise sorted array to final positions
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in,
                                                    d_out,
                                                    d_scan_block_sums,
                                                    d_prefix_sums,
                                                    shift_width,
                                                    d_in_len,
                                                    max_elems_per_block,split,iPerm_in,iPerm_out,Y_in,Y_out,d);

    }else{
      printf("skipped\n" );
    }
  }
    checkCudaErrors(cudaMemcpy(d_out, d_in, sizeof(uint64_t) * d_in_len, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(iPerm_out, iPerm_in, sizeof(uint32_t) * d_in_len, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(Y_out, Y_in, sizeof(double) * d*d_in_len, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaFree(d_scan_block_sums));
    checkCudaErrors(cudaFree(d_block_sums));
    checkCudaErrors(cudaFree(d_prefix_sums));
}
