/*  KRISHNA PRASAD
 *  PORANDLA
 *  KPORANDL
 */

#ifndef A3_HPP
#define A3_HPP

#include <cuda_runtime_api.h>

//Declaring block size as per device config
const int BLOCK_SIZE = 1024;

//Function which can be invoked on device which is used during computation
__device__ float K_cal(float x){
    return (1/sqrt(44.0/7))*exp(-((x*x)/2));
}

//Kernel function to be excuted by each thread on GPU
__global__ void compute(int n, float h, float* d_x, float* d_y){
    //Current position in x
    int ti = blockIdx.x * blockDim.x + threadIdx.x;

    //Current value of x at the particular position
    int current_x = d_x[ti];

    //Shared array to store data in a block 
    __shared__ float buf[BLOCK_SIZE];

    //A varaible to store intermediate solution of x[i]
    float S = 0.0;

    //Iterate so that each time a block gets next set of elements
    for(int i=0;i<gridDim.x;i++){
        //Index of x from which current thread should read data
        int ind = ((blockIdx.x+i)%gridDim.x)*blockDim.x + threadIdx.x ;
        if(ind<n){
            buf[threadIdx.x] = d_x[ind];
        
            __syncthreads();

            //Local thread computation with data in the block
            for(int j=0;j<BLOCK_SIZE;j++){
                S+=K_cal((current_x - buf[j])/h);
            }
            __syncthreads();
        }
    }
    
    //Assigning the final value of y[i]
    d_y[ti] = (S/(n*h));
}

void gaussian_kde(int n, float h, std::vector<float>& x, std::vector<float>& y) {

    int size = n*sizeof(float);

    float* d_x;
    float* d_y;

    //Allocating memory in device for input and output
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    //Copying the input data
    cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);

    //Calculating the number of blocks
    int num_blocks = (n+BLOCK_SIZE - 1)/BLOCK_SIZE;

    //Kernel to perform the gaussian kde computations on gpu
    compute<<<num_blocks,BLOCK_SIZE>>>(n, h, d_x, d_y);

    //Copying the computed gaussian kde value to host
    cudaMemcpy(y.data(), d_y, size, cudaMemcpyDeviceToHost);

    //Free up memory on GPU
    cudaFree(d_x);
    cudaFree(d_y);

} // gaussian_kde

#endif // A3_HPP
