#include <stdio.h>

const int n = 4096;
const int BlockSize = 16;

__global__ void kernel_down(double *A, double *B, int n, int number_row, int number_column){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by*BlockSize + ty;
    int column = bx*BlockSize + tx; 
    if(number_column < column && number_row < row){
        double glav = A[row*n + number_column]/A[number_row*n + number_column];    
        if(number_column == column){
            B[row] -= B[number_row]*glav;
        }
        A[row*n + column] -= glav*A[number_row*n + column];
    }
}

__global__ void kernel_up(double *A, double *B, int n, int number_row, int number_column){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by*BlockSize + ty;
    int column = bx*BlockSize + tx; 
    if(number_column == column && number_row > row){
        double glav = A[row*n + number_column]/A[number_row*n + number_column];    
        B[row] -= B[number_row]*glav;
        A[row*n + column] = 0;
    }
}

int main (int argc, char* argv []){
    double* A = new double[n * n];
    double* b = new double[n];
    double* result = new double[n];


    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    
    cudaEventCreate ( &start );
    cudaEventCreate ( &stop );


    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            A[i*n+j] = i+j+1;
        }
        b[i] = i;
    }

    double *dev_a, *dev_b;

    cudaMalloc((void**) &dev_a, n*n*sizeof(double));
    cudaMalloc((void**) &dev_b, n*sizeof(double));


    cudaEventRecord ( start, 0 );
    cudaMemcpy(dev_a, A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n*sizeof(double), cudaMemcpyHostToDevice);

    dim3 Grid(n/BlockSize, n/BlockSize);
    dim3 Block(BlockSize, BlockSize);

    for(int i=0;i<n-1;++i){
        kernel_down<<<Grid, Block>>>(dev_a, dev_b, n, i, i);
    }
    for(int i=n-1;i>=0;--i){ 
        kernel_up<<<Grid,Block>>>(dev_a, dev_b, n, i, i);
    }

    cudaMemcpy(A, dev_a, n*n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, dev_b, n*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < n - 1; ++i){
        result[i] = b[i]/A[i*n + i];
    }


    cudaEventRecord ( stop, 0 );
    cudaEventSynchronize ( stop );
    cudaEventElapsedTime ( &gpuTime, start, stop );


    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime );


    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaDeviceReset();

    return 0;
}
