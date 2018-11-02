#include<cuda_runtime.h>
#include"matrix.h"
#include<cstdlib>
#include<cmath>
#include"layer.h"
#include<iostream>

// Matrix multiplication kernel called by MatMul()
__device__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < A.height && col < B.width)
    {
    	for(int e = 0; e < A.width; ++e)
    		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    	C.elements[row * C.width + col] = Cvalue;
    }
}



__device__ void AddSigmoidKernel(Matrix A, Matrix B, Matrix C)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < A.height  && col < A.width)
	{
		C.elements[row * C.width + col] = A.elements[row * A.width + col]\
				+ B.elements[row * B.width + col];
		C.elements[row * C.width + col] = \
				1.0 / (1.0 + std::exp(-C.elements[row * C.width + col]));
	}
}



__global__ void forwardKernel(Matrix input, Matrix weight, Matrix bias, \
		Matrix output)
{
	MatMulKernel(input, weight, output);
	__syncthreads();
	AddSigmoidKernel(output, bias, output);
	__syncthreads();
}

extern "C"
int deviceQuery()
{
    // By default, we use device 0, otherwise
	//we override the device ID based on what is provided at the command line
    int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        //printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Use a larger block size for Fermi and above
    int block_size = (deviceProp.major < 2) ? 16 : 32;
    return block_size;
}


extern "C"
void forwardGpu(Matrix input, Net* n, Matrix output)
{
	//std::cout<<"in forwardGPU...\n";
	cudaError_t err;
	// copy the net parameter
	// compute dimGrid, dimBlock
	int block_size = deviceQuery();
	dim3 dimBlock(block_size, block_size);
	//std::cout<<block_size<<" ";
	dim3 dimGrid(4096 / dimBlock.x, 4096 / dimBlock.y);
	//device memory....
	Matrix h_input;
	h_input.width = 1;
	h_input.height = input.height;
	h_input.elements = (float*)malloc(input.height*sizeof(float));
	for(int i = 0; i < input.height; ++i)
		h_input.elements[i] = input.elements[i];

	for(int i = 0; i < n->numLayer; ++i)
	{
		// allocate bias
		int outDim = n->layerOutputDim[i];
		Matrix d_bias;
		d_bias.width = 1;
		d_bias.height = outDim;
		d_bias.elements = NULL;
		size_t size = d_bias.height * sizeof(float);
		err = cudaMalloc((void**)&d_bias.elements, size);
		if(err != cudaSuccess)
		{
			std::cout<<"Malloc failed...\n";
		}
		cudaMemcpy(d_bias.elements, n->layers[i].bias.elements, size,\
						cudaMemcpyHostToDevice);
		// allocate weights
		Matrix d_weight;
		d_weight.width = n->layerOutputDim[i];
		d_weight.height = n->layerInputDim[i];
		// copy weight parameters
		size = d_weight.width * d_weight.height * sizeof(float);

		err = cudaMalloc(&d_weight.elements, size);
		if(err !=cudaSuccess)
			std::cout<<"Malloc failed...\n";
		cudaMemcpy(d_weight.elements, n->layers[i].weight.elements, \
						size, cudaMemcpyHostToDevice);
		// allocate output
		Matrix d_outTemp;
		d_outTemp.width = 1;
		d_outTemp.height = n->layerOutputDim[i];
		size = n->layerOutputDim[i] * sizeof(float);
		err = cudaMalloc(&d_outTemp.elements, size);
		if(err != cudaSuccess)
			std::cout<<"Malloc d_outtemp failed\n";

		// allocate inputs
		Matrix d_input;
		d_input.height = n->layerInputDim[i];
		d_input.width = 1;
		size = d_input.height * sizeof(float);
		err = cudaMalloc(&d_input.elements, size);
		if(err != cudaSuccess)
			std::cout<<"cudamalloc d_input failed";
		cudaMemcpy(d_input.elements, h_input.elements, \
				size, cudaMemcpyHostToDevice);
		// Invoke kernel calls
		forwardKernel<<<dimGrid, dimBlock>>>(d_input, d_weight, d_bias,\
				d_outTemp);
		//std::cout<<"after kernel invoked\n";
		// copy current output;
		Matrix h_output;
		h_output.width = 1;
		h_output.height = n->layerOutputDim[i];
		h_output.elements = new float[h_output.height];
		size = h_output.height * sizeof(float);
		cudaMemcpy(h_output.elements, d_outTemp.elements, size, cudaMemcpyDeviceToHost);
		// copy h_output to h_input;
		delete h_input.elements;
		h_input.width = 1;
		h_input.height = h_output.height;
		h_input.elements = new float[h_output.height];
		for(int j = 0; j < h_output.height; ++j)
			h_input.elements[i] = h_output.elements[i];
		// free memory..
		if(i == n->numLayer -1 )
		{
			for(int k = 0; k < h_output.height; ++k)
				output.elements[k] = h_output.elements[k];
		}
		delete h_output.elements;
		cudaFree(d_input.elements);
		cudaFree(d_outTemp.elements);
		cudaFree(d_bias.elements);
		cudaFree(d_weight.elements);
	}
	// copy output to return
}

