/*
 * layer.cpp
 *
 *  Created on: Nov 26, 2017
 *      Author: deepgray
 */

#include"layer.h"
#include"matrix.h"

// Thread block size
float* matmalloc(int width, int height)
{
	float* e = new float[width*height];
	return e;
}


extern "C"
void forwardGpu(Matrix input, Net* n, Matrix output);

void layerInit(Layer* l, int inputDim, int outputDim)
{
	l->inputDim = inputDim;
	l->outputDim = outputDim;

	l->weight.width = outputDim;
	l->weight.height = inputDim;

	l->bias.height = inputDim;
	l->bias.width = 1;

	l->bias.elements = matmalloc(1, inputDim);
	l->weight.elements = matmalloc(inputDim, outputDim);

	for(int i = 0; i < l->bias.width * l->bias.height; ++i)
		l->bias.elements[i] = 0.01 ;
	//std::cout<<l->bias.elements[10]<<"\n";

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0, 0.001);
	//Initialize weights;
	for(int i = 0; i < inputDim*outputDim; ++i)
		l->weight.elements[i] = d(gen);
	//std::cout<<l->weight.elements[10];
}


void netInit(Net* n)
{
	n->numLayer = 4;
	float tmp[4] = {512, 1024, 2048, 4096};
	float tmp2[4] = {1024, 2048, 4096, 2048};

	for(int i = 0; i< 4; ++i)
	{
		n->layerInputDim[i] = tmp[i];
		n->layerOutputDim[i] = tmp2[i];
	}

	for(int i = 0; i < n->numLayer; ++i)
		layerInit(&(n->layers[i]), n->layerInputDim[i], n->layerOutputDim[i]);
	//std::cout<<"after net init \n";
}


void netDestroy(Net* h)
{
	for(int i = 0; i < h->numLayer; ++i)
	{
		delete h->layers[i].bias.elements;
		delete h->layers[i].weight.elements;
	}
}

void forward(Net* n)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(0, 0.001);
	//Initialize weights;
	Matrix input;
	input.width = 1;
	input.height = n->layerInputDim[0];
	input.elements = matmalloc(1 ,n->layerInputDim[0]);

	for(int i = 0; i < n->layerInputDim[0]; ++i)
		input.elements[i] = d(gen);

	Matrix output;
	output.width = 1;
	output.height = n->layerOutputDim[3];
	output.elements = matmalloc(1, n->layerOutputDim[3]);

	forwardGpu(input, n, output);
	delete input.elements;
	delete output.elements;

	//std::cout<<"after forward...\n";
}
