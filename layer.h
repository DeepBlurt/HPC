/*
 * layer.h
 *
 *  Created on: Nov 26, 2017
 *      Author: deepgray
 */

#ifndef LAYER_H_
#define LAYER_H_

#include<random>
#include"matrix.h"

typedef struct
{
	int inputDim;
	int outputDim;
	Matrix weight;
	Matrix bias;
}Layer;



typedef struct
{
	int numLayer ;
	int layerInputDim[4];
	int layerOutputDim[4];
	Layer layers[4];
}Net;


#endif /* LAYER_H_ */
