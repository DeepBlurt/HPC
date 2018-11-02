/*
 * matrix.h
 *
 *  Created on: Nov 26, 2017
 *      Author: deepgray
 */

#ifndef MATRIX_H_
#define MATRIX_H_

// Matrixs are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct{
    int width;
    int height;
    float* elements;
} Matrix;

#endif /* MATRIX_H_ */
