
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include "compute.h"

// Pointers to host memory, defined in nbody.c
extern vector3 *hPos, *hVel;
extern double *mass;

// Pointers to device memory
vector3 *dPos, *dVel, *dAccels;
double *dMass;

#ifdef __cplusplus
extern "C" {
#endif

__global__ void calculate_accelerations(vector3 *pos, double *mass, vector3 *accels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i == j) {
            FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
        } else {
            vector3 distance;
            distance[0] = pos[i][0] - pos[j][0];
            distance[1] = pos[i][1] - pos[j][1];
            distance[2] = pos[i][2] - pos[j][2];

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            
            if (magnitude > 0) {
                double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
                accels[i * NUMENTITIES + j][0] = accelmag * distance[0] / magnitude;
                accels[i * NUMENTITIES + j][1] = accelmag * distance[1] / magnitude;
                accels[i * NUMENTITIES + j][2] = accelmag * distance[2] / magnitude;
            } else {
                FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
            }
        }
    }
}

__global__ void update_positions(vector3 *pos, vector3 *vel, vector3 *accels, double* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES) {
        vector3 accel_sum = {0, 0, 0};
        for (int j = 0; j < NUMENTITIES; j++) {
            accel_sum[0] += accels[i * NUMENTITIES + j][0];
            accel_sum[1] += accels[i * NUMENTITIES + j][1];
            accel_sum[2] += accels[i * NUMENTITIES + j][2];
        }

        vel[i][0] += accel_sum[0] * INTERVAL;
        vel[i][1] += accel_sum[1] * INTERVAL;
        vel[i][2] += accel_sum[2] * INTERVAL;

        pos[i][0] += vel[i][0] * INTERVAL;
        pos[i][1] += vel[i][1] * INTERVAL;
        pos[i][2] += vel[i][2] * INTERVAL;
    }
}

void compute_cuda_init() {
    cudaMalloc(&dPos, NUMENTITIES * sizeof(vector3));
    cudaMalloc(&dVel, NUMENTITIES * sizeof(vector3));
    cudaMalloc(&dMass, NUMENTITIES * sizeof(double));
    cudaMalloc(&dAccels, NUMENTITIES * NUMENTITIES * sizeof(vector3));

    cudaMemcpy(dPos, hPos, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, NUMENTITIES * sizeof(vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(dMass, mass, NUMENTITIES * sizeof(double), cudaMemcpyHostToDevice);
}

void compute() {
    dim3 blockSize(16, 16);
    dim3 gridSize((NUMENTITIES + blockSize.x - 1) / blockSize.x, (NUMENTITIES + blockSize.y - 1) / blockSize.y);

    calculate_accelerations<<<gridSize, blockSize>>>(dPos, dMass, dAccels);

    dim3 updateBlockSize(256);
    dim3 updateGridSize((NUMENTITIES + updateBlockSize.x - 1) / updateBlockSize.x);

    update_positions<<<updateGridSize, updateBlockSize>>>(dPos, dVel, dAccels, dMass);

    cudaDeviceSynchronize();
}

void compute_cuda_finalize() {
    cudaMemcpy(hPos, dPos, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, NUMENTITIES * sizeof(vector3), cudaMemcpyDeviceToHost);

    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
    cudaFree(dAccels);
}

#ifdef __cplusplus
}
#endif