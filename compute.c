#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "vector.h"
#include "config.h"
#include "compute.h"

static vector3 *d_accels = NULL;
static double *d_mass = NULL;
static int device_initialized = 0;

extern vector3 *d_hPos;
extern vector3 *d_hVel;

// Kernel 1
__global__ void compute_accels_kernel(vector3 *pos, double *mass, vector3 *accels, int n) {
	long long tid = blockIdx.x * blockDim.x + threadIdx.x;
	long long total = (long long)n * (long long)n;
	if (tid >= total) return;

	int i = tid / n;
	int j = tid % n;

	if (i == j) {
		accels[tid][0] = 0.0;
		accels[tid][1] = 0.0;
		accels[tid][2] = 0.0;
		return;
	}

	double dx = pos[i][0] - pos[j][0];
	double dy = pos[i][1] - pos[j][1];
	double dz = pos[i][2] - pos[j][2];

	double mag_sq = dx*dx + dy*dy + dz*dz;
	double mag = sqrt(mag_sq);

	double accelmag = -1.0 * GRAV_CONSTANT * mass[j] / mag_sq;

	accels[tid][0] = accelmag * dx / mag;
	accels[tid][1] = accelmag * dy / mag;
	accels[tid][2] = accelmag * dz / mag;
}

// Kernel 2
__global__ void integrate_kernel(vector3 *pos, vector3 *vel, vector3 *accels, int n, double dt) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	double ax = 0.0, ay = 0.0, az = 0.0;

	for (int j = 0; j < n; ++j) {
		int idx = i * n + j;
		ax += accels[idx][0];
		ay += accels[idx][1];
		az += accels[idx][2];
	}

	vel[i][0] += ax * dt;
	vel[i][1] += ay * dt;
	vel[i][2] += az * dt;

	pos[i][0] += vel[i][0] * dt;
	pos[i][1] += vel[i][1] * dt;
	pos[i][2] += vel[i][2] * dt;
}

static void init_device_memory(void) {
	if (device_initialized) return;
	
	size_t vecBytes = NUMENTITIES * sizeof(vector3);
	size_t massBytes = NUMENTITIES * sizeof(double);
	size_t accelBytes = (size_t)NUMENTITIES * (size_t)NUMENTITIES * sizeof(vector3);

	cudaMalloc((void**)&d_hPos, vecBytes);
	cudaMalloc((void**)&d_hVel, vecBytes);
	cudaMalloc((void**)&d_mass, massBytes);
	cudaMalloc((void**)&d_accels, accelBytes);

	device_initialized = 1;
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){

	init_device_memory();

	size_t vecBytes = NUMENTITIES * sizeof(vector3);
	size_t massBytes = NUMENTITIES * sizeof(double);
	int n = NUMENTITIES;
	double dt = (double)INTERVAL;

	cudaMemcpy(d_hPos, hPos, vecBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, vecBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, massBytes, cudaMemcpyHostToDevice);

	long long total = (long long)n * (long long)n;
	int threadsPerBlock = 256;
	int blocks1 = (int)((total + threadsPerBlock - 1) / threadsPerBlock);

	compute_accels_kernel<<<blocks1, threadsPerBlock>>>(d_hPos, d_mass, d_accels, n);
	cudaDeviceSynchronize();

	int blocks2 = (n + threadsPerBlock - 1) / threadsPerBlock;

	integrate_kernel<<<blocks2, threadsPerBlock>>>(d_hPos, d_hVel, d_accels, n, dt);
	cudaDeviceSynchronize();

	cudaMemcpy(hPos, d_hPos, vecBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_hVel, vecBytes, cudaMemcpyDeviceToHost);
}
