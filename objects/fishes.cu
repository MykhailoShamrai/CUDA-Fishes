#include "fishes.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/helpers.cuh"


void Fishes::h_allocate_memory_for_fishes()
{
	this->x_before_movement = (float*)malloc(sizeof(float) * N);
	this->y_before_movement = (float*)malloc(sizeof(float) * N);
	this->x_vel_before_movement = (float*)malloc(sizeof(float) * N);
	this->y_vel_before_movement = (float*)malloc(sizeof(float) * N);

	this->x_after_movement = (float*)malloc(sizeof(float) * N);
	this->y_after_movement = (float*)malloc(sizeof(float) * N);
	this->x_vel_after_movement = (float*)malloc(sizeof(float) * N);
	this->y_vel_after_movement = (float*)malloc(sizeof(float) * N);

	this->types = (Fishes::FishType*)malloc(sizeof(Fishes::FishType) * N);
} 

void Fishes::d_allocate_memory_for_fishes()
{
	checkCudaErrors(cudaMalloc((void**)&this->x_before_movement, sizeof(float) * N));
	checkCudaErrors(cudaMalloc((void**)&this->y_before_movement, sizeof(float) * N));
	checkCudaErrors(cudaMalloc((void**)&this->x_vel_before_movement, sizeof(float) * N));
	checkCudaErrors(cudaMalloc((void**)&this->y_vel_before_movement, sizeof(float) * N));

	checkCudaErrors(cudaMalloc((void**)&this->x_after_movement, sizeof(float) * N));
	checkCudaErrors(cudaMalloc((void**)&this->y_after_movement, sizeof(float) * N));
	checkCudaErrors(cudaMalloc((void**)&this->x_vel_after_movement, sizeof(float) * N));
	checkCudaErrors(cudaMalloc((void**)&this->y_vel_after_movement, sizeof(float) * N));

	checkCudaErrors(cudaMalloc((void**)&this->types, sizeof(Fishes::FishType) * N));
}

void Fishes::h_clean_memory_for_fishes()
{
	free(this->x_before_movement);
	free(this->y_before_movement);
	free(this->x_vel_before_movement);
	free(this->y_vel_before_movement);

	free(this->x_after_movement);
	free(this->y_after_movement);
	free(this->x_vel_after_movement);
	free(this->y_vel_after_movement);

	free(this->types);
}

void Fishes::d_clean_memory_for_fishes()
{
	checkCudaErrors(cudaFree(this->x_before_movement));
	checkCudaErrors(cudaFree(this->y_before_movement));
	checkCudaErrors(cudaFree(this->x_vel_before_movement));
	checkCudaErrors(cudaFree(this->y_vel_before_movement));

	checkCudaErrors(cudaFree(this->x_after_movement));
	checkCudaErrors(cudaFree(this->y_after_movement));
	checkCudaErrors(cudaFree(this->x_vel_after_movement));
	checkCudaErrors(cudaFree(this->y_vel_after_movement));

	checkCudaErrors(cudaFree(this->types));
}