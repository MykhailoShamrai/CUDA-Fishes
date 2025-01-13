#include "grid.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/helpers.cuh"


Grid::Grid(int nFishes, int radiusForFishes, int width, int heith, bool onGpu):
	onGpu(onGpu), n_fishes(nFishes)
{
	// TODO: Firstly count how many cells there is

	if (onGpu)
	{
		
	}
	else
	{

	}
}

Grid::~Grid()
{
	if (onGpu)
	{
		d_CleanMemory();
	}
	else
	{
		h_CleanMemory();
	}
}

void Grid::h_AllocateMemory()
{
	// Allocate array of ints size number of fishes
	this->cell_id = (int*)malloc(sizeof(int) * this->n_fishes);
	// Allocate array of ints size number of fishes
	this->fish_id - (int*)malloc(sizeof(int) * this->n_fishes);
	// Allocate array of ints size number of cells
	this->cells_starts = (int*)malloc(sizeof(int) * this->n_cells);
	// Allocate array if ints size number of cells
	this->cells_ends = (int*)malloc(sizeof(int) * this->n_cells);
}

void Grid::d_AllocateMemory()
{
	checkCudaErrors(cudaMalloc((void**)&this->cell_id, sizeof(int) * this->n_fishes));
	checkCudaErrors(cudaMalloc((void**)&this->fish_id, sizeof(int) * this->n_fishes));
	checkCudaErrors(cudaMalloc((void**)&this->cells_starts, sizeof(int) * this->n_cells));
	checkCudaErrors(cudaMalloc((void**)&this->cells_ends, sizeof(int) * this->n_cells));	
}

void Grid::h_CleanMemory()
{
	free(this->cell_id);
	free(this->fish_id);
	free(this->cells_starts);
	free(this->cells_ends);	
}

void Grid::d_CleanMemory()
{
	checkCudaErrors(cudaFree(this->cell_id));
	checkCudaErrors(cudaFree(this->fish_id));
	checkCudaErrors(cudaFree(this->cells_starts));
	checkCudaErrors(cudaFree(this->cells_ends));	
}