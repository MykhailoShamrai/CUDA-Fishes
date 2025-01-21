#include "grid.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/helpers.cuh"
#include <assert.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include "fishes.cuh"
#include <set>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>


Grid::Grid(int nFishes, int radiusForFishes, int width, int height, bool onGpu):
	onGpu(onGpu), n_fishes(nFishes), width(width), height(height)
{
	cellSize = radiusForFishes * 2;
	
	// One cell has width = 2 * radius and height = 2 * radius

	this->n_x_cells = (width + cellSize - 1) / cellSize;
	this->n_y_cells = (height + cellSize - 1) / cellSize;
	this->n_cells = n_x_cells * n_y_cells;

	if (onGpu)
	{
		d_AllocateMemory();
	}
	else
	{
		h_AllocateMemory();
	}
}

// Test function for assert
static bool verifyArrayIndices(int* array, int n)
{
	std::set<int> set_test{};
	for (int i = 0; i < n; i++)
	{
		auto res = set_test.insert(array[i]);
		if (!res.second)
		{
			return false;
		}
	}
	return true;
}

// Test
static bool verifyCudaArrayIndices(int* array, int n)
{
	std::vector<int> vec(n, 0);
	checkCudaErrors(cudaMemcpy(vec.data(), array, n * sizeof(int), cudaMemcpyDeviceToHost));
	return verifyArrayIndices(vec.data(), n);
}

void Grid::h_InitialiseArraysIndicesAndFishes()
{
	for (int i = 0; i < n_fishes; i++)
	{
		indices[i] = i;
		fish_id[i] = i;
	}
	assert(verifyArrayIndices(indices, n_fishes));
	assert(verifyArrayIndices(fish_id, n_fishes));
}

void Grid::d_InitialiseArraysIndicesAndFishes(int* initialisedIndexArray)
{
	assert(initialisedIndexArray);
	assert(verifyArrayIndices(initialisedIndexArray, n_fishes));
	checkCudaErrors(cudaMemcpy(indices, initialisedIndexArray, n_fishes * sizeof(float), cudaMemcpyHostToDevice));
	assert(verifyCudaArrayIndices(indices, n_fishes));
	checkCudaErrors(cudaMemcpy(fish_id, indices, n_fishes * sizeof(float), cudaMemcpyDeviceToDevice));
	assert(verifyCudaArrayIndices(fish_id, n_fishes));
}

void Grid::FindCellsForFishes(Fishes fishes)
{
	CellForFishFunctor func = CellForFishFunctor(fishes.x_before_movement,
		fishes.y_before_movement, cellSize, width, height, n_x_cells, n_y_cells);
	QuarterForFishFunctor funcQ = QuarterForFishFunctor(fishes.x_before_movement,
		fishes.y_before_movement, cell_id, cellSize, width, height, n_x_cells, n_y_cells);
	if (onGpu)
	{
		auto dev_ptr_cell_id = thrust::device_pointer_cast(cell_id);
		auto dev_ptr_indices = thrust::device_pointer_cast(indices);
		auto dev_ptr_quarters = thrust::device_pointer_cast(quarter_number);
		thrust::transform(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_cell_id, func);
		cudaDeviceSynchronize();
		assert(verifyCudaArrayIndices(fish_id, n_fishes));
		thrust::transform(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_quarters, funcQ);
		cudaDeviceSynchronize();
		assert(verifyCudaArrayIndices(fish_id, n_fishes));
	}
	else
	{
		thrust::transform(thrust::host, indices, indices + n_fishes, cell_id, func);
		thrust::transform(thrust::host, indices, indices + n_fishes, quarter_number, funcQ);
	}
}


void Grid::SortCellsWithFishes()
{
	if (onGpu)
	{
		auto dev_ptr_cell_id = thrust::device_pointer_cast(cell_id);
		auto dev_ptr_fish_id = thrust::device_pointer_cast(fish_id);
		thrust::sort_by_key(thrust::device, dev_ptr_cell_id, dev_ptr_cell_id + n_fishes, dev_ptr_fish_id);
		cudaDeviceSynchronize();
		assert(verifyCudaArrayIndices(fish_id, n_fishes));
	}
	else
	{
		thrust::sort_by_key(thrust::host, cell_id, cell_id + n_fishes, fish_id);
	}
}

void Grid::FindStartsAndEnds()
{
	FindStartsAndEndsFunctor func = FindStartsAndEndsFunctor(n_fishes, cells_starts, cells_ends, cell_id);
	if (onGpu)
	{
		auto dev_ptr_indices = thrust::device_pointer_cast(indices);
		thrust::transform(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_indices, func);
		cudaDeviceSynchronize();
	}
	else
	{
		thrust::transform(thrust::host, indices, indices + n_fishes, indices, func);
	}
}

void Grid::CleanStartsAndEnds()
{
	CleanStartsAndEndsFunctor func = CleanStartsAndEndsFunctor();
	if (onGpu)
	{
		auto dev_ptr_starts = thrust::device_pointer_cast(cells_starts);
		auto dev_ptr_ends = thrust::device_pointer_cast(cells_ends);
		thrust::transform(thrust::device, dev_ptr_starts, dev_ptr_starts + n_cells, dev_ptr_starts, func);
		cudaDeviceSynchronize();
		thrust::transform(thrust::device, dev_ptr_ends, dev_ptr_ends + n_cells, dev_ptr_ends, func);
		cudaDeviceSynchronize();
	}
	else
	{
		thrust::transform(thrust::host, cells_starts, cells_starts + n_cells, cells_starts, func);
		thrust::transform(thrust::host, cells_ends, cells_ends + n_cells, cells_ends, func);
	}
}

static bool VerifyIfArraysAreNotTheSame(float* array1, float* array2, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (array1[i] == array2[i])
		{
			return false;
		}
	}
	return true;
}

static bool VerifyCudaIfArraysAreNotTheSame(float* array1, float* array2, int n)
{
	std::vector<float> vec1(n, 0.0f);
	std::vector<float> vec2(n, 0.0f);
	checkCudaErrors(cudaMemcpy(vec1.data(), array1, n * sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(vec2.data(), array2, n * sizeof(float), cudaMemcpyDeviceToHost));
	return VerifyIfArraysAreNotTheSame(vec1.data(), vec2.data(), n);
}

void Grid::CleanAfterAllCount(Fishes fishes)
{
	if (onGpu)
	{
		assert(VerifyCudaIfArraysAreNotTheSame(fishes.x_before_movement, fishes.x_after_movement, n_fishes));
		assert(VerifyCudaIfArraysAreNotTheSame(fishes.y_before_movement, fishes.y_after_movement, n_fishes));
		checkCudaErrors(cudaMemcpy(fishes.x_before_movement, fishes.x_after_movement, n_fishes * sizeof(float), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(fishes.y_before_movement, fishes.y_after_movement, n_fishes * sizeof(float), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(fishes.x_vel_before_movement, fishes.x_vel_after_movement, n_fishes * sizeof(float), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(fishes.y_vel_before_movement, fishes.y_vel_after_movement, n_fishes * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	else
	{
		assert(VerifyIfArraysAreNotTheSame(fishes.x_before_movement, fishes.x_after_movement, n_fishes));
		assert(VerifyIfArraysAreNotTheSame(fishes.y_before_movement, fishes.y_after_movement, n_fishes));
		for (int i = 0; i < n_fishes; i++)
		{
			fishes.x_before_movement[i] = fishes.x_after_movement[i];
			fishes.y_before_movement[i] = fishes.y_after_movement[i];
			fishes.x_vel_before_movement[i] = fishes.x_vel_after_movement[i];
			fishes.y_vel_before_movement[i] = fishes.y_vel_after_movement[i];
		}
	}
}



void Grid::h_AllocateMemory()
{
	cell_id = (int*)malloc(sizeof(int) * n_fishes);
	fish_id = (int*)malloc(sizeof(int) * n_fishes);
	cells_starts = (int*)malloc(sizeof(int) * n_cells);
	cells_ends = (int*)malloc(sizeof(int) * n_cells);
	indices = (int*)malloc(sizeof(int) * n_fishes);
	quarter_number = (int*)malloc(sizeof(int) * n_fishes);
}

void Grid::d_AllocateMemory()
{
	checkCudaErrors(cudaMalloc((void**)&cell_id, sizeof(int) * n_fishes));
	checkCudaErrors(cudaMalloc((void**)&fish_id, sizeof(int) * n_fishes));
	checkCudaErrors(cudaMalloc((void**)&cells_starts, sizeof(int) * n_cells));
	checkCudaErrors(cudaMalloc((void**)&cells_ends, sizeof(int) * n_cells));
	checkCudaErrors(cudaMalloc((void**)&indices, sizeof(int) * n_fishes));
	checkCudaErrors(cudaMalloc((void**)&quarter_number, sizeof(int) * n_fishes));
}

void Grid::h_CleanMemory()
{
	free(cell_id);
	free(fish_id);
	free(cells_starts);
	free(cells_ends);
	free(indices);
	free(quarter_number);
}

void Grid::d_CleanMemory()
{
	checkCudaErrors(cudaFree(cell_id));
	checkCudaErrors(cudaFree(fish_id));
	checkCudaErrors(cudaFree(cells_starts));
	checkCudaErrors(cudaFree(cells_ends));
	checkCudaErrors(cudaFree(indices));
	checkCudaErrors(cudaFree(quarter_number));
}