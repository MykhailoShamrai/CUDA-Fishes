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
	// TODO: Firstly count how many cells there is
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
	InitialiseArraysIndicesAndFishes();
}

static bool verifyArray(int* array1, int n)
{
	std::set<int> set_test{};
	for (int i = 0; i < n; i++)
	{
		printf("%d\n", array1[i]);
		auto res = set_test.insert(array1[i]);
		if (!res.second)
		{
			while (i < n)
			{
				//printf("%d\n", array1[i]);
				i++;
			}
			return false;
		}
	}
	return true;
}

static bool verifyCudaArray(int* array, int n)
{
	std::vector<int> vec(n, 0);
	checkCudaErrors(cudaMemcpy(vec.data(), array, n * sizeof(int), cudaMemcpyDeviceToHost));
	return verifyArray(vec.data(), n);
}

void Grid::InitialiseArraysIndicesAndFishes()
{
	InitArraysFunctor func = InitArraysFunctor();
	if (onGpu)
	{
		{
			thrust::device_ptr<int> dev_ptr_indices(indices);
			thrust::device_ptr<int> dev_ptr_fish_id(fish_id);
			thrust::transform(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_indices, func);
			checkCudaErrors(cudaDeviceSynchronize());
		}
		thrust::device_ptr<int> dev_ptr_indices(indices);
		thrust::device_ptr<int> dev_ptr_indices_1(indices);
		thrust::device_ptr<int> dev_ptr_fish_id(fish_id);
		// WTF
		thrust::inclusive_scan(indices, indices + n_fishes, indices);
		checkCudaErrors(cudaDeviceSynchronize());
		thrust::copy_n(thrust::device, dev_ptr_indices_1, n_fishes, dev_ptr_fish_id);
		checkCudaErrors(cudaDeviceSynchronize());
		assert(verifyCudaArray(indices, n_fishes));
	}
	else
	{
		thrust::transform(thrust::host, indices, indices + n_fishes, indices, func);
		thrust::exclusive_scan(thrust::host, indices, indices + n_fishes, indices);
		thrust::copy_n(thrust::host, indices, n_fishes, fish_id);
		assert(verifyArray(indices, n_fishes));
	}
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
		thrust::transform(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_quarters, funcQ);
		cudaDeviceSynchronize();
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

void Grid::CleanAfterAllCount(Fishes fishes)
{
	CopyFishPositionsAndVelocitiesAfterCountFunctor func = 
		CopyFishPositionsAndVelocitiesAfterCountFunctor(fishes.x_before_movement, fishes.y_before_movement,
			fishes.x_vel_before_movement, fishes.y_vel_before_movement, fishes.x_after_movement, fishes.y_after_movement,
			fishes.x_vel_after_movement, fishes.y_vel_after_movement);
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