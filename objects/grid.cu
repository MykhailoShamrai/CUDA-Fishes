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


Grid::Grid(int nFishes, int radiusForFishes, int width, int heith, bool onGpu):
	onGpu(onGpu), n_fishes(nFishes)
{
	// TODO: Firstly count how many cells there is
	cellSize = radiusForFishes * 2;
	
	// One cell has width = 2 * radius and height = 2 * radius

	this->n_x_cells = (width + width - 1) / cellSize;
	this->n_y_cells = (heith + heith - 1) / cellSize;
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

void Grid::InitialiseArraysIndicesAndFishes()
{
	InitArraysFunctor func = InitArraysFunctor();
	if (onGpu)
	{
		auto dev_ptr_indices = thrust::device_pointer_cast(indices);
		auto dev_ptr_fish_id = thrust::device_pointer_cast(indices);
		// Firstly start from writing everywhere -1
		thrust::transform(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_indices, func);
		thrust::exclusive_scan(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_indices);
		thrust::copy_n(thrust::device, dev_ptr_indices, n_fishes, dev_ptr_fish_id);
	}
	else
	{

		thrust::transform(thrust::host, indices, indices + n_fishes, indices, func);
		thrust::exclusive_scan(thrust::host, indices, indices + n_fishes, indices);
		thrust::copy_n(thrust::host, indices, n_fishes, fish_id);
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
		thrust::transform(thrust::device, dev_ptr_indices, dev_ptr_indices + n_fishes, dev_ptr_quarters, funcQ);
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
	}
	else
	{
		thrust::transform(thrust::host, indices, indices + n_fishes, indices, func);
	}
}

void Grid::h_AllocateMemory()
{
	// Allocate array of ints size number of fishes
	cell_id = (int*)malloc(sizeof(int) * n_fishes);
	// Allocate array of ints size number of fishes
	fish_id - (int*)malloc(sizeof(int) * n_fishes);
	// Allocate array of ints size number of cells
	cells_starts = (int*)malloc(sizeof(int) * n_cells);
	// Allocate array if ints size number of cells
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