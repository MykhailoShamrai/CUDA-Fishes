#pragma once 
#include "fishes.cuh"
#include <cuda_runtime.h>
struct Grid
{
private:
	bool onGpu;
	const 
	// Number of cells
	int n_x_cells;
	int n_y_cells;
	int n_cells;
	int n_fishes;
	void h_AllocateMemory();
	void d_AllocateMemory();

	void h_CleanMemory();
	void d_CleanMemory();
public:
	Grid(int nFishes, int radiusForFishes, int width, int heith, bool onGpu);
	~Grid();

	int* cell_id;
	int* fish_id;

	// Starts and ends for cells
	int* cells_starts;
	int* cells_ends;

	void FindCellsForFishes(Fishes fishes);
	void SortCellsWithFishes();
	void FindStartsAndEnds();
	void CleanStartAndEnds();
};