#pragma once

struct Fishes;
struct Grid;
struct Options;

void CountForFishesCpu(Grid& grid, Options& options, Fishes& fishes,
	float* buffer, int n, float cursorPosX, float cursorPosY, bool fearingWithCursor);

void CountCircleForFishesCpu(Fishes& fishes, float* buffer, int n_fishes,
	int n_points, int radius);