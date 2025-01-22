#define _USE_MATH_DEFINES

#pragma once
#include <math.h>

struct __align__(16) Options
{
	int width = 1600;
	int height = 900;

	float separationNormalFishes = 1.0f;
	float alignmentNormalFishes = 1.3f;
	float cohesionNormalFishes = 1.0f;
	float minVelNormalFishes = 1.0f;
	float maxVelNormalFishes = 4.0f;
	float radiusNormalFishes = 40.0f;
	float angleNormalFishes = 3 * M_PI / 4;
};