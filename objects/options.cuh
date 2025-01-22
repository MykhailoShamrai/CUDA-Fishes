#define _USE_MATH_DEFINES

#pragma once
#include <math.h>

struct __align__(16) Options
{
	int width = 1600;
	int height = 900;

	float separationNormalFishes = 0.08f;
	float alignmentNormalFishes = 0.005f;
	float cohesionNormalFishes = 0.005f;
	float minVelNormalFishes = 1.0f;
	float maxVelNormalFishes = 3.0f;
	float radiusNormalFishes = 20.0f;
	float angleNormalFishes = 3 * M_PI / 4;
	float forceForWallAvoidance = 10.0f;
};