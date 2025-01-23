#define _USE_MATH_DEFINES

#pragma once
#include <math.h>

struct __align__(16) Options
{
	int width = 1600;
	int height = 900;

	float separationForFishes = 0.05f;
	float alignmentForFishes = 0.09f;
	float cohesionForFishes = 0.00004f;

	float radiusSeparation = 8.0f;
	float radiusForFishes = 40.0f;

	float angleNormalFishes = 3 * M_PI / 4;
	float forceForWallAvoidance = 0.2f;
	float rangeToBorderToStartTurn = 100.0f;

	float minVelFishes = 0.5f;
	float maxVelFishes = 3.0f;

	float powerOfFearForCursor = 0.01f;

	void resetToDefaults()
	{
		separationForFishes = 0.05f;
		alignmentForFishes = 0.09f;
		cohesionForFishes = 0.00004f;

		radiusSeparation = 8.0f;
		forceForWallAvoidance = 0.2f;
		rangeToBorderToStartTurn = 100.0f;
		maxVelFishes = 3.0f;
	}
};