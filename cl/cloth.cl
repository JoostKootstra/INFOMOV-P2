#include "C:/Users/joost/Documents/UU/GMT/OV/P2/INFOMOV-P2/template/common.h"
#include "C:/Users/joost/Documents/UU/GMT/OV/P2/INFOMOV-P2/cl/tools.cl"

// Point structure but on the GPU :)
typedef struct
{
	float2 pos;
	float2 prev_pos;
	float2 fix;
	int fixed;
	float restlength[4];
} Point;

// Applies gravity
__kernel void gravity(__global float *posx, __global float *posy, __global float *prev_posx, __global float *prev_posy, float magic)
{
	int id = get_global_id(0);

	float curposx = posx[id];
	float curposy = posy[id];
	float prevposx = prev_posx[id];
	float prevposy = prev_posy[id];

	// Apply gravity
	posx[id] += curposx - prevposx;
	posy[id] += (curposy - prevposy) + 0.001f;
	prev_posx[id] = curposx;
	prev_posy[id] = curposy;

	// Get random float between 0 and 1
	uint seed = WangHash(id);
	float r0 = RandomFloat(&seed);

	if ((r0 * 10.0f) < 0.01f)
	{
		// Get two more random floats
		float r1 = RandomFloat(&seed);
		float r2 = RandomFloat(&seed);

		// Apply magic stuff based on random floats (scaled)
		posx[id] += (r1 * (0.02f + magic));
		posy[id] += (r2 * 0.12f);
	}
	return;
}

// Applies constraints
// Generally, this is not the best method to handle this, since it would probably be better to either do:
// 1. Do some "checkerboard"-like calculations by separating into alternating neighborhoods, or
// 2. Rewrite the Point data structure such that it stores x and y separately, which I can then use to atomically do calculations
__kernel void constraint(__global float *posx, __global float *posy, __global float *prev_posx, __global float *prev_posy, __global float *restlength)
{
	// offsets for the neighbors later on
	int xoffset[4] = { 1, -1, 0, 0 };
	int yoffset[4] = { 0, 0, 1, -1 };

	int id = get_global_id(0);

	int x = id % 256;
	int y = id >> 8;

	// In the regular CPU code, the for-loop starts at x and y = 1, and ends at x and y = GRIDSIZE - 1, aka 255.
	if (x == 0 || y == 0 || x == 256 || y == 256)
		return;

	float2 curpos = (float2)(posx[id], posy[id]);

	// Go through the neigobrs
	for (int linknr = 0; linknr < 4; linknr++)
	{
		// Get neighbor and its position, and calculate distance
		int neighbor_index = x + xoffset[linknr] + (y + yoffset[linknr]) * 256;
		float neighborx = posx[neighbor_index];
		float neighbory = posy[neighbor_index];
		float2 neighborpos = (float2)(neighborx, neighbory);
		float dist = length(neighborpos - curpos);

		if (!isfinite(dist))
			continue;

		if (dist > restlength[id * 4 + linknr])
		{
			float extra = dist / (restlength[id * 4 + linknr]) - 1;
			float2 dir = neighborpos - curpos;
			float2 force = (extra * 0.05f) * dir;
			float scale = 0.5f;
			curpos += force;
			posx[neighbor_index] -= force.x * scale;
			posy[neighbor_index] -= force.y * scale;
			//grid[neighbor_index].pos -= force * 0.5f;
			// Note on the line above: since we do not atomically add and subtract the float2s, we need to compensate for that.
			// So, we compensate by scaling the force initially lower than usual (0.25f), and then applying a negative force
			// on the neighbor by an even lower factor.
		}
	}
	posx[id] = curpos.x;
	posy[id] = curpos.y;

	return;
}

// Applies the fixed line for all grid positions where y = 0
__kernel void fix(__global float *posx, __global float *posy, __global float *fixx, __global float *fixy)
{
	int id = get_global_id(0);
	int y = id / 256;

	if (y == 0)
	{
		posx[id] = fixx[id];
		posy[id] = fixy[id];
	}
		


	return;
}