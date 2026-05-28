#include "template/common.h"
#include "cl/tools.cl"

typedef struct
{
	float2 pos;
	float2 prev_pos;
	float2 fix;
	int fixed;
	float restlength[4];
} Point;

__kernel void gravity(__global Point *grid)
{
	int id = get_global_id(0);

	float2 curpos = grid[id].pos;
	float2 prevpos = grid[id].prev_pos;

	grid[id].pos += (curpos - prevpos) + (float2)(0.0f, 0.003f);

	grid[id].prev_pos = curpos;
}