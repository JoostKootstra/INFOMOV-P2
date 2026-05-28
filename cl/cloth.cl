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

__kernel void gravity(__global Point *grid, float magic)
{
	int id = get_global_id(0);

	float2 curpos = grid[id].pos;
	float2 prevpos = grid[id].prev_pos;

	grid[id].pos += (curpos - prevpos) + (float2)(0.0f, 0.003f);

	uint seed = WangHash((id + 1) * 17);
	float r0 = RandomFloat(&seed);

	if ((r0 * 10.0f) < 0.03f)
	{
		float r1 = RandomFloat(&seed);
		float r2 = RandomFloat(&seed);
		grid[id].pos += (float2)((r1 * (0.02f + magic)), (r2 * 0.12f));
	}

	grid[id].prev_pos = curpos;
}