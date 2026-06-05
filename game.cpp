#include "precomp.h"
#include "game.h"
//#include "opencl.cpp"

#define GRIDSIZE 256
int GPU = 1;

// VERLET CLOTH SIMULATION DEMO
// High-level concept: a grid consists of points, each connected to four 
// neighbours. For a simulation step, the position of each point is affected
// by its speed, expressed as (current position - previous position), a
// constant gravity force downwards, and random impulses ("wind").
// The final force is provided by the bonds between points, via the four
// connections.
// Together, this simple scheme yields a pretty convincing cloth simulation.
// The algorithm has been used in games since the game "Thief".

// ASSIGNMENT STEPS:
// 1. SIMD, part 1: in Game::Simulation, convert lines 119 to 126 to SIMD.
//    You receive 2 points if the resulting code is faster than the original.
//    This will probably require a reorganization of the data layout, which
//    may in turn require changes to the rest of the code.
// 2. SIMD, part 2: for an additional 4 points, convert the full Simulation
//    function to SSE. This may require additional changes to the data to
//    avoid concurrency issues when operating on neighbouring points.
//    The resulting code must be at least 2 times faster (using SSE) or 4
//    times faster (using AVX) than the original  to receive the full 4 points.
// 3. GPGPU, part 1: modify Game::Simulation so that it sends the cloth data
//    to the GPU, and execute lines 119 to 126 on the GPU. After this, bring
//    back the cloth data to the CPU and execute the remainder of the Verlet
//    simulation code. You receive 2 points if the code *works* correctly;
//    note that this is expected to be slower due to the data transfers.
// 4. GPGPU, part 2: execute the full Game::Simulation function on the GPU.
//    You receive 4 additional points if this yields a correct simulation
//    that is at least 5x faster than the original code. DO NOT draw the
//    cloth on the GPU; this is (for now) outside the scope of the assignment.
// Note that the GPGPU tasks will benefit from the SIMD tasks.
// Also note that your final grade will be capped at 10.

#if 1
struct Point
{
	float2 pos;				// current position of the point
	float2 prev_pos;		// position of the point in the previous frame
	float2 fix;				// stationary position; used for the top line of points
	bool fixed;				// true if this is a point in the top line of the cloth
	float restlength[4];	// initial distance to neighbours
};

// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
Point& grid(const uint x, const uint y) { return pointGrid[x + y * GRIDSIZE]; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

Kernel* gravity;
Kernel* constraints;
Kernel* fix;

// initialization
void Game::Init()
{
	gravity = new Kernel("cl/cloth.cl", "gravity");
	constraints = new Kernel("cl/cloth.cl", "constraint");
	fix = new Kernel("cl/cloth.cl", "fix");

	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
	{
		grid(x, y).pos.x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
		grid(x, y).pos.y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);
		grid(x, y).prev_pos = grid(x, y).pos; // all points start stationary
		if (y == 0)
		{
			grid(x, y).fixed = true;
			grid(x, y).fix = grid(x, y).pos;
		}
		else
		{
			grid(x, y).fixed = false;
		}
	}
	for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
	{
		// calculate and store distance to four neighbours, allow 15% slack
		for (int c = 0; c < 4; c++)
		{
			grid(x, y).restlength[c] = length(grid(x, y).pos - grid(x + xoffset[c], y + yoffset[c]).pos) * 1.15f;
		}
	}
}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.
void Game::DrawGrid()
{
	// draw the grid
	screen->Clear(0);
	for (int y = 0; y < (GRIDSIZE - 1); y++) for (int x = 1; x < (GRIDSIZE - 2); x++)
	{
		const float2 p1 = grid(x, y).pos;
		const float2 p2 = grid(x + 1, y).pos;
		const float2 p3 = grid(x, y + 1).pos;
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
		screen->Line(p1.x, p1.y, p3.x, p3.y, 0xffffff);
	}
	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		const float2 p1 = grid(GRIDSIZE - 2, y).pos;
		const float2 p2 = grid(GRIDSIZE - 2, y + 1).pos;
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
void Game::Simulation()
{
	Buffer* gridbuffer = new Buffer(GRIDSIZE * GRIDSIZE * sizeof(Point), pointGrid, Buffer::DEFAULT);
	gridbuffer->CopyToDevice(true);

	// simulation is exected three times per frame; do not change this.
	for (int steps = 0; steps < 3; steps++)
	{
		// verlet integration; apply gravity

		// Original CPU code
		/*for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
		{
			float2 curpos = grid( x, y ).pos, prevpos = grid( x, y ).prev_pos;
			grid( x, y ).pos += (curpos - prevpos) + float2( 0, 0.003f ); // gravity
			grid( x, y ).prev_pos = curpos;
			if (Rand( 10 ) < 0.03f) grid( x, y ).pos += float2( Rand( 0.02f + magic ), Rand( 0.12f ) );
		}*/

		// GPU code
		gravity->SetArguments(gridbuffer, magic);
		gravity->Run(GRIDSIZE * GRIDSIZE);
		//gridbuffer->CopyFromDevice(true);

		magic += 0.0002f; // slowly increases the chance of anomalies
		// apply constraints; 4 simulation steps: do not change this number.
		for (int i = 0; i < 4; i++)
		{
			// Original CPU code
			/*for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
			{
				float2 pointpos = grid( x, y ).pos;
				// use springs to four neighbouring points
				for (int linknr = 0; linknr < 4; linknr++)
				{
					Point& neighbour = grid( x + xoffset[linknr], y + yoffset[linknr] );
					float distance = length( neighbour.pos - pointpos );
					if (!isfinite( distance ))
					{
						// warning: this happens; sometimes vertex positions 'explode'.
						continue;
					}
					if (distance > grid( x, y ).restlength[linknr])
					{
						// pull points together
						float extra = distance / (grid( x, y ).restlength[linknr]) - 1;
						float2 dir = neighbour.pos - pointpos;
						pointpos += extra * dir * 0.5f;
						neighbour.pos -= extra * dir * 0.5f;
					}
				}
				grid( x, y ).pos = pointpos;
			}

			// fixed line of points is fixed.
			for (int x = 0; x < GRIDSIZE; x++) grid( x, 0 ).pos = grid( x, 0 ).fix;*/

			// GPU code
			// Apply constraints
			constraints->SetArguments(gridbuffer);
			constraints->Run(GRIDSIZE * GRIDSIZE);

			// Fixed line of points is fixed
			fix->SetArguments(gridbuffer);
			fix->Run(GRIDSIZE);
		}
	}
	gridbuffer->CopyFromDevice(true);

	// Destroy buffer after we're done because otherwise memory issues
	// Alternatively we could use the same buffer for each iteration but I am too lazy to figure this out :)
	gridbuffer->~Buffer();
}
#else
static union { float posx[GRIDSIZE * GRIDSIZE]; __m128 posx4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float posy[GRIDSIZE * GRIDSIZE]; __m128 posy4[GRIDSIZE * GRIDSIZE / 4]; };

static union { float prev_posx[GRIDSIZE * GRIDSIZE]; __m128 prev_posx4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float prev_posy[GRIDSIZE * GRIDSIZE]; __m128 prev_posy4[GRIDSIZE * GRIDSIZE / 4]; };

static union { float fixx[GRIDSIZE * GRIDSIZE]; __m128 fix4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float fixy[GRIDSIZE * GRIDSIZE]; __m128 fixy4[GRIDSIZE * GRIDSIZE / 4]; };

static union { bool fixedb[GRIDSIZE * GRIDSIZE]; __m128 fixed4[GRIDSIZE * GRIDSIZE / 4]; };
static union { float restlength[GRIDSIZE * GRIDSIZE][4]; __m128 restlength4[GRIDSIZE * GRIDSIZE / 4][4]; };


// grid access convenience
//Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
//Point& grid( const uint x, const uint y ) { return pointGrid[x + y * GRIDSIZE]; }
int coord(const uint x, const uint y) { return x + y * GRIDSIZE; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// initialization
void Game::Init()
{
	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
	{
		posx[coord(x, y)] = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
		posy[coord(x, y)] = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);
		//pos[coord(x,y)].x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
		//pos[coord(x,y)].y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);
		prev_posx[coord(x, y)] = posx[coord(x, y)];
		prev_posy[coord(x, y)] = posy[coord(x, y)];
		//prev_pos[coord(x,y)] = pos[coord(x,y)]; // all points start stationary
		if (y == 0)
		{
			fixedb[coord(x, y)] = true;
			fixx[coord(x, y)] = posx[coord(x, y)];
			fixy[coord(x, y)] = posy[coord(x, y)];
		}
		else
		{
			fixedb[coord(x, y)] = false;
		}
	}
	for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
	{
		// calculate and store distance to four neighbours, allow 15% slack
		for (int c = 0; c < 4; c++)
		{
			restlength[coord(x, y)][c]
				= length(
					float2(posx[coord(x, y)], posy[coord(x, y)]) -
					float2(posx[coord(x + xoffset[c], y + yoffset[c])], posy[coord(x + xoffset[c], y + yoffset[c])])) * 1.15f;
		}
	}
}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.
void Game::DrawGrid()
{
	// draw the grid
	screen->Clear(0);
	for (int y = 0; y < (GRIDSIZE - 1); y++) for (int x = 1; x < (GRIDSIZE - 2); x++)
	{
		const float2 p1 = float2(posx[coord(x, y)], posy[coord(x, y)]);
		const float2 p2 = float2(posx[coord(x + 1, y)], posy[coord(x + 1, y)]);
		const float2 p3 = float2(posx[coord(x, y + 1)], posy[coord(x, y + 1)]);
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
		screen->Line(p1.x, p1.y, p3.x, p3.y, 0xffffff);
	}
	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		const float2 p1 = float2(posx[coord(GRIDSIZE - 2, y)], posy[coord(GRIDSIZE - 2, y)]);
		const float2 p2 = float2(posx[coord(GRIDSIZE - 2, y + 1)], posy[coord(GRIDSIZE - 2, y + 1)]);
		screen->Line(p1.x, p1.y, p2.x, p2.y, 0xffffff);
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
void Game::Simulation()
{
	//Kernel* k = new Kernel("cl/cloth.cl", "gravity");

	// simulation is exected three times per frame; do not change this.
	for (int steps = 0; steps < 3; steps++)
	{
		// verlet integration; apply gravity

		const __m128 gravity = _mm_set1_ps(0.0025f);

		for (int i = 0; i < GRIDSIZE * GRIDSIZE / 4; i++)
		{
			__m128 curposx = posx4[i];
			__m128 curposy = posy4[i];
			__m128 prevposx = prev_posx4[i];
			__m128 prevposy = prev_posy4[i];

			posx4[i] = _mm_add_ps(curposx, _mm_sub_ps(curposx, prevposx));
			posy4[i] = _mm_add_ps(curposy, _mm_add_ps(_mm_sub_ps(curposy, prevposy), gravity));
			prev_posx4[i] = curposx;
			prev_posy4[i] = curposy;

			float* px = (float*)&posx4[i];
			float* py = (float*)&posy4[i];

			// apply random force to each lane, could not find good random for __m128
			for (int j = 0; j < 4; j++)
				if (Rand(10) < 0.03f)
				{
					px[j] += Rand(0.09f + magic);
					py[j] += Rand(0.12f);
				}

		}
		/*
		for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
		{
			float curposx = posx[coord(x, y)];
			float curposy = posy[coord(x, y)];
			float prevposx = prev_posx[coord(x, y)];
			float prevposy = prev_posy[coord(x, y)];
			posx[coord(x, y)] += curposx - prevposx;
			posy[coord(x, y)] += (curposy - prevposy) + 0.003f; // gravity
			//pos[coord(x,y)] += (curpos - prevpos) + float2( 0, 0.003f ); // gravity
			prev_posx[coord(x, y)] = curposx;
			prev_posy[coord(x, y)] = curposy;
			//prev_pos[coord(x,y)] = curpos;
			if (Rand(10) < 0.03f) //pos[coord(x, y)] += float2(Rand(0.02f + magic), Rand(0.12f));
			{
				posx[coord(x, y)] += Rand(0.02f + magic);
				posy[coord(x, y)] += Rand(0.12f);
			}

		}
		*/

		// GPU code
		/*
		Buffer* b = new Buffer(GRIDSIZE * GRIDSIZE * sizeof(Point), pointGrid, Buffer::DEFAULT);
		b->CopyToDevice(true);
		k->SetArguments(b, magic);
		k->Run(GRIDSIZE * GRIDSIZE, 256);
		b->CopyFromDevice(true);
		*/

		magic += 0.0002f; // slowly increases the chance of anomalies
		// apply constraints; 4 simulation steps: do not change this number.

		for (int i = 0; i < 4; i++)
		{
			for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
			{
				//float2 pointpos = pos[coord(x,y)];
				float2 pointpos = float2(posx[coord(x, y)], posy[coord(x, y)]);
				// use springs to four neighbouring points
				for (int linknr = 0; linknr < 4; linknr++)
				{
					//float2& neighbour = float2(posx[coord(x + xoffset[linknr], y + yoffset[linknr])], 
											   //posy[coord(x + xoffset[linknr], y + yoffset[linknr])]);

					float& neighbourx = posx[coord(x + xoffset[linknr], y + yoffset[linknr])];
					float& neighboury = posy[coord(x + xoffset[linknr], y + yoffset[linknr])];
					float2 neighbour = float2(neighbourx, neighboury);

					//float2& neighbour = pos[coord(x + xoffset[linknr], y + yoffset[linknr])];
					float distance = length(neighbour - pointpos);
					if (!isfinite(distance))
					{
						// warning: this happens; sometimes vertex positions 'explode'.
						continue;
					}
					if (distance > restlength[coord(x, y)][linknr])
					{
						// pull points together
						float extra = distance / (restlength[coord(x, y)][linknr]) - 1;
						float2 dir = neighbour - pointpos;
						pointpos += extra * dir * 0.5f;
						neighbourx -= extra * dir.x * 0.5f;
						neighboury -= extra * dir.y * 0.5f;
					}
				}
				posx[coord(x, y)] = pointpos.x;
				posy[coord(x, y)] = pointpos.y;
				//pos[coord(x,y)] = pointpos;
			}
			// fixed line of points is fixed.
			for (int x = 0; x < GRIDSIZE; x++) //pos[coord(x,0)] = fix[coord(x,0)];
			{
				posx[coord(x, 0)] = fixx[coord(x, 0)];
				posy[coord(x, 0)] = fixy[coord(x, 0)];
			}
		}

	}
}



#endif


void Game::Tick( float a_DT )
{
	// update the simulation
	Timer tm;
	tm.reset();
	Simulation();
	float elapsed1 = tm.elapsed();

	// draw the grid
	tm.reset();
	DrawGrid();
	float elapsed2 = tm.elapsed();

	// display statistics
	char t[128];
	sprintf( t, "ye olde ruggeth cloth simulation: %5.1f ms", elapsed1 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 24, 0xffffff );
	sprintf( t, "                       rendering: %5.1f ms", elapsed2 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 14, 0xffffff );
}