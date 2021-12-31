struct Camera
{
	float4 matrix[4];
};
typedef struct Camera Camera;

struct Star
{
	float3 position;
	float3 velocity;
	float mass;
	float radius; // Avoid recalculating every step
};
typedef struct Star Star;

float4 matrixByVector(const float4 m[4], const float4 v)
{
	return (float4)(
		m[0].x*v.x + m[1].x*v.y + m[2].x*v.z + m[3].x*v.w,
		m[0].y*v.x + m[1].y*v.y + m[2].y*v.z + m[3].y*v.w,
		m[0].z*v.x + m[1].z*v.y + m[2].z*v.z + m[3].z*v.w,
		m[0].w*v.x + m[1].w*v.y + m[2].w*v.z + m[3].w*v.w
	);
}

void draw_point(int2 p, uchar4 color, __global uchar4 *output, const int width, const int height)
{
	if(p.x < 0 || p.x > width || p.y < 0 || p.y > height) return;

	output[p.y * width + p.x] = color;
}

void draw_line(int2 pstart, int2 pend, uchar4 color, __global uchar4 *output, const int width, const int height)
{
	float x = pend.x - pstart.x;
	float y = pend.y - pstart.y;

	const float maxc = max(fabs(x), fabs(y));
	x /= maxc;
	y /= maxc;

	float2 p = (float2)(pstart.x, pstart.y);

	for(float n = 0.f; n < maxc; n += 1.f)
	{
		draw_point((int2)(p.x, p.y), color, output, width, height);

		p.x += x;
		p.y += y;
	}
}

void draw_circle(int2 p, float radius, uchar4 color, __global uchar4 *output, const int width, const int height)
{
	if(radius < 2.f)
	{
		draw_point(p, color, output, width, height);
		return;
	}

	for(int j = round(-radius); j <= round(radius); j++)
	{
		for(int i = round(-radius); i <= round(radius); i++)
		{
			if(i*i + j*j <= radius*radius)
			{
				draw_point((int2)(p.x + i, p.y + j), color, output, width, height);
			}
		}
	}
}

__kernel void update_position(__global Star *stars, const float deltatime)
{
	const uint id = get_global_id(0);

	__global Star *star = &stars[id];

	star->position += star->velocity * deltatime;
}

__kernel void update_velocity(__global Star *stars, const float deltatime)
{
	const uint id = get_global_id(0);
	const uint num = get_global_size(0);
	
	const float G = 1.f;
	
	__global Star *self = &stars[id];

	for(uint i = 0; i < num; i++)
	{
		if(i == id) continue;

		// F = G * (m1*m2)/d²
		// a = F/m1
		// a = G * m2/d²

		__global Star *other = &stars[i];

		float3 dir = other->position - self->position;
		
		if(any(isnan(dir))) continue;

		float lenSqr = ( dir.x*dir.x + dir.y*dir.y + dir.z*dir.z );

		if(lenSqr <= pown(self->radius+other->radius, 2)) continue; // Avoid division by 0

		self->velocity += normalize(dir) * (G * other->mass / lenSqr) * deltatime;
	}
}


__kernel void render(__global const Star *stars, __global uchar4 *output, const int width, const int height, const Camera camera)
{
	const uint id = get_global_id(0);

	__global const Star *star = &stars[id];

	//float2 prevPos = (float2)(star->position.x, star->position.y);

	//float2 pos = (float2)(star->position.x, star->position.y);

	//draw_line(prevPos, pos, (uchar4)(0xFF, 0xFF, 0xFF, 0xFF), output, width, height);

	float4 pos = (float4)(1.f);
	pos.xyz = star->position;

	// Clip space
	pos = matrixByVector(camera.matrix, pos);
	
	if(pos.z < 0.1f) return;

	float w = pos.w;

	// Screen space (-1 to 1)
	pos /= w;

	float colorFalloff = pow(1.001f, w);

	// Window space
	draw_point((int2)(width * (pos.x+1.f)/2.f, height * (pos.y+1.f)/2.f), convert_uchar4((float4)(0xFF, 0xFF, 0xFF, 0xFF) / colorFalloff), output, width, height);
}

//Clear every pixel in the texture
__kernel void update_texture(__global uchar4 *output, const int width, const int height)
{
	const uint id = get_global_id(0);

	// int2 pos = (int2)(id % width, id / width);

	output[id] = (uchar4)(0, 0, 0, 0xFF);
}
