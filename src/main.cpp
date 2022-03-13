#include <vector>
#include <stack>

#include <sys/time.h>
#include <SDL2/SDL.h>

#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/rotate_vector.hpp>

#define CL_TARGET_OPENCL_VERSION 300
#include <boost/compute.hpp>
#include <boost/compute/utility.hpp>

#define VEC3TOCL(v) ( cl_float3( {{ (v).x, (v).y, (v).z }} ) )
#define VEC4TOCL(v) ( cl_float4( {{ (v).x, (v).y, (v).z, (v).w }} ) )

#define FLOAT3TOVEC3(v) ( glm::vec3( (v).x, (v).y, (v).z ) )

long now()
{
	timeval curr;
	gettimeofday(&curr, 0);

	return curr.tv_sec*1'000'000 + curr.tv_usec;
}

bool point_in_rect(const glm::vec3 &p, const glm::vec3 &a, const glm::vec3 &b)
{
	return p.x >= a.x && p.y >= a.y && p.z >= a.z && p.x < b.x && p.y < b.y && p.z < b.z;
}

struct Star
{
	alignas(cl_float3) glm::vec3 position;
	alignas(cl_float3) glm::vec3 velocity;
	alignas(cl_float3) cl_float mass;
	cl_float radius;

	Star()
	{
		this->position = glm::vec3(0.f);
		this->velocity = glm::vec3(0.f);
		this->mass = 1.f;
		this->radius = 1.f;
	}

	Star(const glm::vec3 &position, const glm::vec3 &velocity, cl_float mass)
	{
		this->position = position;
		this->velocity = velocity;
		this->mass = mass;
		this->radius = glm::pow(3 * mass / (4 * 1.f/*density*/ * glm::pi<float>()), 1.f/3.f); // r = cbrt(3m / 4pπ)
	}
};

struct Node
{
	alignas(cl_float3) glm::vec3 minBound;
	alignas(cl_float3) glm::vec3 maxBound;

	alignas(cl_float3) glm::vec3 centerOfMass;
	alignas(cl_float3) cl_float totalMass;
	
    enum struct State: cl_uchar
    {
     	STATE_EMPTY,
     	STATE_LEAF,
     	STATE_BRANCH
    };
	State type;
	size_t child;
	

	Node(const glm::vec3 &minBound, const glm::vec3 &maxBound)
	{
		this->minBound = minBound;
		this->maxBound = maxBound;

		this->centerOfMass = glm::vec3(0.f);
		this->totalMass = 0;

		child = -1; // overflows to max size_t value

		this->type = State::STATE_EMPTY;
	}

	void insert(std::vector<Node> &nodes, const std::vector<Star> &stars, size_t idx, int depth = 18)
	{
		const Star &star = stars[idx];

		if(!point_in_rect(star.position, minBound, maxBound)) return;

		centerOfMass += stars[idx].position * stars[idx].mass;
		totalMass += stars[idx].mass;

		if(type == State::STATE_EMPTY)
		{
			type = State::STATE_LEAF;
			child = idx;
		}
		else if(depth > 0)
		{
			if(type == State::STATE_LEAF)
			{
				glm::vec3 middle = (minBound + maxBound) / 2.f;

				nodes.push_back(Node(minBound, middle)); // top left

				size_t childIdx = nodes.size() - 1;

				nodes.push_back(Node({ middle.x, minBound.y, minBound.z }, { maxBound.x, middle.y, middle.z })); // top right
				nodes.push_back(Node({ minBound.x, middle.y, minBound.z }, { middle.x, maxBound.y, middle.z })); // bottom left
				nodes.push_back(Node({ middle.x, middle.y, minBound.z }, { maxBound.x, maxBound.y, middle.z })); // bottom right

				nodes.push_back(Node({ minBound.x, minBound.y, middle.z }, { middle.x, middle.y, maxBound.z })); // top left
				nodes.push_back(Node({ middle.x, minBound.y, middle.z }, { maxBound.x, middle.y, maxBound.z })); // top right
				nodes.push_back(Node({ minBound.x, middle.y, middle.z }, { middle.x, maxBound.y, maxBound.z })); // bottom left
				nodes.push_back(Node({ middle.x, middle.y, middle.z }, { maxBound.x, maxBound.y, maxBound.z })); // bottom right

				for(size_t i = 0; i < 8; i++)
				{
					nodes[childIdx + i].insert(nodes, stars, child, depth - 1);
				}

				child = childIdx;

				type = State::STATE_BRANCH;
			}

			for(size_t i = 0; i < 8; i++)
			{
				nodes[child + i].insert(nodes, stars, idx, depth - 1);
			}
		}
	}

	void average_com(std::vector<Node> &nodes)
	{
	 	if(type == State::STATE_EMPTY) return; // If empty, don't do anything

	 	centerOfMass /= totalMass;

	 	if(type == State::STATE_LEAF) return; // If you're a branch, repeat for all children

	 	for(size_t i = 0; i < 8; i++) nodes[child + i].average_com(nodes);
	}
	
	void iterative_average_com(std::vector<Node> &nodes)
	{
		std::vector<Node *> stack;
		stack.reserve(16);
		stack.push_back(this);

		while(!stack.empty())
		{
			Node *curr = stack.back();
			stack.pop_back();

			if(curr->type == State::STATE_EMPTY) continue;

			curr->centerOfMass /= curr->totalMass;

			if(curr->type == State::STATE_LEAF) continue;

			for(size_t i = 0; i < 8; i++) stack.push_back(&nodes[curr->child + i]);
		}
	}

	glm::vec3 calculate_forces(const std::vector<Node> &nodes, const Star &star) const
	{
	 	if(type == State::STATE_EMPTY) return glm::vec3(0.f);

	 	float s = maxBound.x - minBound.x;

	 	glm::vec3 dir = centerOfMass - star.position;
	 	
	 	if(glm::any(glm::isnan(dir))) return glm::vec3(0.f);

	 	float distanceSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
	 
	 	if(distanceSqr <= star.radius) return glm::vec3(0.f);

	 	float d = sqrtf(distanceSqr);

	 	if(type == State::STATE_LEAF || s/d < 1.f)
	 	{
	 		return (dir / d) * ((totalMass) / distanceSqr);
	 	}
	 	else
	 	{
	 		glm::vec3 acc(0);

	 		for(size_t i = 0; i < 8; i++) acc += nodes[child + i].calculate_forces(nodes, star);

	 		return acc;
	 	}
	}

	glm::vec3 iterative_calculate_forces(const std::vector<Node> &nodes, const Star &star) const
	{
		glm::vec3 force(0);

		// node pointer, was parent branch
		std::pair<const Node *, bool> stack[256];
		int sp = 0;

		stack[0] = { this, false };

		while(sp > -1)
		{
			auto popped = stack[sp];
			sp--;
			size_t num = popped.second ? 8 : 1;

			for(size_t i = 0; i < num; i++)
			{
				const Node *curr = popped.first + i;

				if(curr->type == State::STATE_EMPTY) continue;

				float s = curr->maxBound.x - curr->minBound.x;

				glm::vec3 dir = curr->centerOfMass - star.position;

				if(glm::any(glm::isnan(dir))) continue;

				float distanceSqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;

				if(distanceSqr <= star.radius) continue;

				float d = sqrtf(distanceSqr);

				if(curr->type == State::STATE_LEAF || s/d <= 0.f || sp == 255)
				{
					force += (dir / d) * ((curr->totalMass) / distanceSqr);
				}
				else
				{
					sp++;
					stack[sp] = { &nodes[curr->child], true };
				}
			}
		}

		return force;
	}

	void print_tree(FILE *graph, const std::vector<Node> &nodes) const
	{
		if(type == State::STATE_BRANCH)
		{
			for(size_t i = 0; i < 8; i++)
			{
				fprintf(graph, "\t\"%zu\" -- \"%zu\"\n", this - nodes.data(), child + i);
				nodes[child + i].print_tree(graph, nodes);
			}
		}
	}
};

float randf()
{
	return static_cast<float>(rand() % 10000) / 10000.f;
}

int main(void)
{
	namespace compute = boost::compute;
	
	// Init SDL
	SDL_Window *window;
	SDL_Renderer *renderer;

	SDL_CreateWindowAndRenderer(1200, 900, 0, &window, &renderer);
	SDL_SetRelativeMouseMode(SDL_TRUE);

	SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, 1200, 900);

	struct
	{
		glm::vec3 position, direction;
	} camera = { { 0.f, -200.f, -500.f }, { 0.f, 0.f, 0.f } }; // Looking down

	// Init compute
	compute::device device = compute::system::default_device();
	compute::context context(device);

	compute::program program = compute::program::create_with_source_file("kernel.cl", context);
	program.build();

	compute::kernel kUpdatePosition(program, "update_position");
	compute::kernel kUpdateVelocity(program, "update_velocity");
	compute::kernel kUpdateVelocityBarnes(program, "update_velocity_barnes");

	compute::kernel kRender(program, "render");
	compute::kernel kUpdateTexture(program, "update_texture");

	compute::command_queue queue(context, device);
	
	std::vector<Star> stars;
	stars.push_back(Star(glm::vec3(0), glm::vec3(0), 500000));

	const int numStars = 10000;

	float radiusMultiplier = 500.f;

	for(size_t i = 0; i < numStars; i++)
	{
		float mass;
		{
			float x = randf();
			float k = glm::pow(1 - .87f, 3);

			mass = (x * k) / (x * k - x + 1) * 10000;
		}

		//Generate disk coordinate
		float linearRadius = randf();
		float radius = powf(linearRadius, 1.f / 2.f) * radiusMultiplier; //Square root for uniform distribution

		if(radius < stars[0].radius*2) continue;

		float u = randf();
		
		float theta = u * 2.f * M_PI;
		
		auto bodyPosition = glm::vec3(radius * cos(theta), (randf()*radius - radius/2)*.05f, radius * sin(theta));
		 
		float speed = sqrtf( stars[0].mass / radius ) ;

		glm::vec3 initialVelocity = speed * glm::cross(glm::normalize(bodyPosition), glm::vec3(0, 1, 0)); //Normalize vector

		stars.push_back(
			Star(bodyPosition, initialVelocity, mass)
		);
	}

	std::vector<Node> tree;

	compute::buffer starBuffer = compute::buffer(context, sizeof(Star) * stars.size());
	compute::buffer treeBuffer = compute::buffer(context, 0);
	compute::buffer textureBuffer = compute::buffer(context, 1200 * 900 * sizeof(cl_uchar4));

	queue.enqueue_write_buffer(starBuffer, 0, sizeof(Star) * stars.size(), stars.data());

	kUpdatePosition.set_arg(0, starBuffer);
	kUpdateVelocity.set_arg(0, starBuffer);
	kUpdateVelocityBarnes.set_arg(0, starBuffer);
	kUpdateVelocityBarnes.set_arg(1, treeBuffer);

	kRender.set_arg(0, starBuffer);
	kRender.set_arg(1, textureBuffer);
	kRender.set_arg(2, 1200);
	kRender.set_arg(3, 900);

	kUpdateTexture.set_arg(0, textureBuffer);
	kUpdateTexture.set_arg(1, 1200);
	kUpdateTexture.set_arg(2, 900);

	struct
	{
		bool forward, left, right, backwards, up, down;
	} movementKeys = { false, false, false, false, false, false };

	FILE *graph = fopen("output.dot", "w");
	fprintf(graph, "graph {\n");
	
	float deltatime = 0.f;
	float deltaclock = 0.f;

	float speed = 1.f;
	
	double averageFrametime = 0.;
	int ticks = 0;

	bool running = true;
	bool paused = true;
	while(running)
	{
		long frameStart = now();

		SDL_Event event;
		while(SDL_PollEvent(&event))
		{
			if(event.type == SDL_EventType::SDL_QUIT)
			{
				running = false;
			}
			else if(event.type == SDL_EventType::SDL_KEYDOWN)
			{
				switch(event.key.keysym.sym)
				{
					case SDLK_z:
						movementKeys.forward = true;
						break;
					case SDLK_s:
						movementKeys.backwards = true;
						break;
					case SDLK_d:
						movementKeys.right = true;
						break;
					case SDLK_a:
						movementKeys.left = true;
						break;
					case SDLK_c:
						movementKeys.down = true;
						break;
					case SDLK_SPACE:
						movementKeys.up = true;
						break;
				}
			}
			else if(event.type == SDL_EventType::SDL_KEYUP)
			{
				switch(event.key.keysym.sym)
				{
					case SDLK_z:
						movementKeys.forward = false;
						break;
					case SDLK_s:
						movementKeys.backwards = false;
						break;
					case SDLK_d:
						movementKeys.right = false;
						break;
					case SDLK_a:
						movementKeys.left = false;
						break;
					case SDLK_c:
						movementKeys.down = false;
						break;
					case SDLK_SPACE:
						movementKeys.up = false;
						break;
					case SDLK_ESCAPE:
						SDL_SetRelativeMouseMode(SDL_FALSE);
						break;
					case SDLK_p:
						paused = !paused;
						break;
				}
			}
			else if(event.type == SDL_EventType::SDL_MOUSEMOTION)
			{
				if(event.motion.xrel != 0)
				{
					camera.direction.y += glm::pi<float>() * event.motion.xrel / 3000.f;
				}

				if(event.motion.yrel != 0)
				{
					camera.direction.x += glm::pi<float>() * event.motion.yrel / 3000.f;
				}
			}
			else if(event.type == SDL_EventType::SDL_MOUSEBUTTONDOWN)
			{
				if(event.button.button & SDL_BUTTON_LMASK) SDL_SetRelativeMouseMode(SDL_TRUE);
			}
			else if(event.type == SDL_EventType::SDL_MOUSEWHEEL)
			{
				speed += event.wheel.y * .1f;
			}
		}

		if(deltaclock > 1.f/30.f) // Only update 30 times a second
		{
			long renderStart = now();

			queue.enqueue_1d_range_kernel(kUpdateTexture, 0, 1200*900, 0);

			if(!paused)
			{
				tree.clear();

				glm::vec3 minBound(FLT_MAX);
				glm::vec3 maxBound(-FLT_MAX);
				for(auto &star : stars)
				{
					minBound = glm::min(minBound, star.position);
					maxBound = glm::max(maxBound, star.position);
				}

				tree.push_back(Node(minBound, maxBound)); // Push back root

				for(size_t i = 0; i < stars.size(); i++)
				{
					tree[0].insert(tree, stars, i);
				}

				tree[0].average_com(tree);

				// for(auto &star : stars)
				// {
				// 	glm::vec3 force = tree[0].iterative_calculate_forces(tree, star);

				// 	star.velocity += force * deltaclock;
				// }
				// queue.enqueue_write_buffer(starBuffer, 0, sizeof(Star) * stars.size(), stars.data());

				// kUpdateVelocity.set_arg(1, deltaclock);
				// queue.enqueue_1d_range_kernel(kUpdateVelocity, 0, stars.size(), 0);

				if(sizeof(Node) * tree.size() > treeBuffer.size())
				{
					treeBuffer = compute::buffer(context, sizeof(Node) * (tree.size() + 16));
					kUpdateVelocityBarnes.set_arg(1, treeBuffer);
				}

				queue.enqueue_write_buffer(treeBuffer, 0, sizeof(Node) * tree.size(), tree.data());

				kUpdateVelocityBarnes.set_arg(2, deltaclock);
				queue.enqueue_1d_range_kernel(kUpdateVelocityBarnes, 0, stars.size(), 0);
				
				kUpdatePosition.set_arg(1, deltaclock);
				queue.enqueue_1d_range_kernel(kUpdatePosition, 0, stars.size(), 0);

				queue.enqueue_read_buffer(starBuffer, 0, sizeof(Star) * stars.size(), stars.data());
			}

			glm::mat4 orientation = glm::eulerAngleYXZ(-camera.direction.y, -camera.direction.x, 0.f);
			glm::mat4 viewMatrix = glm::lookAt(camera.position, camera.position + glm::vec3(orientation * glm::vec4(0, 0, 1, 1)), glm::vec3(0, 1, 0));

			{
				const glm::vec3 movement = glm::normalize(glm::vec3(orientation * glm::vec4(movementKeys.left - movementKeys.right, movementKeys.down - movementKeys.up, movementKeys.forward - movementKeys.backwards, 0))); // 0 at the end nullify's translation

				if(!glm::all(glm::isnan(movement))) camera.position += speed * movement;
			}

			glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.f), 1.f, .1f, 1000.f);
			projectionMatrix *= viewMatrix;

			cl_float4 cameraMatrix[4] = {
				VEC4TOCL(projectionMatrix[0]),
				VEC4TOCL(projectionMatrix[1]),
				VEC4TOCL(projectionMatrix[2]),
				VEC4TOCL(projectionMatrix[3])
			};

			kRender.set_arg(4, sizeof(cl_float4)*4, cameraMatrix);

			queue.enqueue_1d_range_kernel(kRender, 0, stars.size(), 0);
			
			SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0xFF); // Clear renderer while reading texture data
			SDL_RenderClear(renderer);

			cl_uchar4 *pixels = (cl_uchar4 *)queue.enqueue_map_buffer(textureBuffer, CL_MAP_READ, 0, 1200*900 * sizeof(cl_uchar4));

			SDL_UpdateTexture(texture, NULL, pixels, 1200 * 4);

			queue.enqueue_unmap_buffer(textureBuffer, pixels);

			SDL_RenderCopy(renderer, texture, NULL, NULL);

			SDL_RenderPresent(renderer);

			averageFrametime += now() - renderStart;
			if(++ticks % 60 == 0)
			{
				std::cout << "Frame took " << averageFrametime/60. << " µs to render\n";
				averageFrametime = 0.;
				ticks = 0;
			}

			deltaclock = 0.f;
		}

		deltaclock += deltatime;



		deltatime = static_cast<double>(now() - frameStart) / 1'000'000.0;
	}

	SDL_DestroyTexture(texture);
	SDL_DestroyWindow(window);
	
	SDL_DestroyRenderer(renderer);

	SDL_Quit();

	fprintf(graph, "\n}");
	fclose(graph);

	return 0;
}
