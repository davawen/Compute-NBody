#include <vector>

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

struct Star
{
	cl_float3 position;
	cl_float3 velocity;
	cl_float mass;
	cl_float radius;

	Star()
	{
		this->position = {{ 0.f }};
		this->velocity = {{ 0.f }};
		this->mass = 1.f;
		this->radius = 1.f;
	}

	Star(cl_float3 position, cl_float3 velocity, cl_float mass)
	{
		this->position = position;
		this->velocity = velocity;
		this->mass = mass;
		this->radius = glm::pow(3 * mass / (4 * 1.f/*density*/ * glm::pi<float>()), 1.f/3.f); // r = cbrt(3m / 4pπ)
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

	SDL_CreateWindowAndRenderer(900, 900, 0, &window, &renderer);
	SDL_SetRelativeMouseMode(SDL_TRUE);

	SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, 900, 900);

	struct
	{
		glm::vec3 position, direction;
	} camera = { { 0.f, -200.f, -500.f }, { 0.f, 0.f, 0.f } }; // Looking down

	// Init compute
	compute::device device = compute::system::default_device();
	compute::context context(device);

	compute::program program = compute::program::create_with_source_file("kernel.cl", context);
	program.build();

	compute::kernel k_update_position(program, "update_position");
	compute::kernel k_update_velocity(program, "update_velocity");

	compute::kernel k_render(program, "render");
	compute::kernel k_update_texture(program, "update_texture");

	compute::command_queue queue(context, device);
	
	std::vector<Star> stars;
	stars.push_back(Star({{ 0 }}, {{ 0 }}, 500000));

	const int numStars = 10000;

	float radiusMultiplier = 300.f;

	// a

	for(size_t i = 0; i < numStars; i++)
	{
		float linearRadius = 1.f;
		float radius = powf(linearRadius, 1.f / 2.f) * radiusMultiplier; //Square root for uniform distribution

		float mass;
		{
			float x = randf();
			float k = glm::pow(1 - .87f, 3);

			mass = (x * k) / (x * k - x + 1) * 10000;
		}

		//Generate disk coordinate
		float u = randf();
		float v = randf();
		
		float theta = u * 2.f * M_PI;
		float phi = v * 2.f * M_PI;
		
		auto bodyPosition = glm::vec3(radius * cos(theta) * sin(phi), radius * sin(theta) * sin(phi), radius * cos(theta));
		// auto bodyPosition = glm::vec3(radius * cos(theta), (randf()*radius - radius/2)*.01f, radius * sin(theta));
		// 
		// float speed = sqrtf( stars[0].mass / radius ) ;

		// glm::vec3 initialVelocity = speed * glm::cross(glm::normalize(bodyPosition), glm::vec3(0, 1, 0)); //Normalize vector
		glm::vec3 initialVelocity = glm::normalize(bodyPosition) * 30.f;

		stars.push_back(
			Star({{ bodyPosition.x, bodyPosition.y, bodyPosition.z }}, {{ initialVelocity.x, initialVelocity.y, initialVelocity.z }}, mass)
		);
	}

	compute::buffer starBuffer = compute::buffer(context, sizeof(Star) * stars.size());
	compute::buffer textureBuffer = compute::buffer(context, 900 * 900 * sizeof(cl_uchar4));

	queue.enqueue_write_buffer(starBuffer, 0, sizeof(Star) * stars.size(), stars.data());

	k_update_position.set_arg(0, starBuffer);

	k_update_velocity.set_arg(0, starBuffer);

	k_render.set_arg(0, starBuffer);
	k_render.set_arg(1, textureBuffer);
	k_render.set_arg(2, 900);
	k_render.set_arg(3, 900);

	k_update_texture.set_arg(0, textureBuffer);
	k_update_texture.set_arg(1, 900);
	k_update_texture.set_arg(2, 900);

	struct
	{
		bool forward, left, right, backwards, up, down;
	} movementKeys = { false, false, false, false, false, false };
	
	float deltatime = 0.f;
	float deltaclock = 0.f;
	
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
		}

		if(deltaclock > 1.f/30.f) // Only update 30 times a second
		{
			long renderStart = now();

			queue.enqueue_1d_range_kernel(k_update_texture, 0, 900*900, 0);

			if(!paused)
			{
				k_update_velocity.set_arg(1, deltaclock);
				queue.enqueue_1d_range_kernel(k_update_velocity, 0, stars.size(), 0);

				k_update_position.set_arg(1, deltaclock);
				queue.enqueue_1d_range_kernel(k_update_position, 0, stars.size(), 0);

				Star star;
				queue.enqueue_read_buffer(starBuffer, sizeof(Star)*18, sizeof(Star), &star);

				// std::cout << glm::to_string(FLOAT3TOVEC3(star.position)) << "\n" << glm::to_string(FLOAT3TOVEC3(star.velocity)) << "\n";
			}

			glm::mat4 orientation = glm::eulerAngleYXZ(-camera.direction.y, -camera.direction.x, 0.f);
			glm::mat4 viewMatrix = glm::lookAt(camera.position, camera.position + glm::vec3(orientation * glm::vec4(0, 0, 1, 1)), glm::vec3(0, 1, 0));

			{
				const glm::vec3 movement = glm::normalize(glm::vec3(orientation * glm::vec4(movementKeys.left - movementKeys.right, movementKeys.down - movementKeys.up, movementKeys.forward - movementKeys.backwards, 0))); // 0 at the end nullify's translation

				if(!glm::all(glm::isnan(movement))) camera.position += 5.f * movement;
			}

			glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.f), 1.f, .1f, 1000.f);
			projectionMatrix *= viewMatrix;

			cl_float4 cameraMatrix[4] = {
				VEC4TOCL(projectionMatrix[0]),
				VEC4TOCL(projectionMatrix[1]),
				VEC4TOCL(projectionMatrix[2]),
				VEC4TOCL(projectionMatrix[3])
			};

			k_render.set_arg(4, sizeof(cl_float4)*4, cameraMatrix);

			queue.enqueue_1d_range_kernel(k_render, 0, stars.size(), 0);
			
			SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0xFF); // Clear renderer while reading texture data
			SDL_RenderClear(renderer);

			cl_uchar4 *pixels = (cl_uchar4 *)queue.enqueue_map_buffer(textureBuffer, CL_MAP_READ, 0, 900*900 * sizeof(cl_uchar4));

			SDL_UpdateTexture(texture, NULL, pixels, 900 * 4);

			queue.enqueue_unmap_buffer(textureBuffer, pixels);

			SDL_RenderCopy(renderer, texture, NULL, NULL);

			SDL_RenderPresent(renderer);

			averageFrametime += now() -renderStart;
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

	return 0;
}
