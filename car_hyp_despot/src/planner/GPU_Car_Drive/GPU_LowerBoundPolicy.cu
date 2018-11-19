#include "GPU_LowerBoundPolicy.h"

#include <despot/GPUcore/GPUhistory.h>
#include <despot/GPUrandom_streams.h>
#include <despot/GPUutil/GPUcoord.h>
#include "GPU_Car_Drive.h"
using archaeopteryx::util::numeric_limits;
using despot::Dvc_History;
using despot::Dvc_RandomStreams;
using despot::Dvc_State;

#include <despot/GPUinterface/GPUpomdp.h>

DEVICE int Dvc_PedPomdpSmartPolicy::Action(
		int scenarioID, const Dvc_State* particles,
				Dvc_RandomStreams& streams,
				Dvc_History& history)
{

	const Dvc_PomdpState &state=static_cast<const Dvc_PomdpState&>(particles[0]);
	__shared__ int mindist[32];
	auto& carpos = state.car.pos;

	float carvel = state.car.vel;

	mindist[threadIdx.x]=__float_as_int(archaeopteryx::util::numeric_limits<float>::infinity());

	__syncthreads();

	if (threadIdx.y<state.num) {
		auto& p = state.peds[threadIdx.y];
		bool infront=false;

		if(Dvc_ModelParams::IN_FRONT_ANGLE_DEG >= 180.0) {
			// inFront check is disabled in this case
			infront=true;
		}
		else
		{
			const Dvc_COORD& car_pos = state.car.pos;
			const double car_heading = state.car.heading_dir;
			
			float d0 = Dvc_COORD::EuclideanDistance(car_pos, p.pos);

			if(d0 <= /*0.7*/3.5)
				infront=true;
			else
			{
				float d1 = 1;
				float dot = Dvc_Vector::DotProduct(cos(car_heading), sin(car_heading),
						p.pos.x - car_pos.x, p.pos.y - car_pos.y);
				float cosa = dot / (d0 * d1);
				if(cosa > 1.0 + 1E-6 || cosa < -1.0 - 1E-6)
				{
					;

				}
				infront=cosa > in_front_angle_cos;
			}
		}



		if(infront) {
			float d = Dvc_COORD::EuclideanDistance(carpos, p.pos);
			atomicMin(mindist+threadIdx.x, __float_as_int(d));
		}
	}

	__syncthreads();

	// TODO set as a param
	if (__int_as_float(mindist[threadIdx.x]) < /*2*/3.5) {
		return (carvel <= 0.01) ? 0 : 2;
	}

	if (__int_as_float(mindist[threadIdx.x]) < /*4*/5) {
		if (carvel > 1.0+1e-4) return 2;
		else if (carvel < 0.5-1e-4) return 1;
		else return 0;
	}


	return carvel >= Dvc_ModelParams::VEL_MAX-1e-4 ? 0 : 1;
	return 0;
}
