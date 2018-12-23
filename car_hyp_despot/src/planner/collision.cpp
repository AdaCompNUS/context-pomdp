#include <iostream>
#include "param.h"

using namespace std;

/**
 *    A-----------N-----------B
 *    |           ^           |
 *    |           |           |
 *    |     |-L<--H-----|     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |M          |     |
 *    |     |           |     |
 *    |     |           |     |
 *    |     |-----------|     |
 *    |                       |
 *    |                       |
 *    D-----------------------C
 *
 * H: center of the head of the car
 * N: a point right in front of the car
 * L: a point to the left/right of H
 *
 * A point M is inside the safety zone ABCD iff
 *   ((0 <= HM . HN & (HM . HN)^2 <= (HN . HN) * front_margin^2) || (0 => HM . HN & (HM . HN)^2 < (HN . HN) * back_margin^2))
 *   && (HM . HL)^2 <= (HL . HL) * side_margin^2
 */
bool InRectangle(double car_dir_x, double car_dir_y, double car_ped_x, double car_ped_y, double front_margin, double back_margin, double side_margin) {
	double car_tan_x = - car_dir_y, // direction after 90 degree anticlockwise rotation
				 car_tan_y = car_dir_x;

	double ped_proj_1 = car_ped_x * car_dir_x + car_ped_y * car_dir_y, // HM . HN
				 denom_1 = car_dir_x * car_dir_x + car_dir_y * car_dir_y; // HN . HN
	if (ped_proj_1 >= 0 && ped_proj_1 * ped_proj_1 > denom_1 * front_margin * front_margin)
		return false;
	if (ped_proj_1 <= 0 && ped_proj_1 * ped_proj_1 > denom_1 * back_margin * back_margin)
		return false;

	double ped_proj_2 = car_ped_x * car_tan_x + car_ped_y * car_tan_y, // HM . HL
				 denom_2 = car_tan_x * car_tan_x + car_tan_y * car_tan_y; // HL . HL
	return ped_proj_2 * ped_proj_2 <= denom_2 * side_margin * side_margin;
}

/**
 * H: center of the head of the car
 * N: a point right in front of the car
 * M: an arbitrary point
 *
 * Check whether M is in the safety zone
 */
bool inCollision(double ped_x, double ped_y, double car_x, double car_y, double Ctheta) {

	/// car geometry
	double car_dir_x = cos(Ctheta), // car direction
				 car_dir_y = sin(Ctheta);
	double car_ped_x = ped_x - car_x,
				 car_ped_y = ped_y - car_y;

	double side_margin,front_margin, back_margin;
	if(ModelParams::car_model == "pomdp_car"){
		/// pomdp car
		double car_width = CAR_WIDTH,
					 car_length = CAR_LENGTH;
		side_margin = car_width / 2.0 + CAR_SIDE_MARGIN + PED_SIZE;
		front_margin = CAR_FRONT_MARGIN + PED_SIZE;
		back_margin = car_length + CAR_SIDE_MARGIN + PED_SIZE;
	}else if(ModelParams::car_model == "audi_r8"){

		double car_width = 2.0,
				car_length = 4.4;

		double safe_margin = /*0.8*/0.0, side_safe_margin = 0.1, back_safe_margin = 0.1;
		side_margin = car_width / 2.0 + side_safe_margin;
		front_margin = 3.6 + safe_margin;
		back_margin = 0.8 + back_safe_margin;
	}

	return InRectangle(car_dir_x, car_dir_y, car_ped_x, car_ped_y, front_margin, back_margin, side_margin);
}


bool InFrontRectangle(double HNx, double HNy, double HMx, double HMy, double front_margin, double back_margin, double side_margin) {
	double HLx = - HNy, // direction after 90 degree anticlockwise rotation
				 HLy = HNx;

	double HM_HN = HMx * HNx + HMy * HNy, // HM . HN
				 HN_HN = HNx * HNx + HNy * HNy; // HN . HN

	if (HM_HN <= 0) return false;
	if (HM_HN >= 0 && HM_HN * HM_HN > HN_HN * front_margin * front_margin)
		return false;

	double HM_HL = HMx * HLx + HMy * HLy, // HM . HL
				 HL_HL = HLx * HLx + HLy * HLy; // HL . HL
	return HM_HL * HM_HL <= HL_HL * side_margin * side_margin;
}

bool inRealCollision(double Mx, double My, double Hx, double Hy, double Ctheta) {

	double HNx = cos(Ctheta), // car direction
			     HNy = sin(Ctheta);
	double HMx = Mx - Hx,
				 HMy = My - Hy;

	double side_margin,front_margin, back_margin;

	if(ModelParams::car_model == "pomdp_car"){
		/// pomdp car
		double car_width = CAR_WIDTH,
					 car_length = CAR_LENGTH;

		side_margin = car_width / 2.0 + PED_SIZE;
		front_margin = 0.0 + PED_SIZE;
		back_margin = car_length + PED_SIZE;
	}else if (ModelParams::car_model == "audi_r8"){
		/// audi r8
		double car_width = 1.9,
				car_length = 4.4;

		double safe_margin = 0.0, side_safe_margin = 0.0, back_safe_margin = 0.0;
		side_margin = car_width / 2.0 + side_safe_margin;
		front_margin = 3.6 + safe_margin;
		back_margin = 0.8 + back_safe_margin;
	}

	return InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin);
}

