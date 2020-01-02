#include <iostream>
#include "param.h"
#include "coord.h"
#include <vector>

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
bool InRectangle(double car_dir_x, double car_dir_y, double car_ped_x,
		double car_ped_y, double front_margin, double back_margin,
		double side_margin) {
	double car_tan_x = -car_dir_y, car_tan_y = car_dir_x;
	double ped_proj_1 = car_ped_x * car_dir_x + car_ped_y * car_dir_y; // HM . HN
	double denom_1 = car_dir_x * car_dir_x + car_dir_y * car_dir_y; // HN . HN
	if (ped_proj_1 >= 0
			&& ped_proj_1 * ped_proj_1 > denom_1 * front_margin * front_margin)
		return false;
	if (ped_proj_1 <= 0
			&& ped_proj_1 * ped_proj_1 > denom_1 * back_margin * back_margin)
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
bool InCollision(double ped_x, double ped_y, double car_x, double car_y,
		double Ctheta, bool expand) {
	/// car geometry
	double car_dir_x = cos(Ctheta), car_dir_y = sin(Ctheta);
	double car_ped_x = ped_x - car_x, car_ped_y = ped_y - car_y;

	double expand_size = (expand) ? PED_SIZE : 0.0;
	double side_margin = ModelParams::CAR_WIDTH / 2.0 + CAR_SIDE_MARGIN
			+ expand_size;
	double front_margin = ModelParams::CAR_FRONT + CAR_FRONT_MARGIN
			+ expand_size;
	double back_margin = ModelParams::CAR_FRONT + CAR_SIDE_MARGIN + expand_size;

	return InRectangle(car_dir_x, car_dir_y, car_ped_x, car_ped_y, front_margin,
			back_margin, side_margin);
}

bool inCarlaCollision(double ped_x, double ped_y, double car_x, double car_y,
		double Ctheta, double car_extent_x, double car_extent_y, bool flag) {
	double car_dir_x = cos(Ctheta), car_dir_y = sin(Ctheta);
	double car_ped_x = ped_x - car_x, car_ped_y = ped_y - car_y;

	double side_margin, front_margin, back_margin;
	if (flag == 0) { // in search
		side_margin = car_extent_x + CAR_SIDE_MARGIN;
		front_margin = car_extent_y + CAR_FRONT_MARGIN;
		back_margin = car_extent_y + CAR_SIDE_MARGIN;
	} else { // real collision
		side_margin = car_extent_x;
		front_margin = car_extent_y;
		back_margin = car_extent_y;
	}

	return InRectangle(car_dir_x, car_dir_y, car_ped_x, car_ped_y, front_margin,
			back_margin, side_margin);
}

bool InFrontRectangle(double HNx, double HNy, double HMx, double HMy,
		double front_margin, double back_margin, double side_margin) {
	double HLx = -HNy, HLy = HNx;
	double HM_HN = HMx * HNx + HMy * HNy, HN_HN = HNx * HNx + HNy * HNy; // HN . HN

	if (HM_HN <= 0)
		return false;
	if (HM_HN >= 0 && HM_HN * HM_HN > HN_HN * front_margin * front_margin)
		return false;

	double HM_HL = HMx * HLx + HMy * HLy, HL_HL = HLx * HLx + HLy * HLy; // HL . HL
	return HM_HL * HM_HL <= HL_HL * side_margin * side_margin;
}

bool InRealCollision(double Mx, double My, double Hx, double Hy, double Ctheta,
		bool expand) {
	double HNx = cos(Ctheta), HNy = sin(Ctheta);
	double HMx = Mx - Hx, HMy = My - Hy;

	double expand_size = (expand) ? PED_SIZE : 0.0;
	double side_margin = ModelParams::CAR_WIDTH / 2.0 + expand_size;
	double front_margin = ModelParams::CAR_FRONT + expand_size;
	double back_margin = ModelParams::CAR_FRONT + expand_size;

	return InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin,
			side_margin);
}

bool Xor(bool a, bool b) {
	return ((a && !b) || (!a && b));
}

//is the line segments p0-p1-p2 counter-clockwise;
// ie, is p2 on the left side of the vector p0-p1
bool IsCcw(COORD p0, COORD p1, COORD p2) {
	COORD u = p1 - p0;
	COORD v = p2 - p1;
	return (u.x * v.y - v.x * u.y) > 0; // >0
}

//is segment p0_p1 intersects with segment q0_q1
bool IsIntersecting(COORD p0, COORD p1, COORD q0, COORD q1) {
	return Xor(IsCcw(p0, p1, q0), IsCcw(p0, p1, q1))
			&& Xor(IsCcw(q0, q1, p0), IsCcw(q0, q1, p1));
}

// requires: rect is represented by vertices oriented in counter-clockwise direction
bool InRectangle(COORD p, std::vector<COORD> rect) {
	rect.push_back(rect[0]);
	for (int i = 0; i < rect.size() - 1; i++)
		if (!IsCcw(rect[i], rect[i + 1], p))
			return false;
	return true;
}

std::vector<COORD> ComputeRect(COORD pos, double heading,
		double ref_to_front_side, double ref_to_back_side,
		double ref_front_side_angle, double ref_back_side_angle) {
	COORD front_left, front_right, back_left, back_right;

	front_left.x = pos.x
			+ ref_to_front_side * cos(heading + ref_front_side_angle);
	front_left.y = pos.y
			+ ref_to_front_side * sin(heading + ref_front_side_angle);
	front_right.x = pos.x
			+ ref_to_front_side * cos(heading - ref_front_side_angle);
	front_right.y = pos.y
			+ ref_to_front_side * sin(heading - ref_front_side_angle);
	back_left.x = pos.x
			+ ref_to_back_side * cos(heading + M_PI - ref_back_side_angle);
	back_left.y = pos.y
			+ ref_to_back_side * sin(heading + M_PI - ref_back_side_angle);
	back_right.x = pos.x
			+ ref_to_back_side * cos(heading + M_PI + ref_back_side_angle);
	back_right.y = pos.y
			+ ref_to_back_side * sin(heading + M_PI + ref_back_side_angle);

	std::vector<COORD> rect;
	rect.push_back(front_left);
	rect.push_back(back_left);
	rect.push_back(back_right);
	rect.push_back(front_right);

	return rect;
}

/**
 * There are only two collision cases:
 * 1. one rectangle is totally inside the other, or
 * 2. at least one edge of a rectangle intersects with the edge(s) of the other rectangle
 */
bool InCollision(std::vector<COORD> rect_1, std::vector<COORD> rect_2) {
	//check whether there exists one vertex of rect_1 inside rect_2
	for (int i = 0; i < rect_1.size(); i++)
		if (InRectangle(rect_1[i], rect_2))
			return true;
	//check whether there exists one vertex of rect_2 inside rect_1
	for (int i = 0; i < rect_2.size(); i++)
		if (InRectangle(rect_2[i], rect_1))
			return true;

	return false;
}

