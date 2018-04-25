#include <iostream>

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
bool InRectangle(double HNx, double HNy, double HMx, double HMy, double front_margin, double back_margin, double side_margin) {
	double HLx = - HNy, // direction after 90 degree anticlockwise rotation
				 HLy = HNx;

	double HM_HN = HMx * HNx + HMy * HNy, // HM . HN
				 HN_HN = HNx * HNx + HNy * HNy; // HN . HN
	if (HM_HN >= 0 && HM_HN * HM_HN > HN_HN * front_margin * front_margin)
		return false;
	if (HM_HN <= 0 && HM_HN * HM_HN > HN_HN * back_margin * back_margin)
		return false;

	double HM_HL = HMx * HLx + HMy * HLy, // HM . HL
				 HL_HL = HLx * HLx + HLy * HLy; // HL . HL
	return HM_HL * HM_HL <= HL_HL * side_margin * side_margin;
}

/**
 * H: center of the head of the car
 * N: a point right in front of the car
 * M: an arbitrary point
 *
 * Check whether M is in the safety zone
 */
bool inCollision(double Mx, double My, double Hx, double Hy, double Nx, double Ny) {

	/// for golfcart
/*	double HNx = Nx - Hx, // car direction
				 HNy = Ny - Hy;
	double HMx = Mx - Hx,
				 HMy = My - Hy;

	double car_width = 0.87,
				 car_length = 1.544;

				 double safe_margin = 0.92, side_safe_margin = 0.45, back_safe_margin = 0.15,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = car_length/2.0 + safe_margin,
				 back_margin = car_length/2.0 + back_safe_margin;*/

	/// for audi r8
	double HNx = Nx - Hx, // car direction
				 HNy = Ny - Hy;
	double HMx = Mx - Hx,
				 HMy = My - Hy;

	double car_width = 2.0,
				 car_length = 4.4;

				 double safe_margin = 0.8, side_safe_margin = 0.1, back_safe_margin = 0.1,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = 3.6 + safe_margin,
				 back_margin = 0.8 + back_safe_margin;

	return InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin);
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

bool inRealCollision(double Mx, double My, double Hx, double Hy, double Nx, double Ny) {

	/// for golfcart
/*	double HNx = Nx - Hx, // car direction
				 HNy = Ny - Hy;
	double HMx = Mx - Hx,
				 HMy = My - Hy;

	double car_width = 0.87,
				 car_length = 1.544;
	double safe_margin = 0.0, side_safe_margin = 0.0, back_safe_margin = 0.0,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = car_length/2.0 + safe_margin,
				 back_margin = car_length/2.0 + back_safe_margin;
	cout<<"whole body: "<<InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin)<<endl;

	return InFrontRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin);*/


	/// for audi r8
	/// for audi r8
	double HNx = Nx - Hx, // car direction
				 HNy = Ny - Hy;
	double HMx = Mx - Hx,
				 HMy = My - Hy;

	double car_width = 1.9,
				 car_length = 4.4;

				 double safe_margin = 0.0, side_safe_margin = 0.0, back_safe_margin = 0.0,
				 side_margin = car_width / 2.0 + side_safe_margin,
				 front_margin = 3.6 + safe_margin,
				 back_margin = 0.8 + back_safe_margin;

	return InRectangle(HNx, HNy, HMx, HMy, front_margin, back_margin, side_margin);
}

void testInCollision() {
	double Hx = 0.0, Hy = 0.0, Nx = 0.0, Ny = 10.0;
	double Mx = -1.30, My = 1.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 1" << endl << endl;

	Mx = -1.40, My = 1.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 0" << endl << endl;

	Mx = 1.30, My = 1.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 1" << endl << endl;

	Mx = 1.40, My = 1.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 0" << endl << endl;

	My = -3.0, My = 1.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 0" << endl;

	My = -3.0, My = 1.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 0" << endl << endl;

	Nx = 1, Ny = 1;

	My = -3.0, My = 1.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 0" << endl << endl;

	My = -1.0, My = 0.0;
	cout << "In collision: computed / true = " << inCollision(Mx, My, Hx, Hy, Nx, Ny) << " / 1" << endl << endl;
}

// int main() {
// 	testInCollision();
// 	return 0;
// }
