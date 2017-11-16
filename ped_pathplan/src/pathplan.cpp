#include <ped_pathplan/pathplan.h>
#include <ros/console.h>
#include <vector>
#include <algorithm>
#include <set>
#include <queue>
#include <utility>
#include <iostream>

namespace ped_pathplan {

    using namespace std;

    inline float sqr(float x) { return x*x;}
    inline float normAngle(float a) {
        while(a < -M_PI) a += 2*M_PI;
        while(a > M_PI) a -= 2*M_PI;
        return a;
    }

    inline float angleDist(float a, float b) {
        float d = fabs(normAngle(a-b));
        //if (d > M_PI) d = 2*M_PI - d;
        return d;
    }



    PathPlan::PathPlan(int xs, int ys, float steering_limit_deg, float yaw_res_deg, float cost_steering_deg, int step, int num_search, float discretize_ratio, float discount, float map_res, float carlen)
        : nx(xs), ny(ys), step(step), cost_steering(cost_steering_deg / M_PI * 180), num_search(num_search), discretize_ratio(discretize_ratio), discount(discount), map_res(map_res), carlen(carlen)
    {
        float steering_limit = steering_limit_deg / 180 * M_PI;
        yaw_rln = yaw_res_deg / 180 * M_PI;
        ns = xs * ys;
        costarr = new COSTTYPE[ns]; // cost array, 2d config space
        memset(costarr, 0, ns*sizeof(COSTTYPE));

        steerings.push_back(0);
        for(float f=yaw_rln; f<=steering_limit; f+=yaw_rln) {
            steerings.push_back(-f);
            steerings.push_back(f);
        }

        steplen = step*map_res;
   //     cout << "steplen ******: "<<steplen<<endl;
   //     cout << "map resolution *****: "<<map_res<<endl;
        /*
        for(float f=steering_limit; f>0; f-=yaw_rln) {
            steerings.push_back(f);
            steerings.push_back(-f);
        }
        steerings.push_back(0);
        */
    }

    void PathPlan::setCostmap(const COSTTYPE *cmap, bool isROS, bool allow_unknown)
    {
        // static bool output_map =true;
        COSTTYPE *cm = costarr;
        if (isROS)			// ROS-type cost array
        {
            for (int i=0; i<ny; i++)
            {
                int k=i*nx;
                for (int j=0; j<nx; j++, k++, cmap++, cm++)
                {
                    *cm = COST_OBS;
                    int v = *cmap;
                    if (v < COST_OBS_ROS)
                    {
                        v = COST_NEUTRAL+COST_FACTOR*v;
                        if (v >= COST_OBS)
                            v = COST_OBS-1;
                        *cm = v;
                        //*cm = COST_NEUTRAL;
                    }
                    else if(v == COST_UNKNOWN_ROS && allow_unknown)
                    {
                        v = COST_OBS-1;
                        *cm = v;
                    }
                }
            }
        }
        else				// not a ROS map, just a PGM
        {
            for (int i=0; i<ny; i++)
            {
                int k=i*nx;
                for (int j=0; j<nx; j++, k++, cmap++, cm++)
                {
                    *cm = COST_OBS;
                    if (i<7 || i > ny-8 || j<7 || j > nx-8)
                        continue;	// don't do borders
                    int v = *cmap;
                    if (v < COST_OBS_ROS)
                    {
                        v = COST_NEUTRAL+COST_FACTOR*v;
                        if (v >= COST_OBS)
                            v = COST_OBS-1;
                        *cm = v;
                    }
                    else if(v == COST_UNKNOWN_ROS)
                    {
                        v = COST_OBS-1;
                        *cm = v;
                    }
                }
            }

        }

/*        if ( output_map ){
            cout << "before output costmap"<<endl;

            for (int i=0; i<ny; i++)
            {
                int k=i*nx;
                for (int j=0; j<nx; j++)
                {
                    cout<<(int) costarr[k+j]<<" ";
                }
                cout<<endl;
            }

            cout << "after output costmap"<<endl;

            output_map = false;
        }*/

         
    }

    void PathPlan::setGoal(const State& goal) {
		cout << "goal: " << goal[0] << " " << goal[1] << endl;
        this->goal = goal;
    }

    void PathPlan::setStart(const State& start) {
		cout << "start: " << start[0] << " " << start[1] <<  " " << start[2] << endl;
        this->start = start;
        //this->start[2] = normAngle(this->start[2] + (-2.0/180 * M_PI));
		this->start[2] = normAngle(this->start[2]);
    }

    vector<State> PathPlan::calcPath() {
		cout << "calcPath start" << endl;
        vector<PathItem> items;
        set<DiscreteState> visited;
        priority_queue<QPair, vector<QPair>, greater<QPair>> q;

        PathItem p0 = {start, 0, heuristic(start), 0, -1, 1.0};
		items.push_back(p0);
        visited.insert(discretize(start));
        q.push(QPair(p0.g+p0.h, p0.index));

        bool found_solution = false;
        PathItem goal_item;

		//float curr_min_dist = distToGoal(start);

        while(!q.empty()) {
            QPair qp = q.top();
            q.pop();
            PathItem p = items[qp.second];

			/*
			float curr_dist = distToGoal(p.state);
			if(curr_dist < curr_min_dist) {
				cout << "dist to goal = " << curr_dist << endl;
				cout << "(" <<  p.state[0] << " " << p.state[1] << " " << p.state[2] << ")" << " " << qp.first << endl;
				curr_min_dist = curr_dist;
			}*/


            if(isGoalReached(p.state)) {
                found_solution = true;
                goal_item = p;
				break;
            }

			if(items.size() > num_search) {
                cout << "no solution found after searching " << items.size() << " states!" << endl;
				break;
			}

            for(float t: steerings) {
				bool success;
                PathItem p1 = next(p, t, success);
				if (!success or (! (0 < p1.state[0] && p1.state[0] < nx && 0 < p1.state[1] && p1.state[1] < ny))) {
					continue;
				}
                auto dp1 = discretize(p1.state);
                if (visited.count(dp1)) {
                    break;
                    ////continue;
                }

                p1.index = items.size();
                items.push_back(p1);
                visited.insert(dp1);
                q.push(QPair(p1.g + p1.h, p1.index));
            }
        }

        vector<State> sol;
        if (found_solution) {
            // backtrace path
            int i = goal_item.index;
            while (i >= 0) {
                sol.push_back(items[i].state);
                i = items[i].prev_index;
            }
            reverse(sol.begin(), sol.end());
        }
		cout << "calcPath done pathlen = " << sol.size() << "  searched =" << items.size() << endl;
        return sol;
    }

    PathItem PathPlan::next(const PathItem& p, float t, bool& success) {
        const State& s0 = p.state;
        // note that t is steering angle; steplen is the distance it moves at each time step;
        // and carlen is the distance from the front wheel to the back wheel
        float turn = tan(t) * steplen / carlen;
        PathItem p1;

        if(fabs(turn) < 0.001){ // approximate by straight line motion
            float dx = step * cos(s0[2]);
            float dy = step * sin(s0[2]);
            p1.state.resize(3);
            p1.state[0] = s0[0] + dx;
            p1.state[1] = s0[1] + dy;
            float a = s0[2] + turn;
            p1.state[2] = normAngle(a); //heading direction
        } else{ // approximate bicycle model for motion
            //float radius = steplen / turn;
            float radius = step / turn;
            float cx = s0[0] - sin(s0[2]) * radius;
            float cy = s0[1] + cos(s0[2]) * radius;

            p1.state.resize(3);
            float a = s0[2] + turn;
            p1.state[2] = normAngle(a); //heading direction
            p1.state[0] = cx + sin(p1.state[2]) * radius;
            p1.state[1] = cy - cos(p1.state[2]) * radius;
        }       

        DiscreteState ds1 = discretize(p1.state);
        int mx = int(s0[0]);
        int my = int(s0[1]);
        float cost = costarr[my*nx + mx];
        //float steer_cost = sqr(t) * cost_steering;
        float steer_cost = fabs(t) * cost_steering; //t is steering angle in radian

        success = (cost < COST_OBS * 0.999999);
/*        if(angleDist(angleToGoal(p1.state), p1.state[2]) > M_PI / 180.0 * 60.0) {
            success = false;
        }*/


        p1.d = p.d * discount;
        p1.g = p.g + (cost + steer_cost) * p1.d;
        p1.h = heuristic(p1.state) * p1.d;

        p1.prev_index = p.index;

        return p1;

        // Approximate only by the straight line motion model
/*      const State& s0 = p.state;
        PathItem p1;
        float dx = step * cos(s0[2]);
        float dy = step * sin(s0[2]);
		p1.state.resize(3);
        p1.state[0] = s0[0] + dx;
        p1.state[1] = s0[1] + dy;
        // +t is wrong. Think of if t is big, say 45, but distance is small,
        // say 0.1m, it is impossible for the robot to make its heading direction
        //45 degree larger after running for 0.1m. Should plus turning angle turn
        // instead of steering angle t
        //p1.state[2] = s0[2] + t;
        float a = s0[2] + t;
        p1.state[2] = normAngle(a);

        DiscreteState ds1 = discretize(p1.state);
		int mx = int(s0[0]);
		int my = int(s0[1]);
        float cost = costarr[my*nx + mx];
        //float steer_cost = sqr(t) * cost_steering;
        float steer_cost = fabs(t) * cost_steering; //t is steering angle in radian

		success = (cost < COST_OBS * 0.999999);
		if(angleDist(angleToGoal(p1.state), p1.state[2]) > M_PI / 180.0 * 60.0) {
			success = false;
		}


		p1.d = p.d * discount;
        p1.g = p.g + (cost + steer_cost) * p1.d;
        p1.h = heuristic(p1.state) * p1.d;

        p1.prev_index = p.index;

        return p1;*/
    }

    float PathPlan::angleToGoal(const State& s) {
        float dx = goal[0] - s[0];
        float dy = goal[1] - s[1];
        float a = atan2(dy, dx);
        return a;
    }

    float PathPlan::distToGoal(const State& s) {
        float dist = sqrt(sqr(goal[0]-s[0]) + sqr(goal[1]-s[1]));
        return dist;
    }

    float PathPlan::heuristic(const State& s) {
        float d = distToGoal(s);
        float nstep = d / float(step);
        //float nstep = d / float(step) * 1.2;
        float h_dist = (1 - pow(discount, nstep)) / (1 - discount) * COST_NEUTRAL;
/*        float turn_per_step = normAngle(angleDist(angleToGoal(s), s[2])) / nstep;
        float steer_per_step = fabs(atan(turn_per_step*carlen/steplen));
        cout << (1 - pow(discount, nstep)) / (1 - discount) * steer_per_step*cost_steering << endl;
        h_dist += (1 - pow(discount, nstep)) / (1 - discount) * steer_per_step*cost_steering;*/
        //h_dist += fabs(angleDist(angleToGoal(s), s[2])) * cost_steering;
        //h_dist += pow(discount, nstep)*fabs(angleDist(angleToGoal(s), s[2])) * cost_steering;
        return h_dist;
    }

    bool PathPlan::isGoalReached(const State& s) {
        float d = distToGoal(s);
        return (d < TOLERANCE);
    }

    DiscreteState PathPlan::discretize(const State& s) {
        DiscreteState ds(3);
		float dd = step * discretize_ratio;
        ds[0] = int((s[0] / dd));
        ds[1] = int((s[1] / dd));
        ds[2] = int(((s[2] + M_PI) / yaw_rln));
        return ds;
    }
}
