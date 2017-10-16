#ifndef WORLD_SIMULATOR_H
#define WORLD_SIMULATOR_H
#include "pedestrian_state.h"
#include "SFM.h"
#include "map.h"
#include "window.h"
#include "param.h"
#include "math_utils.h"

//here need to modify the obs type according to different simulator
typedef long long OBS_T;
class WorldSimulator{
public:
	WorldSimulator(double random_num=0.62)
		:window(&world_map,0),car(&world_map),sfm(&world_map,&window),unif(random_num)
    {
		if(ModelParams::debug)
		{
			cout<<"DEBUG: seed "<<random_num<<endl;
			cout<<"num ped in view "<<" "<<NumPedInView()<<endl;
			cout<<"rob map size "<<window.rob_map.size()<<endl;
		}
		car.carPos=robPos=windowOrigin=0;	
		car.w=world_map.global_plan[0][0];
		car.h=world_map.global_plan[0][1];
		velGlobal=1.0;
		cerr << "DEBUG: Finished initializing WorldSimulator" << endl;
    }

	void Init()
	{
		cerr << "DEBUG: Call Init() in WorldSimulator" << endl;
		t_stamp=0;
		car.w=world_map.global_plan[0][0];
		car.h=world_map.global_plan[0][1];
		velGlobal=1.0;


		if(ModelParams::FixedPath)     InitTestPaths();   //This is for init user specific path for the pedestrians
		//InitPedestriansBlank();

		ShiftWindow();
		InitPedestriansBlank();
		ShiftWindow();
		cerr << "DEBUG: Done Init() in WorldSimulator" << endl;
		//sfm.UpdateSFM();
	}
	void InitTestPaths()
	{
		ped_paths[0][0][0]=180;
		ped_paths[0][0][1]=130;

		for(int i=1;i<10;i++)
		{
			ped_paths[0][i][0]=180-i*20;
			ped_paths[0][i][1]=130;
		}
		ped_paths[0][10][0]=-1000;
		ped_paths[0][10][1]=-1000;
	}

	Pedestrian GetPedPose(int i)
	{
		Pedestrian ped;
		window.GlobalToLocal(ped_list[pedInView_list[i]].w,ped_list[pedInView_list[i]].h,ped.w,ped.h);
		ped.id=pedInView_list[i];
		return ped;
	}
	Car GetCarPos()
	{
		Car local_car;
		window.GlobalToLocal(car.w,car.h,local_car.w,local_car.h);
		if(ModelParams::debug) 	cout<<"local car "<<local_car.w<<" "<<local_car.h<<endl;
		return local_car;
	}
	void SetSeed(unsigned new_seed_)
	{
		unif.seed_=new_seed_;
	}
	PedestrianState CalCurrObs()
	{
		return GetCurrState();
		/*
		int X_SIZE=ModelParams::XSIZE;
		int Y_SIZE=ModelParams::YSIZE;
		GetCurrState();
		OBS_T obs=0;// = state.Vel*(X_SIZE*Y_SIZE*rob_map.size())+state.RobPos.Y*(X_SIZE*Y_SIZE)+state.PedPos.X*Y_SIZE+state.PedPos.Y;
		OBS_T robObs=curr_state.Vel+curr_state.RobPos.Y*ModelParams::VEL_N;
		OBS_T robObsMax=ModelParams::VEL_N*ModelParams::RMMax;  //max length of the rob_map
		OBS_T pedObsMax=X_SIZE*Y_SIZE;
		OBS_T pedObs=1;
		//for(int i=0;i<state.PedPoses.size();i++)
		for(int i=0;i<curr_state.num;i++)
		{
			OBS_T this_obs=curr_state.PedPoses[i].first.X*Y_SIZE+curr_state.PedPoses[i].first.Y;	
			pedObs=pedObs*pedObsMax+this_obs;
		}
		obs=pedObs*robObsMax+robObs;
		if(ModelParams::debug) 	cout<<"world observation "<<obs<<endl;
		return obs;
		*/
	}
	PedestrianState GetCurrObs()
	{return curr_obs;}

	bool InSafeZone(int w,int h)
	{
		if(fabs(w-car.w)+fabs(h-car.h)<=ModelParams::map_rln*3) return false;	
		else return true;
	}
	Pedestrian InitOnePed()
	{
		/*	
		int goal=unif.next()*ModelParams::NGOAL;
		bool ok=false;
		int rob_w,rob_h,x,y;
		window.GlobalToLocal(world_map.global_plan[robPos][0],world_map.global_plan[robPos][1],rob_w,rob_h);
		while(!ok)
		{
			x=unif.next()*ModelParams::XSIZE;
			y=unif.next()*ModelParams::YSIZE;
			if(fabs(x-rob_w)+fabs(y-rob_h)>3) ok=true;
		}
		int ped_w,ped_h;
		window.LocalToGlobal(x,y,ped_w,ped_h);
		cout<<"initial ped x y goal "<<ped_w<<" "<<ped_h<<" "<<goal<<endl;
		return Pedestrian(ped_w,ped_h,goal,0);*/
			

		int goal=unif.next()*ModelParams::NGOAL;
		int goal_w,goal_h;
		goal_w=world_map.goal_pos[goal][0];
		goal_h=world_map.goal_pos[goal][1];
		cout<<goal_w<<" "<<goal_h<<endl;
		int rln=ModelParams::rln/2;
		int range=(ModelParams::GOAL_DIST-2)*ModelParams::rln;
		int sum=0;
		for(int x=goal_w-range;x<goal_w+range;x+=rln)
			for(int y=goal_h-range;y<goal_h+range;y+=rln)
			{
				if(world_map.InMap(x,y)&&InSafeZone(x,y))  sum++; 
		//		if(true) sum++;
			}
		int grid=unif.next()*sum;
		sum=0;
		bool inner=false;
		for(int x=goal_w-range;x<goal_w+range&&inner==false;x+=rln)
			for(int y=goal_h-range;y<goal_h+range;y+=rln)
			{
		//		if(true) sum++;
				if(world_map.InMap(x,y)&&InSafeZone(x,y))  sum++; 
				if(sum>grid)  
				{
					inner=true;
					//ped_list.push_back(Pedestrian(x,y,goal,0));
					if(ModelParams::debug)  cout<<"initial ped "<<" "<<x<<" "<<y<<" "<<goal<<endl;
					return Pedestrian(x,y,goal,ped_list.size());
					break;
				}
			}

	}
	void InitPedestriansTest()
	{
		ped_list.clear();
		pedInView_list.clear();
		for(int i=0;i<NumPedTotal;i++)
		{
			ped_list.push_back(InitOnePed());
		}
	}
	void InitPedestriansBlank()
	{
		ped_list.clear();
		pedInView_list.clear();
		for(int i=0;i<NumPedTotal;i++)
		{
			if(ModelParams::FixedPath)
			{
				ped_list.push_back(Pedestrian(ped_paths[0][i][0],ped_paths[0][i][1],2,0));
				cout<<ped_list[i].w<<" "<<ped_list[i].h<<endl;
			}
			else 	ped_list.push_back(InitOnePed());
			if(ModelParams::FixedPath)
			{
				ped_list[i].ts=i;
			}
		}
		cout<<"finish initial pedestrians"<<endl;
	}

	bool OneStep(int action)
	{
		if(GoalReached())       return true;	
		//if(InCollision(action)) return true; 		
		if(InCollision()) return true; 		
		if(ModelParams::FixedPath)
		{
			for(int i=0;i<ped_list.size();i++)
				ped_list[i].ts++;
		}
			
		if(ModelParams::FixedPath)  ChangePath(); 	
		UpdateCarGaussian(action);
		//Display();
		UpdatePed();

		GetCurrState();
			
		if(ModelParams::debug)
		{
			cout<<"before shift window"<<endl;
			cout << "Rob Pos: " << window.rob_map[curr_state.RobPos.Y].first << " " <<window.rob_map[curr_state.RobPos.Y].second <<endl;
			//for(int i=0;i<state.PedPoses.size();i++)
			for(int i=0;i<curr_state.num;i++)
			{
				cout << "Ped Pos: " << curr_state.PedPoses[i].first.X << " " <<curr_state.PedPoses[i].first.Y <<endl;
				cout << "Goal: " << curr_state.PedPoses[i].second << endl;
			}
			cout << "Vel: " << curr_state.Vel << endl;
			Display();
		}
		
		//curr_obs=CalCurrObs();

		//ShiftWindow();
		return false;
	}

	void ChangePath()
	{
		for(int i=0;i<ped_list.size();i++)
		{
			ped_list[i].w=ped_paths[0][ped_list[i].ts][0];
			ped_list[i].h=ped_paths[0][ped_list[i].ts][1];
		}
	}

	void SwitchPath(int p)
	{
		
	}
	void ShiftWindow()
	{
		curr_obs=CalCurrObs();
		if(robPos-windowOrigin>=ModelParams::path_rln*3) 	  windowOrigin=robPos-ModelParams::path_rln/2;
		window.RollWindow(windowOrigin);
		pedInView_list.clear();

		for(int i=0;i<ped_list.size();i++)
		{
			int x,y;
			window.GlobalToLocal(ped_list[i].w,ped_list[i].h,x,y);
		}

		for(int i=0;i<ped_list.size();i++)
		{
			if(ped_list[i].w==-1000&&ped_list[i].h==-1000)
			{
				ped_list[i]=InitOnePed();
			}
			if(window.InWindow(ped_list[i].w,ped_list[i].h)&&(pedInView_list.size()<ModelParams::N_PED_IN))
			{
				pedInView_list.push_back(i);	
			}
		}
		cout<<"Num Ped In View = "<<pedInView_list.size()<<endl;
		sfm.UpdateSFM();//update the precompute staff
		if(ModelParams::debug)
		{
			cout<<"local obs map"<<endl;
			for(int i=0;i<ModelParams::XSIZE;i++)
			{
				for(int j=0;j<ModelParams::YSIZE;j++)
				{
					int w,h;
					window.LocalToGlobal(i,j,w,h);	
					//cout<<w<<" "<<h<<" ";
					cout<<world_map.Free(w,h)<<" ";
				}
				cout<<endl;
			}
		}
		t_stamp++;
	}
	void Clean()
	{
		vector<Pedestrian> ped_list_new;
		//for(vector<Pedestrian>::iterator it=ped_list.begin();it!=ped_list.end();++it)
		for(int i=0;i<ped_list.size();i++)
		{
			bool insert=true;
			//for(vector<Pedestrian>::iterator it2=ped_list.begin();it2!=it;++it2)
			int w1,h1;
			w1=ped_list[i].w;
			h1=ped_list[i].h;
			for(int j=0;j<i;j++)
			{
				int w2,h2;
				w2=ped_list[j].w;
				h2=ped_list[j].h;
				//if(abs(it->w-it2->w)<=1&&abs(it->h-it2->h)<=1)
				if(abs(w1-w2)<=1&&abs(h1-h2)<=1)
				{
					insert=false;	
					break;
				}
			}
			if(t_stamp-ped_list[i].last_update>1) insert=false;
			if(insert)
				ped_list_new.push_back(ped_list[i]);
		}
		ped_list=ped_list_new;
	}
	void ShiftWindow(int pos)
	{
		windowOrigin=pos;
		window.RollWindow(windowOrigin);
		pedInView_list.clear();
		for(int i=0;i<ped_list.size();i++)
		{
			if(ped_list[i].w==-1000&&ped_list[i].h==-1000)
			{
				ped_list[i]=InitOnePed();
			}
			if(window.InWindow(ped_list[i].w,ped_list[i].h)&&(pedInView_list.size()<ModelParams::N_PED_IN))
			{
				pedInView_list.push_back(i);	
			}
		}
		sfm.UpdateSFM();//update the precompute staff
	}
	void UpdateVel(int action)
	{
		int action_vel[3]={0,1,-1};
		double prob=unif.next();		
		if(prob<0.05) ;
		else velGlobal=velGlobal+action_vel[action]*control_time*ModelParams::AccSpeed;

		if(velGlobal<0) velGlobal=0;	
		if(velGlobal>ModelParams::VEL_MAX) velGlobal=ModelParams::VEL_MAX;
	}
	double gaussian(double dist)  const{
		if(dist<0.1) dist=1;
		return 1/dist;
	}
	void UpdateCarGaussian(int action) 
	{


		/*
		   robY += robotNoisyMove[rob_vel][lookup(robotMoveProbs[rob_vel], p)];
		   if(robY >= Y_SIZE) robY = Y_SIZE - 1;
		   p = unif.next();
		   rob_vel = robotVelUpdate[action][rob_vel][lookup(robotUpdateProb[action][rob_vel], p)];
		   */


		double next_center=robPos+velGlobal*ModelParams::path_rln*control_time;
		//int max_grid=ModelParams::path_rln*ModelParams::VEL_MAX*control_time*2;  //double range of velocity noise

		double weight[int(ModelParams::path_rln*5)];
		double weight_sum=0;

		int offset=abs(next_center-robPos)*0.2;
		
		for(int i=next_center-offset;i<world_map.pathLength&&i<=next_center+offset;i++)
		{
			weight[i-robPos]=gaussian(fabs(next_center-i));
			weight_sum+=weight[i-robPos];
		}

		double p = unif.next()*weight_sum;
		weight_sum=0;
		int dist=-1;
		for(int i=next_center-offset;i<world_map.pathLength&&i<=next_center+offset;i++)
		{
			weight_sum+=weight[i-robPos];
			if(weight_sum>p) {
				dist=i;
				break;
			}
		}
		//cout<<dist-robY<<endl;
		robPos=dist;
		UpdateVel(action);
	}
	void UpdateCarGood(int action)
	{
		int &robY=robPos;
		int rob_vel=velGlobal;

		double vel_p=unif.next();
		if(rob_vel==0)
		{
			if(vel_p<0.9)   
			{
			}
			else
			{
			//	robY+=ModelParams::path_rln*0.8;
			}

		}
		else if(rob_vel==1)
		{
			if(vel_p<0.8) robY+=ModelParams::rln;
			else if(vel_p<0.9) robY+=2*ModelParams::rln;
		}
		else
		{
			if(vel_p<0.8) robY+=2*ModelParams::rln;
			else if(vel_p<0.9) robY+=ModelParams::rln;
		}

		//if(robY>rob_map.size()-1) robY=rob_map.size()-1;
		//TODO: terminal condition
		double act_p=unif.next();
		if(action==1) rob_vel++;
		if(action==2) 
		{
			if(act_p<0.5) rob_vel--;
			else 		  rob_vel-=2; 
		}
		
		if(rob_vel<0) rob_vel=0;
		if(rob_vel>2) rob_vel=2;	
		velGlobal=rob_vel;
	}

	void UpdateCar(int action)
	{
		if(ModelParams::goodrob==1) {
			UpdateCarGood(action);
			return;
		}
		int &robY=robPos;
		int rob_vel=velGlobal;

		double vel_p=unif.next();
		if(rob_vel==0)
		{
			if(vel_p<0.9)   
			{
			}
			else
			{
				robY+=ModelParams::rln;
			}

		}
		else if(rob_vel==1)
		{
			if(vel_p<0.8) robY+=ModelParams::rln;
			else if(vel_p<0.9) robY+=2*ModelParams::rln;
		}
		else
		{
			if(vel_p<0.8) robY+=2*ModelParams::rln;
			else if(vel_p<0.9) robY+=ModelParams::rln;
		}

		//if(robY>rob_map.size()-1) robY=rob_map.size()-1;
		//TODO: terminal condition
		double act_p=unif.next();

		if(action==1)
		{
			if(rob_vel==0)
			{
				if(act_p<0.7) rob_vel++;
				else if(act_p<0.8) rob_vel+=2;
				else rob_vel=0;
			}
			else if(rob_vel==1)
			{
				if(act_p<0.8) rob_vel=2;
				else if(act_p<0.9) rob_vel=1;
				else rob_vel=0;
			}
			else if(rob_vel==2)
			{
				if(act_p<0.8) rob_vel=2;
				else if(act_p<0.9) rob_vel=1;
				else rob_vel=0;
			}
		}
		else if(action==2)
		{
			if(rob_vel==0)
			{
				if(act_p<0.9) {}
				else {rob_vel++;}
			}
			else if(rob_vel==1)
			{
				if(act_p<0.9){rob_vel--;}
			}
			else if(rob_vel==2)
			{
				if(act_p<0.7){rob_vel--;}
				else if(act_p<0.8) {}
				else {rob_vel-=2;}
			}	
		}
		else
		{
			if(rob_vel==0)
			{
				if(act_p<0.9) rob_vel=0;
				else rob_vel=1;
			}
			else if(rob_vel==1)
			{
				if(act_p<0.7) rob_vel=1;
				else if(act_p<0.8) rob_vel=2;
				else rob_vel=0;
			}
			else if(rob_vel==2)
			{
				if(act_p<0.8) rob_vel=2;
				else if(act_p<0.9) rob_vel=1;
				else 	rob_vel=0;
			}
		}

			
		
		if(rob_vel<0) rob_vel=0;
		if(rob_vel>2) rob_vel=2;	
		velGlobal=rob_vel;

	}

	void UpdatePed()
	{
		//here we update the ped poses based on the old car position
		sfm.WorldTrans(ped_list,car,unif);	

		
		//car.w=world_map.global_plan[robPos][0];
		//car.h=world_map.global_plan[robPos][1];
		//car.carPos=robPos;
	}

	void UpdatePedPoseReal(Pedestrian ped)
	{
		int i;
	//	cout<<"ped pose in simulator"<<endl;
		for( i=0;i<ped_list.size();i++)
		{
			if(ped_list[i].id==ped.id)
			{
				//found the corresponding ped,update the pose
				ped_list[i].w=ped.w;
				ped_list[i].h=ped.h;
				ped_list[i].last_update=t_stamp;
				break;
			}
			if(abs(ped_list[i].w-ped.w)<=1&&abs(ped_list[i].h-ped.h)<=1)   //overladp 
				return;
	//		cout<<ped_list[i].w<<" "<<ped_list[i].h<<endl;
		}
		if(i==ped_list.size())   //not found, new ped
		{
	//		cout<<"add"<<endl;
			ped.last_update=t_stamp;
			ped_list.push_back(ped);

		}
	///	cout<<"this ped "<<ped.w<<" "<<ped.h<<" "<<ped.id<<endl;
	}

	void UpdateRobPoseReal(Car world_car)
	{
		//need to find the closest point in the pre-defined path
		int next=robPos-1;
		int next_diff=10000;
		for(int i=robPos;i<world_map.pathLength&&i<robPos+ModelParams::path_rln*5;i++)
		{
			int curr_diff=abs(world_map.global_plan[i][0]-world_car.w)+abs(world_map.global_plan[i][1]-world_car.h);
			if(curr_diff<next_diff)
			{
				next_diff=curr_diff;	
				next=i;
			}
		}
		robPos=next;
		car.w=world_map.global_plan[robPos][0];
		car.h=world_map.global_plan[robPos][1];
		car_ground_truth.w=world_car.w;
		car_ground_truth.h=world_car.h;
		cout<<"real rob pos"<<robPos<<endl;
	}
	void UpdateVelReal(double vel)
	{
		if(vel>ModelParams::VEL_MAX) vel=ModelParams::VEL_MAX;	
		if(vel<0) vel=0;
		velGlobal=vel;
	}


	void Display()
	{
		GetCurrState();
		cout<<"Car Pos "<<world_map.global_plan[robPos][0]<<" "<<world_map.global_plan[robPos][1]<<endl;
		cout<<"Window Pos "<<windowOrigin<<endl;
		cout<<"Rob Pos "<<robPos<<endl;
		cout<<"ped num "<<pedInView_list.size()<<endl;
		cout<<"real velocity "<<velGlobal<<endl;
		/*
		for(int i=0;i<pedInView_list.size();i++)
		{
			cout<<"Real Ped Pos "<<ped_list[pedInView_list[i]].w<<" "<<ped_list[pedInView_list[i]].h<<endl;
		}*/
		for(int i=0;i<ped_list.size();i++)
		{
			cout<<"Real Ped Pos "<<ped_list[i].w<<" "<<ped_list[i].h<<" "<<ped_list[i].goal<<endl;
		}
	}
	PedestrianState GetCurrState()
	{
		int x,y;

		window.GlobalToLocal(world_map.global_plan[robPos][0],world_map.global_plan[robPos][1],x,y);
		if(ModelParams::debug)  {
			cout<<"rob pos global "<<world_map.global_plan[robPos][0]<<" "<<world_map.global_plan[robPos][1]<<endl;
			cout<<"rob pos local "<<x<<" "<<y<<endl;
		}


        int min_drift = 100;
        int best_i=curr_state.RobPos.Y;
		for(int i=0;i<window.rob_map.size();i++)
		{
            int dist = abs(window.rob_map[i].first-x) + abs(window.rob_map[i].second-y);
			if(dist < min_drift)
			{
                min_drift = dist;
                best_i = i;
			}
		}
        curr_state.RobPos.Y = best_i;
        if(min_drift> 0) cout << "car drift distance = " << min_drift << endl;

		for(int i=0;i<pedInView_list.size();i++)
		{
			int ped_w,ped_h;
			window.GlobalToLocal(ped_list[pedInView_list[i]].w,ped_list[pedInView_list[i]].h,ped_w,ped_h);
			if(ModelParams::debug)
			{
				cout<<"global ped x y "<<ped_list[pedInView_list[i]].w<<" "<<ped_list[pedInView_list[i]].h<<endl;
				cout<<"local ped x y "<<ped_w<<" "<<ped_h<<endl;
			}
			if(ped_w<0) ped_w=0;
			if(ped_w>ModelParams::XSIZE-1) ped_w=ModelParams::XSIZE-1;
			if(ped_h<0) ped_h=0;
			if(ped_h>ModelParams::YSIZE-1) ped_h=ModelParams::YSIZE-1;
			curr_state.PedPoses[i].first.X=ped_w;
		    curr_state.PedPoses[i].first.Y=ped_h;
			curr_state.PedPoses[i].second=ped_list[pedInView_list[i]].goal;
			//curr_state.PedPoses[i].third=pedInView_list[i];
			curr_state.PedPoses[i].third=ped_list[pedInView_list[i]].id;
		}
		curr_state.num=pedInView_list.size();

		double vmax=ModelParams::VEL_MAX/control_freq*ModelParams::map_rln/ModelParams::rln;
		if(velGlobal<0.05) curr_state.Vel=0;
		else
		{
			curr_state.Vel= floor(velGlobal/ModelParams::VEL_MAX*(ModelParams::VEL_N-1) + 0.5);
			if (curr_state.Vel == 0)
				curr_state.Vel = 1;
			if (curr_state.Vel > ModelParams::VEL_N)
				curr_state.Vel = ModelParams::VEL_N - 1;
		}
		
	/*	
		if(velGlobal<0.1)
		{
			curr_state.Vel=0;
		}
		else if(velGlobal<0.75)
		{
			curr_state.Vel=1;
		}
		else if(velGlobal<1.25)
		{
			curr_state.Vel=2;
		}
		else if(velGlobal<1.75)
		{
			curr_state.Vel=3;
		}
		else
		{
			curr_state.Vel=4;
		}*/
		
		/*
		double delta=(ModelParams::vel_levels[ModelParams::VEL_N-1]-ModelParams::vel_levels[0])/ModelParams::VEL_N;
		for(int i=0;i<ModelParams::VEL_N;i++)
		{
			if(fabs(ModelParams::vel_levels[i]-velGlobal)<=delta/2)  {
				curr_state.Vel=i;
				break;
			}
		}*/

		//curr_state.Vel=velGlobal*2;
		return curr_state;
	}
	
	/*
	bool InCollision(int action)
	{
		GetCurrState();
		for(int i=0;i<pedInView_list.size();i++)
		{

			int pedX=curr_state.PedPoses[i].first.X;
			int pedY=curr_state.PedPoses[i].first.Y;
			int robY=curr_state.RobPos.Y;
			int rob_vel=curr_state.Vel;


			if(pedX==window.rob_map[robY].first&&window.rob_map[robY].second==pedY)
			{
				return true;
			}
			if(pedX==window.rob_map[robY].first&&pedY==window.rob_map[robY].second+1)
			{
				if(rob_vel==0&&action==1)  {return true;}
				if(rob_vel==1&&action<2)   {return true;}
				if(rob_vel==2)     			{return true;}
			}
			if(pedX==window.rob_map[robY].first&&pedY==window.rob_map[robY].second+2)
			{
				if(rob_vel==1&&action==1)  {return true;}
				if(rob_vel==2&&action<2)  {return true;}
			}

		}
		return false;
	}
	*/


	bool InCollision()
	{
		for(int i=0;i<pedInView_list.size();i++)
		{
			int pX=ped_list[pedInView_list[i]].w;	
			int pY=ped_list[pedInView_list[i]].h;
			int rX=car.w;
			int rY=car.h;
			if(abs(pX-rX)<ModelParams::map_rln/2&&abs(pY-rY)<ModelParams::map_rln/2)  return true;
		}
		return false;
	}


	bool InRealCollision(int action)
	{
		GetCurrState();
		for(int i=0;i<pedInView_list.size();i++)
		{

			int pedX=curr_state.PedPoses[i].first.X;
			int pedY=curr_state.PedPoses[i].first.Y;
			int robY=curr_state.RobPos.Y;
			int rob_vel=curr_state.Vel;


			if(pedX==window.rob_map[robY].first&&window.rob_map[robY].second==pedY)
			{
				return true;
			}
		}
		return false;
	}
	
	bool Emergency()
	{
		int car_w=car_ground_truth.w;
		int car_h=car_ground_truth.h;
		int rx,ry;
		window.GlobalToLocal(car_w,car_h,rx,ry);
		for(int i=0;i<ped_list.size();i++)
		{
			int px,py;
			window.GlobalToLocal(ped_list[i].w,ped_list[i].h,px,py);
			if(abs(px-rx)<=1&&py-ry>=-1&&py-ry<=1)
			{
				return true;
			}
		}
		return false;
	}
	bool GoalReached()
	{
		double rate=ModelParams::path_rln/ModelParams::rln;
		if(world_map.pathLength-robPos<ModelParams::path_rln*((ModelParams::YSIZE+5)/rate))	 {
			cout<<"goal reached"<<endl;
			return true;
		}
		//if(robPos>ModelParams::path_rln*(ModelParams::YSIZE-2))	 return true;
		else return false;
	}

	int NumPedInView()
	{
		return pedInView_list.size();
	}


	MyMap  world_map;
	MyWindow  window;
	int robPos;
	Car car;
	Car car_ground_truth;
	SFM sfm;
	int NumPedTotal;
	//int robPos;
	int windowOrigin;
	int t_stamp;
	double velGlobal;
	PedestrianState ped_state;
	vector<Pedestrian> ped_list;
	vector<int> pedInView_list;
	PedestrianState curr_state;
	double control_freq;	
	double control_time;
	PedestrianState curr_obs;
	UtilUniform unif;
	int ped_paths[3][100][2];
};
#endif
