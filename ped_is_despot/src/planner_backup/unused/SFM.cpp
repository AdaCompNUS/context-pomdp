#include "SFM.h"
#include <iostream>



MyVector SFM::PedToGoal(Pedestrian &ped,int goal_w,int goal_h)
{
	//double dw=(goal_w-ped.w)/ModelParams::rln;
	//double dh=(goal_h-ped.h)/ModelParams::rln;
	//double dist=sqrt(dw*dw+dh*dh);
	double goal_angle=atan2(goal_h-ped.h,goal_w-ped.w);
	double goal_length=3;
	//double goal_length=10/dist;  //constant speed move to the goal
	//if(goal_length<1) goal_length=1;
	MyVector goal_vec(goal_angle,goal_length,-1);
	if(debug)
		std::cout<<"goal force "<<goal_vec.dw<<" "<<goal_vec.dh<<std::endl;
	return goal_vec;
}

/*need some modifications here,should exclude the ped from the list*/
MyVector SFM::PedToPed(Pedestrian &ped, vector<Pedestrian> &ped_list)
{
	MyVector vec,out_vec;
	if(debug)
	{
		std::cout<<"debug"<<std::endl;
		std::cout<<"num of peds "<<ped_list.size()<<std::endl;
	}
	for(int i=0;i<ped_list.size();i++)
	{
		//add repel from each pedestrian
		int goal_w1,goal_h1,goal_w2,goal_h2;
		sfm_window->GlobalToLocal(ped_list[i].w,ped_list[i].h,goal_w1,goal_h1);
		sfm_window->GlobalToLocal(sfm_map->goal_pos[ped_list[i].goal][0],sfm_map->goal_pos[ped_list[i].goal][1],goal_w2,goal_h2);
		if(goal_w1==goal_w2&goal_h1==goal_h2) //pedestrian already reach the goal, 
			continue;
		vec.dw=ped.w-ped_list[i].w;
		vec.dh=ped.h-ped_list[i].h;
		double dw=vec.dw/ModelParams::rln;
		double dh=vec.dh/ModelParams::rln;
		double dist=sqrt(dw*dw+dh*dh);
		double force=0;
		if(dist>4)  force=0.1;
		else if(dist<0.1) force=0.0;
		else force=4/(dist*dist);
		vec.AdjustLength(force);
		out_vec=out_vec+vec;
	}
	if(debug)
		std::cout<<"ped force "<<out_vec.dw<<" "<<out_vec.dh<<std::endl;
	return out_vec;
}
/*
MyVector SFM::PedToCar(Pedestrian & ped,Car & car)
{
	double car_w=car.w;

	double car_h=car.h;
	double dw=(ped.w-car_w)/ModelParams::rln;
	double dh=(ped.h-car_h)/ModelParams::rln;
	MyVector vec((ped.w-car_w),(ped.h-car_h));
	double dist=sqrt(dw*dw+dh*dh);
	double force;
	if(dist>6) force=0.1;
	//else force=12/(dist*dist);
	else force=15/(dist*dist);
	vec.AdjustLength(force);
	if(debug)
		std::cout<<"Car Force "<<vec.dw<<" "<<vec.dh<<std::endl;
	return vec;
}*/
MyVector SFM::PedToCar(Pedestrian & ped,int carPos)
{
	MyVector car_vec;
	car_vec.dw=sfm_map->global_plan[int(carPos+ModelParams::path_rln*2)][0]-sfm_map->global_plan[carPos][0];
	car_vec.dh=sfm_map->global_plan[int(carPos+ModelParams::path_rln*2)][1]-sfm_map->global_plan[carPos][1];
	
	//we want the pedestrian to move perpendicular to the car_vec
	//this is a temporary hack for align the global coor with the local coord in order to generate consistent force
	int tempw,temph;
	sfm_window->GlobalToLocal(ped.w,ped.h,tempw,temph);
	sfm_window->LocalToGlobal(tempw,temph,ped.w,ped.h);
	MyVector ped_vec;

	/*
	if(fabs(car_vec.dh-0.0)<0.001) 	{
		ped_vec.dw=0;
		ped_vec.dh=1;
	}
	else 	{
		ped_vec.dw=1;
		ped_vec.dh=-car_vec.dw/car_vec.dh;
	}*/

	double car_w=sfm_map->global_plan[carPos][0];
	double car_h=sfm_map->global_plan[carPos][1];



	MyVector carToPed;	
	carToPed.dw=(ped.w-car_w);
	carToPed.dh=(ped.h-car_h);

	//do cross product   (
	if(CrossProduct(car_vec,carToPed)>0.01)  	{
		ped_vec.SetAngle(car_vec.GetAngle()+ModelParams::pi/2);
	}
	else if(CrossProduct(car_vec,carToPed)<-0.01) {
		ped_vec.SetAngle(car_vec.GetAngle()-ModelParams::pi/2);
	}
	else {       //same line,ped_vec point to its goal side
		int goal_w=sfm_map->goal_pos[ped.goal][0];
		int goal_h=sfm_map->goal_pos[ped.goal][1];
		if(CrossProduct(car_vec,MyVector(goal_w-car_w,goal_h-car_h))>0.01)  {
			ped_vec.SetAngle(car_vec.GetAngle()+ModelParams::pi/2);
		}
		else if(CrossProduct(car_vec,MyVector(goal_w-car_w,goal_h-car_h))<-0.01) {
			ped_vec.SetAngle(car_vec.GetAngle()-ModelParams::pi/2);
		}
		else {
			//random 
			//cerr<<"goal and car on the same line"<<endl;
			ped_vec.SetAngle(car_vec.GetAngle()+ModelParams::pi/2);
		}
	}

	double dist=carToPed.GetLength()/ModelParams::rln;
	double force;
	if(dist>6) force=0.1;
	//else force=12/(dist*dist);
	else force=15/(dist*dist);
	ped_vec.AdjustLength(force);
	if(debug)
		std::cout<<"Car Force "<<ped_vec.dw<<" "<<ped_vec.dh<<std::endl;
	return ped_vec;
}
/*
MyVector SFM::PedToCarGrid(int pedX,int pedY,int carX,int carY)
{
	int dw=carX-pedX;
	int dh=carY-pedY;
	double dist=sqrt(dw*dw+dh*dh);
	MyVector vec((pedX-carX),(pedY-carY));
	if(dist>6) force=0.2;
	else force=12/(dist*dist);
	vec.AdjustLength(force);
	return vec;
}*/

MyVector  SFM::SocialForce(Pedestrian & ped,vector<Pedestrian> & ped_list,Car& car)
{
	MyVector vec;
	int goal_id=ped.goal;
	vec=vec+PedToGoal(ped,sfm_map->goal_pos[goal_id][0],sfm_map->goal_pos[goal_id][1]);
	//vec=vec+PedToPed(ped,ped_list);
	
	if(ModelParams::SocialForceWorld)
		vec=vec+PedToCar(ped,car.carPos);	
	if(debug)
		std::cout<<"total force "<<vec.dw<<" "<<vec.dh<<std::endl;
	return vec;
}


/*
void SFM::ModelTrans(PedestrianState&state,UtilUniform unif)
{
	vector<Pedestrian> ped_list;
	Pedestrian ped;
	//for(int i=0;i<state.PedPoses.size();i++)
	for(int i=0;i<ModelParams::N_PED;i++)
	{
		sfm_window->LocalToGlobal(state.PedPoses[i].first.X,state.PedPoses[i].first.Y,ped.w,ped.h);   //here temporarily use the global variable, need to improve
		ped.goal=state.PedPoses[i].second;
		ped_list.push_back(ped);	
	}
	Car car(sfm_map,state.RobPos.Y*ModelParams::path_rln);     //also use global variable here
	double window_angle=sfm_window->GetYaw();
	//for(int i=0;i<state.PedPoses.size();i++)
	for(int i=0;i<ModelParams::N_PED;i++)
	{
		
		sfm_window->LocalToGlobal(state.PedPoses[i].first.X,state.PedPoses[i].first.Y,ped.w,ped.h); 
		ped.goal=state.PedPoses[i].second;
//If pedestrian reach the goal
		int goal_w1,goal_h1,goal_w2,goal_h2;
		sfm_window->GlobalToLocal(sfm_map->goal_pos[ped.goal][0],sfm_map->goal_pos[ped.goal][1],goal_w1,goal_h1);
		sfm_window->GlobalToLocal(ped.w,ped.h,goal_w2,goal_h2);
		if(goal_w1==goal_w2&&goal_h1==goal_h2)  //reach the goal
			continue;
		
		ped_list.erase(ped_list.begin()+i);
		MyVector vec=SocialForce(ped,ped_list,car);
		ped_list.insert(ped_list.begin()+i,ped);

		double ped_angle=vec.GetAngle();
		if(ped_angle<0) ped_angle+=2*ModelParams::pi;   //[0,2*ModelParams::pi]
		double angle=window_angle-ped_angle;
		if(angle<-ModelParams::pi)     angle=angle+2*ModelParams::pi;
		else if(angle>ModelParams::pi)  angle=-(2*ModelParams::pi-angle);					
		int phase=SelectPhase(angle,vec.GetLength());
		//std::cout<<"phase "<<phase<<std::endl;

		//start state transition
		double prob=unif.next(),sum=0;
		int next_i,next_j,curr_i=state.PedPoses[i].first.X,curr_j=state.PedPoses[i].first.Y;
		double total_weight=0;
		for(int p=0;p<9;p++)
		{
			next_i=curr_i+pri_table[phase][p][0];
			next_j=curr_j+pri_table[phase][p][1];
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->obs_map[next_i][next_j]==0) {
				total_weight+=pri_table[phase][p][2];	
				if(debug)
					std::cout<<"next i next j prob "<<next_i<<" "<<next_j<<" "<<pri_table[phase][p][2]<<std::endl;
			}
		}
		//deal with boundary condition
		//double free_prob=0.5/(free_pos-1);

		for(int p=0;p<9;p++)
		{
			next_i=curr_i+pri_table[phase][p][0];
			next_j=curr_j+pri_table[phase][p][1];
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->obs_map[next_i][next_j]==0) {
				sum+=pri_table[phase][p][2]/total_weight;
				if(sum>prob)  break;
			}
		}
		state.PedPoses[i].first.X=next_i;
		state.PedPoses[i].first.Y=next_j;
		//std::cout<<std::endl<<std::endl;

	}

}*/
void SFM::WorldTrans(vector<Pedestrian> &ped_list,Car&car,UtilUniform &unif)
{
	double window_angle=ModelParams::pi/2;
	vector<Pedestrian> sfm_list=ped_list;
	for(int i=0;i<sfm_list.size();i++)
	{
		Pedestrian ped=sfm_list[i];
		if(!sfm_map->InMap(ped.w,ped.h)) continue;  //ped already not in the world map
		int goal_w1,goal_h1,goal_w2,goal_h2;
		sfm_window->GlobalToLocal(sfm_map->goal_pos[ped.goal][0],sfm_map->goal_pos[ped.goal][1],goal_w1,goal_h1);
		sfm_window->GlobalToLocal(ped.w,ped.h,goal_w2,goal_h2);
		if(goal_w1==goal_w2&&goal_h1==goal_h2)  //reach the goal
		{
			ped_list[i].w=-1000;
			ped_list[i].h=-1000;
			continue;
		}
		
		sfm_list.erase(sfm_list.begin()+i);
		MyVector vec=SocialForce(ped,sfm_list,car);
		sfm_list.insert(sfm_list.begin()+i,ped);

			
		double ped_angle=vec.GetAngle();
		if(debug)  cout<<ped_angle<<" "<<window_angle<<endl;

		if(ped_angle<0) ped_angle+=2*ModelParams::pi;   //[0,2*ModelParams::pi]
		double angle=window_angle-ped_angle;  //[-2*pi,2*pi]
		if(angle<-ModelParams::pi)     angle=angle+2*ModelParams::pi;
		else if(angle>ModelParams::pi)  angle=-(2*ModelParams::pi-angle);   //[-pi,pi]					
		
		if(angle<0)   angle+=2*ModelParams::pi;  //[0,2*pi],from the window angle

		double prob=unif.next(),sum=0;
		int next_i,next_j,curr_i=ped.w,curr_j=ped.h;
		double total_weight=0;
		double weight=1.0;
		if(debug)
		{
			cout<<"curr_i curr_j "<<curr_i<<" "<<curr_j<<endl;
		}
		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0]*ModelParams::rln*0.8;
			next_j=curr_j+next_grid[p][1]*ModelParams::rln*0.8;
			weight=1.0;
			if(sfm_map->InMap(next_i,next_j)&&(!Danger(next_i,next_j,car))){
				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
				if(debug)
				{
					std::cout<<next_i<<" "<<next_j<<" "<<GetAngularWeight(p,angle,vec.GetLength())<<" "<<GetLaneWeight(next_i)<<std::endl;

				}
				total_weight+=weight;
			}
		}


		if(debug)
		{
			for(int p=0;p<9;p++)
			{
				next_i=curr_i+next_grid[p][0]*ModelParams::rln*0.8;
				next_j=curr_j+next_grid[p][1]*ModelParams::rln*0.8;
				weight=1.0;
				if(sfm_map->InMap(next_i,next_j)&&(!Danger(next_i,next_j,car))) {
				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
					std::cout<<"next i next j prob "<<next_i<<" "<<next_j<<" "<<weight/total_weight<<std::endl;
				}

			}
			cout<<"prob "<<prob<<endl;
		}
		//deal with boundary condition
		//double free_prob=0.5/(free_pos-1);

		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0]*ModelParams::rln*0.8;
			next_j=curr_j+next_grid[p][1]*ModelParams::rln*0.8;
			weight=1.0;
			if(sfm_map->InMap(next_i,next_j)&&(!Danger(next_i,next_j,car))) {
				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
				sum+=weight/total_weight;
				if(sum>prob)  break;
			}
		}
		ped_list[i].w=next_i;
		ped_list[i].h=next_j;

	}
}

void SFM::UpdateGoalModel()
{
	for(int x=0;x<ModelParams::XSIZE;x++)
		for(int y=0;y<ModelParams::YSIZE;y++)
			for(int g=0;g<ModelParams::NGOAL;g++)
			{
				int pedw,pedh;
				sfm_window->LocalToGlobal(x,y,pedw,pedh);
				Pedestrian ped(pedw,pedh);
				MyVector vec=PedToGoal(ped,sfm_map->goal_pos[g][0],sfm_map->goal_pos[g][1]);	
				goal_model[x][y][g]=vec;	
			}
}

void SFM::UpdateCrashModel()
{
	for(int x=0;x<ModelParams::XSIZE;x++)
		for(int y=0;y<ModelParams::YSIZE;y++)
		{

			int min_dist=10000;
			for(int ry=0;ry<sfm_window->rob_map.size();ry++)
			{
				int	robx,roby;
				robx=sfm_window->rob_map[ry].first;
				roby=sfm_window->rob_map[ry].second;
				if(abs(roby-y)<min_dist)
				{
					min_dist=abs(roby-y);	
					crash_model[x][y]=ry;
				}
			}
		}
}
void SFM::UpdateCarModel()
{
	for(int x=0;x<ModelParams::XSIZE;x++)
		for(int y=0;y<ModelParams::YSIZE;y++)
			for(int ry=0;ry<sfm_window->rob_map.size();ry++)
			{
				int ped_w,ped_h;
				sfm_window->LocalToGlobal(x,y,ped_w,ped_h);
				//int car_w,car_h;
				//sfm_window->LocalToGlobal(sfm_window->rob_map[ry].first,sfm_window->rob_map[ry].second,car_w,car_h); 

				//Car car(sfm_map,car_w,car_h);
				int carPos=ry*ModelParams::path_rln+sfm_window->wOrigin;
				//MyVector vec=PedToCar(ped,car);	
				for(int g=0;g<ModelParams::NGOAL;g++)
				{
					Pedestrian ped(ped_w,ped_h,g,0);
					MyVector vec=PedToCar(ped,carPos);

					car_model[x][y][ry][g]=vec;
				}
				/*
				if(ry==2)
				{
					cout<<sfm_window->rob_map[ry].first<<" "<<sfm_window->rob_map[ry].second;
					cout<<" "<<car_w<<" "<<car_h<<" ";
					cout<<vec.dw<<" "<<vec.dh<<endl;
				}*/
			}
}
double SFM::ModelTransProb(PedestrianState state,PedestrianState state_new)
{
	double window_angle=w_angle;
	double total_prob=1.0;
	for(int i=0;i<state.num;i++)
	{
		int x,y,g,ry;
		x=state.PedPoses[i].first.X;
		y=state.PedPoses[i].first.Y;

		ry=state.RobPos.Y;
		g=state.PedPoses[i].second;	

		if(x==local_goals[g][0]&&y==local_goals[g][1]) continue;   //reach the goal

		MyVector vec1=goal_model[x][y][g];

		MyVector vec2=car_model[x][y][ry][g];
		MyVector vec;
		if(ModelParams::SocialForceModel)
			vec=vec1+vec2;	
		else
			vec=vec1;//+vec2;
		

			
		double ped_angle=vec.GetAngle();

		if(ped_angle<0) ped_angle+=2*ModelParams::pi;   //[0,2*ModelParams::pi]
		double angle=window_angle-ped_angle;  //[-2*pi,2*pi]
		if(angle<-ModelParams::pi)     angle=angle+2*ModelParams::pi;
		else if(angle>ModelParams::pi)  angle=-(2*ModelParams::pi-angle);   //[-pi,pi]					
		
		if(angle<0)   angle+=2*ModelParams::pi;  //[0,2*pi]

		double sum=0;
		int next_i,next_j,curr_i=state.PedPoses[i].first.X,curr_j=state.PedPoses[i].first.Y;
		double total_weight=0;
		double weight=1.0;
		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0];
			next_j=curr_j+next_grid[p][1];
			weight=1.0;
			int next_w,next_h;
			sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				//pedestrian crashing onto the car, not allowed
				if(next_i==sfm_window->rob_map[ry].first&&next_j==sfm_window->rob_map[ry].second) continue;

				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
				total_weight+=weight;

			}
		}

		double this_prob=0.001;
		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0];
			next_j=curr_j+next_grid[p][1];
			weight=1.0;
			int next_w,next_h;
			sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				//pedestrian crashing onto the car, not allowed
				if(next_i==sfm_window->rob_map[ry].first&&next_j==sfm_window->rob_map[ry].second) continue;

				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
			}
			if(next_i==state_new.PedPoses[i].first.X&&next_j==state_new.PedPoses[i].first.Y)
			{
				this_prob=(weight/total_weight);
				break;
			}

		}
		total_prob*=this_prob;
	}
	return total_prob;
}
void SFM::ModelTransFast(PedestrianState&state,UtilUniform &unif)
{
	double window_angle=w_angle;
	for(int i=0;i<state.num;i++)
	{
		int x,y,g,ry;
		x=state.PedPoses[i].first.X;
		y=state.PedPoses[i].first.Y;

		ry=state.RobPos.Y;
		g=state.PedPoses[i].second;	

		if(x==local_goals[g][0]&&y==local_goals[g][1]) continue;   //reach the goal

		MyVector vec1=goal_model[x][y][g];

		if(debug)
			cout<<"goal force "<<vec1.dw<<" "<<vec1.dh<<endl;
		MyVector vec2=car_model[x][y][ry][g];
		if(debug)
			cout<<"car force "<<vec2.dw<<" "<<vec2.dh<<endl;
		MyVector vec;
		if(ModelParams::SocialForceModel)
			vec=vec1+vec2;	
		else
			vec=vec1;//+vec2;
		if(debug)
			cout<<"total force "<<vec.dw<<" "<<vec.dh<<endl;
		
	/*	
		if(x==2&&y==6&&ry==4&&g==4)
			cout<<"total force "<<vec2.dw<<" "<<vec2.dh<<endl;
			*/
		
		if(g==ModelParams::NGOAL-1)  //stop intention
			vec.AdjustLength(0.5);
			
		double ped_angle=vec.GetAngle();
		if(debug)  cout<<ped_angle<<" "<<window_angle<<endl;

		if(ped_angle<0) ped_angle+=2*ModelParams::pi;   //[0,2*ModelParams::pi]
		double angle=window_angle-ped_angle;  //[-2*pi,2*pi]
		if(angle<-ModelParams::pi)     angle=angle+2*ModelParams::pi;
		else if(angle>ModelParams::pi)  angle=-(2*ModelParams::pi-angle);   //[-pi,pi]					
		
		if(angle<0)   angle+=2*ModelParams::pi;  //[0,2*pi]

		double prob=unif.next(),sum=0;
		int next_i,next_j,curr_i=state.PedPoses[i].first.X,curr_j=state.PedPoses[i].first.Y;
		double total_weight=0;
		double weight=1.0;
		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0];
			next_j=curr_j+next_grid[p][1];
			weight=1.0;
			int next_w,next_h;
			sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				//pedestrian crashing onto the car, not allowed
			//	if(next_i==sfm_window->rob_map[ry].first&&next_j==sfm_window->rob_map[ry].second) continue;

				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
				total_weight+=weight;
				/*
				if(x==2&&y==9)
				{
					cout<<g<<" "<<next_i<<" "<<next_j<<" "<<weight<<endl;
				}*/
			}
		}
		if(debug)
		{
			for(int p=0;p<9;p++)
			{
				next_i=curr_i+next_grid[p][0];
				next_j=curr_j+next_grid[p][1];
				weight=1.0;
				int next_w,next_h;
				sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
				if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				//pedestrian crashing onto the car, not allowed
				//if(next_i==sfm_window->rob_map[ry].first&&next_j==sfm_window->rob_map[ry].second) continue;

				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
					std::cout<<"next i next j prob "<<next_i<<" "<<next_j<<" "<<weight/total_weight<<std::endl;
				}

			}
		}
		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0];
			next_j=curr_j+next_grid[p][1];
			weight=1.0;
			int next_w,next_h;
			sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				//pedestrian crashing onto the car, not allowed
				//if(next_i==sfm_window->rob_map[ry].first&&next_j==sfm_window->rob_map[ry].second) continue;

				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
				sum+=weight/total_weight;
				/*
				if(x==2&&y==9)
				{
					cout<<g<<" "<<next_i<<" "<<next_j<<" "<<weight/total_weight<<endl;
				}*/
				if(sum>prob)  break;
			}
		}
		state.PedPoses[i].first.X=next_i;
		state.PedPoses[i].first.Y=next_j;

	}
}
void SFM::ModelTrans(PedestrianState&state,UtilUniform &unif)
{
	vector<Pedestrian> ped_list;
	Pedestrian ped;
	//for(int i=0;i<state.PedPoses.size();i++)
	for(int i=0;i<state.num;i++)
	{
		sfm_window->LocalToGlobal(state.PedPoses[i].first.X,state.PedPoses[i].first.Y,ped.w,ped.h);   //here temporarily use the global variable, need to improve
		ped.goal=state.PedPoses[i].second;
		ped_list.push_back(ped);	
	}
	int car_w,car_h;
	int robY=state.RobPos.Y;
	sfm_window->LocalToGlobal(sfm_window->rob_map[robY].first,sfm_window->rob_map[robY].second,car_w,car_h); 
	Car car(sfm_map,car_w,car_h);     //also use global variable here
	double window_angle=sfm_window->GetYaw();
	//for(int i=0;i<state.PedPoses.size();i++)
	for(int i=0;i<state.num;i++)
	{
		
		sfm_window->LocalToGlobal(state.PedPoses[i].first.X,state.PedPoses[i].first.Y,ped.w,ped.h); 
		ped.goal=state.PedPoses[i].second;
//If pedestrian reach the goal
		int goal_w1,goal_h1,goal_w2,goal_h2;
		sfm_window->GlobalToLocal(sfm_map->goal_pos[ped.goal][0],sfm_map->goal_pos[ped.goal][1],goal_w1,goal_h1);
		sfm_window->GlobalToLocal(ped.w,ped.h,goal_w2,goal_h2);
		if(goal_w1==goal_w2&&goal_h1==goal_h2)  //reach the goal
			continue;
		
		ped_list.erase(ped_list.begin()+i);
		MyVector vec=SocialForce(ped,ped_list,car);
		ped_list.insert(ped_list.begin()+i,ped);

		double ped_angle=vec.GetAngle();
		if(debug)  cout<<ped_angle<<" "<<window_angle<<endl;

		if(ped_angle<0) ped_angle+=2*ModelParams::pi;   //[0,2*ModelParams::pi]
		double angle=window_angle-ped_angle;  //[-2*pi,2*pi]
		if(angle<-ModelParams::pi)     angle=angle+2*ModelParams::pi;
		else if(angle>ModelParams::pi)  angle=-(2*ModelParams::pi-angle);   //[-pi,pi]					
		
		if(angle<0)   angle+=2*ModelParams::pi;  //[0,2*pi]

		double prob=unif.next(),sum=0;
		int next_i,next_j,curr_i=state.PedPoses[i].first.X,curr_j=state.PedPoses[i].first.Y;
		double total_weight=0;
		double weight=1.0;
		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0];
			next_j=curr_j+next_grid[p][1];
			weight=1.0;
			int next_w,next_h;
			sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
				if(debug)
				{
					std::cout<<next_i<<" "<<next_j<<" "<<GetAngularWeight(p,angle,vec.GetLength())<<" "<<GetLaneWeight(next_i)<<std::endl;

				}
				total_weight+=weight;
			}
		}


		if(debug)
		{
			for(int p=0;p<9;p++)
			{
				next_i=curr_i+next_grid[p][0];
				next_j=curr_j+next_grid[p][1];
				weight=1.0;
				int next_w,next_h;
				sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
				if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
					std::cout<<"next i next j prob "<<next_i<<" "<<next_j<<" "<<weight/total_weight<<std::endl;
				}

			}
		}
		//deal with boundary condition
		//double free_prob=0.5/(free_pos-1);

		for(int p=0;p<9;p++)
		{
			next_i=curr_i+next_grid[p][0];
			next_j=curr_j+next_grid[p][1];
			weight=1.0;
			int next_w,next_h;
			sfm_window->LocalToGlobal(next_i,next_j,next_w,next_h); 
			if(next_i>=0&&next_i<ModelParams::XSIZE&&next_j>=0&&next_j<ModelParams::YSIZE&&sfm_map->Free(next_w,next_h)) {
				weight*=GetAngularWeight(p,angle,vec.GetLength());	
				weight*=GetLaneWeight(next_i);
				sum+=weight/total_weight;
				if(sum>prob)  break;
			}
		}
		state.PedPoses[i].first.X=next_i;
		state.PedPoses[i].first.Y=next_j;
		//std::cout<<std::endl<<std::endl;

	}

}

