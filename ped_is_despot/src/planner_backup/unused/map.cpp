#include "map.h"
#include <iostream>

using namespace std;
void MyMap::LoadMap(const char*map_name)
{
	cout<<"loading the map"<<endl;
	std::ifstream in(map_name);
	in>>width;
	in>>height;

	cout<<"map size "<<width<<" "<<height<<endl;
	//need to check the index order here
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			in>>full_map[j][i];	
			//full_map[j][i]=0;
		}
	}
	
		
	for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
			PreComp(i,j);
			
	/*
	cout<<"generate the precompute file"<<endl;
	ofstream out("obs_info.data");
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<height;j++)
		{
			PreComp(i,j);
			out<<map_obs[i][j]<<" ";
		}
		out<<endl;
	}*/

	/*	
	cout<<"loading the precompute info"<<endl;
	ifstream obs_in("obs_info.data");
	for(int i=0;i<width;i++)
		for(int j=0;j<height;j++)
		{
			obs_in>>map_obs[i][j];
		}
		*/
	//cout<<map_obs[250][200]<<" "<<full_map[250][200]<<endl;
	//cout<<map_obs[110][220]<<" "<<full_map[126][220]<<endl;
}

void MyMap::LoadPath(const char*path_name, int id)
{
	std::ifstream in(path_name);
	in>>pathLength;
	std::cout<<"path length "<<pathLength<<std::endl;
	//pathLength=
	for(int i=0;i<pathLength;i++)
	{
//		in>>global_plan_multi[id][i][0];
//		in>>global_plan_multi[id][i][1];

//		if(id==0)  //use the first path as the default path

		in>>global_plan[i][0];
		in>>global_plan[i][1];
		//std::cout<<global_plan[i][0]<<" "<<global_plan[i][1]<<std::endl;
	}

}


