#include "window.h"

double MyWindow::GetYaw()
{
	double v1_w=(w1-w0)/(ModelParams::XSIZE+0.0);double v1_h=(h1-h0)/(ModelParams::XSIZE+0.0);
	double v2_w=(w3-w0)/(ModelParams::YSIZE+0.0);double v2_h=(h3-h0)/(ModelParams::YSIZE+0.0);
	return  atan2(v2_h,v2_w); 
}


void MyWindow::UpdateRobMap(int windowOrigin)
{
	rob_map.clear();
	int rob_w,rob_h,w_old,h_old;
	int pt=windowOrigin+ModelParams::path_rln/2;
	rob_w=map->global_plan[pt][0];
	rob_h=map->global_plan[pt][1];
	
	GlobalToLocal(map->global_plan[pt][0],map->global_plan[pt][1],rob_w,rob_h);
	if(rob_w<0) rob_w=0;
	if(rob_w>ModelParams::XSIZE-1) rob_w=ModelParams::XSIZE-1;
	if(rob_h<0) rob_h=0;
	if(rob_h>ModelParams::YSIZE-1) rob_h=ModelParams::YSIZE-1;
	w_old=rob_w;
	h_old=rob_h;
	rob_map.push_back(make_pair(rob_w,rob_h));
	if(ModelParams::debug)
	{
		cout<<rob_w<<" "<<rob_h<<endl;
	}
	//pt+=path_ModelParams::rln;
	pt+=ModelParams::rln;  //here we cannot set the step size too small, in case of floating point error
	//pt+=10;
	
	cout<<"robot map rln "<<ModelParams::rln<<endl;

	while(InWindow(map->global_plan[pt][0],map->global_plan[pt][1]))
	{
		GlobalToLocal(map->global_plan[pt][0],map->global_plan[pt][1],rob_w,rob_h);
		bool insert=true;
		for(int i=0;i<rob_map.size();i++)
		{
			if(rob_map[i].first==rob_w&&rob_map[i].second==rob_h) {insert=false;break;}
		}
		/*
		if(insert==true)
		{
			rob_map.push_back(make_pair(rob_w,rob_h));
			cout<<rob_w<<" "<<rob_h<<endl;
		}*/
		
		if(!(rob_w==w_old&&rob_h==h_old))
		{
			rob_map.push_back(make_pair(rob_w,rob_h));
			w_old=rob_w;
			h_old=rob_h;
			if(ModelParams::debug)
			{
				cout<<rob_w<<" "<<rob_h<<endl;
			}
		}
		//pt+=path_ModelParams::rln;
		pt+=ModelParams::rln;
		//pt+=10;
	}
	if(ModelParams::debug) 	cout<<"rob map size "<<rob_map.size()<<endl;
}

void MyWindow::StepN(int w,int h,int & dest_w,int & dest_h,double angle,int step_size)
{
	int step_w=-step_size*sin(angle);
	int step_h=step_size*cos(angle);
	dest_w=step_w+w;
	dest_h=step_h+h;
}

bool MyWindow::LeftHanded(int x,int y,int vx,int vy)  //x,y on the left hand side of vx,vy
{
	return  (x*vy-vx*y)<0;	
}
bool MyWindow::InWindow(int w,int h)
{
	 if(!LeftHanded(w-w0,h-h0,w1-w0,h1-h0)) return false;
	 if(!LeftHanded(w-w1,h-h1,w2-w1,h2-h1)) return false;
	 if(!LeftHanded(w-w2,h-h2,w3-w2,h3-h2)) return false;
	 if(!LeftHanded(w-w3,h-h3,w0-w3,h0-h3)) return false;
	 return true;
}
void MyWindow::MapWindowSplit(double w,double h,double yaw)
{
	int robw=(int)(w);
	//int robh=(int)(h-ModelParams::rln/2);
	int robh=(int)(h);
	double yaw0=asin(yaw)*2+ModelParams::pi/2;
	double yaw1=asin(yaw)*2-ModelParams::pi/2;

	double left_step,right_step;
	if(ModelParams::XSIZE%2==0)
	{
		left_step=(ModelParams::XSIZE-1.0)/2;
		right_step=(ModelParams::XSIZE+1.0)/2;
	}
	else
	{
		left_step=ModelParams::XSIZE/2.0;
		right_step=ModelParams::XSIZE/2.0;
	}
	StepN(robw,robh,w0,h0,yaw0,left_step*ModelParams::rln);    //assume car at the middle of the 2nd lane;
	StepN(robw,robh,w1,h1,yaw1,right_step*ModelParams::rln);  
	
	double yaw2=asin(yaw)*2;    //with respect to left low point
	double yaw3=asin(yaw)*2;    //with respect to right low point
	
	StepN(w0,h0,w3,h3,yaw3,ModelParams::rln*ModelParams::YSIZE);
	StepN(w1,h1,w2,h2,yaw2,ModelParams::rln*ModelParams::YSIZE);
	if(ModelParams::debug)
	{
		cout<<w<<" "<<h<<" "<<yaw<<endl;
		cout<<w0<<" "<<h0<<endl;
		cout<<w1<<" "<<h1<<endl;
		cout<<w2<<" "<<h2<<endl;
		cout<<w3<<" "<<h3<<endl;
	}	
}

//transform the pedestrian poses to the window coord
void MyWindow::GlobalToLocal(int globalW,int globalH,int &localW,int&localH)
{

	double vec01_w=w1-w0,vec01_h=h1-h0;
	double vec03_w=w3-w0,vec03_h=h3-h0;
	double vec_w=globalW-w0,vec_h=globalH-h0;

	//get the uniform vector
	double u01_w,u01_h,u03_w,u03_h;
	Uniform(vec01_w,vec01_h,u01_w,u01_h);
	Uniform(vec03_w,vec03_h,u03_w,u03_h);

	double l01=DotProduct(vec_w,vec_h,u01_w,u01_h);
	double l03=DotProduct(vec_w,vec_h,u03_w,u03_h);

//	cout<<vec_w<<" "<<vec_h<<" "<<vec01_w<<" "<<vec01_h<<" "<<l01<<" "<<l03<<endl;

	double n01=Norm(vec01_w,vec01_h)/ModelParams::XSIZE;
	double n03=Norm(vec03_w,vec03_h)/ModelParams::YSIZE;

	//cout<<n01<<" "<<n03<<" "<<l01<<" "<<l03<<endl;

	localW=floor(l01/n01);
	localH=floor(l03/n03);

	//if (localH == -1) {
		//cout << localW << " " << localH << " " << w0 << " " << h0 << " " << w1 << " << w
	//}
}

void MyWindow::LocalToGlobalPreComp(int localW,int localH,int & globalW,int & globalH)
{
	double v1_w=(w1-w0)/(ModelParams::XSIZE+0.0);double v1_h=(h1-h0)/(ModelParams::XSIZE+0.0);
	double v2_w=(w3-w0)/(ModelParams::YSIZE+0.0);double v2_h=(h3-h0)/(ModelParams::YSIZE+0.0);
	MyVector vec01(v1_w*(localW+0.5),v1_h*(localW+0.5));
	MyVector vec03(v2_w*(localH+0.5),v2_h*(localH+0.5));
	MyVector vec_move=vec01+vec03;
	globalW=w0+vec_move.dw;
	globalH=h0+vec_move.dh;
}

void MyWindow::LocalToGlobal(int localW,int localH,int & globalW,int & globalH)
{

	//store in the table for fast reference
	globalW=LGMap[localW][localH][0];
	globalH=LGMap[localW][localH][1];
}
