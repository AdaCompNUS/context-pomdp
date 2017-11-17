#include <despot/core/history.h>
#include <cstring>
#include <iostream>

namespace despot {

History::History(const History& src)
{
	actions_.resize(src.Size());
	observations_.resize(src.Size());

	memcpy(actions_.data(),src.Action(), src.Size()*sizeof(int));
	memcpy(observations_.data(),src.Observation(), src.Size()*sizeof(OBS_TYPE));
	/*for(int j=0;j<src.Size();j++)
	{
		std::cout<<src.Action(j)<<" "<<src.Observation(j)<<" ";
		std::cout<<actions_[j]<<" "<<observations_[j]<<" ";
	}
	std::cout<<std::endl;*/
}

}
