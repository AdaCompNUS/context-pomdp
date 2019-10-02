/*
 * PedPomdpNode.h
 *
 */

#ifndef PEDESTRIAN_MOMDP_H_
#define PEDESTRIAN_MOMDP_H_

#include "controller.h"

class PedPomdpNode
{
public:
    PedPomdpNode(int argc, char** argv);
    ~PedPomdpNode();
    
    Controller* controller;

	bool pathPublished;
};

#endif /* PEDESTRIAN_MOMDP_H_ */
