/*
 * PedPomdpNode.h
 *
 */

#ifndef CROWD_POMDP_H_
#define CROWD_POMDP_H_

#include "controller.h"

class PedPomdpNode
{
public:
  PedPomdpNode(int argc, char** argv);
  ~PedPomdpNode() {};

  Controller* controller;

	bool pathPublished;
};

#endif /* CROWD_POMDP_H_ */
