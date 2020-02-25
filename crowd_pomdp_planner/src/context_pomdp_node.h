/*
 * ContextPomdpNode.h
 *
 */

#ifndef CROWD_POMDP_H_
#define CROWD_POMDP_H_

#include "controller.h"

class ContextPomdpNode
{
public:
  ContextPomdpNode(int argc, char** argv);
  ~ContextPomdpNode() {};

  Controller* controller;

	bool pathPublished;
};

#endif /* CROWD_POMDP_H_ */
