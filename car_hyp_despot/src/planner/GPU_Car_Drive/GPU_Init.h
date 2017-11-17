#ifndef GPUINIT_H
#define GPUINIT_H

#include <despot/core/policy_graph.h>
#include <despot/core/pomdp.h>
#include <string>
#include <iostream>
using namespace std;
using despot::DSPOMDP;
using despot::PolicyGraph;



void InitializedGPUModel(std::string rollout_type);

void InitializedGPUModel(std::string rollout_type, DSPOMDP* Hst_model);

void InitializeGPUPolicyGraph(PolicyGraph* hostGraph);

void InitializeGPUGlobals();

void DeleteGPUModel();

void DeleteGPUGlobals();

#endif
