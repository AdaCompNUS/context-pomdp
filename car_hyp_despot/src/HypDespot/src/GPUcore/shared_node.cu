#include <despot/GPUcore/shared_node.h>
#include <despot/solver/despot.h>

using namespace std;

namespace despot {
MsgQueque<Shared_VNode> Expand_queue, Print_queue;
/* =============================================================================
 * Shared_VNode class
 * =============================================================================*/

Shared_VNode::Shared_VNode(vector<State*>& particles,std::vector<int> particleIDs, int depth, Shared_QNode* parent,
	OBS_TYPE edge)
{
	lock_guard<mutex> lck(_mutex);
	particles_=particles;
	particleIDs_=particleIDs;
	GPU_particles_=NULL;
	num_GPU_particles_=0;
	belief_=NULL;
	depth_=depth;
	parent_=parent;
	edge_=edge;
	vstar=this;
	likelihood=1;
	logd << "Constructed Shared_VNode with " << particles_.size() << " particles"
		<< endl;
	for (int i = 0; i < particles_.size(); i++) {
		logd << " " << i << " = " <<"("<< particleIDs_[i]<<")"<< *particles_[i] << endl;
	}
	weight_=0;
	exploration_bonus=0;
	value_=0;
	is_waiting_=false;//waiting_change=false;
	visit_count_=0;
}

Shared_VNode::Shared_VNode(Belief* belief, int depth, Shared_QNode* parent, OBS_TYPE edge){
	lock_guard<mutex> lck(_mutex);
	GPU_particles_=NULL;
	num_GPU_particles_=0;
	belief_=belief;
	depth_=depth;
	parent_=parent;
	edge_=edge;
	vstar=this;
	likelihood=1;
	weight_=0;
	exploration_bonus=0;
	value_=0;
	is_waiting_=false;//waiting_change=false;
	visit_count_=0;
}

Shared_VNode::Shared_VNode(int count, double value, int depth, Shared_QNode* parent, OBS_TYPE edge) {
	lock_guard<mutex> lck(_mutex);
	GPU_particles_=NULL;
	num_GPU_particles_=0;
	belief_=NULL;
	depth_=depth;
	parent_=parent;
	edge_=edge;
	count_=count;
	value_=value;
	weight_=0;
	exploration_bonus=0;
	is_waiting_=false;//waiting_change=false;
	visit_count_=0;
}

Shared_VNode::~Shared_VNode() {
	lock_guard<mutex> lck(_mutex);
	for (int a = 0; a < children_.size(); a++) {
		Shared_QNode* child = static_cast<Shared_QNode*>(children_[a]);
		assert(child != NULL);
		delete child;
	}
	children_.clear();

	if (belief_ != NULL)
		delete belief_;
}

Belief* Shared_VNode::belief() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}
	lock_guard<mutex> lck(_mutex);
	if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return belief_;
}

const vector<State*>& Shared_VNode::particles() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}
	lock_guard<mutex> lck(_mutex);
	if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return particles_;
}

const vector<int>& Shared_VNode::particleIDs() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}
	lock_guard<mutex> lck(_mutex);

	if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return particleIDs_;
}
void Shared_VNode::depth(int d) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);

	depth_ = d;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}

int Shared_VNode::depth() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}
	lock_guard<mutex> lck(_mutex);
	if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return depth_;
}

void Shared_VNode::parent(Shared_QNode* parent) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	parent_ = parent;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}

Shared_QNode* Shared_VNode::parent() {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}
	lock_guard<mutex> lck(_mutex);
	if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return static_cast<Shared_QNode*>(parent_);
}

OBS_TYPE Shared_VNode::edge() {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}
	lock_guard<mutex> lck(_mutex);
	if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return edge_;
}

double Shared_VNode::Weight() {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	if(Globals::config.useGPU==false ||!PassGPUThreshold())
		if(GPUWeight()>0)
			return GPUWeight();
		else
			return State::Weight(particles_);
	else /*if(num_GPU_particles_>0)*/
		return GPUWeight();
}
void Shared_VNode::ResizeParticles(int i)
{
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);

	particles_.resize(i);
	particleIDs_.resize(i);

	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}
void Shared_VNode::ReconstructCPUParticles(const DSPOMDP* model,
		RandomStreams& streams, History& history)
{
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	std::vector<int>& particleIDsinParentList=particleIDs_;
	for(int i=0;i<particleIDsinParentList.size();i++)
	{
		int parent_PID=particleIDsinParentList[i];
		int ScenarioID=((VNode*)this)->parent()->parent()->particleIDs()[parent_PID];

		State* particle=NULL;
		VNode* root=this;
		while(root->parent()!=NULL)//not root yet
		{
			root=root->parent()->parent();//trace backward
		}
		//Copy particle from root
		particle=model->Copy(root->particles()[ScenarioID]);
		int depth=0;
		double reward=0;
		OBS_TYPE obs;
		while(depth!=depth_)//not leaf yet
		{
			int action=history.Action(depth);
			model->Step(*particle,streams.Entry(ScenarioID, depth), action,reward,obs);
			if(obs!=history.Observation(depth))//observation matching
				cerr<<__FUNCTION__<<": Wrong recalculated obs with history!"<<endl;
			depth++;
		}
		particles_.push_back(particle);
		particleIDs_.push_back(ScenarioID);
	}

	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}
/*const vector<Shared_QNode*>& Shared_VNode::children() const {
	return static_cast<vector<Shared_QNode*>>(children_);
}

vector<Shared_QNode*>& Shared_VNode::children() {
	return static_cast<vector<Shared_QNode*>>(children_);
}*/

const Shared_QNode* Shared_VNode::Child(int action) const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return static_cast<Shared_QNode*>(children_[action]);
}

Shared_QNode* Shared_VNode::Child(int action) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return static_cast<Shared_QNode*>(children_[action]);
}

int Shared_VNode::Size() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	int size = 1;
	for (int a = 0; a < children_.size(); a++) {
		size += children_[a]->Size();
	}
	return size;
}

int Shared_VNode::PolicyTreeSize() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	if (children_.size() == 0)
		return 0;

	Shared_QNode* best = NULL;
	for (int a = 0; a < children_.size(); a++) {
		Shared_QNode* child = static_cast<Shared_QNode*>(children_[a]);
		if (best == NULL || child->lower_bound() > best->lower_bound())
			best = child;
	}
	return best->PolicyTreeSize();
}

void Shared_VNode::default_move(ValuedAction move) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	default_move_ = move;
}

ValuedAction Shared_VNode::default_move() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	//lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return default_move_;
}

void Shared_VNode::lower_bound(double value) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	lower_bound_ = value;
}

double Shared_VNode::lower_bound() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	//lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return lower_bound_/*+exploration_bonus*/;
}

void Shared_VNode::upper_bound(double value) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	upper_bound_ = value;
}

double Shared_VNode::upper_bound(bool use_Vloss) const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	//lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	if(use_Vloss)
		return upper_bound_+exploration_bonus;
	else
		return upper_bound_;
}
void Shared_VNode::utility_upper_bound(double value){
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	utility_upper_bound_=value;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}
double Shared_VNode::utility_upper_bound() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	//lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return utility_upper_bound_;
}

bool Shared_VNode::IsLeaf() {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return children_.size() == 0;
}

void Shared_VNode::Add(double val) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}

void Shared_VNode::count(int c) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	count_ = c;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}
int Shared_VNode::count() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	//lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return count_;
}
void Shared_VNode::value(double v) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	value_ = v;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}
double Shared_VNode::value() const {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	//lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return value_;
}

void Shared_VNode::Free(const DSPOMDP& model) {
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	for (int i = 0; i < particles_.size(); i++) {
		if(particles_[i])model.Free(particles_[i]);
	}

	for (int a = 0; a < children().size(); a++) {
		Shared_QNode* Shared_QNode = Child(a);
		map<OBS_TYPE, VNode*>& children = Shared_QNode->children();
		for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
			it != children.end(); it++) {
			static_cast<Shared_VNode*>(it->second)->Free(model);
		}
	}
}

void Shared_VNode::PrintPolicyTree(int depth, ostream& os) {
	lock_guard<mutex> lck(_mutex);
	if (depth != -1 && this->depth() > depth)
		return;

	vector<QNode*>& Shared_QNodes = children();
	if (Shared_QNodes.size() == 0) {
		int astar = this->default_move().action;
		os << this << "-a=" << astar << endl;
	} else {
		Shared_QNode* qstar = NULL;
		for (int a = 0; a < Shared_QNodes.size(); a++) {
			Shared_QNode* Shared_qNode = static_cast<Shared_QNode*>(Shared_QNodes[a]);
			if (qstar == NULL || Shared_qNode->lower_bound() > qstar->lower_bound()) {
				qstar = Shared_qNode;
			}
		}

		os << this << "-a=" << qstar->edge() << endl;

		vector<OBS_TYPE> labels;
		map<OBS_TYPE, VNode*>& Shared_VNodes = qstar->children();
		for (map<OBS_TYPE, VNode*>::iterator it = Shared_VNodes.begin();
			it != Shared_VNodes.end(); it++) {
			labels.push_back(it->first);
		}

		for (int i = 0; i < labels.size(); i++) {
			if (depth == -1 || this->depth() + 1 <= depth) {
				os << repeat("|   ", this->depth()) << "| o=" << labels[i]
					<< ": ";
				qstar->Child(labels[i])->PrintPolicyTree(depth, os);
			}
		}
	}
}

/*void Shared_VNode::PrintTree(int depth, ostream& os) {
	if (depth != -1 && this->depth() > depth)
		return;

	if (this->depth() == 0) {
		os << "d - default value" << endl
			<< "l - lower bound" << endl
			<< "u - upper bound" << endl
			<< "r - totol weighted one step reward" << endl
			<< "w - total particle weight" << endl;
	}

	os << "(" << "d:" << this->default_move().value <<
		" l:" << this->lower_bound() << ", u:" << this->upper_bound()
		<< ", w:" << this->Weight() << ", weu:" << DESPOT::WEU(static_cast<VNode*>(this))
		<< ")"
		<< endl;


	vector<QNode*>& Shared_QNodes = children();
	for (int a = 0; a < Shared_QNodes.size(); a++) {
		Shared_QNode* Shared_qNode = static_cast<Shared_QNode*>(Shared_QNodes[a]);

		vector<OBS_TYPE> labels;
		map<OBS_TYPE, VNode*>& Shared_VNodes = Shared_qNode->children();
		for (map<OBS_TYPE, VNode*>::iterator it = Shared_VNodes.begin();
			it != Shared_VNodes.end(); it++) {
			labels.push_back(it->first);
		}

		os << repeat("|   ", this->depth()) << "a="
			<< Shared_qNode->edge() << ": "
			<< "(d:" << Shared_qNode->default_value << ", l:" << Shared_qNode->lower_bound()
			<< ", u:" << Shared_qNode->upper_bound()
			<< ", r:" << Shared_qNode->step_reward << ")" << endl;

		for (int i = 0; i < labels.size(); i++) {
			if (depth == -1 || this->depth() + 1 <= depth) {
				os << repeat("|   ", this->depth()) << "| o=" << labels[i]
					<< ": ";
				Shared_qNode->Child(labels[i])->PrintTree(depth, os);
			}
		}
	}
}*/

void Shared_VNode::AddVirtualLoss(float v)
{
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	exploration_bonus-=v;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}
void Shared_VNode::RemoveVirtualLoss(float v)
{
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	lock_guard<mutex> lck(_mutex);
	exploration_bonus+=v;
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
}
float Shared_VNode::GetVirtualLoss()
{
	/*bool waiting_change=false;
	if(is_waiting_ && use_multi_thread_)
	{
		thread_barrier->DettachThread(__FUNCTION__);
		waiting_change=true;
	}*/
	//lock_guard<mutex> lck(_mutex);
	/*if(waiting_change && use_multi_thread_)
		thread_barrier->AttachThread(__FUNCTION__);*/
	return exploration_bonus;
}

/* =============================================================================
 * Shared_QNode class
 * =============================================================================*/

Shared_QNode::Shared_QNode(Shared_VNode* parent, int edge)
	{
	lock_guard<mutex> lck(_mutex);
	parent_=parent;
	edge_=edge;
	vstar=NULL;
	exploration_bonus=0;
	value_=0;
	//lower_bound_=0;upper_bound_=0;
	visit_count_=0;
	weight_=0;
}

Shared_QNode::Shared_QNode(int count, double value)
	{
	lock_guard<mutex> lck(_mutex);
	count_=count;
	value_=value;
	exploration_bonus=0;
	visit_count_=0;
	weight_=0;
}

Shared_QNode::~Shared_QNode() {
	lock_guard<mutex> lck(_mutex);
	for (map<OBS_TYPE, VNode*>::iterator it = children_.begin();
		it != children_.end(); it++) {
		assert(it->second != NULL);
		delete static_cast<Shared_VNode*>(it->second);
	}
	children_.clear();
}

void Shared_QNode::parent(Shared_VNode* parent) {
	lock_guard<mutex> lck(_mutex);
	parent_ = parent;
}

Shared_VNode* Shared_QNode::parent() {
	//lock_guard<mutex> lck(_mutex);
	return static_cast<Shared_VNode*>(parent_);
}

int Shared_QNode::edge() const{
	//lock_guard<mutex> lck(_mutex);
	return edge_;
}

/*map<OBS_TYPE, Shared_VNode*>& Shared_QNode::children() {
	return static_cast<Shared_QNode*>(children_);
}*/

Shared_VNode* Shared_QNode::Child(OBS_TYPE obs) {
	lock_guard<mutex> lck(_mutex);
	return static_cast<Shared_VNode*>(children_[obs]);
}

int Shared_QNode::Size() const {
	lock_guard<mutex> lck(_mutex);
	int size = 0;
	for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		size += static_cast<Shared_VNode*>(it->second)->Size();
	}
	return size;
}

int Shared_QNode::PolicyTreeSize() const {
	lock_guard<mutex> lck(_mutex);
	int size = 0;
	for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		size += static_cast<Shared_VNode*>(it->second)->PolicyTreeSize();
	}
	return 1 + size;
}

double Shared_QNode::Weight() /*const*/ {
	lock_guard<mutex> lck(_mutex);
	if(weight_>1e-5) return weight_;
	else
	{
		weight_=0;
		for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
			it != children_.end(); it++) {
			weight_ += static_cast<Shared_VNode*>(it->second)->Weight();
		}
		if(weight_>1.001)
		{
			for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
					it != children_.end(); it++) {
				Global_print_value(this_thread::get_id(),
						static_cast<Shared_VNode*>(it->second)->Weight(),
						"Wrong weight from Shared_QNode::Weight()");
				Global_print_value(this_thread::get_id(),
						static_cast<Shared_VNode*>(it->second)->num_GPU_particles_,
						"Wrong weight from Shared_QNode::Weight(), num of particles=");
			}
		}
		return weight_;
	}
}

void Shared_QNode::lower_bound(double value) {
	lock_guard<mutex> lck(_mutex);
	lower_bound_ = value;
}

double Shared_QNode::lower_bound() const {
	//lock_guard<mutex> lck(_mutex);
	return lower_bound_/*+exploration_bonus*/;
}

void Shared_QNode::upper_bound(double value) {
	lock_guard<mutex> lck(_mutex);
	upper_bound_ = value;
}

double Shared_QNode::upper_bound(bool use_Vloss) const {
	//lock_guard<mutex> lck(_mutex);
	if(use_Vloss)
		return upper_bound_+exploration_bonus;
	else
		return upper_bound_;
}
void Shared_QNode::utility_upper_bound(double value){
	lock_guard<mutex> lck(_mutex);
	utility_upper_bound_=value;
}
double Shared_QNode::utility_upper_bound() const {
	//lock_guard<mutex> lck(_mutex);
	return utility_upper_bound_;
}
void Shared_QNode::Add(double val) {
	lock_guard<mutex> lck(_mutex);
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
}

void Shared_QNode::count(int c) {
	lock_guard<mutex> lck(_mutex);
	count_ = c;
}

int Shared_QNode::count() const {
	//lock_guard<mutex> lck(_mutex);
	return count_;
}

void Shared_QNode::value(double v) {
	lock_guard<mutex> lck(_mutex);
	value_ = v;
}

double Shared_QNode::value() const {
	//lock_guard<mutex> lck(_mutex);
	return value_;
}
void Shared_QNode::AddVirtualLoss(float v)
{
	lock_guard<mutex> lck(_mutex);
	exploration_bonus-=v;
}
void Shared_QNode::RemoveVirtualLoss(float v)
{
	lock_guard<mutex> lck(_mutex);
	exploration_bonus+=v;
}
float Shared_QNode::GetVirtualLoss()
{
	//lock_guard<mutex> lck(_mutex);
	return exploration_bonus;
}
} // namespace despot
