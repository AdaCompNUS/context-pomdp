#include <despot/GPUcore/GPUnode.h>
//#include <despot/solver/GPUdespot.h>
//#include <despot/GPUutil/GPUmap.h>
//#include <despot/GPUutil/GPUvector.h>


using namespace std;

namespace despot {


void VNode::AssignGPUparticles( Dvc_State* src, int size)
{
	GPU_particles_=src;
	num_GPU_particles_=size;
}
void VNode::FreeGPUparticles(const DSPOMDP& model)
{
	model.DestroyMemoryPool(1);

	//model.DeleteGPUParticles(GPU_particles_, num_GPU_particles_);
/*
	if(GPU_particles_!=NULL)
	{
		HANDLE_ERROR(cudaFree(GPU_particles_));
		num_GPU_particles_=0;GPU_particles_=NULL;
	}
*/
	/*for (int a = 0; a < children().size(); a++) {
		QNode* qnode = Child(a);
		std::map<OBS_TYPE, VNode*>& children = qnode->children();
		for (std::map<OBS_TYPE, VNode*>::iterator it = children.begin();
			it != children.end(); it++) {
			it->second->FreeGPUparticles(model);
		}
	}*/
}
__global__ void CalWeight(double* result, Dvc_State* particles, int size)
{
	*result=0;
	for(int i=0;i<size;i++)
	{
		Dvc_State* tmp=DvcModelGet_(particles,i);
		*result+=tmp->weight;
	}
}
double VNode::GPUWeight()
{
	/*double result;
	double* tmp_result; cudaMalloc(&tmp_result, sizeof(double));
	dim3 grid(1,1); dim3 threads(1,1);
	CalWeight<<<grid, threads>>>(tmp_result,GPU_particles_, num_GPU_particles_);
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaMemcpy(&result, tmp_result, sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(tmp_result);
    return result;*/

    return weight_;
}
void VNode::ResizeParticles(int i)
{
	particles_.resize(i);
	particleIDs_.resize(i);
}
void VNode::ReconstructCPUParticles(const DSPOMDP* model,
		RandomStreams& streams, History& history)
{
	const std::vector<int>& particleIDsinParentList=particleIDs();
	for(int i=0;i<particleIDsinParentList.size();i++)
	{
		cerr<<__FUNCTION__<<"==================="<<endl;

		int ScenarioID=particleIDsinParentList[i];
		cerr<<__FUNCTION__<<": parent_PID="<<ScenarioID<<endl;


		State* particle=NULL;
		VNode* root=this;
		while(root->parent()!=NULL)//not root yet
		{
			cout<<"depth="<<root->depth()<<endl;
			cout<<"action_edge="<<root->parent()->edge()<<endl;
			cout<<"obs_edge="<<root->edge()<<endl;
			cout<<"history_action="<<history.Action(root->depth()-1)<<endl;
			cout<<"history_obs="<<history.Observation(root->depth()-1)<<endl;
			root=root->parent()->parent();//trace backward
			ScenarioID=root->particleIDs()[ScenarioID];
		}
		cerr<<__FUNCTION__<<": scenarioID="<<ScenarioID<<endl;

		//Copy particle from root
		particle=model->Copy(root->particles()[ScenarioID]);
		cerr<<__FUNCTION__<<": particle->scenario_id="<<particle->scenario_id<<endl;
		cerr<<__FUNCTION__<<"------------------"<<endl;

		int depth=0;
		double reward=0;
		OBS_TYPE obs=0;
		while(depth!=this->depth())//not leaf yet
		{
			int action=history.Action(depth);
			cout<<"depth="<<depth<<endl;
			cout<<"history_action="<<action<<endl;
			cout<<"history_obs="<<history.Observation(depth)<<endl;
			model->Step(*particle,streams.Entry(ScenarioID, depth), action,reward,obs);
			if(obs!=history.Observation(depth))//observation matching
			{
				//obsservation can mismatch because hst and dvc codes are using different rand number generators
				cerr<<__FUNCTION__<<": Wrong recalculated obs with history!"<<endl;
				cout<<"obs="<<obs<<endl;
			}
			depth++;
		}
		particles_[i]=particle;
		particleIDs_[i]=ScenarioID;
	}
}
void VNode::ReadBackCPUParticles(const DSPOMDP* model)
{
	//Resize its own particle list
	//particles_.resize(numParticles);
	//particleIDs_.resize(numParticles);
	//Reconstruct parent's particle list
	//std::vector<State*> ParentHstParticles;
	//ParentHstParticles.resize(numParentParticles);
	//for(int i=0;i<numParentParticles;i++)
	//{
	//	ParentHstParticles[i]=model->Allocate(0,0);//Zeros to be initial values
	//}
	//Get parent's particle contents from GPU
	for(int i=0;i<particles_.size();i++)
	{
		particles_[i]=model->Allocate(0,0);//Zeros to be initial values
	}
	/*int ThreadID = 0;//Debugging
	if (use_multi_thread_)
		ThreadID = MapThread(this_thread::get_id());	 */

	//cout<<__FUNCTION__<<endl;
	/*if(FIX_SCENARIO==1 && edge()==11220829450167354192)
	{CPUDoPrint=true;CPUPrintPID=372;GPUDoPrint=true; GPUPrintPID=372;}*/
	model->ReadBackToCPU(particles_,GetGPUparticles(), true);
	/*if(FIX_SCENARIO==1 && edge()==11220829450167354192)
	{CPUDoPrint=false;GPUDoPrint=false;}*/
	/*if(ThreadID==0)
		printf("Copying from address=%#0x \n", GetGPUparticles());//Debugging*/

	//Update and filter particles
	/*int pos=0;
	for(int i=0;i<numParentParticles;i++)
	{
		State* parent_particle=ParentHstParticles[i];
		OBS_TYPE obs;
		double reward;
		int parent_action = parent()->edge();
		model->Step(*parent_particle, parent_action,
				streams->Entry(parent_particle->scenario_id, streams->position_-1),
				obs);
		if(obs==edge_)//observation matching
		{
			particles_[pos++]=model->Copy(parent_particle);
			particleIDs_[pos]=parent_particle->scenario_id;
		}
		model->Free(parent_particle);
	}*/
	for(int i=0;i<particles_.size();i++)
	{
		particleIDs_[i]=particles()[i]->scenario_id;
	}
}
/* =============================================================================
 * Dvc_VNode class
 * =============================================================================*/
/*

DEVICE Dvc_VNode::Dvc_VNode(Dvc_State**& particles, int depth, Dvc_QNode* parent,
	OBS_TYPE edge) :
	particles_(particles),
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1) {
	//logd << "Constructed vnode with " << num_particles_ << " particles"
		//<< endl;
	for (int i = 0; i < num_particles_; i++) {
		logd << " " << i << " = " << *particles_[i] << endl;
	}
}

DEVICE Dvc_VNode::Dvc_VNode(Dvc_Belief* belief, int depth, Dvc_QNode* parent, OBS_TYPE edge) :
	belief_(belief),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1) {
}

DEVICE Dvc_VNode::Dvc_VNode(int count, double value, int depth, Dvc_QNode* parent, OBS_TYPE edge) :
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	count_(count),
	value_(value) {
}

DEVICE Dvc_VNode::~Dvc_VNode() {
	for (int a = 0; a < num_children_; a++) {
		Dvc_QNode* child = children_[a];
		assert(child != NULL);
		delete child;
	}

	delete [] children_;//.clear();

	if (belief_ != NULL)
		delete belief_;
}

DEVICE Dvc_Belief* Dvc_VNode::belief() const {
	return belief_;
}

DEVICE Dvc_State **const& Dvc_VNode::particles() const {
	return particles_;
}

DEVICE void Dvc_VNode::depth(int d) {
	depth_ = d;
}

DEVICE int Dvc_VNode::depth() const {
	return depth_;
}

DEVICE void Dvc_VNode::parent(Dvc_QNode* parent) {
	parent_ = parent;
}

DEVICE Dvc_QNode* Dvc_VNode::parent() {
	return parent_;
}

DEVICE OBS_TYPE Dvc_VNode::edge() {
	return edge_;
}

DEVICE double Dvc_VNode::Weight() const {
	return Dvc_State::Weight(particles_, num_particles_);
}

DEVICE Dvc_QNode**const & Dvc_VNode::children() const {
	return children_;
}

DEVICE Dvc_QNode**& Dvc_VNode::children() {
	return children_;
}

DEVICE const Dvc_QNode* Dvc_VNode::Child(int action) const {
	return children_[action];
}

DEVICE Dvc_QNode* Dvc_VNode::Child(int action) {
	return children_[action];
}

DEVICE int Dvc_VNode::Size() const {
	int size = 1;
	for (int a = 0; a < num_children_; a++) {
		size += children_[a]->Size();
	}
	return size;
}

DEVICE int Dvc_VNode::PolicyTreeSize() const {
	if (num_children_ == 0)
		return 0;

	Dvc_QNode* best = NULL;
	for (int a = 0; a < num_children_; a++) {
		Dvc_QNode* child = children_[a];
		if (best == NULL || child->lower_bound() > best->lower_bound())
			best = child;
	}
	return best->PolicyTreeSize();
}

DEVICE void Dvc_VNode::default_move(Dvc_ValuedAction move) {
	default_move_ = move;
}

DEVICE Dvc_ValuedAction Dvc_VNode::default_move() const {
	return default_move_;
}

DEVICE void Dvc_VNode::lower_bound(double value) {
	lower_bound_ = value;
}

DEVICE double Dvc_VNode::lower_bound() const {
	return lower_bound_;
}

DEVICE void Dvc_VNode::upper_bound(double value) {
	upper_bound_ = value;
}

DEVICE double Dvc_VNode::upper_bound() const {
	return upper_bound_;
}

DEVICE bool Dvc_VNode::IsLeaf() {
	return num_children_ == 0;
}

DEVICE void Dvc_VNode::Add(double val) {
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
}

DEVICE void Dvc_VNode::count(int c) {
	count_ = c;
}
DEVICE int Dvc_VNode::count() const {
	return count_;
}
DEVICE void Dvc_VNode::value(double v) {
	value_ = v;
}
DEVICE double Dvc_VNode::value() const {
	return value_;
}

DEVICE void Dvc_VNode::Free(const Dvc_DSPOMDP& model) {
	for (int i = 0; i < num_particles_; i++) {
		model.Free(particles_[i]);
	}

	for (int a = 0; a < num_children_; a++) {
		Dvc_QNode* qnode = Child(a);
		archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>& children = qnode->children();
		for (archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>::iterator it = children.begin();
			it != children.end(); it++) {
			it->second->Free(model);
		}
	}
}


DEVICE void Dvc_VNode::PrintPolicyTree(int depth, ostream& os) {
	if (depth != -1 && this->depth() > depth)
		return;

	Dvc_QNode**& qnodes = children();
	if (num_children_ == 0) {
		int astar = this->default_move().action;
		//os << this << "-a=" << astar << endl;
	} else {
		Dvc_QNode* qstar = NULL;
		for (int a = 0; a < num_children_; a++) {
			Dvc_QNode* qnode = qnodes[a];
			if (qstar == NULL || qnode->lower_bound() > qstar->lower_bound()) {
				qstar = qnode;
			}
		}

		//os << this << "-a=" << qstar->edge() << endl;

		archaeopteryx::util::vector<OBS_TYPE> labels;
		archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>& vnodes = qstar->children();
		for (archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
			labels.push_back(it->first);
		}

		for (int i = 0; i < labels.size(); i++) {
			if (depth == -1 || this->depth() + 1 <= depth) {
				//os << repeat("|   ", this->depth()) << "| o=" << labels[i]
				//	<< ": ";
				qstar->Child(labels[i])->PrintPolicyTree(depth, os);
			}
		}
	}
}



DEVICE void Dvc_VNode::PrintTree(int depth, ostream& os) {
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
		<< ", w:" << this->Weight() << ", weu:" << GPUDESPOT::WEU(this)
		<< ")"
		<< endl;


	Dvc_QNode**& qnodes = children();
	for (int a = 0; a < num_children_; a++) {
		Dvc_QNode* qnode = qnodes[a];

		archaeopteryx::util::vector<OBS_TYPE> labels;
		archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>& vnodes = qnode->children();
		for (archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
			labels.push_back(it->first);
		}

		os << repeat("|   ", this->depth()) << "a="
			<< qnode->edge() << ": "
			<< "(d:" << qnode->default_value << ", l:" << qnode->lower_bound()
			<< ", u:" << qnode->upper_bound()
			<< ", r:" << qnode->step_reward << ")" << endl;

		for (int i = 0; i < labels.size(); i++) {
			if (depth == -1 || this->depth() + 1 <= depth) {
				os << repeat("|   ", this->depth()) << "| o=" << labels[i]
					<< ": ";
				qnode->Child(labels[i])->PrintTree(depth, os);
			}
		}
	}
}

HOST void Dvc_VNode::assign(VNode* host_node)
{
	HANDLE_ERROR(cudaMemcpy(this, host_node, sizeof(Dvc_VNode),cudaMemcpyHostToDevice));
}
*/
/* =============================================================================
 * Dvc_QNode class
 * =============================================================================*/
/*

DEVICE Dvc_QNode::Dvc_QNode(Dvc_VNode* parent, int edge) :
	parent_(parent),
	edge_(edge),
	vstar(NULL) {
}

DEVICE Dvc_QNode::Dvc_QNode(int count, double value) :
	count_(count),
	value_(value) {
}

DEVICE Dvc_QNode::~Dvc_QNode() {
	for (archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>::iterator it = children_.begin();
		it != children_.end(); it++) {
		assert(it->second != NULL);
		delete it->second;
	}
	children_.clear();
}

DEVICE void Dvc_QNode::parent(Dvc_VNode* parent) {
	parent_ = parent;
}

DEVICE Dvc_VNode* Dvc_QNode::parent() {
	return parent_;
}
*/

DEVICE int Dvc_QNode::edge() {
	return edge_;
}

/*DEVICE archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>& Dvc_QNode::children()
{
	return children_;
}
DEVICE archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>& Dvc_QNode::children() {
	return children_;
}

DEVICE Dvc_VNode* Dvc_QNode::Child(OBS_TYPE obs) {
	return children_[obs];
}

DEVICE int Dvc_QNode::Size() const {
	int size = 0;
	for (archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		size += it->second->Size();
	}
	return size;
}

DEVICE int Dvc_QNode::PolicyTreeSize() const {
	int size = 0;

	for (archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		size += it->second->PolicyTreeSize();
	}
	return 1 + size;
}

DEVICE double Dvc_QNode::Weight() const {
	double weight = 0;
	for (archaeopteryx::util::map<OBS_TYPE, Dvc_VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		weight += it->second->Weight();
	}
	return weight;
}

DEVICE void Dvc_QNode::lower_bound(double value) {
	lower_bound_ = value;
}

DEVICE double Dvc_QNode::lower_bound() const {
	return lower_bound_;
}

DEVICE void Dvc_QNode::upper_bound(double value) {
	upper_bound_ = value;
}

DEVICE double Dvc_QNode::upper_bound() const {
	return upper_bound_;
}

DEVICE void Dvc_QNode::Add(double val) {
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
}

DEVICE void Dvc_QNode::count(int c) {
	count_ = c;
}

DEVICE int Dvc_QNode::count() const {
	return count_;
}

DEVICE void Dvc_QNode::value(double v) {
	value_ = v;
}

DEVICE double Dvc_QNode::value() const {
	return value_;
}*/
HOST void Dvc_QNode::assign(QNode* host_node)
{
	HANDLE_ERROR(cudaMemcpy(this, host_node, sizeof(Dvc_QNode),cudaMemcpyHostToDevice));
}

__global__ void CopyMembers(Dvc_QNode* Dvc, Dvc_QNode* src)
{
	int x=threadIdx.x;
	int y=threadIdx.y;
	if(x<1 && y<1)
	{
		Dvc->edge_=src->edge();
		Dvc->lower_bound_=src->lower_bound_;
		Dvc->upper_bound_=src->upper_bound_;

		// For POMCP
		Dvc->count_=src->count_; // Number of visits on the node
		Dvc->value_=src->value_; // Value of the node

		Dvc->default_value=src->default_value;
		Dvc->utility_upper_bound=src->utility_upper_bound;
		Dvc->step_reward=src->step_reward;
		Dvc->likelihood=src->likelihood;
	}

	__syncthreads();
}

__global__ void CheckMembers(Dvc_QNode* Dvc)
{
	int i=Dvc->edge_;

	i++;

	__syncthreads();
}
HOST void Dvc_QNode::CopyToGPU(Dvc_QNode* Dvc, const QNode* Hst)
{
	//HANDLE_ERROR(cudaMemcpy((void*)Dvc, (const void*)&(Hst->allocated_), sizeof(Dvc_UncNavigationState), cudaMemcpyHostToDevice));

	//Dvc_QNode* tmp;
	//HANDLE_ERROR(cudaMallocManaged((void**)&tmp,sizeof(Dvc_QNode)));

	/*tmp->edge_=Hst->edge_;
	tmp->lower_bound_=Hst->lower_bound_;
	tmp->upper_bound_=Hst->upper_bound_;

	// For POMCP
	tmp->count_=Hst->count_; // Number of visits on the node
	tmp->value_=Hst->value_; // Value of the node

	tmp->default_value=Hst->default_value;
	tmp->utility_upper_bound=Hst->utility_upper_bound;
	tmp->step_reward=Hst->step_reward;
	tmp->likelihood=Hst->likelihood;*/

	//tmp->children_;
	//tmp->parent_;
	//tmp->vstar;
	/*dim3 grid(1,1);dim3 threads(1,1);
	CopyMembers<<<grid, threads>>>(Dvc,tmp);
	cudaDeviceSynchronize();

	cudaFree(tmp);*/

	Dvc->edge_=Hst->edge();
	//Dvc->lower_bound_=Hst->lower_bound();
	//Dvc->upper_bound_=Hst->upper_bound();

	// For POMCP
	//Dvc->count_=Hst->count(); // Number of visits on the node
	//Dvc->value_=Hst->value(); // Value of the node

	//Dvc->default_value=Hst->default_value;
	//Dvc->utility_upper_bound=Hst->utility_upper_bound;
	//Dvc->step_reward=Hst->step_reward;
	//Dvc->likelihood=Hst->likelihood;

	//CheckMembers<<<1,1,1>>>(Dvc);
	//cudaDeviceSynchronize();
}
} // namespace despot
