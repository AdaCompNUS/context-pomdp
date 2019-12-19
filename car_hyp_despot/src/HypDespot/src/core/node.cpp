#include <despot/core/node.h>
#include <despot/solver/despot.h>

using namespace std;

namespace despot {

/* =============================================================================
 * VNode class
 * =============================================================================*/


VNode::VNode(vector<State*>& particles,std::vector<int> particleIDs, int depth, QNode* parent,
	OBS_TYPE edge) :
	particles_(particles),
	particleIDs_(particleIDs),
	GPU_particles_(NULL),
	num_GPU_particles_(0),
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
	prior_value_(DUMMY_VALUE){
	logv << "Constructed vnode with " << particles_.size() << " particles"
		<< endl;
	for (int i = 0; i < particles_.size(); i++) {
		logv << " " << i << " = " <<"("<< particleIDs_[i]<<")"<< *particles_[i] << endl;
	}
	weight_=0;
	prior_initialized_ = false;
}

void VNode::Initialize(vector<State*>& particles,std::vector<int> particleIDs, int depth, QNode* parent,
	OBS_TYPE edge){
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
	weight_=0;
	prior_initialized_ = false;
}

VNode::VNode(Belief* belief, int depth, QNode* parent, OBS_TYPE edge) :
	GPU_particles_(NULL),
	num_GPU_particles_(0),
	belief_(belief),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
	prior_value_(DUMMY_VALUE) {
	weight_=0;
	prior_initialized_ = false;

}

VNode::VNode(int count, double value, int depth, QNode* parent, OBS_TYPE edge) :
	GPU_particles_(NULL),
	num_GPU_particles_(0),
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	count_(count),
	value_(value),
	prior_value_(DUMMY_VALUE) {
	weight_=0;
	prior_initialized_ = false;
}


VNode::~VNode() {
	// for (int a = 0; a < children_.size(); a++) {
	for (int a : legal_actions_) {
		if (a < children_.size()) {
			QNode* child = children_[a];
			assert(child != NULL);
			if(child->IsAllocated())//allocated by memory pool
				DESPOT::qnode_pool_.Free(child);
			else
				delete child;
		}
	}
	children_.clear();

	if (belief_ != NULL)
		delete belief_;
}

Belief* VNode::belief() const {
	return belief_;
}

const std::vector<State*>& VNode::particles() const {
	return particles_;
}

const std::vector<int>& VNode::particleIDs() const {
	return particleIDs_;
}
void VNode::depth(int d) {
	depth_ = d;
}

int VNode::depth() const {
	return depth_;
}

void VNode::parent(QNode* parent) {
	parent_ = parent;
}

QNode* VNode::parent() {
	return parent_;
}

OBS_TYPE VNode::edge() {
	return edge_;
}

double VNode::Weight() {
	if(Globals::config.useGPU==false ||!PassGPUThreshold())
		if(GPUWeight()>0)
			return GPUWeight();
		else
			return State::Weight(particles_);
	else/* if(num_GPU_particles_>0)*/
		return GPUWeight();
}
bool VNode::PassGPUThreshold(){
	return (particleIDs().size()>Globals::config.expanstion_switch_thresh || depth()<1);
}
const vector<QNode*>& VNode::children() const {
	return children_;
}

vector<QNode*>& VNode::children() {
	return children_;
}

const QNode* VNode::Child(int action) const {
	return children_[action];
}

QNode* VNode::Child(int action) {
	return children_[action];
}

int VNode::Size() const {
	int size = 1;
//	for (int a = 0; a < children_.size(); a++) {
	for (int a : legal_actions_) {
		if (a < children_.size()) {
//			logv << "[VNode::Size] vnode " << this << " legal action " << a <<
//					" with " << children_.size() << " children"<< endl;
//			logv << "[VNode::Size] qnode " << children_[a] << endl;
			size += children_[a]->Size();
		}
	}
	return size;
}

int VNode::PolicyTreeSize() const {
	if (children_.size() == 0)
		return 0;

	QNode* best = NULL;
//	for (int a = 0; a < children_.size(); a++) {
	for (int a: legal_actions_) {
		if (a < children_.size()) {
			QNode* child = children_[a];
			if (best == NULL || child->lower_bound() > best->lower_bound())
				best = child;
		}
	}

	return best->PolicyTreeSize();
}

void VNode::default_move(ValuedAction move) {
	default_move_ = move;
}

ValuedAction VNode::default_move() const {
	return default_move_;
}

void VNode::lower_bound(double value) {
	lower_bound_ = value;
}

double VNode::lower_bound() const {
	return lower_bound_;
}

void VNode::upper_bound(double value) {
	upper_bound_ = value;
}

double VNode::upper_bound() const {
	return upper_bound_;
}
void VNode::utility_upper_bound(double value){
	utility_upper_bound_=value;
}
double VNode::utility_upper_bound() const {
	return utility_upper_bound_;
}

double VNode::prior_value() {
	if(prior_value_ != DUMMY_VALUE){
		return prior_value_;
	} else {
		std::cerr << "prior_value_ uninitialized" << std::endl;
		std::cerr << "node info: depth " << depth() << " parent action "
				<< parent()->edge();

		int sib_id = 0;
		for (std::map<OBS_TYPE, VNode*>::iterator it =
				parent()->children().begin(); it != parent()->children().end();
				it++) {
			OBS_TYPE obs = it->first;
			if (this == it->second)
				break;
			sib_id++;
		}
		std::cerr << " sib_id " << sib_id << std::endl;
		exit(1);
	}
}

ACT_TYPE VNode::max_prob_action(){
	ACT_TYPE astar = -1;
	double prob = 0;
//	for (int a = 0; a < prior_action_probs_.size(); a++) {
	for (map<ACT_TYPE, double>::iterator it = prior_action_probs_.begin();
		        it != prior_action_probs_.end(); it++) {
		ACT_TYPE a = it->first;
		if (it->second > prob){
			prob = it->second;
			astar = a;
		}
	}

	return astar;
}

void VNode::print_action_probs(){
	logi << "vnode " << this << " of level " << depth() << " action_probs with " <<
			prior_action_probs_.size() << " elements:" << endl;
	for (map<ACT_TYPE, double>::iterator it = prior_action_probs_.begin();
			it != prior_action_probs_.end(); it++) {
		ACT_TYPE a = it->first;
		logi << it->second << " ";
	}
	logi << endl;
}

bool VNode::IsLeaf() {
	return children_.size() == 0;
}

void VNode::Add(double val) {
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
}

void VNode::count(int c) {
	count_ = c;
}
int VNode::count() const {
	return count_;
}
void VNode::value(double v) {
	value_ = v;
}
double VNode::value() const {
	return value_;
}

void VNode::Free(const DSPOMDP& model) {
	for (int i = 0; i < particles_.size(); i++) {
		if(particles_[i])model.Free(particles_[i]);
	}

//	cout << "free 1" << endl;

	for (int a = 0; a < children().size(); a++) {
		QNode* qnode = Child(a);

		if(qnode){
			map<OBS_TYPE, VNode*>& children = qnode->children();
			for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
				it != children.end(); it++) {

				if (it->second)
					it->second->Free(model);
				else{
					;//cout <<"free: NULL vnode" << endl;
				}
			}
		}
		else{
			;//cout <<"free: NULL qnode" << endl;
		}
	}
}

void VNode::PrintPolicyTree(int depth, ostream& os) {
	if (depth != -1 && this->depth() > depth)
		return;

	vector<QNode*>& qnodes = children();
	if (qnodes.size() == 0) {
		int astar = this->default_move().action;
		os << this << "-a=" << astar << endl;
	} else {
		QNode* qstar = NULL;
		for (int a = 0; a < qnodes.size(); a++) {
			QNode* qnode = qnodes[a];
			if (qstar == NULL || qnode->lower_bound() > qstar->lower_bound()) {
				qstar = qnode;
			}
		}

		os << this << "-a=" << qstar->edge() << endl;

		vector<OBS_TYPE> labels;
		map<OBS_TYPE, VNode*>& vnodes = qstar->children();
		for (map<OBS_TYPE, VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
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

void VNode::PrintTree(int depth, ostream& os) {
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
		<< ", w:" << this->Weight() << ", weu:" << DESPOT::WEU(this)
		<< ")"
		<< endl;


	vector<QNode*>& qnodes = children();
	for (int a = 0; a < qnodes.size(); a++) {
		QNode* qnode = qnodes[a];

		vector<OBS_TYPE> labels;
		map<OBS_TYPE, VNode*>& vnodes = qnode->children();
		for (map<OBS_TYPE, VNode*>::iterator it = vnodes.begin();
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

/* =============================================================================
 * QNode class
 * =============================================================================*/
QNode::QNode() :
	parent_(NULL),
	edge_(-1),
	vstar(NULL),
	prior_probability_(-1){
	weight_=0;
	prior_values_ready_ = false;
	//lower_bound_=0;upper_bound_=0;
}

QNode::QNode(VNode* parent, int edge) :
	parent_(parent),
	edge_(edge),
	vstar(NULL),
	prior_probability_(-1) {
	weight_=0;
	prior_values_ready_ = false;
}

QNode::QNode(int count, double value) :
	count_(count),
	value_(value),
	prior_probability_(-1) {
	weight_=0;
	prior_values_ready_ = false;
}

QNode::~QNode() {
	for (map<OBS_TYPE, VNode*>::iterator it = children_.begin();
		it != children_.end(); it++) {
		if(it->second){
			if(it->second->IsAllocated())//allocated by memory pool
				DESPOT::vnode_pool_.Free(it->second);
			else
				delete it->second;
		}
	}
	children_.clear();
}

void QNode::parent(VNode* parent) {
	parent_ = parent;
}

VNode* QNode::parent() {
	return parent_;
}

int QNode::edge() const{
	return edge_;
}

map<OBS_TYPE, VNode*>& QNode::children() {
	return children_;
}

VNode* QNode::Child(OBS_TYPE obs) {
	return children_[obs];
}

int QNode::Size() const {
	int size = 0;
	for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		if(it->second)
			size += it->second->Size();
	}
	return size;
}

int QNode::PolicyTreeSize() const {
	int size = 0;
	for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		if (it->second)
			size += it->second->PolicyTreeSize();
	}
	return 1 + size;
}

double QNode::Weight() /*const*/ {
	if(false/*weight_>1e-5*/){
		//assert((parent()->depth()==0 && weight_>1-1e-5) ||
		//		parent()->depth()!=0 && weight_<1+1e-5);

		/*if(parent()->depth()==0 && weight_<=1-1e-5){
			cout<<"Wrong stored weight as depth 1 " << this << " "<<weight_<<endl;
			//exit(-1);
		}*/

		return weight_;
	}
	else
	{
		try{
			weight_=0;
			for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
				it != children_.end(); it++) {
				if (it->second)
					weight_ += it->second->Weight();
				else
					;//cout << "[weight] NULL vnode" << endl;
			}
		} catch (std::exception e) {
			cout << "Error: " << e.what() << endl;
			raise(SIGABRT);
		}

		/*if(parent()->depth()==0 && weight_<=1-1e-5){
			cout<<"Wrong calcuated weight as depth 1 " << this << " "<<weight_<<endl;
			//exit(-1);
		}*/

//		cout << __FUNCTION__ << " return" << endl;
		return weight_;
	}
}

void QNode::lower_bound(double value) {
	lower_bound_ = value;
}

double QNode::lower_bound() const {
	return lower_bound_;
}

void QNode::upper_bound(double value) {
	upper_bound_ = value;
}

double QNode::upper_bound() const {
	return upper_bound_;
}
void QNode::utility_upper_bound(double value){
	utility_upper_bound_=value;
}
double QNode::utility_upper_bound() const {
	return utility_upper_bound_;
}

void QNode::Add(double val) {
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
}

void QNode::count(int c) {
	count_ = c;
}

int QNode::count() const {
	return count_;
}

void QNode::value(double v) {
	value_ = v;
}

double QNode::value() const {
	return value_;
}

//Let's drive

void QNode::prior_probability(double p) {
	prior_probability_ = p;
}

double QNode::prior_probability() const {
	return prior_probability_;
}
//

} // namespace despot
