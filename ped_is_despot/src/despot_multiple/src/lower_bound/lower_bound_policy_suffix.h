#ifndef LOWER_BOUND_POLICY_SUFFIX_H
#define LOWER_BOUND_POLICY_SUFFIX_H

#include "lower_bound/lower_bound_policy.h"
#include "lower_bound/lower_bound_policy_mode.h"
#include <iostream>
//#include "SSTree.h"
//#include "Tools.h"
#include "string.h"
#include "history.h"
#include <queue>
#include <vector>
#include <stdlib.h>
#include "globals.h"

using namespace std;

template<typename T>
class Solver;

template<typename T>
class SuffixPolicyLowerBound : public PolicyLowerBound<T> {
private:
	mutable unsigned action_root_seed_;
	PolicyLowerBound<T>* default_policy;
	mutable int defaultCount;
	mutable int suffixCount;
public:
	SuffixPolicyLowerBound(const RandomStreams& streams, int num_states) : PolicyLowerBound<T>(streams) {
		default_policy = new ModePolicyLowerBound<T>(streams, num_states);
		defaultCount = 0;
		suffixCount = 0;
	}

public:
  int Action(const vector<Particle<T>*>& particles,
             const Model<T>& model,
             const History& history) const {

		vector<int> actions = SearchForLongestSuffixMatchActions(history);
		if(actions.size() == 0) {
			defaultCount ++;
			return default_policy->Action(particles, model, history);
			//return rand_r(&action_root_seed_) % model.NumActions();
		}

		suffixCount ++;
		return actions[rand_r(&action_root_seed_) % actions.size()];
	}

	void CollectSearchInformation(Solver<T>* solver) {
		cout << "Clearing previously stored traces..."; cout.flush();
		Clear();
		cout << "Done." << endl; cout.flush();

		cout << "DefaultCount = " << defaultCount << endl;
		cout << "SuffixCount = " << suffixCount << endl;

		cout << "Retrieving history-action mapping..."; cout.flush();
		solver->RetrieveHistoryActionMapping(static_cast<SuffixPolicyLowerBound<T>*>(this));
		cout << Count() << " pairs are retrieved." << endl; cout.flush();

		cout << "Constructing suffix tree...";
		ConstructMapping();
		cout << "Done." << endl; cout.flush();
	}

  static string Name() { return "suffix"; }

	static SuffixPolicyLowerBound<T>* Instance(string name, const RandomStreams& streams, int num_states, unsigned action_root_seed);

	virtual int Count() = 0;
	virtual void Clear() = 0;
	virtual void Put(const History& history, int action) = 0;
	virtual void ConstructMapping() = 0;
	virtual vector<int> SearchForLongestSuffixMatchActions(const History& history) const = 0;
};

/*
template<typename T>
class SuffixPolicyLowerBoundSuffixTreeImpl : public SuffixPolicyLowerBound<T> {
private:
	SSTree* policy;
	vector<uchar*> runs;
	mutable unsigned action_root_seed_;

	const static int CODE_LENGTH = 16;
	const static char OBS_PREFIX = 'O';
	const static char ACTION_PREFIX = 'A';
	const static int BASE = 26;
	const static char FIRST = 'a';

	static void ObsToString(uint64_t obs, uchar* buf, int pos) {
		buf[pos] = OBS_PREFIX;
		for(int i=1; i<CODE_LENGTH; i++) {
			buf[pos+i] = FIRST + obs%BASE;
			obs /= BASE;
		}
	}
	static void ActionToString(int action, uchar* buf, int pos) {
		buf[pos] = ACTION_PREFIX;
		for(int i=1; i<CODE_LENGTH; i++) {
			buf[pos+i] = FIRST + action%BASE;
			action /= BASE;
		}
	}
	static int StringToAction(const uchar* buf) {
		int action = 0,
				length = strlen((char*) buf),
				pow = 1;
		for(int i=0; i<length; i++) {
			if(buf[i] == '$')
				break;

			action += (buf[i] - FIRST) * pow;
			pow *= BASE;
		}
		return action;
	}

	static uchar* toString(const History& history, int action) {
		return toString(history, action, 0);
	}
	static uchar* toString(const History& history, int action, int start) {
		uchar* buf = new uchar[CODE_LENGTH *
			(2 * (history.Size() - start)
			 + (action == -1 ? 0 : 1))
			+ (action == -1 ? 2 : 1)];
		int pos = 0;
		for(int i=start; i<history.Size(); i++) {
			ActionToString(history.Action(i), buf, pos);
			pos += CODE_LENGTH;
			ObsToString(history.Observation(i), buf, pos);
			pos += CODE_LENGTH;
		}
		if(action != -1) {
			ActionToString(action, buf, pos);
			buf[pos] = 'R';
			pos += CODE_LENGTH;
		} else {
			buf[pos] = 'R';
			pos ++;
		}
		buf[pos] = '\0';
		return buf;
	}

	static void Copy(uchar* from, uchar* to, int& pos) {
		int length = strlen((char*) from);
		for(int i=0; i<length; i++)
			to[pos++] = from[i];
	}

	static bool Matches(const uchar* text, const uchar* pattern, int& pos) {
		int length = strlen((const char*)pattern);

		if(pos + length > (int)strlen((const char*) text))
			return false;

		for(int i=0; i<length; i++)
			if(text[pos++] != pattern[i])
				return false;
		return true;
	}

	static ulong Search(SSTree* sst, uchar* str) {
		ulong node = sst->root();
		int pos = 0, length = strlen((char*) str);
		while(true) {
			node = sst->child(node, str[pos]);

			if(node == 0) // no match
				return 0;

			uchar* label = sst->edge(node);

			if(!Matches(str, label, pos))
				break;
			if(pos == length)
				return node;
		}
		return 0;
	}

public:
	SuffixPolicyLowerBoundSuffixTreeImpl(const RandomStreams& streams, int num_states, unsigned action_root_seed) 
		: SuffixPolicyLowerBound<T>(streams, num_states),
		action_root_seed_(action_root_seed) {
			policy = NULL;
	}

  static string Name() { return "suffix"; }

public:
	void Clear() {
		runs.clear();
	}

	int Count() { return runs.size(); }

	void Put(const History& history, int action) {
		runs.push_back(toString(history, action));
	}

	void ConstructMapping() {
		if(runs.size() == 0) {
			policy = NULL;
			return;
		}

		int length = 0;
		for(unsigned int i=0; i<runs.size(); i++)
			length += strlen((char*)runs[i]) + 1;
		uchar* runsStr = new uchar[length];
		int pos = 0;
		for(unsigned int i=0; i<runs.size(); i++) {
			Copy(runs[i], runsStr, pos);
			runsStr[pos] = '$';
			pos++;
		}
		runsStr[pos-1] = '\0';

		// cout << "Length: " << length << " " << pos << endl;
		// cout << "RunsString: " << runsStr << endl;

		policy = new SSTree(runsStr, pos);

		// policy->PrintTree(policy->root(), 0);
	}

	vector<int> SearchForLongestSuffixMatchActions(const History& history) const {
		vector<int> actions;

		if(policy == NULL) return actions;

		for(int i=0; i<history.Size(); i++) {
			uchar* str = toString(history, -1, i);
			// cout << "Query: " << str << endl; cout.flush();
			ulong match = Search(static_cast<SSTree*>(policy), str);
			// cout << "Match: " << match << " " << policy->pathlabel(match) << endl; cout.flush();

			if(match != 0) {
				ulong child = policy->firstChild(match);

				while(child != 0) {
					actions.push_back(StringToAction(policy->edge(child)));

					child = policy->sibling(child);
				}

				break;
			}
		}

		return actions;
	}
};
*/

template<typename T>
class SuffixPolicyLowerBoundMapImpl : public SuffixPolicyLowerBound<T> {
private:
	mutable unsigned action_root_seed_;
	map<History, vector<int>> suffix_action_map;
	int count;

public:
	SuffixPolicyLowerBoundMapImpl(const RandomStreams& streams, int num_states, unsigned action_root_seed) 
		: SuffixPolicyLowerBound<T>(streams, num_states), 
		action_root_seed_(action_root_seed), 
		count(0) {
	}

	static string Name() { return "suffix"; }

public:
	void Clear()  {
		suffix_action_map.clear();
		count = 0;
	}

	int Count() { return count; }

	void Put(const History& history, int action) {
		count ++;
		for(int i=0; i<history.Size()-1; i++)
			suffix_action_map[history.Suffix(i)].push_back(action);
	}

	void ConstructMapping() {
	}

	vector<int> SearchForLongestSuffixMatchActions(const History& history) const {
		if(suffix_action_map.size() == 0)
			return vector<int>();

		for(int i=0; i<history.Size()-10; i++) {
			map<History, vector<int>>::const_iterator iter = suffix_action_map.find(history.Suffix(i));
			if(iter != suffix_action_map.end())
				return iter->second;
		}

		return vector<int>();
	}
};

template<typename T>
SuffixPolicyLowerBound<T>* SuffixPolicyLowerBound<T>::Instance(string name, const RandomStreams& streams, int num_states, unsigned action_root_seed) {
	//if(name == "suffix_tree") return new SuffixPolicyLowerBoundMapImpl<T>(streams, num_states, action_root_seed);

	return new SuffixPolicyLowerBoundMapImpl<T>(streams, num_states, action_root_seed);
}

#endif
