/*	\file   Knob.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 7, 2012
	\brief  The source file for the Knob class
*/

// Archaeopteryx Includes
#include <despot/GPUutil/GPUKnob.h>

#include <despot/GPUutil/GPUmap.h>
#include <despot/GPUutil/GPUdebug.h>

namespace archaeopteryx
{

namespace util
{

__device__ Knob::Knob(const util::string& name, const util::string& value)
: _name(name), _value(value)
{

}

__device__ const util::string& Knob::name() const
{
	return _name;
}

__device__ const util::string& Knob::value() const
{
	return _value;
}

typedef util::map<util::string, Knob*> KnobMap;

static __device__ KnobMap* knobDatabaseImplementation = 0;

__device__ void KnobDatabase::addKnob(Knob* base)
{
	KnobMap::iterator knob = knobDatabaseImplementation->find(base->name());
	
	if(knob == knobDatabaseImplementation->end())
	{
		knobDatabaseImplementation->insert(util::make_pair(base->name(), base));
	}
	else
	{
		delete knob->second;
		knob->second = base;
	}
}

__device__ void KnobDatabase::removeKnob(const Knob& base)
{
	KnobMap::iterator knob = knobDatabaseImplementation->find(base.name());

	if(knob == knobDatabaseImplementation->end())
	{
		delete knob->second;
		knobDatabaseImplementation->erase(knob);
	}
}

__device__ const Knob& KnobDatabase::getKnobBase(const util::string& name)
{
	KnobMap::iterator knob = knobDatabaseImplementation->find(name);

	if(knob == knobDatabaseImplementation->end())
	{
		std::printf("ERROR: No knob '%s' declared.\n", name.c_str());
	}

	device_assert(knob != knobDatabaseImplementation->end());
	
	return *knob->second;
}

__device__ void KnobDatabase::create()
{
	knobDatabaseImplementation = new KnobMap;
}

__device__ void KnobDatabase::destroy()
{
	delete knobDatabaseImplementation;
}

}

}


