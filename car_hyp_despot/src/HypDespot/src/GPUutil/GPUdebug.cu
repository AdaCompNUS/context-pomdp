/*! \file   debug.cpp
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for archaeopteryx debug functions.
*/

// Archaeopteryx Includes
#include <despot/GPUutil/GPUdebug.h>

// Standard Library Includes
#include <cstdio> 

namespace archaeopteryx
{

namespace util
{

__device__ void _assert(bool condition, const char* expression,
	const char* filename, int line)
{
	if(!condition)
	{
		printf("%s:%i - assertion '%s' failed!\n", filename, line, expression);
		asm("trap;");
	}
}

}

}

