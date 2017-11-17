/*! \file   cstdlib.cu
	\date   Sunday December 9, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for device implementation of
			C standard library functions.
*/

#include <despot/GPUutil/GPUcstdlib.h>

namespace archaeopteryx
{

namespace util
{

__device__ int atoi(const char* s)
{
	int value = 0;
	int base = 10;

	if(s[0] != '\0' && s[1] != '\0' && s[0] == '0' && s[1] == 'x')
	{
		base = 16;
		
		s += 2;
	}

	while(*s != '\0')
	{
		if(*s <= '9')
		{
			value = value * base;
		
			value += *s - '0';
		}
		else if(*s <= 'F')
		{
			value = value * base;
		
			value += *s - 'A' + 10;
		}
		else if(*s <= 'f')
		{
			value = value * base;
		
			value += *s - 'a' + 10;
		}
		
		++s;
	}

	return value;
}

}

}


