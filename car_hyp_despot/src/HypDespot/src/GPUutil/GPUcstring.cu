/*! \file   string.cpp
	\date   Tuesday June 28, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for device string functions.
*/

// Archaeopteryx Includes
#include <despot/GPUutil/GPUcstring.h>

namespace archaeopteryx
{

namespace util
{

__device__ void strlcpy(char* destination, const char* source, size_t max)
{
	max = max == 0 ? 1 : max;
	const char* end = source + (max - 1);
	for( ; source != end; ++source, ++destination)
	{
		*destination = *source;
		if( *source == '\0' ) return;
	}
	*destination = '\0';
}

__device__ int strcmp(const char* left, const char* right)
{
	while(*left != '\0' && *right != '\0')
	{
		if(*left != *right) return -1;
		
		++left; ++right;
	}

	if(*left != *right) return -1;

	return 0;
}

__device__ int memcmp(const void* s1, const void* s2, size_t n)
{
	if(n)
	{
		const unsigned char* p1 = (const unsigned char*)s1;
		const unsigned char* p2 = (const unsigned char*)s2;

		do
		{
			if(*p1++ != *p2++)
			{
				return (*--p1 - *--p2);
			}
		}
		while (--n);
	}
	
	return 0;
}

__device__ void* memcpy(void* s1, const void* s2, size_t n)
{
	      unsigned char* p1 =       (unsigned char*)s1;
	const unsigned char* p2 = (const unsigned char*)s2;

	while(n--)
	{
		*p1 = *p2;
		++p1;
		++p2;
	}
	
	return p1;
}

__device__ size_t strlen(const char* s)
{
	size_t size = 0;
	
	while(*s++ != '\0') ++size;

	return size;
}

__device__ void* memset(void* s, int c, size_t n)
{
	unsigned char* p1 = (unsigned char*)s;
	
	for(unsigned int i = 0; i < n; ++i, ++p1)
	{
		*p1 = c;
	}
	
	return p1;
}

}

}

