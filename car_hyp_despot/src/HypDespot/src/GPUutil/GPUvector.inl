/*	\file   vector.inl
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for the vector class
*/

// Archaeopteryx Includes
#include <despot/GPUutil/GPUvector.h>

namespace archaeopteryx
{

namespace util
{

template <class T, class A>
__device__ vector<T, A>::vector()
: _begin(0), _end(0), _capacityEnd(0)
{

}

template <class T, class A>
__device__ vector<T, A>::vector(const allocator_type& a)
: _begin(0), _end(0), _capacityEnd(0), _allocator(a)
{

}

template <class T, class A>
__device__ vector<T, A>::vector(size_type n)
: _begin(0), _end(0), _capacityEnd(0)
{
	if(n > 0)
	{
		allocate(n);
		constructAtEnd(n);
	}
}

template <class T, class A>
__device__ vector<T, A>::vector(size_type n, const value_type& value,
	const allocator_type& a)
: _begin(0), _end(0), _capacityEnd(0), _allocator(a)
{
	if(n > 0)
	{
		allocate(n);
		constructAtEnd(n, value);
	}
}

template <class T, class A>
template <class InputIterator>
	__device__ vector<T, A>::vector(InputIterator first, InputIterator last,
		const allocator_type& a)
: _begin(0), _end(0), _capacityEnd(0), _allocator(a)
{
	for(; first != last; ++first)
	{
		push_back(*first);
	}
}

template <class T, class A>    
__device__ vector<T, A>::vector(const vector& x)
: _begin(0), _end(0), _capacityEnd(0), _allocator(x.allocator())
{
	size_t n = x.size();
	
	if(n > 0)
	{
		allocate(n);
		constructAtEnd(x.begin(), x.end());
	}
}

template <class T, class A>    
__device__ vector<T, A>::~vector()
{
	if(_begin != 0)
    {
        clear();
        _allocator->deallocate(_begin, capacity());
    }
}
    
template <class T, class A>    
__device__ vector<T, A>& vector<T, A>::operator=(const vector& x)
{
	assign(x.begin(), x.end());
	return *this;
}

template <class T, class A>    
template <class InputIterator>
__device__ void vector<T, A>::assign(InputIterator first, InputIterator last)
{
	clear();

	for(; first != last; ++first)
	{
		push_back(*first);
	}
}

template <class T, class A>
__device__ void vector<T, A>::assign(size_type n, const value_type& u)
{
	clear();

	for(; first != last; ++first)
	{
		push_back(*first);
	}
}
	
template <class T, class A>
__device__ vector<T, A>::iterator vector<T, A>::begin()
{
	return _begin;
}

template <class T, class A>
__device__ vector<T, A>::const_iterator vector<T, A>::begin() const
{
	return _begin;
}

template <class T, class A>
__device__ vector<T, A>::iterator vector<T, A>::end()
{
	return _end;
}

template <class T, class A>
__device__ vector<T, A>::const_iterator vector<T, A>::end() const
{
	return _end;
}

template <class T, class A>
__device__ vector<T, A>::reverse_iterator vector<T, A>::rbegin()
{
	return reverse_iterator(end());
}

template <class T, class A>
__device__ vector<T, A>::const_reverse_iterator vector<T, A>::rbegin() const
{
	return const_reverse_iterator(end());
}

template <class T, class A>
__device__ vector<T, A>::reverse_iterator vector<T, A>::rend()
{
	return reverse_iterator(begin()); 
}

template <class T, class A>
__device__ vector<T, A>::const_reverse_iterator vector<T, A>::rend() const
{
	return const_reverse_iterator(begin());
}

template <class T, class A>
__device__ vector<T, A>::size_type vector<T, A>::size() const
{
	return _end - _begin;
}

template <class T, class A>
__device__ vector<T, A>::size_type vector<T, A>::max_size() const
{
	return std::min(allocator()->max_size(),
		std::numeric_limits<size_type>::max() / 2);
}

template <class T, class A>
    __device__ vector<T, A>::size_type vector<T, A>::capacity() const
{
	return _capacityEnd - _begin;
}

template <class T, class A>
    __device__ bool vector<T, A>::empty() const
{
	return size() == 0;
}

template <class T, class A>
__device__ void vector<T, A>::reserve(size_type n)
{
	assertM(false, "Not implemented.");
}

template <class T, class A>
__device__ void vector<T, A>::shrink_to_fit()
{
	assertM(false, "Not implemented.");
}

template <class T, class A>
__device__ void vector<T, A>::resize(size_type sz)
{
	assertM(false, "Not implemented.");
}

template <class T, class A>
__device__ void vector<T, A>::resize(size_type sz, const value_type& c)
{
	assertM(false, "Not implemented.");
}

template <class T, class A>
__device__ vector<T, A>::reference vector<T, A>::operator[](size_type n)
{
	return _begin[n];
}

template <class T, class A>
__device__ vector<T, A>::const_reference vector<T, A>::operator[](size_type n) const
{
	return _begin[n];
}

template <class T, class A>
__device__ vector<T, A>::reference vector<T, A>::at(size_type n)
{
	return _begin[n];
}

template <class T, class A>
__device__ vector<T, A>::const_reference vector<T, A>::at(size_type n) const
{
	return _begin[n];
}

template <class T, class A>
__device__ vector<T, A>::reference vector<T, A>::front()
{
	return *_begin;
}

template <class T, class A>
__device__ vector<T, A>::const_reference vector<T, A>::front() const
{
	return *_begin;
}

template <class T, class A>
__device__ vector<T, A>::reference vector<T, A>::back()
{
	return *(_end - 1);
}

template <class T, class A>
__device__ vector<T, A>::const_reference vector<T, A>::back() const
{
	return *(_end - 1);
}

template <class T, class A>
__device__ vector<T, A>::value_type* vector<T, A>::data()
{
	return _begin;
}

template <class T, class A>
__device__ const vector<T, A>::value_type* vector<T, A>::data() const
{
	return _begin;
}

template <class T, class A>
__device__ void vector<T, A>::push_back(const value_type& x)
{
	if(_end == _capacityEnd)
	{
		//slow path
		 size_t oldSize     = size();
		 size_t newCapacity = recommend(oldSize + 1);
		pointer newBase     = _allocator.allocate(newCapacity);
		
		std::copy(begin(), end(), newBase);
		
		destroy();
		
		_begin       = newBase;
		_end         = newBase + oldSize;
		_capacityEnd = newBase + newCapacity;
	}
	// fast path	
	_allocator.construct(_end++, x);
}

template <class T, class A>
__device__ void vector<T, A>::pop_back()
{
	_allocator->destroy(_end--);
}

template <class T, class A>
__device__ vector<T, A>::iterator vector<T, A>::insert(const_iterator position, const value_type& x)
{
	assertM(false, "Not implemented.");
	return position;
}

template <class T, class A>
__device__ vector<T, A>::iterator vector<T, A>::insert(const_iterator position, size_type n, const value_type& x)
{
	assertM(false, "Not implemented.");
	return position;
}

template <class T, class A>
template <class InputIterator>
__device__ vector<T, A>::iterator vector<T, A>::insert(const_iterator position, InputIterator first,
	InputIterator last)
{
	assertM(false, "Not implemented.");
	return position;
}

template <class T, class A>
__device__ vector<T, A>::iterator vector<T, A>::erase(const_iterator position)
{
	assertM(false, "Not implemented.");
	return position;
}

template <class T, class A>
__device__ vector<T, A>::iterator vector<T, A>::erase(const_iterator first, const_iterator last)
{
	assertM(false, "Not implemented.");
	return position;
}

template <class T, class A>
__device__ void vector<T, A>::clear()
{
	while(_end != _begin) { _allocator.destroy(_end--); }
	allocator.deallocate(_begin, capacity());

	_begin       = 0;
	_end         = 0;
    _capacityEnd = 0;
}

template <class T, class A>
__device__ void vector<T, A>::swap(vector& v)
{
	std::swap(_begin,       v._begin      );
	std::swap(_end,         v._end        );
	std::swap(_capacityEnd, v._capacityEnd);
}

template <class T, class A>
__device__ vector<T, A>::allocator_type vector<T, A>::get_allocator() const
{
	return _allocator;
}

}

}


