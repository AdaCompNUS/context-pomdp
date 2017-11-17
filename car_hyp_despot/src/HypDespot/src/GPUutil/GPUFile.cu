/*! \file   File.cpp
	\date   Sunday June 26, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the File class.
*/

// Archaeoperyx Includes
#include <despot/GPUutil/GPUFile.h>
#include <despot/GPUutil/GPUcstring.h>
#include <despot/GPUutil/GPUdebug.h>

#include <despot/GPUutil/GPUalgorithm.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace archaeopteryx
{

namespace util
{

__device__ File::File(const char* fileName, const char* mode)
{
	device_report("Opening file '%s' with mode '%s' on the gpu\n",
		fileName, mode);

	OpenMessage open(fileName, mode);
	
	HostReflectionDevice::sendSynchronous(open);
	
	OpenReply reply;
	
	HostReflectionDevice::receive(reply);
	
	_handle = reply.handle();
	_size   = reply.size();
	_put    = 0;
	_get    = 0;

	if(_handle == 0)
	{
		device_report(" failed to open file...\n");
	
		device_assert(_handle != 0);
	}
	
	device_report(" file opened, current size is %d\n", _size);
}

__device__ File::~File()
{
	if(_handle != (Handle)-1)
	{
		TeardownMessage teardown(_handle);
	
		HostReflectionDevice::sendSynchronous(teardown);
	}
}

__device__ void File::write(const void* data, size_t bytes)
{
	const char* pointer = reinterpret_cast<const char*>(data);

	while(bytes > 0)
	{
		size_t written = writeSome(pointer, bytes);
		
		pointer += written;
		bytes   -= written;
	}
}

__device__ size_t File::writeSome(const void* data, size_t bytes)
{	
	size_t attemptedSize =
		util::min(bytes, util::max((size_t)1,
		(size_t)(HostReflectionDevice::maxMessageSize() / 2)));
	
	WriteMessage message(data, attemptedSize, _put, _handle);
	
	device_report("sending file write message (%d size, %d pointer, %p handle)\n",
		attemptedSize, _put, _handle);
	
	HostReflectionDevice::sendSynchronous(message);
	
	_put += attemptedSize;
	
	if(_put > _size)
	{
		_size = _put;
	}
	
	return attemptedSize;
}

__device__ void File::read(void* data, size_t bytes)
{
	if(_get + bytes > size())
	{
		bytes = size() - _get;
	}

	char* pointer = reinterpret_cast<char*>(data);

	device_report("performing file read (%d size, %d pointer)\n",
		(int)bytes, (int)_get);

	while(bytes > 0)
	{
		size_t bytesRead = readSome(pointer, bytes);
	
		pointer += bytesRead;
		bytes   -= bytesRead;
	}
}

__device__ size_t File::readSome(void* data, size_t bytes)
{
	if(_get + bytes > size())
	{
		bytes = size() - _get;
	}

	size_t attemptedSize =
		util::min(bytes, util::max((size_t)1,
			(size_t)(HostReflectionDevice::maxMessageSize() / 2)));

	device_report(" sending file read message (%d size, %d pointer, %p handle)\n",
		(int)attemptedSize, (int)_get, _handle);
	
	ReadMessage message(attemptedSize, _get, _handle);
	
	HostReflectionDevice::sendSynchronous(message);
	
	ReadReply reply(attemptedSize);
	
	HostReflectionDevice::receive(reply);
	
	util::memcpy(data, reply.payload(), attemptedSize);
	
	_get += attemptedSize;
	
	return attemptedSize;
}

__device__ size_t File::size() const
{
	return _size;
}
	
__device__ size_t File::tellg() const
{
	return _get;
}
	
__device__ size_t File::tellp() const
{
	return _put;
}

__device__ void File::seekg(size_t g)
{
	if(g > size())
	{
		g = size();
	}

	device_report("seeking get pointer from %d to %d\n", (int)_get, (int)g);
	
	_get = g;
}

__device__ void File::seekp(size_t p)
{
	if(p > size())
	{
		p = size();
	}
	
	_put = p;
}

__device__ File::OpenMessage::OpenMessage(const char* f, const char* m)
{
	util::memset(_filename, 0, payloadSize());
	strlcpy(_filename, f, payloadSize());
	size_t offset = util::strlen(f) + 1;
	strlcpy(_filename + offset, m, payloadSize() - offset);
}

__device__ File::OpenMessage::~OpenMessage()
{

}

__device__ void* File::OpenMessage::payload() const
{
	return (void*)_filename;
}

__device__ size_t File::OpenMessage::payloadSize() const
{
	return sizeof(_filename);
}

__device__ HostReflectionDevice::HandlerId File::OpenMessage::handler() const
{
	return HostReflectionDevice::OpenFileMessageHandler;
}

__device__ File::OpenReply::OpenReply()
{

}

__device__ File::OpenReply::~OpenReply()
{

}

__device__ File::Handle File::OpenReply::handle() const
{
	return _data.handle;
}

__device__ size_t File::OpenReply::size() const
{
	return _data.size;
}

__device__ void* File::OpenReply::payload() const
{
	return (void*)&_data;
}

__device__ size_t File::OpenReply::payloadSize() const
{
	return sizeof(Payload);
}

__device__ HostReflectionDevice::HandlerId File::OpenReply::handler() const
{
	return (HostReflectionDevice::HandlerId)
		HostReflectionDevice::InvalidMessageHandler;
}

__device__ File::TeardownMessage::TeardownMessage(Handle h)
: _handle(h)
{

}

__device__ File::TeardownMessage::~TeardownMessage()
{
	
}

__device__ void* File::TeardownMessage::payload() const
{
	return (void*)&_handle;
}

__device__ size_t File::TeardownMessage::payloadSize() const
{
	return sizeof(Handle);
}

__device__ HostReflectionDevice::HandlerId File::TeardownMessage::handler() const
{
	return HostReflectionDevice::TeardownFileMessageHandler;
}

__device__ File::WriteMessage::WriteMessage(const void* data, size_t size,
	size_t pointer, Handle handle)
{
	size_t bytes = size + sizeof(Header);

	_payload = new char[bytes];
	
	Header header;
	
	header.size    = bytes;
	header.pointer = pointer;
	header.handle  = handle;

	util::memcpy(_payload, &header, sizeof(Header));
	util::memcpy((char*)_payload + sizeof(Header), data, size);
}

__device__ File::WriteMessage::~WriteMessage()
{
	delete[] _payload;
}

__device__ void* File::WriteMessage::payload() const
{
	return _payload;
}

__device__ size_t File::WriteMessage::payloadSize() const
{
	Header* header = (Header*)_payload;

	return header->size;
}

__device__ HostReflectionDevice::HandlerId File::WriteMessage::handler() const
{
	return HostReflectionDevice::FileWriteMessageHandler;
}

__device__ File::ReadMessage::ReadMessage(
	size_t size, size_t pointer, Handle handle)
{
	_payload.size    = size;
	_payload.pointer = pointer;
	_payload.handle  = handle;
}
	
__device__ File::ReadMessage::~ReadMessage()
{

}

__device__ void* File::ReadMessage::payload() const
{
	return (void*)&_payload;
}

__device__ size_t File::ReadMessage::payloadSize() const
{
	return sizeof(Payload);
}

__device__ HostReflectionDevice::HandlerId File::ReadMessage::handler() const
{
	return HostReflectionDevice::FileReadMessageHandler;
}

__device__ File::ReadReply::ReadReply(size_t size)
: _size(size), _data(new char[size])
{

}

__device__ File::ReadReply::~ReadReply()
{
	delete[] _data;
}

__device__ void* File::ReadReply::payload() const
{
	return _data;
}

__device__ size_t File::ReadReply::payloadSize() const
{
	return _size;
}

__device__ HostReflectionDevice::HandlerId File::ReadReply::handler() const
{
	return HostReflectionDevice::FileReadReplyHandler;
}

}

}

