/*
 * msgQueue.h
 *
 *  Created on: 19 Jul, 2017
 *      Author: panpan
 */

#ifndef MSGQUEUE_H_
#define MSGQUEUE_H_


#include "thread_globals.h"
using namespace std;
template< class T>
class MsgQueque{
	deque<T*> _queue;
	condition_variable _cond;
	mutex _mutex;
public:

	MsgQueque()
	{
		cout<<"new message queue:"<<this<<endl;
	}
	void send(T* msg)
	{
		{
			lock_guard<mutex> lck(_mutex);
			Global_print_mutex(this_thread::get_id(),this, __FUNCTION__, 0);
			_queue.push_front(msg);
			Global_print_mutex(this_thread::get_id(),this, __FUNCTION__, 1);
		}
		_cond.notify_one();
	}
	void WakeOneThread()
	{
		lock_guard<mutex> lck(_mutex);
		_cond.notify_one();
	}

	T* receive (bool is_expansion_thread, float timeout)
	{
		unique_lock<mutex> lck(_mutex);
		T* msg=NULL;
		Global_print_mutex(this_thread::get_id(),this, __FUNCTION__, 0);
		_cond.wait(lck,[this, timeout]{
			if (_queue.empty())
			{
				Global_print_mutex(this_thread::get_id(),this, "receive::wait", 1);
				Global_print_queue(this_thread::get_id(),this, _queue.empty(), Active_thread_count);
			}
			//cout<<"thread "<<this_thread::get_id()<<" "<<this<<"::"<<", _queue.empty()="<<_queue.empty()<<", Active_thread_count= "<<Active_thread_count<<endl;
			return !_queue.empty()||(_queue.empty() && Active_thread_count==0)||Timeout(timeout);
		});


		if(!_queue.empty())
		{
			if(is_expansion_thread)
				AddActiveThread();
			msg= move(_queue.back());
			Global_print_mutex(this_thread::get_id(),msg, __FUNCTION__, 2);

			//msg->AddVLoss(pow(sqrt(NUM_CHILDREN),msg->GetDepth()));
			_queue.pop_back();
		}
		else
		{
			;
		}
		Global_print_mutex(this_thread::get_id(),msg, __FUNCTION__, 3);
		return msg;
	}
	bool empty()
	{
		unique_lock<mutex> lck(_mutex);
		Global_print_mutex(this_thread::get_id(),this, __FUNCTION__, 0);
		Global_print_mutex(this_thread::get_id(),this, __FUNCTION__, 1);
		return _queue.empty();
	}


};

#endif /* MSGQUEUE_H_ */
