/*
 * globals.h
 *
 *  Created on: 19 Jul, 2017
 *      Author: panpan
 */

#ifndef THREAD_GLOBALS_H_
#define THREAD_GLOBALS_H_
#include <stdio.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <deque>
#include <algorithm>

#include <future>
#include <condition_variable>

#include <chrono>
#include <despot/GPUcore/CudaInclude.h>
#include <signal.h>
#include <limits.h>
extern int Active_thread_count;
extern int Expansion_Count;
extern float Serial_expansion_time;
//extern float Time_limit;
extern float discount;
extern std::chrono::time_point<std::chrono::system_clock> start_time;
extern int kernelNum;
extern bool use_multi_thread_;
extern int Concurrency_threashold;
extern struct sigaction sa;
extern unsigned long long int* Record;
extern unsigned long long int* record;



extern int NUM_THREADS;
const int MAX_DEPTH=50;
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::seconds sec;
typedef std::chrono::microseconds us;
typedef std::chrono::nanoseconds ns;
typedef std::chrono::duration<float> fsec;
void segfault_sigaction(int sig, siginfo_t *info, void *c);
/*#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}*/
void AddActiveThread();
void MinusActiveThread();
void AddExpanded();
void ResetExpanded();
int CountExpanded();
void AddSerialTime(float);
bool Timeout(float);
void SetUpCPUThreads(bool useCPU, int ThreadNum);
void AdvanceStreamCounter(int stride);
int GetCurrentStream();
void lock_process();
void unlock_process();


void Global_print_mutex(std::thread::id threadIdx,void* address,const  char* func, int mode );
template<typename T>
void Global_print_node(std::thread::id threadIdx,void* node_address,int depth,float step_reward,
		float value,float ub,float uub, float v_loss,float weight,T edge,float weu, const char* msg);
void Global_print_child(std::thread::id threadIdx,void* node_address,int depth, int v_star);
void Global_print_expand(std::thread::id threadIdx,void* node_address,int depth, int obs);
void Global_print_queue(std::thread::id threadIdx,void* node_address,bool empty_queue, int Active_thread_count);
void Global_print_down(std::thread::id threadIdx,void* node_address,int depth);
void Global_print_deleteT(std::thread::id threadIdx,int mode, int the_case);
void Global_print_GPUuse(std::thread::id threadIdx);
void Global_print_message(std::thread::id threadIdx, char* msg);
void Global_print_value(std::thread::id threadIdx, double value, char* msg);

void AddMappedThread(std::thread::id the_id, int mapped_id);
int MapThread(std::thread::id the_id);

void DebugRandom(std::ostream& fout, char* msg);
void InitRandGen();
void DestroyRandGen();
float RandGeneration1(float seed);

#ifdef __CUDACC__
extern cudaStream_t* cuda_streams;

#endif

class spinlock_barrier
{
public:
  spinlock_barrier(const spinlock_barrier&) = delete;
  spinlock_barrier& operator=(const spinlock_barrier&) = delete;

  explicit spinlock_barrier(unsigned int count); /*:
    m_count(check_counter(count)), m_generation(0),
    m_count_reset_value(count)
  {
  }*/

  void count_down_and_wait();
  void reset_barrier(unsigned int count);
  /*{
    unsigned int gen = m_generation.load();

    if (--m_count == 0)
    {
      if (m_generation.compare_exchange_weak(gen, gen + 1))
      {
        m_count = m_count_reset_value;
      }
      return;
    }

    while ((gen == m_generation) && (m_count != 0))
      std::this_thread::yield();
  }*/
  void DettachThread(const char *msg);
  void AttachThread(const char* msg);
#ifdef __CUDACC__


#endif


private:
  static inline unsigned int check_counter(unsigned int count)
    {
  	if (count == 0) {std::cout<<"Wrong counter value!"<<std::endl;exit(1);}
  	return count;
    }
  std::atomic<unsigned int> m_count;
  std::atomic<unsigned int> m_generation;
  unsigned int m_count_reset_value;
};
extern spinlock_barrier* thread_barrier;

#ifdef __CUDACC__

extern __device__ __managed__ bool GPUDoPrint;
extern __device__ __managed__ bool GPUPrintPID;

#endif

extern int PrintThreadID;

#endif /* THREAD_GLOBALS_H_ */
