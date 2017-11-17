#include <despot/core/policy_graph.h>
#include <despot/GPUcore/thread_globals.h>
#include <map>
#include <ucontext.h>
using namespace std;


mutex global_mutex;
int Active_thread_count=0;
int Expansion_Count=0;
float Serial_expansion_time=0;
//float Time_limit=3;
std::chrono::time_point<std::chrono::system_clock> start_time;
int kernelNum=0;
map<std::thread::id, int > ThreadIdMap;
bool use_multi_thread_=false;
int NUM_THREADS =15;
static int stream_counter = 0;
cudaStream_t* cuda_streams = NULL;
int Concurrency_threashold=INT_MAX;
spinlock_barrier* thread_barrier=NULL;
unsigned long long int* Record;
unsigned long long int* record=NULL;
struct sigaction sa;
__device__ __managed__ bool GPUDoPrint=false;
__device__ __managed__ bool GPUPrintPID=1;
int PrintThreadID=0;

static const char *gregs[] = {
	"GS",
	"FS",
	"ES",
	"DS",
	"EDI",
	"ESI",
	"EBP",
	"ESP",
	"EBX",
	"EDX",
	"ECX",
	"EAX",
	"TRAPNO",
	"ERR",
	"EIP",
	"CS",
	"EFL",
	"UESP",
	"SS"
};

void segfault_sigaction(int sig, siginfo_t *info, void *c)
{
	ucontext_t *context = (ucontext_t *)c;

	fprintf(stderr,
		"si_signo:  %d\n"
		"si_code:   %s\n"
		"si_errno:  %d\n"
		"si_pid:    %d\n"
		"si_uid:    %d\n"
		"si_addr:   %p\n"
		"si_status: %d\n"
		"si_band:   %ld\n",
		info->si_signo,
		(info->si_code == SEGV_MAPERR) ? "SEGV_MAPERR" : "SEGV_ACCERR",
		info->si_errno, info->si_pid, info->si_uid, info->si_addr,
		info->si_status, info->si_band
	);

	fprintf(stderr,
		"uc_flags:  0x%x\n"
		"ss_sp:     %p\n"
		"ss_size:   %d\n"
		"ss_flags:  0x%X\n",
		context->uc_flags,
		context->uc_stack.ss_sp,
		context->uc_stack.ss_size,
		context->uc_stack.ss_flags
	);

	fprintf(stderr, "General Registers:\n");
	for(int i = 0; i < 19; i++)
		fprintf(stderr, "\t%7s: 0x%x\n", gregs[i], context->uc_mcontext.gregs[i]);
	//fprintf(stderr, "\tOLDMASK: 0x%x\n", context->uc_mcontext.oldmask);
	//fprintf(stderr, "\t    CR2: 0x%x\n", context->uc_mcontext.cr2);

	exit(-1);
}


void AddSerialTime(float used_time)
{
	lock_guard<mutex> lck(global_mutex);
	Serial_expansion_time+=used_time;
}
void AddMappedThread(std::thread::id the_id, int mapped_id)
{
	ThreadIdMap[the_id]=mapped_id;
	//cout<< "Add thread "<<the_id<< "as id "<< mapped_id<<endl;
}

int MapThread(std::thread::id the_id)
{
	return ThreadIdMap[the_id];
}

void AddActiveThread()
{
	lock_guard<mutex> lck(global_mutex);
	Active_thread_count++;
}

void MinusActiveThread()
{
	lock_guard<mutex> lck(global_mutex);
	Active_thread_count--;
}

void AddExpanded()
{
	lock_guard<mutex> lck(global_mutex);
	Expansion_Count++;
}

void ResetExpanded()
{
	lock_guard<mutex> lck(global_mutex);
	Expansion_Count=0;
}
int CountExpanded()
{
	lock_guard<mutex> lck(global_mutex);
	return Expansion_Count;
}

bool Timeout(float timeout)
{
	//lock_guard<mutex> lck(global_mutex);
	auto t1 = Time::now();
	fsec fs = t1 - start_time;
	ns d = std::chrono::duration_cast<ns>(fs);
	return d.count()/1000000000.0>=timeout;
}

void SetUpCPUThreads(bool useCPU, int ThreadNum)
{
	use_multi_thread_=useCPU;
	NUM_THREADS=ThreadNum;
}

void AdvanceStreamCounter(int stride) {
	stream_counter += stride;
	if (stream_counter >= NUM_THREADS)
		stream_counter = 0;
}
int GetCurrentStream()
{
	return stream_counter;
}
void lock_process()
{
	global_mutex.lock();
}
void unlock_process()
{
	global_mutex.unlock();
}

void Global_print_mutex(std::thread::id threadIdx,void* address,const  char* func, int mode )
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		cout<<"thread "<<threadID<<" instance "<<address<<"::"<<func<<": ";
		switch(mode)
		{
		case 0:  cout<< "lock"; break;
		case 1:  cout<< "unlock"; break;
		case 2:  cout<< "relock with msg"; break;
		case 3:  cout<< "un-relock with msg"; break;
		}
		cout<<endl;
	}
}
template<typename T>
void Global_print_node(std::thread::id threadIdx,void* node_address,int depth,float step_reward,
		float value,float ub,float uub, float v_loss,float weight,T edge,float WEU, const char* msg)
{
	if(false || /*node_address ==NULL ||*/FIX_SCENARIO==1)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		if(threadID==PrintThreadID/*true*/)
		{
			cout.precision(4);
			if(weight!=0)
			{
				cout<<"thread "<<threadID<<" "<<msg<<" get old node "<<node_address<<" at depth "<<depth
					<<": weight="<<weight<<", reward="<<step_reward/weight
					<<", lb="<<value/weight<<", ub="<<ub/weight;

				if(uub>-1000)
					cout <<", uub="<<uub/weight;
				cout<<", edge="<< edge ;
				if(WEU>-1000)
					cout <<", WEU="<<WEU;
				cout<<", v_loss="<<v_loss/weight;
			}
			else
			{
				cout<<"thread "<<threadID<<" "<<msg<<" get old node "<<node_address<<" at depth "<<depth
					<<": weight="<<weight<<", reward="<<step_reward
					<<", lb="<<value<<", ub="<<ub;
				if(uub>-1000)
					cout <<", uub="<<uub;
				cout<<", edge="<< edge ;
				if(WEU>-1000)
					cout <<", WEU="<<WEU;
				cout <<", v_loss="<<v_loss;
			}
			cout<<endl;
		}
	}
}


template void Global_print_node<int>(std::thread::id threadIdx,void* node_address,
		int depth,float step_reward, float value, float ub, float uub, float v_loss,float weight,
		int edge, float weu, const char* msg);

template void Global_print_node<uint64_t>(std::thread::id threadIdx,void* node_address,
		int depth,float step_reward, float value, float ub, float uub, float v_loss,float weight,
		uint64_t edge, float weu, const char* msg);

void Global_print_child(std::thread::id threadIdx,void* node_address,int depth, int v_star)
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		if(/*threadID==PrintThreadID*/true)
		{
			cout<<"thread "<<threadID<<" node "<<node_address<<" at depth "<<depth<<" select optimal child "<<v_star;
			cout<<endl;
		}
	}
}
void Global_print_expand(std::thread::id threadIdx,void* node_address,int depth, int obs)
{
	if(/*false||*/FIX_SCENARIO==1)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=0;
		if(use_multi_thread_)
			threadID=MapThread(threadIdx);
		if(threadID==PrintThreadID/*true*/)
		{
			cout<<"thread "<<threadID<<" expand node "<<node_address<<" at depth "<<depth<< " edge "<<obs;
			cout<<endl;
		}
	}
}

void Global_print_queue(std::thread::id threadIdx,void* node_address,bool empty_queue, int Active_thread_count)
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		cout<<"thread "<<threadID<<" "<<node_address<<"::"<<", _queue.empty()="<<empty_queue<<", Active_thread_count= "<<Active_thread_count<<endl;
	}
}
void Global_print_down(std::thread::id threadIdx,void* node_address,int depth)
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		cout<<"thread "<<threadID<<" trace to node "<<node_address<<" at depth "<<depth;
		cout<<endl;
	}
}
void Global_print_deleteT(std::thread::id threadIdx,int mode, int the_case)
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		switch(mode)
		{
		case 0:cout<<"Delete expansion thread "<<threadID;break;
		case 1:cout<<"Delete printing thread "<<threadID;break;
		};
		switch(the_case)
		{
		case 0:cout<<" due to NULL ptr"<<endl;break;
		case 1:cout<<" due to time out"<<endl;break;
		};
		cout<<endl;
	}
}
void Global_print_GPUuse(std::thread::id threadIdx)
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		cout<<"call GPU func with expansion thread "<<threadID<<endl;
	}
}

void Global_print_message(std::thread::id threadIdx, char* msg)
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		cout<<"Msg from thread "<<threadID<<" "<<msg<<endl;
	}
}

void Global_print_value(std::thread::id threadIdx, double value, char* msg)
{
	if(false)
	{
		lock_guard<mutex> lck(global_mutex);
		int threadID=MapThread(threadIdx);
		if(threadID==PrintThreadID/*true*/)
		{
			cout.precision(4);
			cout<<msg<<" Value from thread "<<threadID<<" "<<value<<endl;
		}
	}
}

void spinlock_barrier::count_down_and_wait()
{
	unsigned int gen = m_generation.load();
	/*int threadID=MapThread(this_thread::get_id());
	cout<<"Thread "<<threadID<<" wait "<<__FUNCTION__<<" m_count="<<m_count-1<<",m_count_reset_value="<<m_count_reset_value<<endl;*/
	if (--m_count <= 0)
	{
	  if (m_generation.compare_exchange_weak(gen, gen + 1))
	  {
		m_count = m_count_reset_value;
		/*lock_guard<mutex> lck(global_mutex);
		cout<<"Thread "<<threadID<<" restart all "<<__FUNCTION__<<" m_count="<<m_count-1<<",m_count_reset_value="<<m_count_reset_value<<endl;*/
	  }
	  return;
	}

	while ((gen == m_generation) && (m_count > 0))
	  std::this_thread::yield();

	//m_count = m_count_reset_value;

}

/*explicit*/ spinlock_barrier::spinlock_barrier(unsigned int count) :
    m_count(check_counter(count)), m_generation(0),
    m_count_reset_value(count)
  {
  }
void spinlock_barrier::reset_barrier(unsigned int count)
  {
	m_count=count;
	m_generation=0;
	m_count_reset_value=count;
  }
void spinlock_barrier::AttachThread(const char* msg)
{
	  unsigned int temp=m_generation.load();//prevent this to be called with count_down_and_wait
	  ++m_count_reset_value;
	  m_count++;
	  /*unsigned int count=m_count.load();
	  if(count>1)
		  m_count.compare_exchange_weak(count, count - 1);*/
	  /*lock_guard<mutex> lck(global_mutex);
	  int threadID=MapThread(this_thread::get_id());
	  cout<<"Thread "<<threadID<<" "<<msg<<" "<<__FUNCTION__<<" m_count="<<m_count<<",m_count_reset_value="<<m_count_reset_value<<endl;*/
	  m_generation.compare_exchange_weak(temp, temp);
}
void spinlock_barrier::DettachThread(const char* msg)
{
	  unsigned int temp=m_generation.load();//prevent this to be called with count_down_and_wait
	  --m_count_reset_value;
	  if (--m_count <= 0)
	  {
		  if(m_generation.compare_exchange_weak(temp, temp+1))
			m_count = m_count_reset_value;
		//lock_guard<mutex> lck(global_mutex);
		//cout<<"Thread "<<threadID<<" restart all "<<__FUNCTION__<<" m_count="<<m_count-1<<",m_count_reset_value="<<m_count_reset_value<<endl;
	  }
	  /*unsigned int count=m_count.load();
	  if(count>1)
		  m_count.compare_exchange_weak(count, count - 1);*/
	  /*lock_guard<mutex> lck(global_mutex);
	  int threadID=MapThread(this_thread::get_id());
	  cout<<"Thread "<<threadID<<" "<<msg<<" "<<__FUNCTION__<<" m_count="<<m_count<<",m_count_reset_value="<<m_count_reset_value<<endl;*/
}
float RandGeneration1(float seed)
{
	if(use_multi_thread_)
	{
		unsigned long long int &global_record=record[MapThread(this_thread::get_id())];
		//float value between 0 and 1
		global_record+= seed*ULLONG_MAX/1000.0;
		float record_db=0;
		global_record=16807*global_record;
		//if(record<0)
		//	record=-record;
		global_record=global_record%2147483647;
		record_db=((double)global_record)/2147483647;
		return record_db;
	}
	else
	{
		unsigned long long int &global_record=record[0];
		//float value between 0 and 1
		global_record+= seed*ULLONG_MAX/1000.0;
		float record_db=0;
		global_record=16807*global_record;
		//if(record<0)
		//	record=-record;
		global_record=global_record%2147483647;
		record_db=((double)global_record)/2147483647;
		return record_db;
	}
}




/*__global__ void TestRandom(int* CountArray  , float* seeds,unsigned long long int *record)
{//<<<10,1>>>
	float index = threadIdx.x;//max 10
	CountArray[threadIdx.x]=0;
	for (int i=0;i<100;i++)
	{
		float randfloat=RandGeneration(record, seeds[threadIdx.x]);
		if(randfloat<(index+1)/10.0f && randfloat>index/10.0f)
			CountArray[threadIdx.x]++;
	}
	__syncthreads();
}*/
static
__device__ float RandGeneration(unsigned long long int *record, float seed)
{
	//float value between 0 and 1
	//seed have to be within 0 and 1
	record[0]+=seed*ULLONG_MAX/1000.0;
	//unsigned long long int record_l=atomicAdd(record,t);//atomic add returns old value of record

	//record_l+=t;
	float record_f=0;
	record[0]*=16807;
	record[0]=record[0]%2147483647;
	record_f=((double)record[0])/2147483647;
	return record_f;
}
__global__ void TestRandom(float* Results , float* seeds,int* CountArray,unsigned long long int *record, int bins)
{//<<<10,1>>>
	Results[threadIdx.x]=0;
	for (int index=0;index<bins;index++)
	{
		CountArray[index]=0;
	}
	__syncthreads();

	unsigned long long int Temp=123456;
	float rand=seeds[threadIdx.x];
	rand=RandGeneration(&Temp, rand);
	rand=RandGeneration(&Temp, rand);
	rand=RandGeneration(&Temp, rand);
	Results[threadIdx.x]=RandGeneration(&Temp, rand);

	for (int index=0;index<bins;index++)
	{
		if(Results[threadIdx.x]<(((float)index)+1)/((float)bins)
				&& Results[threadIdx.x]>((float)index)/((float)bins))
			atomicAdd(CountArray+index,1);
			//CountArray[index]++;
	}
}
void DebugRandom(ostream& fout, char* msg)
{
	const int bin_size=100;
	int* CountArray=new int[bin_size];
	const int size=1000;
	float* HstRands=new float[size];

	float* hst_seeds=new float[size];
	for(int i=0;i<size;i++)
	{
		hst_seeds[i]=i/((float)size)/*0.9*/;
	}

	int* DvcCountArray;
	float* DvcRands;
	HANDLE_ERROR(cudaMalloc((void**)&DvcCountArray,bin_size*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&DvcRands,size*sizeof(float)));

	float* Dvc_seeds;
	HANDLE_ERROR(cudaMalloc((void**)&Dvc_seeds,size*sizeof(float)));


	HANDLE_ERROR(cudaMemcpy(Dvc_seeds,hst_seeds,size*sizeof(float),cudaMemcpyHostToDevice));


	TestRandom<<<1,size>>>(DvcRands,Dvc_seeds,DvcCountArray,Record, bin_size);
	HANDLE_ERROR(cudaMemcpy(CountArray,DvcCountArray,bin_size*sizeof(int),cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(HstRands,DvcRands,size*sizeof(float),cudaMemcpyDeviceToHost));

	fout<< msg<<endl;
	for(int i=0;i<size;i++)
	{
		if(i!=0) fout << ",";
		fout<< HstRands[i];
	}
	fout<< endl;
	for(int i=0;i<bin_size;i++)
	{
		if(i!=0) fout << ",";
		fout<< CountArray[i];
	}
	fout<< endl;

	for (int index=0;index<bin_size;index++)
	{
		CountArray[index]=0;
	}

	for(int i=0;i<size;i++)
	{
		//record=123456;
		if(use_multi_thread_)
		{
			record[MapThread(this_thread::get_id())]=123456;
		}
		else
			record[0]=123456;
		double rand=hst_seeds[i];
		rand=RandGeneration1(rand);
		rand=RandGeneration1(rand);
		rand=RandGeneration1(rand);
		rand=RandGeneration1(rand);
		if(i!=0) fout << ",";
		fout<< rand;
		for (int index=0;index<bin_size;index++)
		{
			if(rand<(((float)index)+1)/((float)bin_size)
					&& rand>((float)index)/((float)bin_size))
				CountArray[index]++;
		}
	}
	fout<< endl;
	for(int i=0;i<bin_size;i++)
	{
		if(i!=0) fout << ",";
		fout<< CountArray[i];
	}
	fout<< endl;
	HANDLE_ERROR(cudaFree(DvcCountArray));
	HANDLE_ERROR(cudaFree(DvcRands));
	HANDLE_ERROR(cudaFree(Dvc_seeds));

	delete [] CountArray;
	delete [] HstRands;
	delete [] hst_seeds;




}
__global__ void InitRecord(unsigned long long int* _Record)
{
	_Record[0]=1;
}
void InitRandGen()
{
	HANDLE_ERROR(cudaMalloc(    (void**)&Record,        sizeof(unsigned long long int)));
	InitRecord<<<1,1,1>>>(Record);
	HANDLE_ERROR(cudaDeviceSynchronize());
}

void DestroyRandGen()
{
	if(Record!=NULL)		{cudaFree(Record);		 Record=NULL;		 }
}


