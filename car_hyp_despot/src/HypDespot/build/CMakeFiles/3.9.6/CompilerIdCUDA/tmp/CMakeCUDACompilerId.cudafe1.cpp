# 1 "CMakeCUDACompilerId.cu"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
# 1
#pragma GCC diagnostic push
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"
# 1
#pragma GCC diagnostic ignored "-Wunused-function"
# 1
static char __nv_inited_managed_rt = 0; static void **__nv_fatbinhandle_for_managed_rt; static void __nv_save_fatbinhandle_for_managed_rt(void **in){__nv_fatbinhandle_for_managed_rt = in;} static char __nv_init_managed_rt_with_module(void **); static inline void __nv_init_managed_rt(void) { __nv_inited_managed_rt = (__nv_inited_managed_rt ? __nv_inited_managed_rt                 : __nv_init_managed_rt_with_module(__nv_fatbinhandle_for_managed_rt));}
# 1
#pragma GCC diagnostic pop
# 1
#pragma GCC diagnostic ignored "-Wunused-variable"

# 1
#define __nv_is_extended_device_lambda_closure_type(X) false
#define __nv_is_extended_host_device_lambda_closure_type(X) false

# 1
# 56 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic push
# 59
#pragma GCC diagnostic ignored "-Wunused-function"
# 61 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_types.h"
#if 0
# 61
enum cudaRoundMode { 
# 63
cudaRoundNearest, 
# 64
cudaRoundZero, 
# 65
cudaRoundPosInf, 
# 66
cudaRoundMinInf
# 67
}; 
#endif
# 93 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 93
struct char1 { 
# 95
signed char x; 
# 96
}; 
#endif
# 98 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 98
struct uchar1 { 
# 100
unsigned char x; 
# 101
}; 
#endif
# 104 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 104
struct __attribute((aligned(2))) char2 { 
# 106
signed char x, y; 
# 107
}; 
#endif
# 109 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 109
struct __attribute((aligned(2))) uchar2 { 
# 111
unsigned char x, y; 
# 112
}; 
#endif
# 114 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 114
struct char3 { 
# 116
signed char x, y, z; 
# 117
}; 
#endif
# 119 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 119
struct uchar3 { 
# 121
unsigned char x, y, z; 
# 122
}; 
#endif
# 124 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 124
struct __attribute((aligned(4))) char4 { 
# 126
signed char x, y, z, w; 
# 127
}; 
#endif
# 129 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 129
struct __attribute((aligned(4))) uchar4 { 
# 131
unsigned char x, y, z, w; 
# 132
}; 
#endif
# 134 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 134
struct short1 { 
# 136
short x; 
# 137
}; 
#endif
# 139 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 139
struct ushort1 { 
# 141
unsigned short x; 
# 142
}; 
#endif
# 144 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 144
struct __attribute((aligned(4))) short2 { 
# 146
short x, y; 
# 147
}; 
#endif
# 149 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 149
struct __attribute((aligned(4))) ushort2 { 
# 151
unsigned short x, y; 
# 152
}; 
#endif
# 154 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 154
struct short3 { 
# 156
short x, y, z; 
# 157
}; 
#endif
# 159 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 159
struct ushort3 { 
# 161
unsigned short x, y, z; 
# 162
}; 
#endif
# 164 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 164
struct __attribute((aligned(8))) short4 { short x; short y; short z; short w; }; 
#endif
# 165 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 165
struct __attribute((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; }; 
#endif
# 167 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 167
struct int1 { 
# 169
int x; 
# 170
}; 
#endif
# 172 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 172
struct uint1 { 
# 174
unsigned x; 
# 175
}; 
#endif
# 177 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 177
struct __attribute((aligned(8))) int2 { int x; int y; }; 
#endif
# 178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 178
struct __attribute((aligned(8))) uint2 { unsigned x; unsigned y; }; 
#endif
# 180 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 180
struct int3 { 
# 182
int x, y, z; 
# 183
}; 
#endif
# 185 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 185
struct uint3 { 
# 187
unsigned x, y, z; 
# 188
}; 
#endif
# 190 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 190
struct __attribute((aligned(16))) int4 { 
# 192
int x, y, z, w; 
# 193
}; 
#endif
# 195 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 195
struct __attribute((aligned(16))) uint4 { 
# 197
unsigned x, y, z, w; 
# 198
}; 
#endif
# 200 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 200
struct long1 { 
# 202
long x; 
# 203
}; 
#endif
# 205 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 205
struct ulong1 { 
# 207
unsigned long x; 
# 208
}; 
#endif
# 215 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 215
struct __attribute((aligned((2) * sizeof(long)))) long2 { 
# 217
long x, y; 
# 218
}; 
#endif
# 220 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 220
struct __attribute((aligned((2) * sizeof(unsigned long)))) ulong2 { 
# 222
unsigned long x, y; 
# 223
}; 
#endif
# 227 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 227
struct long3 { 
# 229
long x, y, z; 
# 230
}; 
#endif
# 232 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 232
struct ulong3 { 
# 234
unsigned long x, y, z; 
# 235
}; 
#endif
# 237 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 237
struct __attribute((aligned(16))) long4 { 
# 239
long x, y, z, w; 
# 240
}; 
#endif
# 242 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 242
struct __attribute((aligned(16))) ulong4 { 
# 244
unsigned long x, y, z, w; 
# 245
}; 
#endif
# 247 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 247
struct float1 { 
# 249
float x; 
# 250
}; 
#endif
# 269 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 269
struct __attribute((aligned(8))) float2 { float x; float y; }; 
#endif
# 274 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 274
struct float3 { 
# 276
float x, y, z; 
# 277
}; 
#endif
# 279 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 279
struct __attribute((aligned(16))) float4 { 
# 281
float x, y, z, w; 
# 282
}; 
#endif
# 284 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 284
struct longlong1 { 
# 286
long long x; 
# 287
}; 
#endif
# 289 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 289
struct ulonglong1 { 
# 291
unsigned long long x; 
# 292
}; 
#endif
# 294 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 294
struct __attribute((aligned(16))) longlong2 { 
# 296
long long x, y; 
# 297
}; 
#endif
# 299 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 299
struct __attribute((aligned(16))) ulonglong2 { 
# 301
unsigned long long x, y; 
# 302
}; 
#endif
# 304 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 304
struct longlong3 { 
# 306
long long x, y, z; 
# 307
}; 
#endif
# 309 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 309
struct ulonglong3 { 
# 311
unsigned long long x, y, z; 
# 312
}; 
#endif
# 314 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 314
struct __attribute((aligned(16))) longlong4 { 
# 316
long long x, y, z, w; 
# 317
}; 
#endif
# 319 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 319
struct __attribute((aligned(16))) ulonglong4 { 
# 321
unsigned long long x, y, z, w; 
# 322
}; 
#endif
# 324 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 324
struct double1 { 
# 326
double x; 
# 327
}; 
#endif
# 329 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 329
struct __attribute((aligned(16))) double2 { 
# 331
double x, y; 
# 332
}; 
#endif
# 334 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 334
struct double3 { 
# 336
double x, y, z; 
# 337
}; 
#endif
# 339 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 339
struct __attribute((aligned(16))) double4 { 
# 341
double x, y, z, w; 
# 342
}; 
#endif
# 356 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char1 
# 356
char1; 
#endif
# 357 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar1 
# 357
uchar1; 
#endif
# 358 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char2 
# 358
char2; 
#endif
# 359 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar2 
# 359
uchar2; 
#endif
# 360 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char3 
# 360
char3; 
#endif
# 361 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar3 
# 361
uchar3; 
#endif
# 362 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef char4 
# 362
char4; 
#endif
# 363 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uchar4 
# 363
uchar4; 
#endif
# 364 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short1 
# 364
short1; 
#endif
# 365 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort1 
# 365
ushort1; 
#endif
# 366 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short2 
# 366
short2; 
#endif
# 367 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort2 
# 367
ushort2; 
#endif
# 368 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short3 
# 368
short3; 
#endif
# 369 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort3 
# 369
ushort3; 
#endif
# 370 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef short4 
# 370
short4; 
#endif
# 371 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ushort4 
# 371
ushort4; 
#endif
# 372 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int1 
# 372
int1; 
#endif
# 373 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint1 
# 373
uint1; 
#endif
# 374 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int2 
# 374
int2; 
#endif
# 375 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint2 
# 375
uint2; 
#endif
# 376 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int3 
# 376
int3; 
#endif
# 377 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint3 
# 377
uint3; 
#endif
# 378 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef int4 
# 378
int4; 
#endif
# 379 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef uint4 
# 379
uint4; 
#endif
# 380 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long1 
# 380
long1; 
#endif
# 381 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong1 
# 381
ulong1; 
#endif
# 382 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long2 
# 382
long2; 
#endif
# 383 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong2 
# 383
ulong2; 
#endif
# 384 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long3 
# 384
long3; 
#endif
# 385 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong3 
# 385
ulong3; 
#endif
# 386 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef long4 
# 386
long4; 
#endif
# 387 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulong4 
# 387
ulong4; 
#endif
# 388 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float1 
# 388
float1; 
#endif
# 389 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float2 
# 389
float2; 
#endif
# 390 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float3 
# 390
float3; 
#endif
# 391 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef float4 
# 391
float4; 
#endif
# 392 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong1 
# 392
longlong1; 
#endif
# 393 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong1 
# 393
ulonglong1; 
#endif
# 394 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong2 
# 394
longlong2; 
#endif
# 395 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong2 
# 395
ulonglong2; 
#endif
# 396 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong3 
# 396
longlong3; 
#endif
# 397 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong3 
# 397
ulonglong3; 
#endif
# 398 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef longlong4 
# 398
longlong4; 
#endif
# 399 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef ulonglong4 
# 399
ulonglong4; 
#endif
# 400 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double1 
# 400
double1; 
#endif
# 401 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double2 
# 401
double2; 
#endif
# 402 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double3 
# 402
double3; 
#endif
# 403 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef double4 
# 403
double4; 
#endif
# 411 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
# 411
struct dim3 { 
# 413
unsigned x, y, z; 
# 419
}; 
#endif
# 421 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_types.h"
#if 0
typedef dim3 
# 421
dim3; 
#endif
# 149 "/usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h" 3
typedef long ptrdiff_t; 
# 216 "/usr/lib/gcc/x86_64-linux-gnu/5/include/stddef.h" 3
typedef unsigned long size_t; 
#include "crt/host_runtime.h"
# 171 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 171
enum cudaError { 
# 178
cudaSuccess, 
# 184
cudaErrorMissingConfiguration, 
# 190
cudaErrorMemoryAllocation, 
# 196
cudaErrorInitializationError, 
# 206 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchFailure, 
# 215 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorPriorLaunchFailure, 
# 226 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchTimeout, 
# 235 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchOutOfResources, 
# 241
cudaErrorInvalidDeviceFunction, 
# 250 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorInvalidConfiguration, 
# 256
cudaErrorInvalidDevice, 
# 262
cudaErrorInvalidValue, 
# 268
cudaErrorInvalidPitchValue, 
# 274
cudaErrorInvalidSymbol, 
# 279
cudaErrorMapBufferObjectFailed, 
# 284
cudaErrorUnmapBufferObjectFailed, 
# 290
cudaErrorInvalidHostPointer, 
# 296
cudaErrorInvalidDevicePointer, 
# 302
cudaErrorInvalidTexture, 
# 308
cudaErrorInvalidTextureBinding, 
# 315
cudaErrorInvalidChannelDescriptor, 
# 321
cudaErrorInvalidMemcpyDirection, 
# 331 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorAddressOfConstant, 
# 340 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorTextureFetchFailed, 
# 349 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorTextureNotBound, 
# 358 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSynchronizationError, 
# 364
cudaErrorInvalidFilterSetting, 
# 370
cudaErrorInvalidNormSetting, 
# 378
cudaErrorMixedDeviceExecution, 
# 385
cudaErrorCudartUnloading, 
# 390
cudaErrorUnknown, 
# 398
cudaErrorNotYetImplemented, 
# 407 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorMemoryValueTooLarge, 
# 414
cudaErrorInvalidResourceHandle, 
# 422
cudaErrorNotReady, 
# 429
cudaErrorInsufficientDriver, 
# 442 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSetOnActiveProcess, 
# 448
cudaErrorInvalidSurface, 
# 454
cudaErrorNoDevice, 
# 460
cudaErrorECCUncorrectable, 
# 465
cudaErrorSharedObjectSymbolNotFound, 
# 470
cudaErrorSharedObjectInitFailed, 
# 476
cudaErrorUnsupportedLimit, 
# 482
cudaErrorDuplicateVariableName, 
# 488
cudaErrorDuplicateTextureName, 
# 494
cudaErrorDuplicateSurfaceName, 
# 504 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorDevicesUnavailable, 
# 509
cudaErrorInvalidKernelImage, 
# 517
cudaErrorNoKernelImageForDevice, 
# 530 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorIncompatibleDriverContext, 
# 537
cudaErrorPeerAccessAlreadyEnabled, 
# 544
cudaErrorPeerAccessNotEnabled, 
# 550
cudaErrorDeviceAlreadyInUse = 54, 
# 557
cudaErrorProfilerDisabled, 
# 565
cudaErrorProfilerNotInitialized, 
# 572
cudaErrorProfilerAlreadyStarted, 
# 579
cudaErrorProfilerAlreadyStopped, 
# 587
cudaErrorAssert, 
# 594
cudaErrorTooManyPeers, 
# 600
cudaErrorHostMemoryAlreadyRegistered, 
# 606
cudaErrorHostMemoryNotRegistered, 
# 611
cudaErrorOperatingSystem, 
# 617
cudaErrorPeerAccessUnsupported, 
# 624
cudaErrorLaunchMaxDepthExceeded, 
# 632
cudaErrorLaunchFileScopedTex, 
# 640
cudaErrorLaunchFileScopedSurf, 
# 655 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorSyncDepthExceeded, 
# 667 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorLaunchPendingCountExceeded, 
# 672
cudaErrorNotPermitted, 
# 678
cudaErrorNotSupported, 
# 687 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorHardwareStackError, 
# 695
cudaErrorIllegalInstruction, 
# 704 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorMisalignedAddress, 
# 715 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorInvalidAddressSpace, 
# 723
cudaErrorInvalidPc, 
# 731
cudaErrorIllegalAddress, 
# 737
cudaErrorInvalidPtx, 
# 742
cudaErrorInvalidGraphicsContext, 
# 748
cudaErrorNvlinkUncorrectable, 
# 755
cudaErrorJitCompilerNotFound, 
# 764 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
cudaErrorCooperativeLaunchTooLarge, 
# 769
cudaErrorStartupFailure = 127, 
# 777
cudaErrorApiFailureBase = 10000
# 778
}; 
#endif
# 783 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 783
enum cudaChannelFormatKind { 
# 785
cudaChannelFormatKindSigned, 
# 786
cudaChannelFormatKindUnsigned, 
# 787
cudaChannelFormatKindFloat, 
# 788
cudaChannelFormatKindNone
# 789
}; 
#endif
# 794 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 794
struct cudaChannelFormatDesc { 
# 796
int x; 
# 797
int y; 
# 798
int z; 
# 799
int w; 
# 800
cudaChannelFormatKind f; 
# 801
}; 
#endif
# 806 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
typedef struct cudaArray *cudaArray_t; 
# 811
typedef const cudaArray *cudaArray_const_t; 
# 813
struct cudaArray; 
# 818
typedef struct cudaMipmappedArray *cudaMipmappedArray_t; 
# 823
typedef const cudaMipmappedArray *cudaMipmappedArray_const_t; 
# 825
struct cudaMipmappedArray; 
# 830
#if 0
# 830
enum cudaMemoryType { 
# 832
cudaMemoryTypeHost = 1, 
# 833
cudaMemoryTypeDevice
# 834
}; 
#endif
# 839 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 839
enum cudaMemcpyKind { 
# 841
cudaMemcpyHostToHost, 
# 842
cudaMemcpyHostToDevice, 
# 843
cudaMemcpyDeviceToHost, 
# 844
cudaMemcpyDeviceToDevice, 
# 845
cudaMemcpyDefault
# 846
}; 
#endif
# 853 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 853
struct cudaPitchedPtr { 
# 855
void *ptr; 
# 856
size_t pitch; 
# 857
size_t xsize; 
# 858
size_t ysize; 
# 859
}; 
#endif
# 866 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 866
struct cudaExtent { 
# 868
size_t width; 
# 869
size_t height; 
# 870
size_t depth; 
# 871
}; 
#endif
# 878 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 878
struct cudaPos { 
# 880
size_t x; 
# 881
size_t y; 
# 882
size_t z; 
# 883
}; 
#endif
# 888 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 888
struct cudaMemcpy3DParms { 
# 890
cudaArray_t srcArray; 
# 891
cudaPos srcPos; 
# 892
cudaPitchedPtr srcPtr; 
# 894
cudaArray_t dstArray; 
# 895
cudaPos dstPos; 
# 896
cudaPitchedPtr dstPtr; 
# 898
cudaExtent extent; 
# 899
cudaMemcpyKind kind; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 900
}; 
#endif
# 905 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 905
struct cudaMemcpy3DPeerParms { 
# 907
cudaArray_t srcArray; 
# 908
cudaPos srcPos; 
# 909
cudaPitchedPtr srcPtr; 
# 910
int srcDevice; 
# 912
cudaArray_t dstArray; 
# 913
cudaPos dstPos; 
# 914
cudaPitchedPtr dstPtr; 
# 915
int dstDevice; 
# 917
cudaExtent extent; 
# 918
}; 
#endif
# 923 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
struct cudaGraphicsResource; 
# 928
#if 0
# 928
enum cudaGraphicsRegisterFlags { 
# 930
cudaGraphicsRegisterFlagsNone, 
# 931
cudaGraphicsRegisterFlagsReadOnly, 
# 932
cudaGraphicsRegisterFlagsWriteDiscard, 
# 933
cudaGraphicsRegisterFlagsSurfaceLoadStore = 4, 
# 934
cudaGraphicsRegisterFlagsTextureGather = 8
# 935
}; 
#endif
# 940 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 940
enum cudaGraphicsMapFlags { 
# 942
cudaGraphicsMapFlagsNone, 
# 943
cudaGraphicsMapFlagsReadOnly, 
# 944
cudaGraphicsMapFlagsWriteDiscard
# 945
}; 
#endif
# 950 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 950
enum cudaGraphicsCubeFace { 
# 952
cudaGraphicsCubeFacePositiveX, 
# 953
cudaGraphicsCubeFaceNegativeX, 
# 954
cudaGraphicsCubeFacePositiveY, 
# 955
cudaGraphicsCubeFaceNegativeY, 
# 956
cudaGraphicsCubeFacePositiveZ, 
# 957
cudaGraphicsCubeFaceNegativeZ
# 958
}; 
#endif
# 963 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 963
enum cudaResourceType { 
# 965
cudaResourceTypeArray, 
# 966
cudaResourceTypeMipmappedArray, 
# 967
cudaResourceTypeLinear, 
# 968
cudaResourceTypePitch2D
# 969
}; 
#endif
# 974 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 974
enum cudaResourceViewFormat { 
# 976
cudaResViewFormatNone, 
# 977
cudaResViewFormatUnsignedChar1, 
# 978
cudaResViewFormatUnsignedChar2, 
# 979
cudaResViewFormatUnsignedChar4, 
# 980
cudaResViewFormatSignedChar1, 
# 981
cudaResViewFormatSignedChar2, 
# 982
cudaResViewFormatSignedChar4, 
# 983
cudaResViewFormatUnsignedShort1, 
# 984
cudaResViewFormatUnsignedShort2, 
# 985
cudaResViewFormatUnsignedShort4, 
# 986
cudaResViewFormatSignedShort1, 
# 987
cudaResViewFormatSignedShort2, 
# 988
cudaResViewFormatSignedShort4, 
# 989
cudaResViewFormatUnsignedInt1, 
# 990
cudaResViewFormatUnsignedInt2, 
# 991
cudaResViewFormatUnsignedInt4, 
# 992
cudaResViewFormatSignedInt1, 
# 993
cudaResViewFormatSignedInt2, 
# 994
cudaResViewFormatSignedInt4, 
# 995
cudaResViewFormatHalf1, 
# 996
cudaResViewFormatHalf2, 
# 997
cudaResViewFormatHalf4, 
# 998
cudaResViewFormatFloat1, 
# 999
cudaResViewFormatFloat2, 
# 1000
cudaResViewFormatFloat4, 
# 1001
cudaResViewFormatUnsignedBlockCompressed1, 
# 1002
cudaResViewFormatUnsignedBlockCompressed2, 
# 1003
cudaResViewFormatUnsignedBlockCompressed3, 
# 1004
cudaResViewFormatUnsignedBlockCompressed4, 
# 1005
cudaResViewFormatSignedBlockCompressed4, 
# 1006
cudaResViewFormatUnsignedBlockCompressed5, 
# 1007
cudaResViewFormatSignedBlockCompressed5, 
# 1008
cudaResViewFormatUnsignedBlockCompressed6H, 
# 1009
cudaResViewFormatSignedBlockCompressed6H, 
# 1010
cudaResViewFormatUnsignedBlockCompressed7
# 1011
}; 
#endif
# 1016 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1016
struct cudaResourceDesc { 
# 1017
cudaResourceType resType; 
# 1019
union { 
# 1020
struct { 
# 1021
cudaArray_t array; 
# 1022
} array; 
# 1023
struct { 
# 1024
cudaMipmappedArray_t mipmap; 
# 1025
} mipmap; 
# 1026
struct { 
# 1027
void *devPtr; 
# 1028
cudaChannelFormatDesc desc; 
# 1029
size_t sizeInBytes; 
# 1030
} linear; 
# 1031
struct { 
# 1032
void *devPtr; 
# 1033
cudaChannelFormatDesc desc; 
# 1034
size_t width; 
# 1035
size_t height; 
# 1036
size_t pitchInBytes; 
# 1037
} pitch2D; 
# 1038
} res; 
# 1039
}; 
#endif
# 1044 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1044
struct cudaResourceViewDesc { 
# 1046
cudaResourceViewFormat format; 
# 1047
size_t width; 
# 1048
size_t height; 
# 1049
size_t depth; 
# 1050
unsigned firstMipmapLevel; 
# 1051
unsigned lastMipmapLevel; 
# 1052
unsigned firstLayer; 
# 1053
unsigned lastLayer; 
# 1054
}; 
#endif
# 1059 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1059
struct cudaPointerAttributes { 
# 1065
cudaMemoryType memoryType; 
# 1076 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
int device; 
# 1082
void *devicePointer; 
# 1088
void *hostPointer; 
# 1093
int isManaged; __pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)__pad__(volatile char:8;)
# 1094
}; 
#endif
# 1099 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1099
struct cudaFuncAttributes { 
# 1106
size_t sharedSizeBytes; 
# 1112
size_t constSizeBytes; 
# 1117
size_t localSizeBytes; 
# 1124
int maxThreadsPerBlock; 
# 1129
int numRegs; 
# 1136
int ptxVersion; 
# 1143
int binaryVersion; 
# 1149
int cacheModeCA; 
# 1156
int maxDynamicSharedSizeBytes; 
# 1163
int preferredShmemCarveout; 
# 1164
}; 
#endif
# 1169 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1169
enum cudaFuncAttribute { 
# 1171
cudaFuncAttributeMaxDynamicSharedMemorySize = 8, 
# 1172
cudaFuncAttributePreferredSharedMemoryCarveout, 
# 1173
cudaFuncAttributeMax
# 1174
}; 
#endif
# 1179 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1179
enum cudaFuncCache { 
# 1181
cudaFuncCachePreferNone, 
# 1182
cudaFuncCachePreferShared, 
# 1183
cudaFuncCachePreferL1, 
# 1184
cudaFuncCachePreferEqual
# 1185
}; 
#endif
# 1191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1191
enum cudaSharedMemConfig { 
# 1193
cudaSharedMemBankSizeDefault, 
# 1194
cudaSharedMemBankSizeFourByte, 
# 1195
cudaSharedMemBankSizeEightByte
# 1196
}; 
#endif
# 1201 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1201
enum cudaSharedCarveout { 
# 1202
cudaSharedmemCarveoutDefault = (-1), 
# 1203
cudaSharedmemCarveoutMaxShared = 100, 
# 1204
cudaSharedmemCarveoutMaxL1 = 0
# 1205
}; 
#endif
# 1210 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1210
enum cudaComputeMode { 
# 1212
cudaComputeModeDefault, 
# 1213
cudaComputeModeExclusive, 
# 1214
cudaComputeModeProhibited, 
# 1215
cudaComputeModeExclusiveProcess
# 1216
}; 
#endif
# 1221 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1221
enum cudaLimit { 
# 1223
cudaLimitStackSize, 
# 1224
cudaLimitPrintfFifoSize, 
# 1225
cudaLimitMallocHeapSize, 
# 1226
cudaLimitDevRuntimeSyncDepth, 
# 1227
cudaLimitDevRuntimePendingLaunchCount
# 1228
}; 
#endif
# 1233 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1233
enum cudaMemoryAdvise { 
# 1235
cudaMemAdviseSetReadMostly = 1, 
# 1236
cudaMemAdviseUnsetReadMostly, 
# 1237
cudaMemAdviseSetPreferredLocation, 
# 1238
cudaMemAdviseUnsetPreferredLocation, 
# 1239
cudaMemAdviseSetAccessedBy, 
# 1240
cudaMemAdviseUnsetAccessedBy
# 1241
}; 
#endif
# 1246 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1246
enum cudaMemRangeAttribute { 
# 1248
cudaMemRangeAttributeReadMostly = 1, 
# 1249
cudaMemRangeAttributePreferredLocation, 
# 1250
cudaMemRangeAttributeAccessedBy, 
# 1251
cudaMemRangeAttributeLastPrefetchLocation
# 1252
}; 
#endif
# 1257 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1257
enum cudaOutputMode { 
# 1259
cudaKeyValuePair, 
# 1260
cudaCSV
# 1261
}; 
#endif
# 1266 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1266
enum cudaDeviceAttr { 
# 1268
cudaDevAttrMaxThreadsPerBlock = 1, 
# 1269
cudaDevAttrMaxBlockDimX, 
# 1270
cudaDevAttrMaxBlockDimY, 
# 1271
cudaDevAttrMaxBlockDimZ, 
# 1272
cudaDevAttrMaxGridDimX, 
# 1273
cudaDevAttrMaxGridDimY, 
# 1274
cudaDevAttrMaxGridDimZ, 
# 1275
cudaDevAttrMaxSharedMemoryPerBlock, 
# 1276
cudaDevAttrTotalConstantMemory, 
# 1277
cudaDevAttrWarpSize, 
# 1278
cudaDevAttrMaxPitch, 
# 1279
cudaDevAttrMaxRegistersPerBlock, 
# 1280
cudaDevAttrClockRate, 
# 1281
cudaDevAttrTextureAlignment, 
# 1282
cudaDevAttrGpuOverlap, 
# 1283
cudaDevAttrMultiProcessorCount, 
# 1284
cudaDevAttrKernelExecTimeout, 
# 1285
cudaDevAttrIntegrated, 
# 1286
cudaDevAttrCanMapHostMemory, 
# 1287
cudaDevAttrComputeMode, 
# 1288
cudaDevAttrMaxTexture1DWidth, 
# 1289
cudaDevAttrMaxTexture2DWidth, 
# 1290
cudaDevAttrMaxTexture2DHeight, 
# 1291
cudaDevAttrMaxTexture3DWidth, 
# 1292
cudaDevAttrMaxTexture3DHeight, 
# 1293
cudaDevAttrMaxTexture3DDepth, 
# 1294
cudaDevAttrMaxTexture2DLayeredWidth, 
# 1295
cudaDevAttrMaxTexture2DLayeredHeight, 
# 1296
cudaDevAttrMaxTexture2DLayeredLayers, 
# 1297
cudaDevAttrSurfaceAlignment, 
# 1298
cudaDevAttrConcurrentKernels, 
# 1299
cudaDevAttrEccEnabled, 
# 1300
cudaDevAttrPciBusId, 
# 1301
cudaDevAttrPciDeviceId, 
# 1302
cudaDevAttrTccDriver, 
# 1303
cudaDevAttrMemoryClockRate, 
# 1304
cudaDevAttrGlobalMemoryBusWidth, 
# 1305
cudaDevAttrL2CacheSize, 
# 1306
cudaDevAttrMaxThreadsPerMultiProcessor, 
# 1307
cudaDevAttrAsyncEngineCount, 
# 1308
cudaDevAttrUnifiedAddressing, 
# 1309
cudaDevAttrMaxTexture1DLayeredWidth, 
# 1310
cudaDevAttrMaxTexture1DLayeredLayers, 
# 1311
cudaDevAttrMaxTexture2DGatherWidth = 45, 
# 1312
cudaDevAttrMaxTexture2DGatherHeight, 
# 1313
cudaDevAttrMaxTexture3DWidthAlt, 
# 1314
cudaDevAttrMaxTexture3DHeightAlt, 
# 1315
cudaDevAttrMaxTexture3DDepthAlt, 
# 1316
cudaDevAttrPciDomainId, 
# 1317
cudaDevAttrTexturePitchAlignment, 
# 1318
cudaDevAttrMaxTextureCubemapWidth, 
# 1319
cudaDevAttrMaxTextureCubemapLayeredWidth, 
# 1320
cudaDevAttrMaxTextureCubemapLayeredLayers, 
# 1321
cudaDevAttrMaxSurface1DWidth, 
# 1322
cudaDevAttrMaxSurface2DWidth, 
# 1323
cudaDevAttrMaxSurface2DHeight, 
# 1324
cudaDevAttrMaxSurface3DWidth, 
# 1325
cudaDevAttrMaxSurface3DHeight, 
# 1326
cudaDevAttrMaxSurface3DDepth, 
# 1327
cudaDevAttrMaxSurface1DLayeredWidth, 
# 1328
cudaDevAttrMaxSurface1DLayeredLayers, 
# 1329
cudaDevAttrMaxSurface2DLayeredWidth, 
# 1330
cudaDevAttrMaxSurface2DLayeredHeight, 
# 1331
cudaDevAttrMaxSurface2DLayeredLayers, 
# 1332
cudaDevAttrMaxSurfaceCubemapWidth, 
# 1333
cudaDevAttrMaxSurfaceCubemapLayeredWidth, 
# 1334
cudaDevAttrMaxSurfaceCubemapLayeredLayers, 
# 1335
cudaDevAttrMaxTexture1DLinearWidth, 
# 1336
cudaDevAttrMaxTexture2DLinearWidth, 
# 1337
cudaDevAttrMaxTexture2DLinearHeight, 
# 1338
cudaDevAttrMaxTexture2DLinearPitch, 
# 1339
cudaDevAttrMaxTexture2DMipmappedWidth, 
# 1340
cudaDevAttrMaxTexture2DMipmappedHeight, 
# 1341
cudaDevAttrComputeCapabilityMajor, 
# 1342
cudaDevAttrComputeCapabilityMinor, 
# 1343
cudaDevAttrMaxTexture1DMipmappedWidth, 
# 1344
cudaDevAttrStreamPrioritiesSupported, 
# 1345
cudaDevAttrGlobalL1CacheSupported, 
# 1346
cudaDevAttrLocalL1CacheSupported, 
# 1347
cudaDevAttrMaxSharedMemoryPerMultiprocessor, 
# 1348
cudaDevAttrMaxRegistersPerMultiprocessor, 
# 1349
cudaDevAttrManagedMemory, 
# 1350
cudaDevAttrIsMultiGpuBoard, 
# 1351
cudaDevAttrMultiGpuBoardGroupID, 
# 1352
cudaDevAttrHostNativeAtomicSupported, 
# 1353
cudaDevAttrSingleToDoublePrecisionPerfRatio, 
# 1354
cudaDevAttrPageableMemoryAccess, 
# 1355
cudaDevAttrConcurrentManagedAccess, 
# 1356
cudaDevAttrComputePreemptionSupported, 
# 1357
cudaDevAttrCanUseHostPointerForRegisteredMem, 
# 1358
cudaDevAttrReserved92, 
# 1359
cudaDevAttrReserved93, 
# 1360
cudaDevAttrReserved94, 
# 1361
cudaDevAttrCooperativeLaunch, 
# 1362
cudaDevAttrCooperativeMultiDeviceLaunch, 
# 1363
cudaDevAttrMaxSharedMemoryPerBlockOptin
# 1364
}; 
#endif
# 1370 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1370
enum cudaDeviceP2PAttr { 
# 1371
cudaDevP2PAttrPerformanceRank = 1, 
# 1372
cudaDevP2PAttrAccessSupported, 
# 1373
cudaDevP2PAttrNativeAtomicSupported
# 1374
}; 
#endif
# 1378 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1378
struct cudaDeviceProp { 
# 1380
char name[256]; 
# 1381
size_t totalGlobalMem; 
# 1382
size_t sharedMemPerBlock; 
# 1383
int regsPerBlock; 
# 1384
int warpSize; 
# 1385
size_t memPitch; 
# 1386
int maxThreadsPerBlock; 
# 1387
int maxThreadsDim[3]; 
# 1388
int maxGridSize[3]; 
# 1389
int clockRate; 
# 1390
size_t totalConstMem; 
# 1391
int major; 
# 1392
int minor; 
# 1393
size_t textureAlignment; 
# 1394
size_t texturePitchAlignment; 
# 1395
int deviceOverlap; 
# 1396
int multiProcessorCount; 
# 1397
int kernelExecTimeoutEnabled; 
# 1398
int integrated; 
# 1399
int canMapHostMemory; 
# 1400
int computeMode; 
# 1401
int maxTexture1D; 
# 1402
int maxTexture1DMipmap; 
# 1403
int maxTexture1DLinear; 
# 1404
int maxTexture2D[2]; 
# 1405
int maxTexture2DMipmap[2]; 
# 1406
int maxTexture2DLinear[3]; 
# 1407
int maxTexture2DGather[2]; 
# 1408
int maxTexture3D[3]; 
# 1409
int maxTexture3DAlt[3]; 
# 1410
int maxTextureCubemap; 
# 1411
int maxTexture1DLayered[2]; 
# 1412
int maxTexture2DLayered[3]; 
# 1413
int maxTextureCubemapLayered[2]; 
# 1414
int maxSurface1D; 
# 1415
int maxSurface2D[2]; 
# 1416
int maxSurface3D[3]; 
# 1417
int maxSurface1DLayered[2]; 
# 1418
int maxSurface2DLayered[3]; 
# 1419
int maxSurfaceCubemap; 
# 1420
int maxSurfaceCubemapLayered[2]; 
# 1421
size_t surfaceAlignment; 
# 1422
int concurrentKernels; 
# 1423
int ECCEnabled; 
# 1424
int pciBusID; 
# 1425
int pciDeviceID; 
# 1426
int pciDomainID; 
# 1427
int tccDriver; 
# 1428
int asyncEngineCount; 
# 1429
int unifiedAddressing; 
# 1430
int memoryClockRate; 
# 1431
int memoryBusWidth; 
# 1432
int l2CacheSize; 
# 1433
int maxThreadsPerMultiProcessor; 
# 1434
int streamPrioritiesSupported; 
# 1435
int globalL1CacheSupported; 
# 1436
int localL1CacheSupported; 
# 1437
size_t sharedMemPerMultiprocessor; 
# 1438
int regsPerMultiprocessor; 
# 1439
int managedMemory; 
# 1440
int isMultiGpuBoard; 
# 1441
int multiGpuBoardGroupID; 
# 1442
int hostNativeAtomicSupported; 
# 1443
int singleToDoublePrecisionPerfRatio; 
# 1444
int pageableMemoryAccess; 
# 1445
int concurrentManagedAccess; 
# 1446
int computePreemptionSupported; 
# 1447
int canUseHostPointerForRegisteredMem; 
# 1448
int cooperativeLaunch; 
# 1449
int cooperativeMultiDeviceLaunch; 
# 1450
size_t sharedMemPerBlockOptin; 
# 1451
}; 
#endif
# 1539 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 1536
struct cudaIpcEventHandle_st { 
# 1538
char reserved[64]; 
# 1539
} cudaIpcEventHandle_t; 
#endif
# 1547 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef 
# 1544
struct cudaIpcMemHandle_st { 
# 1546
char reserved[64]; 
# 1547
} cudaIpcMemHandle_t; 
#endif
# 1558 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaError 
# 1558
cudaError_t; 
#endif
# 1563 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUstream_st *
# 1563
cudaStream_t; 
#endif
# 1568 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUevent_st *
# 1568
cudaEvent_t; 
#endif
# 1573 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaGraphicsResource *
# 1573
cudaGraphicsResource_t; 
#endif
# 1578 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef struct CUuuid_st 
# 1578
cudaUUID_t; 
#endif
# 1583 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
typedef cudaOutputMode 
# 1583
cudaOutputMode_t; 
#endif
# 1588 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1588
enum cudaCGScope { 
# 1589
cudaCGScopeInvalid, 
# 1590
cudaCGScopeGrid, 
# 1591
cudaCGScopeMultiGrid
# 1592
}; 
#endif
# 1597 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_types.h"
#if 0
# 1597
struct cudaLaunchParams { 
# 1599
void *func; 
# 1600
dim3 gridDim; 
# 1601
dim3 blockDim; 
# 1602
void **args; 
# 1603
size_t sharedMem; 
# 1604
cudaStream_t stream; 
# 1605
}; 
#endif
# 84 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 84
enum cudaSurfaceBoundaryMode { 
# 86
cudaBoundaryModeZero, 
# 87
cudaBoundaryModeClamp, 
# 88
cudaBoundaryModeTrap
# 89
}; 
#endif
# 94 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 94
enum cudaSurfaceFormatMode { 
# 96
cudaFormatModeForced, 
# 97
cudaFormatModeAuto
# 98
}; 
#endif
# 103 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
# 103
struct surfaceReference { 
# 108
cudaChannelFormatDesc channelDesc; 
# 109
}; 
#endif
# 114 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_types.h"
#if 0
typedef unsigned long long 
# 114
cudaSurfaceObject_t; 
#endif
# 84 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 84
enum cudaTextureAddressMode { 
# 86
cudaAddressModeWrap, 
# 87
cudaAddressModeClamp, 
# 88
cudaAddressModeMirror, 
# 89
cudaAddressModeBorder
# 90
}; 
#endif
# 95 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 95
enum cudaTextureFilterMode { 
# 97
cudaFilterModePoint, 
# 98
cudaFilterModeLinear
# 99
}; 
#endif
# 104 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 104
enum cudaTextureReadMode { 
# 106
cudaReadModeElementType, 
# 107
cudaReadModeNormalizedFloat
# 108
}; 
#endif
# 113 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 113
struct textureReference { 
# 118
int normalized; 
# 122
cudaTextureFilterMode filterMode; 
# 126
cudaTextureAddressMode addressMode[3]; 
# 130
cudaChannelFormatDesc channelDesc; 
# 134
int sRGB; 
# 138
unsigned maxAnisotropy; 
# 142
cudaTextureFilterMode mipmapFilterMode; 
# 146
float mipmapLevelBias; 
# 150
float minMipmapLevelClamp; 
# 154
float maxMipmapLevelClamp; 
# 155
int __cudaReserved[15]; 
# 156
}; 
#endif
# 161 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
# 161
struct cudaTextureDesc { 
# 166
cudaTextureAddressMode addressMode[3]; 
# 170
cudaTextureFilterMode filterMode; 
# 174
cudaTextureReadMode readMode; 
# 178
int sRGB; 
# 182
float borderColor[4]; 
# 186
int normalizedCoords; 
# 190
unsigned maxAnisotropy; 
# 194
cudaTextureFilterMode mipmapFilterMode; 
# 198
float mipmapLevelBias; 
# 202
float minMipmapLevelClamp; 
# 206
float maxMipmapLevelClamp; 
# 207
}; 
#endif
# 212 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_types.h"
#if 0
typedef unsigned long long 
# 212
cudaTextureObject_t; 
#endif
# 70 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/library_types.h"
typedef 
# 54
enum cudaDataType_t { 
# 56
CUDA_R_16F = 2, 
# 57
CUDA_C_16F = 6, 
# 58
CUDA_R_32F = 0, 
# 59
CUDA_C_32F = 4, 
# 60
CUDA_R_64F = 1, 
# 61
CUDA_C_64F = 5, 
# 62
CUDA_R_8I = 3, 
# 63
CUDA_C_8I = 7, 
# 64
CUDA_R_8U, 
# 65
CUDA_C_8U, 
# 66
CUDA_R_32I, 
# 67
CUDA_C_32I, 
# 68
CUDA_R_32U, 
# 69
CUDA_C_32U
# 70
} cudaDataType; 
# 78
typedef 
# 73
enum libraryPropertyType_t { 
# 75
MAJOR_VERSION, 
# 76
MINOR_VERSION, 
# 77
PATCH_LEVEL
# 78
} libraryPropertyType; 
# 121 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
extern "C" {
# 123
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 124
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 125
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 126
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 127
extern cudaError_t cudaDeviceSynchronize(); 
# 128
extern cudaError_t cudaGetLastError(); 
# 129
extern cudaError_t cudaPeekAtLastError(); 
# 130
extern const char *cudaGetErrorString(cudaError_t error); 
# 131
extern const char *cudaGetErrorName(cudaError_t error); 
# 132
extern cudaError_t cudaGetDeviceCount(int * count); 
# 133
extern cudaError_t cudaGetDevice(int * device); 
# 134
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 135
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 136
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 137
__attribute__((unused)) extern cudaError_t cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 138
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 139
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream); 
# 140
__attribute__((unused)) extern cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream); 
# 141
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 142
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 143
extern cudaError_t cudaFree(void * devPtr); 
# 144
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 145
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 146
__attribute__((unused)) extern cudaError_t cudaMemcpyAsync_ptsz(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream); 
# 147
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 148
__attribute__((unused)) extern cudaError_t cudaMemcpy2DAsync_ptsz(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream); 
# 149
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 150
__attribute__((unused)) extern cudaError_t cudaMemcpy3DAsync_ptsz(const cudaMemcpy3DParms * p, cudaStream_t stream); 
# 151
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 152
__attribute__((unused)) extern cudaError_t cudaMemsetAsync_ptsz(void * devPtr, int value, size_t count, cudaStream_t stream); 
# 153
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 154
__attribute__((unused)) extern cudaError_t cudaMemset2DAsync_ptsz(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream); 
# 155
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 156
__attribute__((unused)) extern cudaError_t cudaMemset3DAsync_ptsz(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream); 
# 157
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBuffer(size_t alignment, size_t size); 
# 206 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern void *cudaGetParameterBufferV2(void * func, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize); 
# 207
__attribute__((unused)) extern cudaError_t cudaLaunchDevice_ptsz(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 208
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2_ptsz(void * parameterBuffer, cudaStream_t stream); 
# 226 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_device_runtime_api.h"
__attribute__((unused)) extern cudaError_t cudaLaunchDevice(void * func, void * parameterBuffer, dim3 gridDimension, dim3 blockDimension, unsigned sharedMemSize, cudaStream_t stream); 
# 227
__attribute__((unused)) extern cudaError_t cudaLaunchDeviceV2(void * parameterBuffer, cudaStream_t stream); 
# 230
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize); 
# 231
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 233
__attribute__((unused)) extern unsigned long long cudaCGGetIntrinsicHandle(cudaCGScope scope); 
# 234
__attribute__((unused)) extern cudaError_t cudaCGSynchronize(unsigned long long handle, unsigned flags); 
# 235
__attribute__((unused)) extern cudaError_t cudaCGGetSize(unsigned * numThreads, unsigned * numGrids, unsigned long long handle); 
# 236
__attribute__((unused)) extern cudaError_t cudaCGGetRank(unsigned * threadRank, unsigned * gridRank, unsigned long long handle); 
# 237
}
# 239
template< class T> static inline cudaError_t cudaMalloc(T ** devPtr, size_t size); 
# 240
template< class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, T * entry); 
# 241
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize); 
# 242
template< class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, T func, int blockSize, size_t dynamicSmemSize, unsigned flags); 
# 218 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern "C" {
# 251 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceReset(); 
# 270 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSynchronize(); 
# 347 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value); 
# 378 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetLimit(size_t * pValue, cudaLimit limit); 
# 410 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 446 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetStreamPriorityRange(int * leastPriority, int * greatestPriority); 
# 489 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig); 
# 519 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig); 
# 562 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config); 
# 587 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetByPCIBusId(int * device, const char * pciBusId); 
# 615 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetPCIBusId(char * pciBusId, int len, int device); 
# 660 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t * handle, cudaEvent_t event); 
# 698 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenEventHandle(cudaEvent_t * event, cudaIpcEventHandle_t handle); 
# 739 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t * handle, void * devPtr); 
# 792 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcOpenMemHandle(void ** devPtr, cudaIpcMemHandle_t handle, unsigned flags); 
# 825 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaIpcCloseMemHandle(void * devPtr); 
# 865 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadExit(); 
# 889 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadSynchronize(); 
# 936 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadSetLimit(cudaLimit limit, size_t value); 
# 967 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadGetLimit(size_t * pValue, cudaLimit limit); 
# 1002 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadGetCacheConfig(cudaFuncCache * pCacheConfig); 
# 1048 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig); 
# 1104 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetLastError(); 
# 1147 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaPeekAtLastError(); 
# 1163 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern const char *cudaGetErrorName(cudaError_t error); 
# 1179 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern const char *cudaGetErrorString(cudaError_t error); 
# 1210 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceCount(int * count); 
# 1470 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device); 
# 1647 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetAttribute(int * value, cudaDeviceAttr attr, int device); 
# 1683 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceGetP2PAttribute(int * value, cudaDeviceP2PAttr attr, int srcDevice, int dstDevice); 
# 1702 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaChooseDevice(int * device, const cudaDeviceProp * prop); 
# 1737 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDevice(int device); 
# 1755 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDevice(int * device); 
# 1784 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetValidDevices(int * device_arr, int len); 
# 1847 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDeviceFlags(unsigned flags); 
# 1890 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetDeviceFlags(unsigned * flags); 
# 1928 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreate(cudaStream_t * pStream); 
# 1958 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned flags); 
# 2002 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamCreateWithPriority(cudaStream_t * pStream, unsigned flags, int priority); 
# 2027 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetPriority(cudaStream_t hStream, int * priority); 
# 2050 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamGetFlags(cudaStream_t hStream, unsigned * flags); 
# 2079 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamDestroy(cudaStream_t stream); 
# 2113 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags); 
# 2127 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void * userData); 
# 2185 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void * userData, unsigned flags); 
# 2207 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamSynchronize(cudaStream_t stream); 
# 2230 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamQuery(cudaStream_t stream); 
# 2302 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void * devPtr, size_t length = 0, unsigned flags = 4); 
# 2339 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreate(cudaEvent_t * event); 
# 2374 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned flags); 
# 2406 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0); 
# 2439 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventQuery(cudaEvent_t event); 
# 2472 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventSynchronize(cudaEvent_t event); 
# 2498 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventDestroy(cudaEvent_t event); 
# 2540 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end); 
# 2603 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 2658 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernel(const void * func, dim3 gridDim, dim3 blockDim, void ** args, size_t sharedMem, cudaStream_t stream); 
# 2755 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams * launchParamsList, unsigned numDevices, unsigned flags = 0); 
# 2804 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetCacheConfig(const void * func, cudaFuncCache cacheConfig); 
# 2859 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetSharedMemConfig(const void * func, cudaSharedMemConfig config); 
# 2894 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncGetAttributes(cudaFuncAttributes * attr, const void * func); 
# 2930 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFuncSetAttribute(const void * func, cudaFuncAttribute attr, int value); 
# 2954 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDoubleForDevice(double * d); 
# 2978 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetDoubleForHost(double * d); 
# 3033 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize); 
# 3077 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int * numBlocks, const void * func, int blockSize, size_t dynamicSMemSize, unsigned flags); 
# 3127 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0); 
# 3156 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaSetupArgument(const void * arg, size_t size, size_t offset); 
# 3197 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaLaunch(const void * func); 
# 3316 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocManaged(void ** devPtr, size_t size, unsigned flags = 1); 
# 3344 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc(void ** devPtr, size_t size); 
# 3375 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocHost(void ** ptr, size_t size); 
# 3416 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocPitch(void ** devPtr, size_t * pitch, size_t width, size_t height); 
# 3460 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, size_t width, size_t height = 0, unsigned flags = 0); 
# 3488 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFree(void * devPtr); 
# 3510 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeHost(void * ptr); 
# 3533 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeArray(cudaArray_t array); 
# 3556 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray); 
# 3617 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostAlloc(void ** pHost, size_t size, unsigned flags); 
# 3698 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostRegister(void * ptr, size_t size, unsigned flags); 
# 3719 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostUnregister(void * ptr); 
# 3762 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetDevicePointer(void ** pDevice, void * pHost, unsigned flags); 
# 3782 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaHostGetFlags(unsigned * pFlags, void * pHost); 
# 3819 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3D(cudaPitchedPtr * pitchedDevPtr, cudaExtent extent); 
# 3956 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMalloc3DArray(cudaArray_t * array, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned flags = 0); 
# 4093 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMallocMipmappedArray(cudaMipmappedArray_t * mipmappedArray, const cudaChannelFormatDesc * desc, cudaExtent extent, unsigned numLevels, unsigned flags = 0); 
# 4120 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetMipmappedArrayLevel(cudaArray_t * levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned level); 
# 4223 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p); 
# 4252 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeer(const cudaMemcpy3DPeerParms * p); 
# 4368 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms * p, cudaStream_t stream = 0); 
# 4392 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms * p, cudaStream_t stream = 0); 
# 4413 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemGetInfo(size_t * free, size_t * total); 
# 4437 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaArrayGetInfo(cudaChannelFormatDesc * desc, cudaExtent * extent, unsigned * flags, cudaArray_t array); 
# 4478 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy(void * dst, const void * src, size_t count, cudaMemcpyKind kind); 
# 4511 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count); 
# 4550 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind); 
# 4588 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromArray(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind); 
# 4627 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 4673 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2D(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 4720 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind); 
# 4767 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArray(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind); 
# 4812 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice); 
# 4853 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbol(const void * symbol, const void * src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice); 
# 4894 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbol(void * dst, const void * symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost); 
# 4948 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 4981 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyPeerAsync(void * dst, int dstDevice, const void * src, int srcDevice, size_t count, cudaStream_t stream = 0); 
# 5028 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5074 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromArrayAsync(void * dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5134 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DAsync(void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5189 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void * src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5243 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpy2DFromArrayAsync(void * dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5292 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyToSymbolAsync(const void * symbol, const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5341 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemcpyFromSymbolAsync(void * dst, const void * symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0); 
# 5368 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset(void * devPtr, int value, size_t count); 
# 5400 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2D(void * devPtr, size_t pitch, int value, size_t width, size_t height); 
# 5442 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent); 
# 5476 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemsetAsync(void * devPtr, int value, size_t count, cudaStream_t stream = 0); 
# 5515 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset2DAsync(void * devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = 0); 
# 5564 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream = 0); 
# 5590 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolAddress(void ** devPtr, const void * symbol); 
# 5615 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSymbolSize(size_t * size, const void * symbol); 
# 5683 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemPrefetchAsync(const void * devPtr, size_t count, int dstDevice, cudaStream_t stream = 0); 
# 5770 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemAdvise(const void * devPtr, size_t count, cudaMemoryAdvise advice, int device); 
# 5827 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttribute(void * data, size_t dataSize, cudaMemRangeAttribute attribute, const void * devPtr, size_t count); 
# 5864 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaMemRangeGetAttributes(void ** data, size_t * dataSizes, cudaMemRangeAttribute * attributes, size_t numAttributes, const void * devPtr, size_t count); 
# 6018 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attributes, const void * ptr); 
# 6057 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceCanAccessPeer(int * canAccessPeer, int device, int peerDevice); 
# 6097 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned flags); 
# 6117 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDeviceDisablePeerAccess(int peerDevice); 
# 6178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource); 
# 6211 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource, unsigned flags); 
# 6248 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 6281 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t * resources, cudaStream_t stream = 0); 
# 6311 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedPointer(void ** devPtr, size_t * size, cudaGraphicsResource_t resource); 
# 6347 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t * array, cudaGraphicsResource_t resource, unsigned arrayIndex, unsigned mipLevel); 
# 6374 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray_t * mipmappedArray, cudaGraphicsResource_t resource); 
# 6414 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc * desc, cudaArray_const_t array); 
# 6450 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f); 
# 6501 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTexture(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t size = ((2147483647) * 2U) + 1U); 
# 6556 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTexture2D(size_t * offset, const textureReference * texref, const void * devPtr, const cudaChannelFormatDesc * desc, size_t width, size_t height, size_t pitch); 
# 6590 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTextureToArray(const textureReference * texref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 6626 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindTextureToMipmappedArray(const textureReference * texref, cudaMipmappedArray_const_t mipmappedArray, const cudaChannelFormatDesc * desc); 
# 6647 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaUnbindTexture(const textureReference * texref); 
# 6672 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureAlignmentOffset(size_t * offset, const textureReference * texref); 
# 6698 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureReference(const textureReference ** texref, const void * symbol); 
# 6739 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaBindSurfaceToArray(const surfaceReference * surfref, cudaArray_const_t array, const cudaChannelFormatDesc * desc); 
# 6760 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceReference(const surfaceReference ** surfref, const void * symbol); 
# 6988 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateTextureObject(cudaTextureObject_t * pTexObject, const cudaResourceDesc * pResDesc, const cudaTextureDesc * pTexDesc, const cudaResourceViewDesc * pResViewDesc); 
# 7005 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject); 
# 7023 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc * pResDesc, cudaTextureObject_t texObject); 
# 7041 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc * pTexDesc, cudaTextureObject_t texObject); 
# 7060 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc * pResViewDesc, cudaTextureObject_t texObject); 
# 7101 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t * pSurfObject, const cudaResourceDesc * pResDesc); 
# 7118 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject); 
# 7135 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc * pResDesc, cudaSurfaceObject_t surfObject); 
# 7164 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaDriverGetVersion(int * driverVersion); 
# 7183 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
extern cudaError_t cudaRuntimeGetVersion(int * runtimeVersion); 
# 7188
extern cudaError_t cudaGetExportTable(const void ** ppExportTable, const cudaUUID_t * pExportTableId); 
# 7416 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime_api.h"
}
# 107 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template< class T> inline cudaChannelFormatDesc cudaCreateChannelDesc() 
# 108
{ 
# 109
return cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindNone); 
# 110
} 
# 112
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf() 
# 113
{ 
# 114
int e = (((int)sizeof(unsigned short)) * 8); 
# 116
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 117
} 
# 119
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf1() 
# 120
{ 
# 121
int e = (((int)sizeof(unsigned short)) * 8); 
# 123
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 124
} 
# 126
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf2() 
# 127
{ 
# 128
int e = (((int)sizeof(unsigned short)) * 8); 
# 130
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 131
} 
# 133
static inline cudaChannelFormatDesc cudaCreateChannelDescHalf4() 
# 134
{ 
# 135
int e = (((int)sizeof(unsigned short)) * 8); 
# 137
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 138
} 
# 140
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char> () 
# 141
{ 
# 142
int e = (((int)sizeof(char)) * 8); 
# 147
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 149
} 
# 151
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< signed char> () 
# 152
{ 
# 153
int e = (((int)sizeof(signed char)) * 8); 
# 155
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 156
} 
# 158
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned char> () 
# 159
{ 
# 160
int e = (((int)sizeof(unsigned char)) * 8); 
# 162
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 163
} 
# 165
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char1> () 
# 166
{ 
# 167
int e = (((int)sizeof(signed char)) * 8); 
# 169
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 170
} 
# 172
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar1> () 
# 173
{ 
# 174
int e = (((int)sizeof(unsigned char)) * 8); 
# 176
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 177
} 
# 179
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char2> () 
# 180
{ 
# 181
int e = (((int)sizeof(signed char)) * 8); 
# 183
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 184
} 
# 186
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar2> () 
# 187
{ 
# 188
int e = (((int)sizeof(unsigned char)) * 8); 
# 190
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 191
} 
# 193
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< char4> () 
# 194
{ 
# 195
int e = (((int)sizeof(signed char)) * 8); 
# 197
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 198
} 
# 200
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uchar4> () 
# 201
{ 
# 202
int e = (((int)sizeof(unsigned char)) * 8); 
# 204
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 205
} 
# 207
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short> () 
# 208
{ 
# 209
int e = (((int)sizeof(short)) * 8); 
# 211
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 212
} 
# 214
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned short> () 
# 215
{ 
# 216
int e = (((int)sizeof(unsigned short)) * 8); 
# 218
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 219
} 
# 221
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short1> () 
# 222
{ 
# 223
int e = (((int)sizeof(short)) * 8); 
# 225
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 226
} 
# 228
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort1> () 
# 229
{ 
# 230
int e = (((int)sizeof(unsigned short)) * 8); 
# 232
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 233
} 
# 235
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short2> () 
# 236
{ 
# 237
int e = (((int)sizeof(short)) * 8); 
# 239
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 240
} 
# 242
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort2> () 
# 243
{ 
# 244
int e = (((int)sizeof(unsigned short)) * 8); 
# 246
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 247
} 
# 249
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< short4> () 
# 250
{ 
# 251
int e = (((int)sizeof(short)) * 8); 
# 253
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 254
} 
# 256
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< ushort4> () 
# 257
{ 
# 258
int e = (((int)sizeof(unsigned short)) * 8); 
# 260
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 261
} 
# 263
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int> () 
# 264
{ 
# 265
int e = (((int)sizeof(int)) * 8); 
# 267
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 268
} 
# 270
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< unsigned> () 
# 271
{ 
# 272
int e = (((int)sizeof(unsigned)) * 8); 
# 274
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 275
} 
# 277
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int1> () 
# 278
{ 
# 279
int e = (((int)sizeof(int)) * 8); 
# 281
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindSigned); 
# 282
} 
# 284
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint1> () 
# 285
{ 
# 286
int e = (((int)sizeof(unsigned)) * 8); 
# 288
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned); 
# 289
} 
# 291
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int2> () 
# 292
{ 
# 293
int e = (((int)sizeof(int)) * 8); 
# 295
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindSigned); 
# 296
} 
# 298
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint2> () 
# 299
{ 
# 300
int e = (((int)sizeof(unsigned)) * 8); 
# 302
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindUnsigned); 
# 303
} 
# 305
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< int4> () 
# 306
{ 
# 307
int e = (((int)sizeof(int)) * 8); 
# 309
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindSigned); 
# 310
} 
# 312
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< uint4> () 
# 313
{ 
# 314
int e = (((int)sizeof(unsigned)) * 8); 
# 316
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindUnsigned); 
# 317
} 
# 379 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/channel_descriptor.h"
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float> () 
# 380
{ 
# 381
int e = (((int)sizeof(float)) * 8); 
# 383
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 384
} 
# 386
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float1> () 
# 387
{ 
# 388
int e = (((int)sizeof(float)) * 8); 
# 390
return cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindFloat); 
# 391
} 
# 393
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float2> () 
# 394
{ 
# 395
int e = (((int)sizeof(float)) * 8); 
# 397
return cudaCreateChannelDesc(e, e, 0, 0, cudaChannelFormatKindFloat); 
# 398
} 
# 400
template<> inline cudaChannelFormatDesc cudaCreateChannelDesc< float4> () 
# 401
{ 
# 402
int e = (((int)sizeof(float)) * 8); 
# 404
return cudaCreateChannelDesc(e, e, e, e, cudaChannelFormatKindFloat); 
# 405
} 
# 79 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz) 
# 80
{ 
# 81
cudaPitchedPtr s; 
# 83
(s.ptr) = d; 
# 84
(s.pitch) = p; 
# 85
(s.xsize) = xsz; 
# 86
(s.ysize) = ysz; 
# 88
return s; 
# 89
} 
# 106 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaPos make_cudaPos(size_t x, size_t y, size_t z) 
# 107
{ 
# 108
cudaPos p; 
# 110
(p.x) = x; 
# 111
(p.y) = y; 
# 112
(p.z) = z; 
# 114
return p; 
# 115
} 
# 132 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/driver_functions.h"
static inline cudaExtent make_cudaExtent(size_t w, size_t h, size_t d) 
# 133
{ 
# 134
cudaExtent e; 
# 136
(e.width) = w; 
# 137
(e.height) = h; 
# 138
(e.depth) = d; 
# 140
return e; 
# 141
} 
# 75 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_functions.h"
static inline char1 make_char1(signed char x); 
# 77
static inline uchar1 make_uchar1(unsigned char x); 
# 79
static inline char2 make_char2(signed char x, signed char y); 
# 81
static inline uchar2 make_uchar2(unsigned char x, unsigned char y); 
# 83
static inline char3 make_char3(signed char x, signed char y, signed char z); 
# 85
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z); 
# 87
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w); 
# 89
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w); 
# 91
static inline short1 make_short1(short x); 
# 93
static inline ushort1 make_ushort1(unsigned short x); 
# 95
static inline short2 make_short2(short x, short y); 
# 97
static inline ushort2 make_ushort2(unsigned short x, unsigned short y); 
# 99
static inline short3 make_short3(short x, short y, short z); 
# 101
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z); 
# 103
static inline short4 make_short4(short x, short y, short z, short w); 
# 105
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w); 
# 107
static inline int1 make_int1(int x); 
# 109
static inline uint1 make_uint1(unsigned x); 
# 111
static inline int2 make_int2(int x, int y); 
# 113
static inline uint2 make_uint2(unsigned x, unsigned y); 
# 115
static inline int3 make_int3(int x, int y, int z); 
# 117
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z); 
# 119
static inline int4 make_int4(int x, int y, int z, int w); 
# 121
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w); 
# 123
static inline long1 make_long1(long x); 
# 125
static inline ulong1 make_ulong1(unsigned long x); 
# 127
static inline long2 make_long2(long x, long y); 
# 129
static inline ulong2 make_ulong2(unsigned long x, unsigned long y); 
# 131
static inline long3 make_long3(long x, long y, long z); 
# 133
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z); 
# 135
static inline long4 make_long4(long x, long y, long z, long w); 
# 137
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w); 
# 139
static inline float1 make_float1(float x); 
# 141
static inline float2 make_float2(float x, float y); 
# 143
static inline float3 make_float3(float x, float y, float z); 
# 145
static inline float4 make_float4(float x, float y, float z, float w); 
# 147
static inline longlong1 make_longlong1(long long x); 
# 149
static inline ulonglong1 make_ulonglong1(unsigned long long x); 
# 151
static inline longlong2 make_longlong2(long long x, long long y); 
# 153
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y); 
# 155
static inline longlong3 make_longlong3(long long x, long long y, long long z); 
# 157
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z); 
# 159
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w); 
# 161
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w); 
# 163
static inline double1 make_double1(double x); 
# 165
static inline double2 make_double2(double x, double y); 
# 167
static inline double3 make_double3(double x, double y, double z); 
# 169
static inline double4 make_double4(double x, double y, double z, double w); 
# 75 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/vector_functions.hpp"
static inline char1 make_char1(signed char x) 
# 76
{ 
# 77
char1 t; (t.x) = x; return t; 
# 78
} 
# 80
static inline uchar1 make_uchar1(unsigned char x) 
# 81
{ 
# 82
uchar1 t; (t.x) = x; return t; 
# 83
} 
# 85
static inline char2 make_char2(signed char x, signed char y) 
# 86
{ 
# 87
char2 t; (t.x) = x; (t.y) = y; return t; 
# 88
} 
# 90
static inline uchar2 make_uchar2(unsigned char x, unsigned char y) 
# 91
{ 
# 92
uchar2 t; (t.x) = x; (t.y) = y; return t; 
# 93
} 
# 95
static inline char3 make_char3(signed char x, signed char y, signed char z) 
# 96
{ 
# 97
char3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 98
} 
# 100
static inline uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) 
# 101
{ 
# 102
uchar3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 103
} 
# 105
static inline char4 make_char4(signed char x, signed char y, signed char z, signed char w) 
# 106
{ 
# 107
char4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 108
} 
# 110
static inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) 
# 111
{ 
# 112
uchar4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 113
} 
# 115
static inline short1 make_short1(short x) 
# 116
{ 
# 117
short1 t; (t.x) = x; return t; 
# 118
} 
# 120
static inline ushort1 make_ushort1(unsigned short x) 
# 121
{ 
# 122
ushort1 t; (t.x) = x; return t; 
# 123
} 
# 125
static inline short2 make_short2(short x, short y) 
# 126
{ 
# 127
short2 t; (t.x) = x; (t.y) = y; return t; 
# 128
} 
# 130
static inline ushort2 make_ushort2(unsigned short x, unsigned short y) 
# 131
{ 
# 132
ushort2 t; (t.x) = x; (t.y) = y; return t; 
# 133
} 
# 135
static inline short3 make_short3(short x, short y, short z) 
# 136
{ 
# 137
short3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 138
} 
# 140
static inline ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) 
# 141
{ 
# 142
ushort3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 143
} 
# 145
static inline short4 make_short4(short x, short y, short z, short w) 
# 146
{ 
# 147
short4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 148
} 
# 150
static inline ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) 
# 151
{ 
# 152
ushort4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 153
} 
# 155
static inline int1 make_int1(int x) 
# 156
{ 
# 157
int1 t; (t.x) = x; return t; 
# 158
} 
# 160
static inline uint1 make_uint1(unsigned x) 
# 161
{ 
# 162
uint1 t; (t.x) = x; return t; 
# 163
} 
# 165
static inline int2 make_int2(int x, int y) 
# 166
{ 
# 167
int2 t; (t.x) = x; (t.y) = y; return t; 
# 168
} 
# 170
static inline uint2 make_uint2(unsigned x, unsigned y) 
# 171
{ 
# 172
uint2 t; (t.x) = x; (t.y) = y; return t; 
# 173
} 
# 175
static inline int3 make_int3(int x, int y, int z) 
# 176
{ 
# 177
int3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 178
} 
# 180
static inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) 
# 181
{ 
# 182
uint3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 183
} 
# 185
static inline int4 make_int4(int x, int y, int z, int w) 
# 186
{ 
# 187
int4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 188
} 
# 190
static inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) 
# 191
{ 
# 192
uint4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 193
} 
# 195
static inline long1 make_long1(long x) 
# 196
{ 
# 197
long1 t; (t.x) = x; return t; 
# 198
} 
# 200
static inline ulong1 make_ulong1(unsigned long x) 
# 201
{ 
# 202
ulong1 t; (t.x) = x; return t; 
# 203
} 
# 205
static inline long2 make_long2(long x, long y) 
# 206
{ 
# 207
long2 t; (t.x) = x; (t.y) = y; return t; 
# 208
} 
# 210
static inline ulong2 make_ulong2(unsigned long x, unsigned long y) 
# 211
{ 
# 212
ulong2 t; (t.x) = x; (t.y) = y; return t; 
# 213
} 
# 215
static inline long3 make_long3(long x, long y, long z) 
# 216
{ 
# 217
long3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 218
} 
# 220
static inline ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) 
# 221
{ 
# 222
ulong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 223
} 
# 225
static inline long4 make_long4(long x, long y, long z, long w) 
# 226
{ 
# 227
long4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 228
} 
# 230
static inline ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) 
# 231
{ 
# 232
ulong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 233
} 
# 235
static inline float1 make_float1(float x) 
# 236
{ 
# 237
float1 t; (t.x) = x; return t; 
# 238
} 
# 240
static inline float2 make_float2(float x, float y) 
# 241
{ 
# 242
float2 t; (t.x) = x; (t.y) = y; return t; 
# 243
} 
# 245
static inline float3 make_float3(float x, float y, float z) 
# 246
{ 
# 247
float3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 248
} 
# 250
static inline float4 make_float4(float x, float y, float z, float w) 
# 251
{ 
# 252
float4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 253
} 
# 255
static inline longlong1 make_longlong1(long long x) 
# 256
{ 
# 257
longlong1 t; (t.x) = x; return t; 
# 258
} 
# 260
static inline ulonglong1 make_ulonglong1(unsigned long long x) 
# 261
{ 
# 262
ulonglong1 t; (t.x) = x; return t; 
# 263
} 
# 265
static inline longlong2 make_longlong2(long long x, long long y) 
# 266
{ 
# 267
longlong2 t; (t.x) = x; (t.y) = y; return t; 
# 268
} 
# 270
static inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) 
# 271
{ 
# 272
ulonglong2 t; (t.x) = x; (t.y) = y; return t; 
# 273
} 
# 275
static inline longlong3 make_longlong3(long long x, long long y, long long z) 
# 276
{ 
# 277
longlong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 278
} 
# 280
static inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) 
# 281
{ 
# 282
ulonglong3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 283
} 
# 285
static inline longlong4 make_longlong4(long long x, long long y, long long z, long long w) 
# 286
{ 
# 287
longlong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 288
} 
# 290
static inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) 
# 291
{ 
# 292
ulonglong4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 293
} 
# 295
static inline double1 make_double1(double x) 
# 296
{ 
# 297
double1 t; (t.x) = x; return t; 
# 298
} 
# 300
static inline double2 make_double2(double x, double y) 
# 301
{ 
# 302
double2 t; (t.x) = x; (t.y) = y; return t; 
# 303
} 
# 305
static inline double3 make_double3(double x, double y, double z) 
# 306
{ 
# 307
double3 t; (t.x) = x; (t.y) = y; (t.z) = z; return t; 
# 308
} 
# 310
static inline double4 make_double4(double x, double y, double z, double w) 
# 311
{ 
# 312
double4 t; (t.x) = x; (t.y) = y; (t.z) = z; (t.w) = w; return t; 
# 313
} 
# 27 "/usr/include/string.h" 3
extern "C" {
# 42 "/usr/include/string.h" 3
extern void *memcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 43
 __attribute((__nonnull__(1, 2))); 
# 46
extern void *memmove(void * __dest, const void * __src, size_t __n) throw()
# 47
 __attribute((__nonnull__(1, 2))); 
# 54
extern void *memccpy(void *__restrict__ __dest, const void *__restrict__ __src, int __c, size_t __n) throw()
# 56
 __attribute((__nonnull__(1, 2))); 
# 62
extern void *memset(void * __s, int __c, size_t __n) throw() __attribute((__nonnull__(1))); 
# 65
extern int memcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 66
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 70
extern "C++" {
# 72
extern void *memchr(void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 73
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 74
extern const void *memchr(const void * __s, int __c, size_t __n) throw() __asm__("memchr")
# 75
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 90 "/usr/include/string.h" 3
}
# 101
extern "C++" void *rawmemchr(void * __s, int __c) throw() __asm__("rawmemchr")
# 102
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 103
extern "C++" const void *rawmemchr(const void * __s, int __c) throw() __asm__("rawmemchr")
# 104
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 112
extern "C++" void *memrchr(void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 113
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 114
extern "C++" const void *memrchr(const void * __s, int __c, size_t __n) throw() __asm__("memrchr")
# 115
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 125
extern char *strcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 126
 __attribute((__nonnull__(1, 2))); 
# 128
extern char *strncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 130
 __attribute((__nonnull__(1, 2))); 
# 133
extern char *strcat(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 134
 __attribute((__nonnull__(1, 2))); 
# 136
extern char *strncat(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 137
 __attribute((__nonnull__(1, 2))); 
# 140
extern int strcmp(const char * __s1, const char * __s2) throw()
# 141
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 143
extern int strncmp(const char * __s1, const char * __s2, size_t __n) throw()
# 144
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 147
extern int strcoll(const char * __s1, const char * __s2) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 150
extern size_t strxfrm(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 152
 __attribute((__nonnull__(2))); 
# 39 "/usr/include/xlocale.h" 3
typedef 
# 27
struct __locale_struct { 
# 30
struct __locale_data *__locales[13]; 
# 33
const unsigned short *__ctype_b; 
# 34
const int *__ctype_tolower; 
# 35
const int *__ctype_toupper; 
# 38
const char *__names[13]; 
# 39
} *__locale_t; 
# 42
typedef __locale_t locale_t; 
# 162 "/usr/include/string.h" 3
extern int strcoll_l(const char * __s1, const char * __s2, __locale_t __l) throw()
# 163
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 165
extern size_t strxfrm_l(char * __dest, const char * __src, size_t __n, __locale_t __l) throw()
# 166
 __attribute((__nonnull__(2, 4))); 
# 171
extern char *strdup(const char * __s) throw()
# 172
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 179
extern char *strndup(const char * __string, size_t __n) throw()
# 180
 __attribute((__malloc__)) __attribute((__nonnull__(1))); 
# 209 "/usr/include/string.h" 3
extern "C++" {
# 211
extern char *strchr(char * __s, int __c) throw() __asm__("strchr")
# 212
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 213
extern const char *strchr(const char * __s, int __c) throw() __asm__("strchr")
# 214
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 229 "/usr/include/string.h" 3
}
# 236
extern "C++" {
# 238
extern char *strrchr(char * __s, int __c) throw() __asm__("strrchr")
# 239
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 240
extern const char *strrchr(const char * __s, int __c) throw() __asm__("strrchr")
# 241
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 256 "/usr/include/string.h" 3
}
# 267
extern "C++" char *strchrnul(char * __s, int __c) throw() __asm__("strchrnul")
# 268
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 269
extern "C++" const char *strchrnul(const char * __s, int __c) throw() __asm__("strchrnul")
# 270
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 280
extern size_t strcspn(const char * __s, const char * __reject) throw()
# 281
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 284
extern size_t strspn(const char * __s, const char * __accept) throw()
# 285
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 288
extern "C++" {
# 290
extern char *strpbrk(char * __s, const char * __accept) throw() __asm__("strpbrk")
# 291
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 292
extern const char *strpbrk(const char * __s, const char * __accept) throw() __asm__("strpbrk")
# 293
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 308 "/usr/include/string.h" 3
}
# 315
extern "C++" {
# 317
extern char *strstr(char * __haystack, const char * __needle) throw() __asm__("strstr")
# 318
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 319
extern const char *strstr(const char * __haystack, const char * __needle) throw() __asm__("strstr")
# 320
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 335 "/usr/include/string.h" 3
}
# 343
extern char *strtok(char *__restrict__ __s, const char *__restrict__ __delim) throw()
# 344
 __attribute((__nonnull__(2))); 
# 349
extern char *__strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 352
 __attribute((__nonnull__(2, 3))); 
# 354
extern char *strtok_r(char *__restrict__ __s, const char *__restrict__ __delim, char **__restrict__ __save_ptr) throw()
# 356
 __attribute((__nonnull__(2, 3))); 
# 362
extern "C++" char *strcasestr(char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 363
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 364
extern "C++" const char *strcasestr(const char * __haystack, const char * __needle) throw() __asm__("strcasestr")
# 366
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 377 "/usr/include/string.h" 3
extern void *memmem(const void * __haystack, size_t __haystacklen, const void * __needle, size_t __needlelen) throw()
# 379
 __attribute((__pure__)) __attribute((__nonnull__(1, 3))); 
# 383
extern void *__mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 385
 __attribute((__nonnull__(1, 2))); 
# 386
extern void *mempcpy(void *__restrict__ __dest, const void *__restrict__ __src, size_t __n) throw()
# 388
 __attribute((__nonnull__(1, 2))); 
# 394
extern size_t strlen(const char * __s) throw()
# 395
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 401
extern size_t strnlen(const char * __string, size_t __maxlen) throw()
# 402
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 408
extern char *strerror(int __errnum) throw(); 
# 433 "/usr/include/string.h" 3
extern char *strerror_r(int __errnum, char * __buf, size_t __buflen) throw()
# 434
 __attribute((__nonnull__(2))); 
# 440
extern char *strerror_l(int __errnum, __locale_t __l) throw(); 
# 446
extern void __bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 450
extern void bcopy(const void * __src, void * __dest, size_t __n) throw()
# 451
 __attribute((__nonnull__(1, 2))); 
# 454
extern void bzero(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 457
extern int bcmp(const void * __s1, const void * __s2, size_t __n) throw()
# 458
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 462
extern "C++" {
# 464
extern char *index(char * __s, int __c) throw() __asm__("index")
# 465
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 466
extern const char *index(const char * __s, int __c) throw() __asm__("index")
# 467
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 482 "/usr/include/string.h" 3
}
# 490
extern "C++" {
# 492
extern char *rindex(char * __s, int __c) throw() __asm__("rindex")
# 493
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 494
extern const char *rindex(const char * __s, int __c) throw() __asm__("rindex")
# 495
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 510 "/usr/include/string.h" 3
}
# 518
extern int ffs(int __i) throw() __attribute((const)); 
# 523
extern int ffsl(long __l) throw() __attribute((const)); 
# 524
__extension__ extern int ffsll(long long __ll) throw()
# 525
 __attribute((const)); 
# 529
extern int strcasecmp(const char * __s1, const char * __s2) throw()
# 530
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 533
extern int strncasecmp(const char * __s1, const char * __s2, size_t __n) throw()
# 534
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 540
extern int strcasecmp_l(const char * __s1, const char * __s2, __locale_t __loc) throw()
# 542
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 3))); 
# 544
extern int strncasecmp_l(const char * __s1, const char * __s2, size_t __n, __locale_t __loc) throw()
# 546
 __attribute((__pure__)) __attribute((__nonnull__(1, 2, 4))); 
# 552
extern char *strsep(char **__restrict__ __stringp, const char *__restrict__ __delim) throw()
# 554
 __attribute((__nonnull__(1, 2))); 
# 559
extern char *strsignal(int __sig) throw(); 
# 562
extern char *__stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 563
 __attribute((__nonnull__(1, 2))); 
# 564
extern char *stpcpy(char *__restrict__ __dest, const char *__restrict__ __src) throw()
# 565
 __attribute((__nonnull__(1, 2))); 
# 569
extern char *__stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 571
 __attribute((__nonnull__(1, 2))); 
# 572
extern char *stpncpy(char *__restrict__ __dest, const char *__restrict__ __src, size_t __n) throw()
# 574
 __attribute((__nonnull__(1, 2))); 
# 579
extern int strverscmp(const char * __s1, const char * __s2) throw()
# 580
 __attribute((__pure__)) __attribute((__nonnull__(1, 2))); 
# 583
extern char *strfry(char * __string) throw() __attribute((__nonnull__(1))); 
# 586
extern void *memfrob(void * __s, size_t __n) throw() __attribute((__nonnull__(1))); 
# 594
extern "C++" char *basename(char * __filename) throw() __asm__("basename")
# 595
 __attribute((__nonnull__(1))); 
# 596
extern "C++" const char *basename(const char * __filename) throw() __asm__("basename")
# 597
 __attribute((__nonnull__(1))); 
# 658 "/usr/include/string.h" 3
}
# 29 "/usr/include/time.h" 3
extern "C" {
# 30 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef unsigned char __u_char; 
# 31
typedef unsigned short __u_short; 
# 32
typedef unsigned __u_int; 
# 33
typedef unsigned long __u_long; 
# 36
typedef signed char __int8_t; 
# 37
typedef unsigned char __uint8_t; 
# 38
typedef signed short __int16_t; 
# 39
typedef unsigned short __uint16_t; 
# 40
typedef signed int __int32_t; 
# 41
typedef unsigned __uint32_t; 
# 43
typedef signed long __int64_t; 
# 44
typedef unsigned long __uint64_t; 
# 52
typedef long __quad_t; 
# 53
typedef unsigned long __u_quad_t; 
# 124 "/usr/include/x86_64-linux-gnu/bits/types.h" 3
typedef unsigned long __dev_t; 
# 125
typedef unsigned __uid_t; 
# 126
typedef unsigned __gid_t; 
# 127
typedef unsigned long __ino_t; 
# 128
typedef unsigned long __ino64_t; 
# 129
typedef unsigned __mode_t; 
# 130
typedef unsigned long __nlink_t; 
# 131
typedef long __off_t; 
# 132
typedef long __off64_t; 
# 133
typedef int __pid_t; 
# 134
typedef struct { int __val[2]; } __fsid_t; 
# 135
typedef long __clock_t; 
# 136
typedef unsigned long __rlim_t; 
# 137
typedef unsigned long __rlim64_t; 
# 138
typedef unsigned __id_t; 
# 139
typedef long __time_t; 
# 140
typedef unsigned __useconds_t; 
# 141
typedef long __suseconds_t; 
# 143
typedef int __daddr_t; 
# 144
typedef int __key_t; 
# 147
typedef int __clockid_t; 
# 150
typedef void *__timer_t; 
# 153
typedef long __blksize_t; 
# 158
typedef long __blkcnt_t; 
# 159
typedef long __blkcnt64_t; 
# 162
typedef unsigned long __fsblkcnt_t; 
# 163
typedef unsigned long __fsblkcnt64_t; 
# 166
typedef unsigned long __fsfilcnt_t; 
# 167
typedef unsigned long __fsfilcnt64_t; 
# 170
typedef long __fsword_t; 
# 172
typedef long __ssize_t; 
# 175
typedef long __syscall_slong_t; 
# 177
typedef unsigned long __syscall_ulong_t; 
# 181
typedef __off64_t __loff_t; 
# 182
typedef __quad_t *__qaddr_t; 
# 183
typedef char *__caddr_t; 
# 186
typedef long __intptr_t; 
# 189
typedef unsigned __socklen_t; 
# 30 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
struct timeval { 
# 32
__time_t tv_sec; 
# 33
__suseconds_t tv_usec; 
# 34
}; 
# 25 "/usr/include/x86_64-linux-gnu/bits/timex.h" 3
struct timex { 
# 27
unsigned modes; 
# 28
__syscall_slong_t offset; 
# 29
__syscall_slong_t freq; 
# 30
__syscall_slong_t maxerror; 
# 31
__syscall_slong_t esterror; 
# 32
int status; 
# 33
__syscall_slong_t constant; 
# 34
__syscall_slong_t precision; 
# 35
__syscall_slong_t tolerance; 
# 36
timeval time; 
# 37
__syscall_slong_t tick; 
# 38
__syscall_slong_t ppsfreq; 
# 39
__syscall_slong_t jitter; 
# 40
int shift; 
# 41
__syscall_slong_t stabil; 
# 42
__syscall_slong_t jitcnt; 
# 43
__syscall_slong_t calcnt; 
# 44
__syscall_slong_t errcnt; 
# 45
__syscall_slong_t stbcnt; 
# 47
int tai; 
# 50
int:32; int:32; int:32; int:32; 
# 51
int:32; int:32; int:32; int:32; 
# 52
int:32; int:32; int:32; 
# 53
}; 
# 90 "/usr/include/x86_64-linux-gnu/bits/time.h" 3
extern "C" {
# 93
extern int clock_adjtime(__clockid_t __clock_id, timex * __utx) throw(); 
# 95
}
# 59 "/usr/include/time.h" 3
typedef __clock_t clock_t; 
# 75 "/usr/include/time.h" 3
typedef __time_t time_t; 
# 91 "/usr/include/time.h" 3
typedef __clockid_t clockid_t; 
# 103 "/usr/include/time.h" 3
typedef __timer_t timer_t; 
# 120 "/usr/include/time.h" 3
struct timespec { 
# 122
__time_t tv_sec; 
# 123
__syscall_slong_t tv_nsec; 
# 124
}; 
# 133
struct tm { 
# 135
int tm_sec; 
# 136
int tm_min; 
# 137
int tm_hour; 
# 138
int tm_mday; 
# 139
int tm_mon; 
# 140
int tm_year; 
# 141
int tm_wday; 
# 142
int tm_yday; 
# 143
int tm_isdst; 
# 146
long tm_gmtoff; 
# 147
const char *tm_zone; 
# 152
}; 
# 161
struct itimerspec { 
# 163
timespec it_interval; 
# 164
timespec it_value; 
# 165
}; 
# 168
struct sigevent; 
# 174
typedef __pid_t pid_t; 
# 189 "/usr/include/time.h" 3
extern clock_t clock() throw(); 
# 192
extern time_t time(time_t * __timer) throw(); 
# 195
extern double difftime(time_t __time1, time_t __time0) throw()
# 196
 __attribute((const)); 
# 199
extern time_t mktime(tm * __tp) throw(); 
# 205
extern size_t strftime(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp) throw(); 
# 213
extern char *strptime(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp) throw(); 
# 223
extern size_t strftime_l(char *__restrict__ __s, size_t __maxsize, const char *__restrict__ __format, const tm *__restrict__ __tp, __locale_t __loc) throw(); 
# 230
extern char *strptime_l(const char *__restrict__ __s, const char *__restrict__ __fmt, tm * __tp, __locale_t __loc) throw(); 
# 239
extern tm *gmtime(const time_t * __timer) throw(); 
# 243
extern tm *localtime(const time_t * __timer) throw(); 
# 249
extern tm *gmtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 254
extern tm *localtime_r(const time_t *__restrict__ __timer, tm *__restrict__ __tp) throw(); 
# 261
extern char *asctime(const tm * __tp) throw(); 
# 264
extern char *ctime(const time_t * __timer) throw(); 
# 272
extern char *asctime_r(const tm *__restrict__ __tp, char *__restrict__ __buf) throw(); 
# 276
extern char *ctime_r(const time_t *__restrict__ __timer, char *__restrict__ __buf) throw(); 
# 282
extern char *__tzname[2]; 
# 283
extern int __daylight; 
# 284
extern long __timezone; 
# 289
extern char *tzname[2]; 
# 293
extern void tzset() throw(); 
# 297
extern int daylight; 
# 298
extern long timezone; 
# 304
extern int stime(const time_t * __when) throw(); 
# 319 "/usr/include/time.h" 3
extern time_t timegm(tm * __tp) throw(); 
# 322
extern time_t timelocal(tm * __tp) throw(); 
# 325
extern int dysize(int __year) throw() __attribute((const)); 
# 334 "/usr/include/time.h" 3
extern int nanosleep(const timespec * __requested_time, timespec * __remaining); 
# 339
extern int clock_getres(clockid_t __clock_id, timespec * __res) throw(); 
# 342
extern int clock_gettime(clockid_t __clock_id, timespec * __tp) throw(); 
# 345
extern int clock_settime(clockid_t __clock_id, const timespec * __tp) throw(); 
# 353
extern int clock_nanosleep(clockid_t __clock_id, int __flags, const timespec * __req, timespec * __rem); 
# 358
extern int clock_getcpuclockid(pid_t __pid, clockid_t * __clock_id) throw(); 
# 363
extern int timer_create(clockid_t __clock_id, sigevent *__restrict__ __evp, timer_t *__restrict__ __timerid) throw(); 
# 368
extern int timer_delete(timer_t __timerid) throw(); 
# 371
extern int timer_settime(timer_t __timerid, int __flags, const itimerspec *__restrict__ __value, itimerspec *__restrict__ __ovalue) throw(); 
# 376
extern int timer_gettime(timer_t __timerid, itimerspec * __value) throw(); 
# 380
extern int timer_getoverrun(timer_t __timerid) throw(); 
# 386
extern int timespec_get(timespec * __ts, int __base) throw()
# 387
 __attribute((__nonnull__(1))); 
# 403 "/usr/include/time.h" 3
extern int getdate_err; 
# 412 "/usr/include/time.h" 3
extern tm *getdate(const char * __string); 
# 426 "/usr/include/time.h" 3
extern int getdate_r(const char *__restrict__ __string, tm *__restrict__ __resbufp); 
# 430
}
# 70 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern "C" {
# 73
extern clock_t clock() throw(); 
# 78 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern void *memset(void *, int, size_t) throw(); 
# 79 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/common_functions.h"
extern void *memcpy(void *, const void *, size_t) throw(); 
# 81 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/common_functions.h"
}
# 98 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern "C" {
# 168 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int abs(int) throw(); 
# 169 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long labs(long) throw(); 
# 170 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llabs(long long) throw(); 
# 220 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fabs(double x) throw(); 
# 261 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fabsf(float x) throw(); 
# 265 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline int min(int, int); 
# 267
extern inline unsigned umin(unsigned, unsigned); 
# 268
extern inline long long llmin(long long, long long); 
# 269
extern inline unsigned long long ullmin(unsigned long long, unsigned long long); 
# 290 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fminf(float x, float y) throw(); 
# 310 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmin(double x, double y) throw(); 
# 317 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline int max(int, int); 
# 319
extern inline unsigned umax(unsigned, unsigned); 
# 320
extern inline long long llmax(long long, long long); 
# 321
extern inline unsigned long long ullmax(unsigned long long, unsigned long long); 
# 342 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmaxf(float x, float y) throw(); 
# 362 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmax(double, double) throw(); 
# 406 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sin(double x) throw(); 
# 439 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cos(double x) throw(); 
# 458 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincos(double x, double * sptr, double * cptr) throw(); 
# 474 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincosf(float x, float * sptr, float * cptr) throw(); 
# 519 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tan(double x) throw(); 
# 588 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sqrt(double x) throw(); 
# 660 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rsqrt(double x); 
# 730 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rsqrtf(float x); 
# 786 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log2(double x) throw(); 
# 811 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp2(double x) throw(); 
# 836 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float exp2f(float x) throw(); 
# 863 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp10(double x) throw(); 
# 886 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float exp10f(float x) throw(); 
# 932 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double expm1(double x) throw(); 
# 977 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float expm1f(float x) throw(); 
# 1032 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log2f(float x) throw(); 
# 1086 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log10(double x) throw(); 
# 1157 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log(double x) throw(); 
# 1251 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double log1p(double x) throw(); 
# 1348 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log1pf(float x) throw(); 
# 1423 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double floor(double x) throw(); 
# 1462 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double exp(double x) throw(); 
# 1493 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cosh(double x) throw(); 
# 1523 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sinh(double x) throw(); 
# 1553 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tanh(double x) throw(); 
# 1588 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double acosh(double x) throw(); 
# 1626 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float acoshf(float x) throw(); 
# 1642 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double asinh(double x) throw(); 
# 1658 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float asinhf(float x) throw(); 
# 1712 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atanh(double x) throw(); 
# 1766 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atanhf(float x) throw(); 
# 1825 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double ldexp(double x, int exp) throw(); 
# 1881 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ldexpf(float x, int exp) throw(); 
# 1933 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double logb(double x) throw(); 
# 1988 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float logbf(float x) throw(); 
# 2018 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int ilogb(double x) throw(); 
# 2048 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int ilogbf(float x) throw(); 
# 2124 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double scalbn(double x, int n) throw(); 
# 2200 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float scalbnf(float x, int n) throw(); 
# 2276 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double scalbln(double x, long n) throw(); 
# 2352 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float scalblnf(float x, long n) throw(); 
# 2430 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double frexp(double x, int * nptr) throw(); 
# 2505 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float frexpf(float x, int * nptr) throw(); 
# 2519 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double round(double x) throw(); 
# 2536 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float roundf(float x) throw(); 
# 2554 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lround(double x) throw(); 
# 2572 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lroundf(float x) throw(); 
# 2590 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llround(double x) throw(); 
# 2608 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llroundf(float x) throw(); 
# 2660 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rintf(float x) throw(); 
# 2676 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lrint(double x) throw(); 
# 2692 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long lrintf(float x) throw(); 
# 2708 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llrint(double x) throw(); 
# 2724 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern long long llrintf(float x) throw(); 
# 2777 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nearbyint(double x) throw(); 
# 2830 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nearbyintf(float x) throw(); 
# 2892 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double ceil(double x) throw(); 
# 2904 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double trunc(double x) throw(); 
# 2919 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float truncf(float x) throw(); 
# 2945 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fdim(double x, double y) throw(); 
# 2971 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fdimf(float x, float y) throw(); 
# 3007 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atan2(double y, double x) throw(); 
# 3038 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double atan(double x) throw(); 
# 3061 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double acos(double x) throw(); 
# 3093 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double asin(double x) throw(); 
# 3139 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double hypot(double x, double y) throw(); 
# 3191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rhypot(double x, double y) throw(); 
# 3237 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float hypotf(float x, float y) throw(); 
# 3289 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rhypotf(float x, float y) throw(); 
# 3333 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double norm3d(double a, double b, double c) throw(); 
# 3384 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rnorm3d(double a, double b, double c) throw(); 
# 3433 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double norm4d(double a, double b, double c, double d) throw(); 
# 3489 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rnorm4d(double a, double b, double c, double d) throw(); 
# 3534 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double norm(int dim, const double * t) throw(); 
# 3585 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rnorm(int dim, const double * t) throw(); 
# 3637 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rnormf(int dim, const float * a) throw(); 
# 3681 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normf(int dim, const float * a) throw(); 
# 3726 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float norm3df(float a, float b, float c) throw(); 
# 3777 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rnorm3df(float a, float b, float c) throw(); 
# 3826 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float norm4df(float a, float b, float c, float d) throw(); 
# 3882 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rnorm4df(float a, float b, float c, float d) throw(); 
# 3969 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cbrt(double x) throw(); 
# 4055 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cbrtf(float x) throw(); 
# 4110 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double rcbrt(double x); 
# 4160 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float rcbrtf(float x); 
# 4220 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double sinpi(double x); 
# 4280 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinpif(float x); 
# 4332 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cospi(double x); 
# 4384 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cospif(float x); 
# 4414 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincospi(double x, double * sptr, double * cptr); 
# 4444 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern void sincospif(float x, float * sptr, float * cptr); 
# 4756 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double pow(double x, double y) throw(); 
# 4812 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double modf(double x, double * iptr) throw(); 
# 4871 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fmod(double x, double y) throw(); 
# 4957 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double remainder(double x, double y) throw(); 
# 5047 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float remainderf(float x, float y) throw(); 
# 5101 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double remquo(double x, double y, int * quo) throw(); 
# 5155 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float remquof(float x, float y, int * quo) throw(); 
# 5196 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double j0(double x) throw(); 
# 5238 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float j0f(float x) throw(); 
# 5299 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double j1(double x) throw(); 
# 5360 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float j1f(float x) throw(); 
# 5403 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double jn(int n, double x) throw(); 
# 5446 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float jnf(int n, float x) throw(); 
# 5498 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double y0(double x) throw(); 
# 5550 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float y0f(float x) throw(); 
# 5602 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double y1(double x) throw(); 
# 5654 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float y1f(float x) throw(); 
# 5707 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double yn(int n, double x) throw(); 
# 5760 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ynf(int n, float x) throw(); 
# 5787 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cyl_bessel_i0(double x) throw(); 
# 5813 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cyl_bessel_i0f(float x) throw(); 
# 5840 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double cyl_bessel_i1(double x) throw(); 
# 5866 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cyl_bessel_i1f(float x) throw(); 
# 5949 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erf(double x) throw(); 
# 6031 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erff(float x) throw(); 
# 6095 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfinv(double y); 
# 6152 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfinvf(float y); 
# 6191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfc(double x) throw(); 
# 6229 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcf(float x) throw(); 
# 6357 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double lgamma(double x) throw(); 
# 6420 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfcinv(double y); 
# 6476 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcinvf(float y); 
# 6534 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double normcdfinv(double y); 
# 6592 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normcdfinvf(float y); 
# 6635 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double normcdf(double y); 
# 6678 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float normcdff(float y); 
# 6753 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double erfcx(double x); 
# 6828 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float erfcxf(float x); 
# 6962 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float lgammaf(float x) throw(); 
# 7071 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double tgamma(double x) throw(); 
# 7180 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tgammaf(float x) throw(); 
# 7193 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double copysign(double x, double y) throw(); 
# 7206 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float copysignf(float x, float y) throw(); 
# 7243 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nextafter(double x, double y) throw(); 
# 7280 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nextafterf(float x, float y) throw(); 
# 7296 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double nan(const char * tagp) throw(); 
# 7312 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float nanf(const char * tagp) throw(); 
# 7319 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isinff(float) throw(); 
# 7320 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isnanf(float) throw(); 
# 7330 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finite(double) throw(); 
# 7331 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finitef(float) throw(); 
# 7332 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __signbit(double) throw(); 
# 7333 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isnan(double) throw(); 
# 7334 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isinf(double) throw(); 
# 7337 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __signbitf(float) throw(); 
# 7496 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern double fma(double x, double y, double z) throw(); 
# 7654 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmaf(float x, float y, float z) throw(); 
# 7665 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __signbitl(long double) throw(); 
# 7671 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __finitel(long double) throw(); 
# 7672 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isinfl(long double) throw(); 
# 7673 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern int __isnanl(long double) throw(); 
# 7723 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float acosf(float x) throw(); 
# 7763 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float asinf(float x) throw(); 
# 7803 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atanf(float x) throw(); 
# 7836 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float atan2f(float y, float x) throw(); 
# 7860 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float cosf(float x) throw(); 
# 7902 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinf(float x) throw(); 
# 7944 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tanf(float x) throw(); 
# 7968 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float coshf(float x) throw(); 
# 8009 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sinhf(float x) throw(); 
# 8039 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float tanhf(float x) throw(); 
# 8090 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float logf(float x) throw(); 
# 8140 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float expf(float x) throw(); 
# 8191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float log10f(float x) throw(); 
# 8246 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float modff(float x, float * iptr) throw(); 
# 8554 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float powf(float x, float y) throw(); 
# 8623 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float sqrtf(float x) throw(); 
# 8682 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float ceilf(float x) throw(); 
# 8754 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float floorf(float x) throw(); 
# 8813 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern float fmodf(float x, float y) throw(); 
# 8828 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 28 "/usr/include/math.h" 3
extern "C" {
# 28 "/usr/include/x86_64-linux-gnu/bits/mathdef.h" 3
typedef float float_t; 
# 29
typedef double double_t; 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern double acos(double __x) throw(); extern double __acos(double __x) throw(); 
# 56
extern double asin(double __x) throw(); extern double __asin(double __x) throw(); 
# 58
extern double atan(double __x) throw(); extern double __atan(double __x) throw(); 
# 60
extern double atan2(double __y, double __x) throw(); extern double __atan2(double __y, double __x) throw(); 
# 63
extern double cos(double __x) throw(); extern double __cos(double __x) throw(); 
# 65
extern double sin(double __x) throw(); extern double __sin(double __x) throw(); 
# 67
extern double tan(double __x) throw(); extern double __tan(double __x) throw(); 
# 72
extern double cosh(double __x) throw(); extern double __cosh(double __x) throw(); 
# 74
extern double sinh(double __x) throw(); extern double __sinh(double __x) throw(); 
# 76
extern double tanh(double __x) throw(); extern double __tanh(double __x) throw(); 
# 81
extern void sincos(double __x, double * __sinx, double * __cosx) throw(); extern void __sincos(double __x, double * __sinx, double * __cosx) throw(); 
# 88
extern double acosh(double __x) throw(); extern double __acosh(double __x) throw(); 
# 90
extern double asinh(double __x) throw(); extern double __asinh(double __x) throw(); 
# 92
extern double atanh(double __x) throw(); extern double __atanh(double __x) throw(); 
# 100
extern double exp(double __x) throw(); extern double __exp(double __x) throw(); 
# 103
extern double frexp(double __x, int * __exponent) throw(); extern double __frexp(double __x, int * __exponent) throw(); 
# 106
extern double ldexp(double __x, int __exponent) throw(); extern double __ldexp(double __x, int __exponent) throw(); 
# 109
extern double log(double __x) throw(); extern double __log(double __x) throw(); 
# 112
extern double log10(double __x) throw(); extern double __log10(double __x) throw(); 
# 115
extern double modf(double __x, double * __iptr) throw(); extern double __modf(double __x, double * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern double exp10(double __x) throw(); extern double __exp10(double __x) throw(); 
# 122
extern double pow10(double __x) throw(); extern double __pow10(double __x) throw(); 
# 128
extern double expm1(double __x) throw(); extern double __expm1(double __x) throw(); 
# 131
extern double log1p(double __x) throw(); extern double __log1p(double __x) throw(); 
# 134
extern double logb(double __x) throw(); extern double __logb(double __x) throw(); 
# 141
extern double exp2(double __x) throw(); extern double __exp2(double __x) throw(); 
# 144
extern double log2(double __x) throw(); extern double __log2(double __x) throw(); 
# 153
extern double pow(double __x, double __y) throw(); extern double __pow(double __x, double __y) throw(); 
# 156
extern double sqrt(double __x) throw(); extern double __sqrt(double __x) throw(); 
# 162
extern double hypot(double __x, double __y) throw(); extern double __hypot(double __x, double __y) throw(); 
# 169
extern double cbrt(double __x) throw(); extern double __cbrt(double __x) throw(); 
# 178
extern double ceil(double __x) throw() __attribute((const)); extern double __ceil(double __x) throw() __attribute((const)); 
# 181
extern double fabs(double __x) throw() __attribute((const)); extern double __fabs(double __x) throw() __attribute((const)); 
# 184
extern double floor(double __x) throw() __attribute((const)); extern double __floor(double __x) throw() __attribute((const)); 
# 187
extern double fmod(double __x, double __y) throw(); extern double __fmod(double __x, double __y) throw(); 
# 192
extern int __isinf(double __value) throw() __attribute((const)); 
# 195
extern int __finite(double __value) throw() __attribute((const)); 
# 204
extern inline int isinf(double __value) throw() __attribute((const)); 
# 208
extern int finite(double __value) throw() __attribute((const)); 
# 211
extern double drem(double __x, double __y) throw(); extern double __drem(double __x, double __y) throw(); 
# 215
extern double significand(double __x) throw(); extern double __significand(double __x) throw(); 
# 221
extern double copysign(double __x, double __y) throw() __attribute((const)); extern double __copysign(double __x, double __y) throw() __attribute((const)); 
# 228
extern double nan(const char * __tagb) throw() __attribute((const)); extern double __nan(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnan(double __value) throw() __attribute((const)); 
# 241
extern inline int isnan(double __value) throw() __attribute((const)); 
# 247
extern double j0(double) throw(); extern double __j0(double) throw(); 
# 248
extern double j1(double) throw(); extern double __j1(double) throw(); 
# 249
extern double jn(int, double) throw(); extern double __jn(int, double) throw(); 
# 250
extern double y0(double) throw(); extern double __y0(double) throw(); 
# 251
extern double y1(double) throw(); extern double __y1(double) throw(); 
# 252
extern double yn(int, double) throw(); extern double __yn(int, double) throw(); 
# 259
extern double erf(double) throw(); extern double __erf(double) throw(); 
# 260
extern double erfc(double) throw(); extern double __erfc(double) throw(); 
# 261
extern double lgamma(double) throw(); extern double __lgamma(double) throw(); 
# 268
extern double tgamma(double) throw(); extern double __tgamma(double) throw(); 
# 274
extern double gamma(double) throw(); extern double __gamma(double) throw(); 
# 281
extern double lgamma_r(double, int * __signgamp) throw(); extern double __lgamma_r(double, int * __signgamp) throw(); 
# 289
extern double rint(double __x) throw(); extern double __rint(double __x) throw(); 
# 292
extern double nextafter(double __x, double __y) throw() __attribute((const)); extern double __nextafter(double __x, double __y) throw() __attribute((const)); 
# 294
extern double nexttoward(double __x, long double __y) throw() __attribute((const)); extern double __nexttoward(double __x, long double __y) throw() __attribute((const)); 
# 298
extern double remainder(double __x, double __y) throw(); extern double __remainder(double __x, double __y) throw(); 
# 302
extern double scalbn(double __x, int __n) throw(); extern double __scalbn(double __x, int __n) throw(); 
# 306
extern int ilogb(double __x) throw(); extern int __ilogb(double __x) throw(); 
# 311
extern double scalbln(double __x, long __n) throw(); extern double __scalbln(double __x, long __n) throw(); 
# 315
extern double nearbyint(double __x) throw(); extern double __nearbyint(double __x) throw(); 
# 319
extern double round(double __x) throw() __attribute((const)); extern double __round(double __x) throw() __attribute((const)); 
# 323
extern double trunc(double __x) throw() __attribute((const)); extern double __trunc(double __x) throw() __attribute((const)); 
# 328
extern double remquo(double __x, double __y, int * __quo) throw(); extern double __remquo(double __x, double __y, int * __quo) throw(); 
# 335
extern long lrint(double __x) throw(); extern long __lrint(double __x) throw(); 
# 337
__extension__ extern long long llrint(double __x) throw(); extern long long __llrint(double __x) throw(); 
# 341
extern long lround(double __x) throw(); extern long __lround(double __x) throw(); 
# 343
__extension__ extern long long llround(double __x) throw(); extern long long __llround(double __x) throw(); 
# 347
extern double fdim(double __x, double __y) throw(); extern double __fdim(double __x, double __y) throw(); 
# 350
extern double fmax(double __x, double __y) throw() __attribute((const)); extern double __fmax(double __x, double __y) throw() __attribute((const)); 
# 353
extern double fmin(double __x, double __y) throw() __attribute((const)); extern double __fmin(double __x, double __y) throw() __attribute((const)); 
# 357
extern int __fpclassify(double __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbit(double __value) throw()
# 362
 __attribute((const)); 
# 366
extern double fma(double __x, double __y, double __z) throw(); extern double __fma(double __x, double __y, double __z) throw(); 
# 375
extern int __issignaling(double __value) throw()
# 376
 __attribute((const)); 
# 383
extern double scalb(double __x, double __n) throw(); extern double __scalb(double __x, double __n) throw(); 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern float acosf(float __x) throw(); extern float __acosf(float __x) throw(); 
# 56
extern float asinf(float __x) throw(); extern float __asinf(float __x) throw(); 
# 58
extern float atanf(float __x) throw(); extern float __atanf(float __x) throw(); 
# 60
extern float atan2f(float __y, float __x) throw(); extern float __atan2f(float __y, float __x) throw(); 
# 63
extern float cosf(float __x) throw(); 
# 65
extern float sinf(float __x) throw(); 
# 67
extern float tanf(float __x) throw(); 
# 72
extern float coshf(float __x) throw(); extern float __coshf(float __x) throw(); 
# 74
extern float sinhf(float __x) throw(); extern float __sinhf(float __x) throw(); 
# 76
extern float tanhf(float __x) throw(); extern float __tanhf(float __x) throw(); 
# 81
extern void sincosf(float __x, float * __sinx, float * __cosx) throw(); 
# 88
extern float acoshf(float __x) throw(); extern float __acoshf(float __x) throw(); 
# 90
extern float asinhf(float __x) throw(); extern float __asinhf(float __x) throw(); 
# 92
extern float atanhf(float __x) throw(); extern float __atanhf(float __x) throw(); 
# 100
extern float expf(float __x) throw(); 
# 103
extern float frexpf(float __x, int * __exponent) throw(); extern float __frexpf(float __x, int * __exponent) throw(); 
# 106
extern float ldexpf(float __x, int __exponent) throw(); extern float __ldexpf(float __x, int __exponent) throw(); 
# 109
extern float logf(float __x) throw(); 
# 112
extern float log10f(float __x) throw(); 
# 115
extern float modff(float __x, float * __iptr) throw(); extern float __modff(float __x, float * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern float exp10f(float __x) throw(); 
# 122
extern float pow10f(float __x) throw(); extern float __pow10f(float __x) throw(); 
# 128
extern float expm1f(float __x) throw(); extern float __expm1f(float __x) throw(); 
# 131
extern float log1pf(float __x) throw(); extern float __log1pf(float __x) throw(); 
# 134
extern float logbf(float __x) throw(); extern float __logbf(float __x) throw(); 
# 141
extern float exp2f(float __x) throw(); extern float __exp2f(float __x) throw(); 
# 144
extern float log2f(float __x) throw(); 
# 153
extern float powf(float __x, float __y) throw(); 
# 156
extern float sqrtf(float __x) throw(); extern float __sqrtf(float __x) throw(); 
# 162
extern float hypotf(float __x, float __y) throw(); extern float __hypotf(float __x, float __y) throw(); 
# 169
extern float cbrtf(float __x) throw(); extern float __cbrtf(float __x) throw(); 
# 178
extern float ceilf(float __x) throw() __attribute((const)); extern float __ceilf(float __x) throw() __attribute((const)); 
# 181
extern float fabsf(float __x) throw() __attribute((const)); extern float __fabsf(float __x) throw() __attribute((const)); 
# 184
extern float floorf(float __x) throw() __attribute((const)); extern float __floorf(float __x) throw() __attribute((const)); 
# 187
extern float fmodf(float __x, float __y) throw(); extern float __fmodf(float __x, float __y) throw(); 
# 192
extern int __isinff(float __value) throw() __attribute((const)); 
# 195
extern int __finitef(float __value) throw() __attribute((const)); 
# 204
extern int isinff(float __value) throw() __attribute((const)); 
# 208
extern int finitef(float __value) throw() __attribute((const)); 
# 211
extern float dremf(float __x, float __y) throw(); extern float __dremf(float __x, float __y) throw(); 
# 215
extern float significandf(float __x) throw(); extern float __significandf(float __x) throw(); 
# 221
extern float copysignf(float __x, float __y) throw() __attribute((const)); extern float __copysignf(float __x, float __y) throw() __attribute((const)); 
# 228
extern float nanf(const char * __tagb) throw() __attribute((const)); extern float __nanf(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnanf(float __value) throw() __attribute((const)); 
# 241
extern int isnanf(float __value) throw() __attribute((const)); 
# 247
extern float j0f(float) throw(); extern float __j0f(float) throw(); 
# 248
extern float j1f(float) throw(); extern float __j1f(float) throw(); 
# 249
extern float jnf(int, float) throw(); extern float __jnf(int, float) throw(); 
# 250
extern float y0f(float) throw(); extern float __y0f(float) throw(); 
# 251
extern float y1f(float) throw(); extern float __y1f(float) throw(); 
# 252
extern float ynf(int, float) throw(); extern float __ynf(int, float) throw(); 
# 259
extern float erff(float) throw(); extern float __erff(float) throw(); 
# 260
extern float erfcf(float) throw(); extern float __erfcf(float) throw(); 
# 261
extern float lgammaf(float) throw(); extern float __lgammaf(float) throw(); 
# 268
extern float tgammaf(float) throw(); extern float __tgammaf(float) throw(); 
# 274
extern float gammaf(float) throw(); extern float __gammaf(float) throw(); 
# 281
extern float lgammaf_r(float, int * __signgamp) throw(); extern float __lgammaf_r(float, int * __signgamp) throw(); 
# 289
extern float rintf(float __x) throw(); extern float __rintf(float __x) throw(); 
# 292
extern float nextafterf(float __x, float __y) throw() __attribute((const)); extern float __nextafterf(float __x, float __y) throw() __attribute((const)); 
# 294
extern float nexttowardf(float __x, long double __y) throw() __attribute((const)); extern float __nexttowardf(float __x, long double __y) throw() __attribute((const)); 
# 298
extern float remainderf(float __x, float __y) throw(); extern float __remainderf(float __x, float __y) throw(); 
# 302
extern float scalbnf(float __x, int __n) throw(); extern float __scalbnf(float __x, int __n) throw(); 
# 306
extern int ilogbf(float __x) throw(); extern int __ilogbf(float __x) throw(); 
# 311
extern float scalblnf(float __x, long __n) throw(); extern float __scalblnf(float __x, long __n) throw(); 
# 315
extern float nearbyintf(float __x) throw(); extern float __nearbyintf(float __x) throw(); 
# 319
extern float roundf(float __x) throw() __attribute((const)); extern float __roundf(float __x) throw() __attribute((const)); 
# 323
extern float truncf(float __x) throw() __attribute((const)); extern float __truncf(float __x) throw() __attribute((const)); 
# 328
extern float remquof(float __x, float __y, int * __quo) throw(); extern float __remquof(float __x, float __y, int * __quo) throw(); 
# 335
extern long lrintf(float __x) throw(); extern long __lrintf(float __x) throw(); 
# 337
__extension__ extern long long llrintf(float __x) throw(); extern long long __llrintf(float __x) throw(); 
# 341
extern long lroundf(float __x) throw(); extern long __lroundf(float __x) throw(); 
# 343
__extension__ extern long long llroundf(float __x) throw(); extern long long __llroundf(float __x) throw(); 
# 347
extern float fdimf(float __x, float __y) throw(); extern float __fdimf(float __x, float __y) throw(); 
# 350
extern float fmaxf(float __x, float __y) throw() __attribute((const)); extern float __fmaxf(float __x, float __y) throw() __attribute((const)); 
# 353
extern float fminf(float __x, float __y) throw() __attribute((const)); extern float __fminf(float __x, float __y) throw() __attribute((const)); 
# 357
extern int __fpclassifyf(float __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbitf(float __value) throw()
# 362
 __attribute((const)); 
# 366
extern float fmaf(float __x, float __y, float __z) throw(); extern float __fmaf(float __x, float __y, float __z) throw(); 
# 375
extern int __issignalingf(float __value) throw()
# 376
 __attribute((const)); 
# 383
extern float scalbf(float __x, float __n) throw(); extern float __scalbf(float __x, float __n) throw(); 
# 54 "/usr/include/x86_64-linux-gnu/bits/mathcalls.h" 3
extern long double acosl(long double __x) throw(); extern long double __acosl(long double __x) throw(); 
# 56
extern long double asinl(long double __x) throw(); extern long double __asinl(long double __x) throw(); 
# 58
extern long double atanl(long double __x) throw(); extern long double __atanl(long double __x) throw(); 
# 60
extern long double atan2l(long double __y, long double __x) throw(); extern long double __atan2l(long double __y, long double __x) throw(); 
# 63
extern long double cosl(long double __x) throw(); extern long double __cosl(long double __x) throw(); 
# 65
extern long double sinl(long double __x) throw(); extern long double __sinl(long double __x) throw(); 
# 67
extern long double tanl(long double __x) throw(); extern long double __tanl(long double __x) throw(); 
# 72
extern long double coshl(long double __x) throw(); extern long double __coshl(long double __x) throw(); 
# 74
extern long double sinhl(long double __x) throw(); extern long double __sinhl(long double __x) throw(); 
# 76
extern long double tanhl(long double __x) throw(); extern long double __tanhl(long double __x) throw(); 
# 81
extern void sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); extern void __sincosl(long double __x, long double * __sinx, long double * __cosx) throw(); 
# 88
extern long double acoshl(long double __x) throw(); extern long double __acoshl(long double __x) throw(); 
# 90
extern long double asinhl(long double __x) throw(); extern long double __asinhl(long double __x) throw(); 
# 92
extern long double atanhl(long double __x) throw(); extern long double __atanhl(long double __x) throw(); 
# 100
extern long double expl(long double __x) throw(); extern long double __expl(long double __x) throw(); 
# 103
extern long double frexpl(long double __x, int * __exponent) throw(); extern long double __frexpl(long double __x, int * __exponent) throw(); 
# 106
extern long double ldexpl(long double __x, int __exponent) throw(); extern long double __ldexpl(long double __x, int __exponent) throw(); 
# 109
extern long double logl(long double __x) throw(); extern long double __logl(long double __x) throw(); 
# 112
extern long double log10l(long double __x) throw(); extern long double __log10l(long double __x) throw(); 
# 115
extern long double modfl(long double __x, long double * __iptr) throw(); extern long double __modfl(long double __x, long double * __iptr) throw() __attribute((__nonnull__(2))); 
# 120
extern long double exp10l(long double __x) throw(); extern long double __exp10l(long double __x) throw(); 
# 122
extern long double pow10l(long double __x) throw(); extern long double __pow10l(long double __x) throw(); 
# 128
extern long double expm1l(long double __x) throw(); extern long double __expm1l(long double __x) throw(); 
# 131
extern long double log1pl(long double __x) throw(); extern long double __log1pl(long double __x) throw(); 
# 134
extern long double logbl(long double __x) throw(); extern long double __logbl(long double __x) throw(); 
# 141
extern long double exp2l(long double __x) throw(); extern long double __exp2l(long double __x) throw(); 
# 144
extern long double log2l(long double __x) throw(); extern long double __log2l(long double __x) throw(); 
# 153
extern long double powl(long double __x, long double __y) throw(); extern long double __powl(long double __x, long double __y) throw(); 
# 156
extern long double sqrtl(long double __x) throw(); extern long double __sqrtl(long double __x) throw(); 
# 162
extern long double hypotl(long double __x, long double __y) throw(); extern long double __hypotl(long double __x, long double __y) throw(); 
# 169
extern long double cbrtl(long double __x) throw(); extern long double __cbrtl(long double __x) throw(); 
# 178
extern long double ceill(long double __x) throw() __attribute((const)); extern long double __ceill(long double __x) throw() __attribute((const)); 
# 181
extern long double fabsl(long double __x) throw() __attribute((const)); extern long double __fabsl(long double __x) throw() __attribute((const)); 
# 184
extern long double floorl(long double __x) throw() __attribute((const)); extern long double __floorl(long double __x) throw() __attribute((const)); 
# 187
extern long double fmodl(long double __x, long double __y) throw(); extern long double __fmodl(long double __x, long double __y) throw(); 
# 192
extern int __isinfl(long double __value) throw() __attribute((const)); 
# 195
extern int __finitel(long double __value) throw() __attribute((const)); 
# 204
extern int isinfl(long double __value) throw() __attribute((const)); 
# 208
extern int finitel(long double __value) throw() __attribute((const)); 
# 211
extern long double dreml(long double __x, long double __y) throw(); extern long double __dreml(long double __x, long double __y) throw(); 
# 215
extern long double significandl(long double __x) throw(); extern long double __significandl(long double __x) throw(); 
# 221
extern long double copysignl(long double __x, long double __y) throw() __attribute((const)); extern long double __copysignl(long double __x, long double __y) throw() __attribute((const)); 
# 228
extern long double nanl(const char * __tagb) throw() __attribute((const)); extern long double __nanl(const char * __tagb) throw() __attribute((const)); 
# 234
extern int __isnanl(long double __value) throw() __attribute((const)); 
# 241
extern int isnanl(long double __value) throw() __attribute((const)); 
# 247
extern long double j0l(long double) throw(); extern long double __j0l(long double) throw(); 
# 248
extern long double j1l(long double) throw(); extern long double __j1l(long double) throw(); 
# 249
extern long double jnl(int, long double) throw(); extern long double __jnl(int, long double) throw(); 
# 250
extern long double y0l(long double) throw(); extern long double __y0l(long double) throw(); 
# 251
extern long double y1l(long double) throw(); extern long double __y1l(long double) throw(); 
# 252
extern long double ynl(int, long double) throw(); extern long double __ynl(int, long double) throw(); 
# 259
extern long double erfl(long double) throw(); extern long double __erfl(long double) throw(); 
# 260
extern long double erfcl(long double) throw(); extern long double __erfcl(long double) throw(); 
# 261
extern long double lgammal(long double) throw(); extern long double __lgammal(long double) throw(); 
# 268
extern long double tgammal(long double) throw(); extern long double __tgammal(long double) throw(); 
# 274
extern long double gammal(long double) throw(); extern long double __gammal(long double) throw(); 
# 281
extern long double lgammal_r(long double, int * __signgamp) throw(); extern long double __lgammal_r(long double, int * __signgamp) throw(); 
# 289
extern long double rintl(long double __x) throw(); extern long double __rintl(long double __x) throw(); 
# 292
extern long double nextafterl(long double __x, long double __y) throw() __attribute((const)); extern long double __nextafterl(long double __x, long double __y) throw() __attribute((const)); 
# 294
extern long double nexttowardl(long double __x, long double __y) throw() __attribute((const)); extern long double __nexttowardl(long double __x, long double __y) throw() __attribute((const)); 
# 298
extern long double remainderl(long double __x, long double __y) throw(); extern long double __remainderl(long double __x, long double __y) throw(); 
# 302
extern long double scalbnl(long double __x, int __n) throw(); extern long double __scalbnl(long double __x, int __n) throw(); 
# 306
extern int ilogbl(long double __x) throw(); extern int __ilogbl(long double __x) throw(); 
# 311
extern long double scalblnl(long double __x, long __n) throw(); extern long double __scalblnl(long double __x, long __n) throw(); 
# 315
extern long double nearbyintl(long double __x) throw(); extern long double __nearbyintl(long double __x) throw(); 
# 319
extern long double roundl(long double __x) throw() __attribute((const)); extern long double __roundl(long double __x) throw() __attribute((const)); 
# 323
extern long double truncl(long double __x) throw() __attribute((const)); extern long double __truncl(long double __x) throw() __attribute((const)); 
# 328
extern long double remquol(long double __x, long double __y, int * __quo) throw(); extern long double __remquol(long double __x, long double __y, int * __quo) throw(); 
# 335
extern long lrintl(long double __x) throw(); extern long __lrintl(long double __x) throw(); 
# 337
__extension__ extern long long llrintl(long double __x) throw(); extern long long __llrintl(long double __x) throw(); 
# 341
extern long lroundl(long double __x) throw(); extern long __lroundl(long double __x) throw(); 
# 343
__extension__ extern long long llroundl(long double __x) throw(); extern long long __llroundl(long double __x) throw(); 
# 347
extern long double fdiml(long double __x, long double __y) throw(); extern long double __fdiml(long double __x, long double __y) throw(); 
# 350
extern long double fmaxl(long double __x, long double __y) throw() __attribute((const)); extern long double __fmaxl(long double __x, long double __y) throw() __attribute((const)); 
# 353
extern long double fminl(long double __x, long double __y) throw() __attribute((const)); extern long double __fminl(long double __x, long double __y) throw() __attribute((const)); 
# 357
extern int __fpclassifyl(long double __value) throw()
# 358
 __attribute((const)); 
# 361
extern int __signbitl(long double __value) throw()
# 362
 __attribute((const)); 
# 366
extern long double fmal(long double __x, long double __y, long double __z) throw(); extern long double __fmal(long double __x, long double __y, long double __z) throw(); 
# 375
extern int __issignalingl(long double __value) throw()
# 376
 __attribute((const)); 
# 383
extern long double scalbl(long double __x, long double __n) throw(); extern long double __scalbl(long double __x, long double __n) throw(); 
# 168 "/usr/include/math.h" 3
extern int signgam; 
# 210 "/usr/include/math.h" 3
enum { 
# 211
FP_NAN, 
# 214
FP_INFINITE, 
# 217
FP_ZERO, 
# 220
FP_SUBNORMAL, 
# 223
FP_NORMAL
# 226
}; 
# 354 "/usr/include/math.h" 3
typedef 
# 348
enum { 
# 349
_IEEE_ = (-1), 
# 350
_SVID_ = 0, 
# 351
_XOPEN_, 
# 352
_POSIX_, 
# 353
_ISOC_
# 354
} _LIB_VERSION_TYPE; 
# 359
extern _LIB_VERSION_TYPE _LIB_VERSION; 
# 370 "/usr/include/math.h" 3
struct __exception { 
# 375
int type; 
# 376
char *name; 
# 377
double arg1; 
# 378
double arg2; 
# 379
double retval; 
# 380
}; 
# 383
extern int matherr(__exception * __exc) throw(); 
# 534 "/usr/include/math.h" 3
}
# 34 "/usr/include/stdlib.h" 3
extern "C" {
# 55 "/usr/include/x86_64-linux-gnu/bits/waitflags.h" 3
typedef 
# 51
enum { 
# 52
P_ALL, 
# 53
P_PID, 
# 54
P_PGID
# 55
} idtype_t; 
# 45 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3
static inline unsigned __bswap_32(unsigned __bsx) 
# 46
{ 
# 47
return __builtin_bswap32(__bsx); 
# 48
} 
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 3
static inline __uint64_t __bswap_64(__uint64_t __bsx) 
# 110
{ 
# 111
return __builtin_bswap64(__bsx); 
# 112
} 
# 66 "/usr/include/x86_64-linux-gnu/bits/waitstatus.h" 3
union wait { 
# 68
int w_status; 
# 70
struct { 
# 72
unsigned __w_termsig:7; 
# 73
unsigned __w_coredump:1; 
# 74
unsigned __w_retcode:8; 
# 75
unsigned:16; 
# 83
} __wait_terminated; 
# 85
struct { 
# 87
unsigned __w_stopval:8; 
# 88
unsigned __w_stopsig:8; 
# 89
unsigned:16; 
# 96
} __wait_stopped; 
# 97
}; 
# 101 "/usr/include/stdlib.h" 3
typedef 
# 98
struct { 
# 99
int quot; 
# 100
int rem; 
# 101
} div_t; 
# 109
typedef 
# 106
struct { 
# 107
long quot; 
# 108
long rem; 
# 109
} ldiv_t; 
# 121
__extension__ typedef 
# 118
struct { 
# 119
long long quot; 
# 120
long long rem; 
# 121
} lldiv_t; 
# 139 "/usr/include/stdlib.h" 3
extern size_t __ctype_get_mb_cur_max() throw(); 
# 144
extern double atof(const char * __nptr) throw()
# 145
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 147
extern int atoi(const char * __nptr) throw()
# 148
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 150
extern long atol(const char * __nptr) throw()
# 151
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 157
__extension__ extern long long atoll(const char * __nptr) throw()
# 158
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 164
extern double strtod(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 166
 __attribute((__nonnull__(1))); 
# 172
extern float strtof(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 173
 __attribute((__nonnull__(1))); 
# 175
extern long double strtold(const char *__restrict__ __nptr, char **__restrict__ __endptr) throw()
# 177
 __attribute((__nonnull__(1))); 
# 183
extern long strtol(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 185
 __attribute((__nonnull__(1))); 
# 187
extern unsigned long strtoul(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 189
 __attribute((__nonnull__(1))); 
# 195
__extension__ extern long long strtoq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 197
 __attribute((__nonnull__(1))); 
# 200
__extension__ extern unsigned long long strtouq(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 202
 __attribute((__nonnull__(1))); 
# 209
__extension__ extern long long strtoll(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 211
 __attribute((__nonnull__(1))); 
# 214
__extension__ extern unsigned long long strtoull(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base) throw()
# 216
 __attribute((__nonnull__(1))); 
# 239 "/usr/include/stdlib.h" 3
extern long strtol_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 241
 __attribute((__nonnull__(1, 4))); 
# 243
extern unsigned long strtoul_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 246
 __attribute((__nonnull__(1, 4))); 
# 249
__extension__ extern long long strtoll_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 252
 __attribute((__nonnull__(1, 4))); 
# 255
__extension__ extern unsigned long long strtoull_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, int __base, __locale_t __loc) throw()
# 258
 __attribute((__nonnull__(1, 4))); 
# 260
extern double strtod_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 262
 __attribute((__nonnull__(1, 3))); 
# 264
extern float strtof_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 266
 __attribute((__nonnull__(1, 3))); 
# 268
extern long double strtold_l(const char *__restrict__ __nptr, char **__restrict__ __endptr, __locale_t __loc) throw()
# 271
 __attribute((__nonnull__(1, 3))); 
# 305 "/usr/include/stdlib.h" 3
extern char *l64a(long __n) throw(); 
# 308
extern long a64l(const char * __s) throw()
# 309
 __attribute((__pure__)) __attribute((__nonnull__(1))); 
# 27 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
extern "C" {
# 33
typedef __u_char u_char; 
# 34
typedef __u_short u_short; 
# 35
typedef __u_int u_int; 
# 36
typedef __u_long u_long; 
# 37
typedef __quad_t quad_t; 
# 38
typedef __u_quad_t u_quad_t; 
# 39
typedef __fsid_t fsid_t; 
# 44
typedef __loff_t loff_t; 
# 48
typedef __ino_t ino_t; 
# 55
typedef __ino64_t ino64_t; 
# 60
typedef __dev_t dev_t; 
# 65
typedef __gid_t gid_t; 
# 70
typedef __mode_t mode_t; 
# 75
typedef __nlink_t nlink_t; 
# 80
typedef __uid_t uid_t; 
# 86
typedef __off_t off_t; 
# 93
typedef __off64_t off64_t; 
# 104 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __id_t id_t; 
# 109
typedef __ssize_t ssize_t; 
# 115
typedef __daddr_t daddr_t; 
# 116
typedef __caddr_t caddr_t; 
# 122
typedef __key_t key_t; 
# 136 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __useconds_t useconds_t; 
# 140
typedef __suseconds_t suseconds_t; 
# 150 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef unsigned long ulong; 
# 151
typedef unsigned short ushort; 
# 152
typedef unsigned uint; 
# 194 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef signed char int8_t __attribute((__mode__(__QI__))); 
# 195
typedef short int16_t __attribute((__mode__(__HI__))); 
# 196
typedef int int32_t __attribute((__mode__(__SI__))); 
# 197
typedef long int64_t __attribute((__mode__(__DI__))); 
# 200
typedef unsigned char u_int8_t __attribute((__mode__(__QI__))); 
# 201
typedef unsigned short u_int16_t __attribute((__mode__(__HI__))); 
# 202
typedef unsigned u_int32_t __attribute((__mode__(__SI__))); 
# 203
typedef unsigned long u_int64_t __attribute((__mode__(__DI__))); 
# 205
typedef long register_t __attribute((__mode__(__word__))); 
# 22 "/usr/include/x86_64-linux-gnu/bits/sigset.h" 3
typedef int __sig_atomic_t; 
# 30
typedef 
# 28
struct { 
# 29
unsigned long __val[(1024) / ((8) * sizeof(unsigned long))]; 
# 30
} __sigset_t; 
# 37 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef __sigset_t sigset_t; 
# 54 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef long __fd_mask; 
# 75 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
typedef 
# 65
struct { 
# 69
__fd_mask fds_bits[1024 / (8 * ((int)sizeof(__fd_mask)))]; 
# 75
} fd_set; 
# 82
typedef __fd_mask fd_mask; 
# 96 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern "C" {
# 106 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern int select(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, timeval *__restrict__ __timeout); 
# 118 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
extern int pselect(int __nfds, fd_set *__restrict__ __readfds, fd_set *__restrict__ __writefds, fd_set *__restrict__ __exceptfds, const timespec *__restrict__ __timeout, const __sigset_t *__restrict__ __sigmask); 
# 131 "/usr/include/x86_64-linux-gnu/sys/select.h" 3
}
# 24 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
extern "C" {
# 27
__extension__ extern unsigned gnu_dev_major(unsigned long long __dev) throw()
# 28
 __attribute((const)); 
# 30
__extension__ extern unsigned gnu_dev_minor(unsigned long long __dev) throw()
# 31
 __attribute((const)); 
# 33
__extension__ extern unsigned long long gnu_dev_makedev(unsigned __major, unsigned __minor) throw()
# 35
 __attribute((const)); 
# 58 "/usr/include/x86_64-linux-gnu/sys/sysmacros.h" 3
}
# 228 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __blksize_t blksize_t; 
# 235
typedef __blkcnt_t blkcnt_t; 
# 239
typedef __fsblkcnt_t fsblkcnt_t; 
# 243
typedef __fsfilcnt_t fsfilcnt_t; 
# 262 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
typedef __blkcnt64_t blkcnt64_t; 
# 263
typedef __fsblkcnt64_t fsblkcnt64_t; 
# 264
typedef __fsfilcnt64_t fsfilcnt64_t; 
# 60 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef unsigned long pthread_t; 
# 63
union pthread_attr_t { 
# 65
char __size[56]; 
# 66
long __align; 
# 67
}; 
# 69
typedef pthread_attr_t pthread_attr_t; 
# 79
typedef 
# 75
struct __pthread_internal_list { 
# 77
__pthread_internal_list *__prev; 
# 78
__pthread_internal_list *__next; 
# 79
} __pthread_list_t; 
# 128 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef 
# 91 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union { 
# 92
struct __pthread_mutex_s { 
# 94
int __lock; 
# 95
unsigned __count; 
# 96
int __owner; 
# 98
unsigned __nusers; 
# 102
int __kind; 
# 104
short __spins; 
# 105
short __elision; 
# 106
__pthread_list_t __list; 
# 125 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
} __data; 
# 126
char __size[40]; 
# 127
long __align; 
# 128
} pthread_mutex_t; 
# 134
typedef 
# 131
union { 
# 132
char __size[4]; 
# 133
int __align; 
# 134
} pthread_mutexattr_t; 
# 154
typedef 
# 140
union { 
# 142
struct { 
# 143
int __lock; 
# 144
unsigned __futex; 
# 145
__extension__ unsigned long long __total_seq; 
# 146
__extension__ unsigned long long __wakeup_seq; 
# 147
__extension__ unsigned long long __woken_seq; 
# 148
void *__mutex; 
# 149
unsigned __nwaiters; 
# 150
unsigned __broadcast_seq; 
# 151
} __data; 
# 152
char __size[48]; 
# 153
__extension__ long long __align; 
# 154
} pthread_cond_t; 
# 160
typedef 
# 157
union { 
# 158
char __size[4]; 
# 159
int __align; 
# 160
} pthread_condattr_t; 
# 164
typedef unsigned pthread_key_t; 
# 168
typedef int pthread_once_t; 
# 222 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
typedef 
# 175 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
union { 
# 178
struct { 
# 179
int __lock; 
# 180
unsigned __nr_readers; 
# 181
unsigned __readers_wakeup; 
# 182
unsigned __writer_wakeup; 
# 183
unsigned __nr_readers_queued; 
# 184
unsigned __nr_writers_queued; 
# 185
int __writer; 
# 186
int __shared; 
# 187
signed char __rwelision; 
# 192
unsigned char __pad1[7]; 
# 195
unsigned long __pad2; 
# 198
unsigned __flags; 
# 200
} __data; 
# 220 "/usr/include/x86_64-linux-gnu/bits/pthreadtypes.h" 3
char __size[56]; 
# 221
long __align; 
# 222
} pthread_rwlock_t; 
# 228
typedef 
# 225
union { 
# 226
char __size[8]; 
# 227
long __align; 
# 228
} pthread_rwlockattr_t; 
# 234
typedef volatile int pthread_spinlock_t; 
# 243
typedef 
# 240
union { 
# 241
char __size[32]; 
# 242
long __align; 
# 243
} pthread_barrier_t; 
# 249
typedef 
# 246
union { 
# 247
char __size[4]; 
# 248
int __align; 
# 249
} pthread_barrierattr_t; 
# 273 "/usr/include/x86_64-linux-gnu/sys/types.h" 3
}
# 321 "/usr/include/stdlib.h" 3
extern long random() throw(); 
# 324
extern void srandom(unsigned __seed) throw(); 
# 330
extern char *initstate(unsigned __seed, char * __statebuf, size_t __statelen) throw()
# 331
 __attribute((__nonnull__(2))); 
# 335
extern char *setstate(char * __statebuf) throw() __attribute((__nonnull__(1))); 
# 343
struct random_data { 
# 345
int32_t *fptr; 
# 346
int32_t *rptr; 
# 347
int32_t *state; 
# 348
int rand_type; 
# 349
int rand_deg; 
# 350
int rand_sep; 
# 351
int32_t *end_ptr; 
# 352
}; 
# 354
extern int random_r(random_data *__restrict__ __buf, int32_t *__restrict__ __result) throw()
# 355
 __attribute((__nonnull__(1, 2))); 
# 357
extern int srandom_r(unsigned __seed, random_data * __buf) throw()
# 358
 __attribute((__nonnull__(2))); 
# 360
extern int initstate_r(unsigned __seed, char *__restrict__ __statebuf, size_t __statelen, random_data *__restrict__ __buf) throw()
# 363
 __attribute((__nonnull__(2, 4))); 
# 365
extern int setstate_r(char *__restrict__ __statebuf, random_data *__restrict__ __buf) throw()
# 367
 __attribute((__nonnull__(1, 2))); 
# 374
extern int rand() throw(); 
# 376
extern void srand(unsigned __seed) throw(); 
# 381
extern int rand_r(unsigned * __seed) throw(); 
# 389
extern double drand48() throw(); 
# 390
extern double erand48(unsigned short  __xsubi[3]) throw() __attribute((__nonnull__(1))); 
# 393
extern long lrand48() throw(); 
# 394
extern long nrand48(unsigned short  __xsubi[3]) throw()
# 395
 __attribute((__nonnull__(1))); 
# 398
extern long mrand48() throw(); 
# 399
extern long jrand48(unsigned short  __xsubi[3]) throw()
# 400
 __attribute((__nonnull__(1))); 
# 403
extern void srand48(long __seedval) throw(); 
# 404
extern unsigned short *seed48(unsigned short  __seed16v[3]) throw()
# 405
 __attribute((__nonnull__(1))); 
# 406
extern void lcong48(unsigned short  __param[7]) throw() __attribute((__nonnull__(1))); 
# 412
struct drand48_data { 
# 414
unsigned short __x[3]; 
# 415
unsigned short __old_x[3]; 
# 416
unsigned short __c; 
# 417
unsigned short __init; 
# 418
__extension__ unsigned long long __a; 
# 420
}; 
# 423
extern int drand48_r(drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 424
 __attribute((__nonnull__(1, 2))); 
# 425
extern int erand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, double *__restrict__ __result) throw()
# 427
 __attribute((__nonnull__(1, 2))); 
# 430
extern int lrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 432
 __attribute((__nonnull__(1, 2))); 
# 433
extern int nrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 436
 __attribute((__nonnull__(1, 2))); 
# 439
extern int mrand48_r(drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 441
 __attribute((__nonnull__(1, 2))); 
# 442
extern int jrand48_r(unsigned short  __xsubi[3], drand48_data *__restrict__ __buffer, long *__restrict__ __result) throw()
# 445
 __attribute((__nonnull__(1, 2))); 
# 448
extern int srand48_r(long __seedval, drand48_data * __buffer) throw()
# 449
 __attribute((__nonnull__(2))); 
# 451
extern int seed48_r(unsigned short  __seed16v[3], drand48_data * __buffer) throw()
# 452
 __attribute((__nonnull__(1, 2))); 
# 454
extern int lcong48_r(unsigned short  __param[7], drand48_data * __buffer) throw()
# 456
 __attribute((__nonnull__(1, 2))); 
# 466
extern void *malloc(size_t __size) throw() __attribute((__malloc__)); 
# 468
extern void *calloc(size_t __nmemb, size_t __size) throw()
# 469
 __attribute((__malloc__)); 
# 480
extern void *realloc(void * __ptr, size_t __size) throw()
# 481
 __attribute((__warn_unused_result__)); 
# 483
extern void free(void * __ptr) throw(); 
# 488
extern void cfree(void * __ptr) throw(); 
# 26 "/usr/include/alloca.h" 3
extern "C" {
# 32
extern void *alloca(size_t __size) throw(); 
# 38
}
# 498 "/usr/include/stdlib.h" 3
extern void *valloc(size_t __size) throw() __attribute((__malloc__)); 
# 503
extern int posix_memalign(void ** __memptr, size_t __alignment, size_t __size) throw()
# 504
 __attribute((__nonnull__(1))); 
# 509
extern void *aligned_alloc(size_t __alignment, size_t __size) throw()
# 510
 __attribute((__malloc__)) __attribute((__alloc_size__(2))); 
# 515
extern void abort() throw() __attribute((__noreturn__)); 
# 519
extern int atexit(void (* __func)(void)) throw() __attribute((__nonnull__(1))); 
# 524
extern "C++" int at_quick_exit(void (* __func)(void)) throw() __asm__("at_quick_exit")
# 525
 __attribute((__nonnull__(1))); 
# 535
extern int on_exit(void (* __func)(int __status, void * __arg), void * __arg) throw()
# 536
 __attribute((__nonnull__(1))); 
# 543
extern void exit(int __status) throw() __attribute((__noreturn__)); 
# 549
extern void quick_exit(int __status) throw() __attribute((__noreturn__)); 
# 557
extern void _Exit(int __status) throw() __attribute((__noreturn__)); 
# 564
extern char *getenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 570
extern char *secure_getenv(const char * __name) throw()
# 571
 __attribute((__nonnull__(1))); 
# 578
extern int putenv(char * __string) throw() __attribute((__nonnull__(1))); 
# 584
extern int setenv(const char * __name, const char * __value, int __replace) throw()
# 585
 __attribute((__nonnull__(2))); 
# 588
extern int unsetenv(const char * __name) throw() __attribute((__nonnull__(1))); 
# 595
extern int clearenv() throw(); 
# 606 "/usr/include/stdlib.h" 3
extern char *mktemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 619 "/usr/include/stdlib.h" 3
extern int mkstemp(char * __template) __attribute((__nonnull__(1))); 
# 629 "/usr/include/stdlib.h" 3
extern int mkstemp64(char * __template) __attribute((__nonnull__(1))); 
# 641 "/usr/include/stdlib.h" 3
extern int mkstemps(char * __template, int __suffixlen) __attribute((__nonnull__(1))); 
# 651 "/usr/include/stdlib.h" 3
extern int mkstemps64(char * __template, int __suffixlen)
# 652
 __attribute((__nonnull__(1))); 
# 662 "/usr/include/stdlib.h" 3
extern char *mkdtemp(char * __template) throw() __attribute((__nonnull__(1))); 
# 673 "/usr/include/stdlib.h" 3
extern int mkostemp(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 683 "/usr/include/stdlib.h" 3
extern int mkostemp64(char * __template, int __flags) __attribute((__nonnull__(1))); 
# 693 "/usr/include/stdlib.h" 3
extern int mkostemps(char * __template, int __suffixlen, int __flags)
# 694
 __attribute((__nonnull__(1))); 
# 705 "/usr/include/stdlib.h" 3
extern int mkostemps64(char * __template, int __suffixlen, int __flags)
# 706
 __attribute((__nonnull__(1))); 
# 716
extern int system(const char * __command); 
# 723
extern char *canonicalize_file_name(const char * __name) throw()
# 724
 __attribute((__nonnull__(1))); 
# 733 "/usr/include/stdlib.h" 3
extern char *realpath(const char *__restrict__ __name, char *__restrict__ __resolved) throw(); 
# 741
typedef int (*__compar_fn_t)(const void *, const void *); 
# 744
typedef __compar_fn_t comparison_fn_t; 
# 748
typedef int (*__compar_d_fn_t)(const void *, const void *, void *); 
# 754
extern void *bsearch(const void * __key, const void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 756
 __attribute((__nonnull__(1, 2, 5))); 
# 764
extern void qsort(void * __base, size_t __nmemb, size_t __size, __compar_fn_t __compar)
# 765
 __attribute((__nonnull__(1, 4))); 
# 767
extern void qsort_r(void * __base, size_t __nmemb, size_t __size, __compar_d_fn_t __compar, void * __arg)
# 769
 __attribute((__nonnull__(1, 4))); 
# 774
extern int abs(int __x) throw() __attribute((const)); 
# 775
extern long labs(long __x) throw() __attribute((const)); 
# 779
__extension__ extern long long llabs(long long __x) throw()
# 780
 __attribute((const)); 
# 788
extern div_t div(int __numer, int __denom) throw()
# 789
 __attribute((const)); 
# 790
extern ldiv_t ldiv(long __numer, long __denom) throw()
# 791
 __attribute((const)); 
# 796
__extension__ extern lldiv_t lldiv(long long __numer, long long __denom) throw()
# 798
 __attribute((const)); 
# 811 "/usr/include/stdlib.h" 3
extern char *ecvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 812
 __attribute((__nonnull__(3, 4))); 
# 817
extern char *fcvt(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 818
 __attribute((__nonnull__(3, 4))); 
# 823
extern char *gcvt(double __value, int __ndigit, char * __buf) throw()
# 824
 __attribute((__nonnull__(3))); 
# 829
extern char *qecvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 831
 __attribute((__nonnull__(3, 4))); 
# 832
extern char *qfcvt(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign) throw()
# 834
 __attribute((__nonnull__(3, 4))); 
# 835
extern char *qgcvt(long double __value, int __ndigit, char * __buf) throw()
# 836
 __attribute((__nonnull__(3))); 
# 841
extern int ecvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 843
 __attribute((__nonnull__(3, 4, 5))); 
# 844
extern int fcvt_r(double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 846
 __attribute((__nonnull__(3, 4, 5))); 
# 848
extern int qecvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 851
 __attribute((__nonnull__(3, 4, 5))); 
# 852
extern int qfcvt_r(long double __value, int __ndigit, int *__restrict__ __decpt, int *__restrict__ __sign, char *__restrict__ __buf, size_t __len) throw()
# 855
 __attribute((__nonnull__(3, 4, 5))); 
# 862
extern int mblen(const char * __s, size_t __n) throw(); 
# 865
extern int mbtowc(wchar_t *__restrict__ __pwc, const char *__restrict__ __s, size_t __n) throw(); 
# 869
extern int wctomb(char * __s, wchar_t __wchar) throw(); 
# 873
extern size_t mbstowcs(wchar_t *__restrict__ __pwcs, const char *__restrict__ __s, size_t __n) throw(); 
# 876
extern size_t wcstombs(char *__restrict__ __s, const wchar_t *__restrict__ __pwcs, size_t __n) throw(); 
# 887
extern int rpmatch(const char * __response) throw() __attribute((__nonnull__(1))); 
# 898 "/usr/include/stdlib.h" 3
extern int getsubopt(char **__restrict__ __optionp, char *const *__restrict__ __tokens, char **__restrict__ __valuep) throw()
# 901
 __attribute((__nonnull__(1, 2, 3))); 
# 907
extern void setkey(const char * __key) throw() __attribute((__nonnull__(1))); 
# 915
extern int posix_openpt(int __oflag); 
# 923
extern int grantpt(int __fd) throw(); 
# 927
extern int unlockpt(int __fd) throw(); 
# 932
extern char *ptsname(int __fd) throw(); 
# 939
extern int ptsname_r(int __fd, char * __buf, size_t __buflen) throw()
# 940
 __attribute((__nonnull__(2))); 
# 943
extern int getpt(); 
# 950
extern int getloadavg(double  __loadavg[], int __nelem) throw()
# 951
 __attribute((__nonnull__(1))); 
# 967 "/usr/include/stdlib.h" 3
}
# 194 "/usr/include/x86_64-linux-gnu/c++/5/bits/c++config.h" 3
namespace std { 
# 196
typedef unsigned long size_t; 
# 197
typedef long ptrdiff_t; 
# 202
}
# 216 "/usr/include/x86_64-linux-gnu/c++/5/bits/c++config.h" 3
namespace std { 
# 218
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 219
}
# 220
namespace __gnu_cxx { 
# 222
inline namespace __cxx11 __attribute((__abi_tag__("cxx11"))) { }
# 223
}
# 68 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 72
template< class _Iterator, class _Container> class __normal_iterator; 
# 76
}
# 78
namespace std __attribute((__visibility__("default"))) { 
# 82
struct __true_type { }; 
# 83
struct __false_type { }; 
# 85
template< bool > 
# 86
struct __truth_type { 
# 87
typedef __false_type __type; }; 
# 90
template<> struct __truth_type< true>  { 
# 91
typedef __true_type __type; }; 
# 95
template< class _Sp, class _Tp> 
# 96
struct __traitor { 
# 98
enum { __value = ((bool)_Sp::__value) || ((bool)_Tp::__value)}; 
# 99
typedef typename __truth_type< __value> ::__type __type; 
# 100
}; 
# 103
template< class , class > 
# 104
struct __are_same { 
# 106
enum { __value}; 
# 107
typedef __false_type __type; 
# 108
}; 
# 110
template< class _Tp> 
# 111
struct __are_same< _Tp, _Tp>  { 
# 113
enum { __value = 1}; 
# 114
typedef __true_type __type; 
# 115
}; 
# 118
template< class _Tp> 
# 119
struct __is_void { 
# 121
enum { __value}; 
# 122
typedef __false_type __type; 
# 123
}; 
# 126
template<> struct __is_void< void>  { 
# 128
enum { __value = 1}; 
# 129
typedef __true_type __type; 
# 130
}; 
# 135
template< class _Tp> 
# 136
struct __is_integer { 
# 138
enum { __value}; 
# 139
typedef __false_type __type; 
# 140
}; 
# 147
template<> struct __is_integer< bool>  { 
# 149
enum { __value = 1}; 
# 150
typedef __true_type __type; 
# 151
}; 
# 154
template<> struct __is_integer< char>  { 
# 156
enum { __value = 1}; 
# 157
typedef __true_type __type; 
# 158
}; 
# 161
template<> struct __is_integer< signed char>  { 
# 163
enum { __value = 1}; 
# 164
typedef __true_type __type; 
# 165
}; 
# 168
template<> struct __is_integer< unsigned char>  { 
# 170
enum { __value = 1}; 
# 171
typedef __true_type __type; 
# 172
}; 
# 176
template<> struct __is_integer< wchar_t>  { 
# 178
enum { __value = 1}; 
# 179
typedef __true_type __type; 
# 180
}; 
# 200 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
template<> struct __is_integer< short>  { 
# 202
enum { __value = 1}; 
# 203
typedef __true_type __type; 
# 204
}; 
# 207
template<> struct __is_integer< unsigned short>  { 
# 209
enum { __value = 1}; 
# 210
typedef __true_type __type; 
# 211
}; 
# 214
template<> struct __is_integer< int>  { 
# 216
enum { __value = 1}; 
# 217
typedef __true_type __type; 
# 218
}; 
# 221
template<> struct __is_integer< unsigned>  { 
# 223
enum { __value = 1}; 
# 224
typedef __true_type __type; 
# 225
}; 
# 228
template<> struct __is_integer< long>  { 
# 230
enum { __value = 1}; 
# 231
typedef __true_type __type; 
# 232
}; 
# 235
template<> struct __is_integer< unsigned long>  { 
# 237
enum { __value = 1}; 
# 238
typedef __true_type __type; 
# 239
}; 
# 242
template<> struct __is_integer< long long>  { 
# 244
enum { __value = 1}; 
# 245
typedef __true_type __type; 
# 246
}; 
# 249
template<> struct __is_integer< unsigned long long>  { 
# 251
enum { __value = 1}; 
# 252
typedef __true_type __type; 
# 253
}; 
# 270 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
template<> struct __is_integer< __int128_t>  { enum { __value = 1}; typedef __true_type __type; }; template<> struct __is_integer< __uint128_t>  { enum { __value = 1}; typedef __true_type __type; }; 
# 287 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
template< class _Tp> 
# 288
struct __is_floating { 
# 290
enum { __value}; 
# 291
typedef __false_type __type; 
# 292
}; 
# 296
template<> struct __is_floating< float>  { 
# 298
enum { __value = 1}; 
# 299
typedef __true_type __type; 
# 300
}; 
# 303
template<> struct __is_floating< double>  { 
# 305
enum { __value = 1}; 
# 306
typedef __true_type __type; 
# 307
}; 
# 310
template<> struct __is_floating< long double>  { 
# 312
enum { __value = 1}; 
# 313
typedef __true_type __type; 
# 314
}; 
# 319
template< class _Tp> 
# 320
struct __is_pointer { 
# 322
enum { __value}; 
# 323
typedef __false_type __type; 
# 324
}; 
# 326
template< class _Tp> 
# 327
struct __is_pointer< _Tp *>  { 
# 329
enum { __value = 1}; 
# 330
typedef __true_type __type; 
# 331
}; 
# 336
template< class _Tp> 
# 337
struct __is_normal_iterator { 
# 339
enum { __value}; 
# 340
typedef __false_type __type; 
# 341
}; 
# 343
template< class _Iterator, class _Container> 
# 344
struct __is_normal_iterator< __gnu_cxx::__normal_iterator< _Iterator, _Container> >  { 
# 347
enum { __value = 1}; 
# 348
typedef __true_type __type; 
# 349
}; 
# 354
template< class _Tp> 
# 355
struct __is_arithmetic : public __traitor< __is_integer< _Tp> , __is_floating< _Tp> >  { 
# 357
}; 
# 362
template< class _Tp> 
# 363
struct __is_scalar : public __traitor< __is_arithmetic< _Tp> , __is_pointer< _Tp> >  { 
# 365
}; 
# 370
template< class _Tp> 
# 371
struct __is_char { 
# 373
enum { __value}; 
# 374
typedef __false_type __type; 
# 375
}; 
# 378
template<> struct __is_char< char>  { 
# 380
enum { __value = 1}; 
# 381
typedef __true_type __type; 
# 382
}; 
# 386
template<> struct __is_char< wchar_t>  { 
# 388
enum { __value = 1}; 
# 389
typedef __true_type __type; 
# 390
}; 
# 393
template< class _Tp> 
# 394
struct __is_byte { 
# 396
enum { __value}; 
# 397
typedef __false_type __type; 
# 398
}; 
# 401
template<> struct __is_byte< char>  { 
# 403
enum { __value = 1}; 
# 404
typedef __true_type __type; 
# 405
}; 
# 408
template<> struct __is_byte< signed char>  { 
# 410
enum { __value = 1}; 
# 411
typedef __true_type __type; 
# 412
}; 
# 415
template<> struct __is_byte< unsigned char>  { 
# 417
enum { __value = 1}; 
# 418
typedef __true_type __type; 
# 419
}; 
# 424
template< class _Tp> 
# 425
struct __is_move_iterator { 
# 427
enum { __value}; 
# 428
typedef __false_type __type; 
# 429
}; 
# 444 "/usr/include/c++/5/bits/cpp_type_traits.h" 3
}
# 37 "/usr/include/c++/5/ext/type_traits.h" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 42
template< bool , class > 
# 43
struct __enable_if { 
# 44
}; 
# 46
template< class _Tp> 
# 47
struct __enable_if< true, _Tp>  { 
# 48
typedef _Tp __type; }; 
# 52
template< bool _Cond, class _Iftrue, class _Iffalse> 
# 53
struct __conditional_type { 
# 54
typedef _Iftrue __type; }; 
# 56
template< class _Iftrue, class _Iffalse> 
# 57
struct __conditional_type< false, _Iftrue, _Iffalse>  { 
# 58
typedef _Iffalse __type; }; 
# 62
template< class _Tp> 
# 63
struct __add_unsigned { 
# 66
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 69
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 70
}; 
# 73
template<> struct __add_unsigned< char>  { 
# 74
typedef unsigned char __type; }; 
# 77
template<> struct __add_unsigned< signed char>  { 
# 78
typedef unsigned char __type; }; 
# 81
template<> struct __add_unsigned< short>  { 
# 82
typedef unsigned short __type; }; 
# 85
template<> struct __add_unsigned< int>  { 
# 86
typedef unsigned __type; }; 
# 89
template<> struct __add_unsigned< long>  { 
# 90
typedef unsigned long __type; }; 
# 93
template<> struct __add_unsigned< long long>  { 
# 94
typedef unsigned long long __type; }; 
# 98
template<> struct __add_unsigned< bool> ; 
# 101
template<> struct __add_unsigned< wchar_t> ; 
# 105
template< class _Tp> 
# 106
struct __remove_unsigned { 
# 109
private: typedef __enable_if< std::__is_integer< _Tp> ::__value, _Tp>  __if_type; 
# 112
public: typedef typename __enable_if< std::__is_integer< _Tp> ::__value, _Tp> ::__type __type; 
# 113
}; 
# 116
template<> struct __remove_unsigned< char>  { 
# 117
typedef signed char __type; }; 
# 120
template<> struct __remove_unsigned< unsigned char>  { 
# 121
typedef signed char __type; }; 
# 124
template<> struct __remove_unsigned< unsigned short>  { 
# 125
typedef short __type; }; 
# 128
template<> struct __remove_unsigned< unsigned>  { 
# 129
typedef int __type; }; 
# 132
template<> struct __remove_unsigned< unsigned long>  { 
# 133
typedef long __type; }; 
# 136
template<> struct __remove_unsigned< unsigned long long>  { 
# 137
typedef long long __type; }; 
# 141
template<> struct __remove_unsigned< bool> ; 
# 144
template<> struct __remove_unsigned< wchar_t> ; 
# 148
template< class _Type> inline bool 
# 150
__is_null_pointer(_Type *__ptr) 
# 151
{ return __ptr == 0; } 
# 153
template< class _Type> inline bool 
# 155
__is_null_pointer(_Type) 
# 156
{ return false; } 
# 165 "/usr/include/c++/5/ext/type_traits.h" 3
template< class _Tp, bool  = std::__is_integer< _Tp> ::__value> 
# 166
struct __promote { 
# 167
typedef double __type; }; 
# 172
template< class _Tp> 
# 173
struct __promote< _Tp, false>  { 
# 174
}; 
# 177
template<> struct __promote< long double>  { 
# 178
typedef long double __type; }; 
# 181
template<> struct __promote< double>  { 
# 182
typedef double __type; }; 
# 185
template<> struct __promote< float>  { 
# 186
typedef float __type; }; 
# 188
template< class _Tp, class _Up, class 
# 189
_Tp2 = typename __promote< _Tp> ::__type, class 
# 190
_Up2 = typename __promote< _Up> ::__type> 
# 191
struct __promote_2 { 
# 193
typedef __typeof__(_Tp2() + _Up2()) __type; 
# 194
}; 
# 196
template< class _Tp, class _Up, class _Vp, class 
# 197
_Tp2 = typename __promote< _Tp> ::__type, class 
# 198
_Up2 = typename __promote< _Up> ::__type, class 
# 199
_Vp2 = typename __promote< _Vp> ::__type> 
# 200
struct __promote_3 { 
# 202
typedef __typeof__((_Tp2() + _Up2()) + _Vp2()) __type; 
# 203
}; 
# 205
template< class _Tp, class _Up, class _Vp, class _Wp, class 
# 206
_Tp2 = typename __promote< _Tp> ::__type, class 
# 207
_Up2 = typename __promote< _Up> ::__type, class 
# 208
_Vp2 = typename __promote< _Vp> ::__type, class 
# 209
_Wp2 = typename __promote< _Wp> ::__type> 
# 210
struct __promote_4 { 
# 212
typedef __typeof__(((_Tp2() + _Up2()) + _Vp2()) + _Wp2()) __type; 
# 213
}; 
# 216
}
# 75 "/usr/include/c++/5/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 81
inline double abs(double __x) 
# 82
{ return __builtin_fabs(__x); } 
# 87
inline float abs(float __x) 
# 88
{ return __builtin_fabsf(__x); } 
# 91
inline long double abs(long double __x) 
# 92
{ return __builtin_fabsl(__x); } 
# 95
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 99
abs(_Tp __x) 
# 100
{ return __builtin_fabs(__x); } 
# 102
using ::acos;
# 106
inline float acos(float __x) 
# 107
{ return __builtin_acosf(__x); } 
# 110
inline long double acos(long double __x) 
# 111
{ return __builtin_acosl(__x); } 
# 114
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 118
acos(_Tp __x) 
# 119
{ return __builtin_acos(__x); } 
# 121
using ::asin;
# 125
inline float asin(float __x) 
# 126
{ return __builtin_asinf(__x); } 
# 129
inline long double asin(long double __x) 
# 130
{ return __builtin_asinl(__x); } 
# 133
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 137
asin(_Tp __x) 
# 138
{ return __builtin_asin(__x); } 
# 140
using ::atan;
# 144
inline float atan(float __x) 
# 145
{ return __builtin_atanf(__x); } 
# 148
inline long double atan(long double __x) 
# 149
{ return __builtin_atanl(__x); } 
# 152
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 156
atan(_Tp __x) 
# 157
{ return __builtin_atan(__x); } 
# 159
using ::atan2;
# 163
inline float atan2(float __y, float __x) 
# 164
{ return __builtin_atan2f(__y, __x); } 
# 167
inline long double atan2(long double __y, long double __x) 
# 168
{ return __builtin_atan2l(__y, __x); } 
# 171
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 174
atan2(_Tp __y, _Up __x) 
# 175
{ 
# 176
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 177
return atan2((__type)__y, (__type)__x); 
# 178
} 
# 180
using ::ceil;
# 184
inline float ceil(float __x) 
# 185
{ return __builtin_ceilf(__x); } 
# 188
inline long double ceil(long double __x) 
# 189
{ return __builtin_ceill(__x); } 
# 192
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 196
ceil(_Tp __x) 
# 197
{ return __builtin_ceil(__x); } 
# 199
using ::cos;
# 203
inline float cos(float __x) 
# 204
{ return __builtin_cosf(__x); } 
# 207
inline long double cos(long double __x) 
# 208
{ return __builtin_cosl(__x); } 
# 211
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 215
cos(_Tp __x) 
# 216
{ return __builtin_cos(__x); } 
# 218
using ::cosh;
# 222
inline float cosh(float __x) 
# 223
{ return __builtin_coshf(__x); } 
# 226
inline long double cosh(long double __x) 
# 227
{ return __builtin_coshl(__x); } 
# 230
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 234
cosh(_Tp __x) 
# 235
{ return __builtin_cosh(__x); } 
# 237
using ::exp;
# 241
inline float exp(float __x) 
# 242
{ return __builtin_expf(__x); } 
# 245
inline long double exp(long double __x) 
# 246
{ return __builtin_expl(__x); } 
# 249
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 253
exp(_Tp __x) 
# 254
{ return __builtin_exp(__x); } 
# 256
using ::fabs;
# 260
inline float fabs(float __x) 
# 261
{ return __builtin_fabsf(__x); } 
# 264
inline long double fabs(long double __x) 
# 265
{ return __builtin_fabsl(__x); } 
# 268
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 272
fabs(_Tp __x) 
# 273
{ return __builtin_fabs(__x); } 
# 275
using ::floor;
# 279
inline float floor(float __x) 
# 280
{ return __builtin_floorf(__x); } 
# 283
inline long double floor(long double __x) 
# 284
{ return __builtin_floorl(__x); } 
# 287
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 291
floor(_Tp __x) 
# 292
{ return __builtin_floor(__x); } 
# 294
using ::fmod;
# 298
inline float fmod(float __x, float __y) 
# 299
{ return __builtin_fmodf(__x, __y); } 
# 302
inline long double fmod(long double __x, long double __y) 
# 303
{ return __builtin_fmodl(__x, __y); } 
# 306
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 309
fmod(_Tp __x, _Up __y) 
# 310
{ 
# 311
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 312
return fmod((__type)__x, (__type)__y); 
# 313
} 
# 315
using ::frexp;
# 319
inline float frexp(float __x, int *__exp) 
# 320
{ return __builtin_frexpf(__x, __exp); } 
# 323
inline long double frexp(long double __x, int *__exp) 
# 324
{ return __builtin_frexpl(__x, __exp); } 
# 327
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 331
frexp(_Tp __x, int *__exp) 
# 332
{ return __builtin_frexp(__x, __exp); } 
# 334
using ::ldexp;
# 338
inline float ldexp(float __x, int __exp) 
# 339
{ return __builtin_ldexpf(__x, __exp); } 
# 342
inline long double ldexp(long double __x, int __exp) 
# 343
{ return __builtin_ldexpl(__x, __exp); } 
# 346
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 350
ldexp(_Tp __x, int __exp) 
# 351
{ return __builtin_ldexp(__x, __exp); } 
# 353
using ::log;
# 357
inline float log(float __x) 
# 358
{ return __builtin_logf(__x); } 
# 361
inline long double log(long double __x) 
# 362
{ return __builtin_logl(__x); } 
# 365
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 369
log(_Tp __x) 
# 370
{ return __builtin_log(__x); } 
# 372
using ::log10;
# 376
inline float log10(float __x) 
# 377
{ return __builtin_log10f(__x); } 
# 380
inline long double log10(long double __x) 
# 381
{ return __builtin_log10l(__x); } 
# 384
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 388
log10(_Tp __x) 
# 389
{ return __builtin_log10(__x); } 
# 391
using ::modf;
# 395
inline float modf(float __x, float *__iptr) 
# 396
{ return __builtin_modff(__x, __iptr); } 
# 399
inline long double modf(long double __x, long double *__iptr) 
# 400
{ return __builtin_modfl(__x, __iptr); } 
# 403
using ::pow;
# 407
inline float pow(float __x, float __y) 
# 408
{ return __builtin_powf(__x, __y); } 
# 411
inline long double pow(long double __x, long double __y) 
# 412
{ return __builtin_powl(__x, __y); } 
# 418
inline double pow(double __x, int __i) 
# 419
{ return __builtin_powi(__x, __i); } 
# 422
inline float pow(float __x, int __n) 
# 423
{ return __builtin_powif(__x, __n); } 
# 426
inline long double pow(long double __x, int __n) 
# 427
{ return __builtin_powil(__x, __n); } 
# 431
template< class _Tp, class _Up> inline typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type 
# 434
pow(_Tp __x, _Up __y) 
# 435
{ 
# 436
typedef typename __gnu_cxx::__promote_2< _Tp, _Up> ::__type __type; 
# 437
return pow((__type)__x, (__type)__y); 
# 438
} 
# 440
using ::sin;
# 444
inline float sin(float __x) 
# 445
{ return __builtin_sinf(__x); } 
# 448
inline long double sin(long double __x) 
# 449
{ return __builtin_sinl(__x); } 
# 452
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 456
sin(_Tp __x) 
# 457
{ return __builtin_sin(__x); } 
# 459
using ::sinh;
# 463
inline float sinh(float __x) 
# 464
{ return __builtin_sinhf(__x); } 
# 467
inline long double sinh(long double __x) 
# 468
{ return __builtin_sinhl(__x); } 
# 471
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 475
sinh(_Tp __x) 
# 476
{ return __builtin_sinh(__x); } 
# 478
using ::sqrt;
# 482
inline float sqrt(float __x) 
# 483
{ return __builtin_sqrtf(__x); } 
# 486
inline long double sqrt(long double __x) 
# 487
{ return __builtin_sqrtl(__x); } 
# 490
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 494
sqrt(_Tp __x) 
# 495
{ return __builtin_sqrt(__x); } 
# 497
using ::tan;
# 501
inline float tan(float __x) 
# 502
{ return __builtin_tanf(__x); } 
# 505
inline long double tan(long double __x) 
# 506
{ return __builtin_tanl(__x); } 
# 509
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 513
tan(_Tp __x) 
# 514
{ return __builtin_tan(__x); } 
# 516
using ::tanh;
# 520
inline float tanh(float __x) 
# 521
{ return __builtin_tanhf(__x); } 
# 524
inline long double tanh(long double __x) 
# 525
{ return __builtin_tanhl(__x); } 
# 528
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_integer< _Tp> ::__value, double> ::__type 
# 532
tanh(_Tp __x) 
# 533
{ return __builtin_tanh(__x); } 
# 536
}
# 555 "/usr/include/c++/5/cmath" 3
namespace std __attribute((__visibility__("default"))) { 
# 831 "/usr/include/c++/5/cmath" 3
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 834
fpclassify(_Tp __f) 
# 835
{ 
# 836
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 837
return __builtin_fpclassify(0, 1, 4, 3, 2, (__type)__f); 
# 839
} 
# 841
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 844
isfinite(_Tp __f) 
# 845
{ 
# 846
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 847
return __builtin_isfinite((__type)__f); 
# 848
} 
# 850
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 853
isinf(_Tp __f) 
# 854
{ 
# 855
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 856
return __builtin_isinf((__type)__f); 
# 857
} 
# 859
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 862
isnan(_Tp __f) 
# 863
{ 
# 864
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 865
return __builtin_isnan((__type)__f); 
# 866
} 
# 868
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 871
isnormal(_Tp __f) 
# 872
{ 
# 873
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 874
return __builtin_isnormal((__type)__f); 
# 875
} 
# 877
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 880
signbit(_Tp __f) 
# 881
{ 
# 882
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 883
return (sizeof(__type) == sizeof(float)) ? __builtin_signbitf((__type)__f) : ((sizeof(__type) == sizeof(double)) ? __builtin_signbit((__type)__f) : __builtin_signbitl((__type)__f)); 
# 888
} 
# 890
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 893
isgreater(_Tp __f1, _Tp __f2) 
# 894
{ 
# 895
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 896
return __builtin_isgreater((__type)__f1, (__type)__f2); 
# 897
} 
# 899
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 902
isgreaterequal(_Tp __f1, _Tp __f2) 
# 903
{ 
# 904
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 905
return __builtin_isgreaterequal((__type)__f1, (__type)__f2); 
# 906
} 
# 908
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 911
isless(_Tp __f1, _Tp __f2) 
# 912
{ 
# 913
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 914
return __builtin_isless((__type)__f1, (__type)__f2); 
# 915
} 
# 917
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 920
islessequal(_Tp __f1, _Tp __f2) 
# 921
{ 
# 922
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 923
return __builtin_islessequal((__type)__f1, (__type)__f2); 
# 924
} 
# 926
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 929
islessgreater(_Tp __f1, _Tp __f2) 
# 930
{ 
# 931
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 932
return __builtin_islessgreater((__type)__f1, (__type)__f2); 
# 933
} 
# 935
template< class _Tp> inline typename __gnu_cxx::__enable_if< __is_arithmetic< _Tp> ::__value, int> ::__type 
# 938
isunordered(_Tp __f1, _Tp __f2) 
# 939
{ 
# 940
typedef typename __gnu_cxx::__promote< _Tp> ::__type __type; 
# 941
return __builtin_isunordered((__type)__f1, (__type)__f2); 
# 942
} 
# 947
}
# 114 "/usr/include/c++/5/cstdlib" 3
namespace std __attribute((__visibility__("default"))) { 
# 118
using ::div_t;
# 119
using ::ldiv_t;
# 121
using ::abort;
# 122
using ::abs;
# 123
using ::atexit;
# 129
using ::atof;
# 130
using ::atoi;
# 131
using ::atol;
# 132
using ::bsearch;
# 133
using ::calloc;
# 134
using ::div;
# 135
using ::exit;
# 136
using ::free;
# 137
using ::getenv;
# 138
using ::labs;
# 139
using ::ldiv;
# 140
using ::malloc;
# 142
using ::mblen;
# 143
using ::mbstowcs;
# 144
using ::mbtowc;
# 146
using ::qsort;
# 152
using ::rand;
# 153
using ::realloc;
# 154
using ::srand;
# 155
using ::strtod;
# 156
using ::strtol;
# 157
using ::strtoul;
# 158
using ::system;
# 160
using ::wcstombs;
# 161
using ::wctomb;
# 166
inline long abs(long __i) { return __builtin_labs(__i); } 
# 169
inline ldiv_t div(long __i, long __j) { return ldiv(__i, __j); } 
# 174
inline long long abs(long long __x) { return __builtin_llabs(__x); } 
# 179
inline __int128_t abs(__int128_t __x) { return (__x >= (0)) ? __x : (-__x); } 
# 196 "/usr/include/c++/5/cstdlib" 3
}
# 209 "/usr/include/c++/5/cstdlib" 3
namespace __gnu_cxx __attribute((__visibility__("default"))) { 
# 214
using ::lldiv_t;
# 220
using ::_Exit;
# 224
using ::llabs;
# 227
inline lldiv_t div(long long __n, long long __d) 
# 228
{ lldiv_t __q; (__q.quot) = (__n / __d); (__q.rem) = (__n % __d); return __q; } 
# 230
using ::lldiv;
# 241 "/usr/include/c++/5/cstdlib" 3
using ::atoll;
# 242
using ::strtoll;
# 243
using ::strtoull;
# 245
using ::strtof;
# 246
using ::strtold;
# 249
}
# 251
namespace std { 
# 254
using __gnu_cxx::lldiv_t;
# 256
using __gnu_cxx::_Exit;
# 258
using __gnu_cxx::llabs;
# 259
using __gnu_cxx::div;
# 260
using __gnu_cxx::lldiv;
# 262
using __gnu_cxx::atoll;
# 263
using __gnu_cxx::strtof;
# 264
using __gnu_cxx::strtoll;
# 265
using __gnu_cxx::strtoull;
# 266
using __gnu_cxx::strtold;
# 267
}
# 8942 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
__attribute((always_inline)) inline int signbit(float x); 
# 8946
__attribute((always_inline)) inline int signbit(double x); 
# 8948
__attribute((always_inline)) inline int signbit(long double x); 
# 8950
__attribute((always_inline)) inline int isfinite(float x); 
# 8954
__attribute((always_inline)) inline int isfinite(double x); 
# 8956
__attribute((always_inline)) inline int isfinite(long double x); 
# 8958
__attribute((always_inline)) inline int isnan(float x); 
# 8962
extern "C" __attribute((always_inline)) inline int isnan(double x) throw(); 
# 8964
__attribute((always_inline)) inline int isnan(long double x); 
# 8966
__attribute((always_inline)) inline int isinf(float x); 
# 8970
extern "C" __attribute((always_inline)) inline int isinf(double x) throw(); 
# 8972
__attribute((always_inline)) inline int isinf(long double x); 
# 9029 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9031
template< class T> extern T __pow_helper(T, int); 
# 9032
template< class T> extern T __cmath_power(T, unsigned); 
# 9033
}
# 9035
using std::abs;
# 9036
using std::fabs;
# 9037
using std::ceil;
# 9038
using std::floor;
# 9039
using std::sqrt;
# 9041
using std::pow;
# 9043
using std::log;
# 9044
using std::log10;
# 9045
using std::fmod;
# 9046
using std::modf;
# 9047
using std::exp;
# 9048
using std::frexp;
# 9049
using std::ldexp;
# 9050
using std::asin;
# 9051
using std::sin;
# 9052
using std::sinh;
# 9053
using std::acos;
# 9054
using std::cos;
# 9055
using std::cosh;
# 9056
using std::atan;
# 9057
using std::atan2;
# 9058
using std::tan;
# 9059
using std::tanh;
# 9424 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
namespace std { 
# 9433 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline long long abs(long long); 
# 9443 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline long abs(long); 
# 9444
extern inline float abs(float); 
# 9445
extern inline double abs(double); 
# 9446
extern inline float fabs(float); 
# 9447
extern inline float ceil(float); 
# 9448
extern inline float floor(float); 
# 9449
extern inline float sqrt(float); 
# 9450
extern inline float pow(float, float); 
# 9459 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
extern inline float pow(float, int); 
# 9460
extern inline double pow(double, int); 
# 9465
extern inline float log(float); 
# 9466
extern inline float log10(float); 
# 9467
extern inline float fmod(float, float); 
# 9468
extern inline float modf(float, float *); 
# 9469
extern inline float exp(float); 
# 9470
extern inline float frexp(float, int *); 
# 9471
extern inline float ldexp(float, int); 
# 9472
extern inline float asin(float); 
# 9473
extern inline float sin(float); 
# 9474
extern inline float sinh(float); 
# 9475
extern inline float acos(float); 
# 9476
extern inline float cos(float); 
# 9477
extern inline float cosh(float); 
# 9478
extern inline float atan(float); 
# 9479
extern inline float atan2(float, float); 
# 9480
extern inline float tan(float); 
# 9481
extern inline float tanh(float); 
# 9555 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
}
# 9698 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float logb(float a); 
# 9700
static inline int ilogb(float a); 
# 9702
static inline float scalbn(float a, int b); 
# 9704
static inline float scalbln(float a, long b); 
# 9706
static inline float exp2(float a); 
# 9708
static inline float expm1(float a); 
# 9710
static inline float log2(float a); 
# 9712
static inline float log1p(float a); 
# 9714
static inline float acosh(float a); 
# 9716
static inline float asinh(float a); 
# 9718
static inline float atanh(float a); 
# 9720
static inline float hypot(float a, float b); 
# 9722
static inline float norm3d(float a, float b, float c); 
# 9724
static inline float norm4d(float a, float b, float c, float d); 
# 9726
static inline float cbrt(float a); 
# 9728
static inline float erf(float a); 
# 9730
static inline float erfc(float a); 
# 9732
static inline float lgamma(float a); 
# 9734
static inline float tgamma(float a); 
# 9736
static inline float copysign(float a, float b); 
# 9738
static inline float nextafter(float a, float b); 
# 9740
static inline float remainder(float a, float b); 
# 9742
static inline float remquo(float a, float b, int * quo); 
# 9744
static inline float round(float a); 
# 9746
static inline long lround(float a); 
# 9748
static inline long long llround(float a); 
# 9750
static inline float trunc(float a); 
# 9752
static inline float rint(float a); 
# 9754
static inline long lrint(float a); 
# 9756
static inline long long llrint(float a); 
# 9758
static inline float nearbyint(float a); 
# 9760
static inline float fdim(float a, float b); 
# 9762
static inline float fma(float a, float b, float c); 
# 9764
static inline float fmax(float a, float b); 
# 9766
static inline float fmin(float a, float b); 
# 9807 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.h"
static inline float exp10(float a); 
# 9809
static inline float rsqrt(float a); 
# 9811
static inline float rcbrt(float a); 
# 9813
static inline float sinpi(float a); 
# 9815
static inline float cospi(float a); 
# 9817
static inline void sincospi(float a, float * sptr, float * cptr); 
# 9819
static inline void sincos(float a, float * sptr, float * cptr); 
# 9821
static inline float j0(float a); 
# 9823
static inline float j1(float a); 
# 9825
static inline float jn(int n, float a); 
# 9827
static inline float y0(float a); 
# 9829
static inline float y1(float a); 
# 9831
static inline float yn(int n, float a); 
# 9833
static inline float cyl_bessel_i0(float a); 
# 9835
static inline float cyl_bessel_i1(float a); 
# 9837
static inline float erfinv(float a); 
# 9839
static inline float erfcinv(float a); 
# 9841
static inline float normcdfinv(float a); 
# 9843
static inline float normcdf(float a); 
# 9845
static inline float erfcx(float a); 
# 9847
static inline double copysign(double a, float b); 
# 9849
static inline double copysign(float a, double b); 
# 9851
static inline unsigned min(unsigned a, unsigned b); 
# 9853
static inline unsigned min(int a, unsigned b); 
# 9855
static inline unsigned min(unsigned a, int b); 
# 9857
static inline long min(long a, long b); 
# 9859
static inline unsigned long min(unsigned long a, unsigned long b); 
# 9861
static inline unsigned long min(long a, unsigned long b); 
# 9863
static inline unsigned long min(unsigned long a, long b); 
# 9865
static inline long long min(long long a, long long b); 
# 9867
static inline unsigned long long min(unsigned long long a, unsigned long long b); 
# 9869
static inline unsigned long long min(long long a, unsigned long long b); 
# 9871
static inline unsigned long long min(unsigned long long a, long long b); 
# 9873
static inline float min(float a, float b); 
# 9875
static inline double min(double a, double b); 
# 9877
static inline double min(float a, double b); 
# 9879
static inline double min(double a, float b); 
# 9881
static inline unsigned max(unsigned a, unsigned b); 
# 9883
static inline unsigned max(int a, unsigned b); 
# 9885
static inline unsigned max(unsigned a, int b); 
# 9887
static inline long max(long a, long b); 
# 9889
static inline unsigned long max(unsigned long a, unsigned long b); 
# 9891
static inline unsigned long max(long a, unsigned long b); 
# 9893
static inline unsigned long max(unsigned long a, long b); 
# 9895
static inline long long max(long long a, long long b); 
# 9897
static inline unsigned long long max(unsigned long long a, unsigned long long b); 
# 9899
static inline unsigned long long max(long long a, unsigned long long b); 
# 9901
static inline unsigned long long max(unsigned long long a, long long b); 
# 9903
static inline float max(float a, float b); 
# 9905
static inline double max(double a, double b); 
# 9907
static inline double max(float a, double b); 
# 9909
static inline double max(double a, float b); 
# 281 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int signbit(float x) { return __signbitf(x); } 
# 285
__attribute((always_inline)) inline int signbit(double x) { return __signbit(x); } 
# 287
__attribute((always_inline)) inline int signbit(long double x) { return __signbitl(x); } 
# 298 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(float x) { return __finitef(x); } 
# 313 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(double x) { return __finite(x); } 
# 326 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
__attribute((always_inline)) inline int isfinite(long double x) { return __finitel(x); } 
# 329
__attribute((always_inline)) inline int isnan(float x) { return __isnanf(x); } 
# 333
extern "C" __attribute((always_inline)) inline int isnan(double x) throw() { return __isnan(x); } 
# 335
__attribute((always_inline)) inline int isnan(long double x) { return __isnanl(x); } 
# 337
__attribute((always_inline)) inline int isinf(float x) { return __isinff(x); } 
# 341
extern "C" __attribute((always_inline)) inline int isinf(double x) throw() { return __isinf(x); } 
# 343
__attribute((always_inline)) inline int isinf(long double x) { return __isinfl(x); } 
# 549 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
static inline float logb(float a) 
# 550
{ 
# 551
return logbf(a); 
# 552
} 
# 554
static inline int ilogb(float a) 
# 555
{ 
# 556
return ilogbf(a); 
# 557
} 
# 559
static inline float scalbn(float a, int b) 
# 560
{ 
# 561
return scalbnf(a, b); 
# 562
} 
# 564
static inline float scalbln(float a, long b) 
# 565
{ 
# 566
return scalblnf(a, b); 
# 567
} 
# 569
static inline float exp2(float a) 
# 570
{ 
# 571
return exp2f(a); 
# 572
} 
# 574
static inline float expm1(float a) 
# 575
{ 
# 576
return expm1f(a); 
# 577
} 
# 579
static inline float log2(float a) 
# 580
{ 
# 581
return log2f(a); 
# 582
} 
# 584
static inline float log1p(float a) 
# 585
{ 
# 586
return log1pf(a); 
# 587
} 
# 589
static inline float acosh(float a) 
# 590
{ 
# 591
return acoshf(a); 
# 592
} 
# 594
static inline float asinh(float a) 
# 595
{ 
# 596
return asinhf(a); 
# 597
} 
# 599
static inline float atanh(float a) 
# 600
{ 
# 601
return atanhf(a); 
# 602
} 
# 604
static inline float hypot(float a, float b) 
# 605
{ 
# 606
return hypotf(a, b); 
# 607
} 
# 609
static inline float norm3d(float a, float b, float c) 
# 610
{ 
# 611
return norm3df(a, b, c); 
# 612
} 
# 614
static inline float norm4d(float a, float b, float c, float d) 
# 615
{ 
# 616
return norm4df(a, b, c, d); 
# 617
} 
# 619
static inline float cbrt(float a) 
# 620
{ 
# 621
return cbrtf(a); 
# 622
} 
# 624
static inline float erf(float a) 
# 625
{ 
# 626
return erff(a); 
# 627
} 
# 629
static inline float erfc(float a) 
# 630
{ 
# 631
return erfcf(a); 
# 632
} 
# 634
static inline float lgamma(float a) 
# 635
{ 
# 636
return lgammaf(a); 
# 637
} 
# 639
static inline float tgamma(float a) 
# 640
{ 
# 641
return tgammaf(a); 
# 642
} 
# 644
static inline float copysign(float a, float b) 
# 645
{ 
# 646
return copysignf(a, b); 
# 647
} 
# 649
static inline float nextafter(float a, float b) 
# 650
{ 
# 651
return nextafterf(a, b); 
# 652
} 
# 654
static inline float remainder(float a, float b) 
# 655
{ 
# 656
return remainderf(a, b); 
# 657
} 
# 659
static inline float remquo(float a, float b, int *quo) 
# 660
{ 
# 661
return remquof(a, b, quo); 
# 662
} 
# 664
static inline float round(float a) 
# 665
{ 
# 666
return roundf(a); 
# 667
} 
# 669
static inline long lround(float a) 
# 670
{ 
# 671
return lroundf(a); 
# 672
} 
# 674
static inline long long llround(float a) 
# 675
{ 
# 676
return llroundf(a); 
# 677
} 
# 679
static inline float trunc(float a) 
# 680
{ 
# 681
return truncf(a); 
# 682
} 
# 684
static inline float rint(float a) 
# 685
{ 
# 686
return rintf(a); 
# 687
} 
# 689
static inline long lrint(float a) 
# 690
{ 
# 691
return lrintf(a); 
# 692
} 
# 694
static inline long long llrint(float a) 
# 695
{ 
# 696
return llrintf(a); 
# 697
} 
# 699
static inline float nearbyint(float a) 
# 700
{ 
# 701
return nearbyintf(a); 
# 702
} 
# 704
static inline float fdim(float a, float b) 
# 705
{ 
# 706
return fdimf(a, b); 
# 707
} 
# 709
static inline float fma(float a, float b, float c) 
# 710
{ 
# 711
return fmaf(a, b, c); 
# 712
} 
# 714
static inline float fmax(float a, float b) 
# 715
{ 
# 716
return fmaxf(a, b); 
# 717
} 
# 719
static inline float fmin(float a, float b) 
# 720
{ 
# 721
return fminf(a, b); 
# 722
} 
# 730
static inline float exp10(float a) 
# 731
{ 
# 732
return exp10f(a); 
# 733
} 
# 735
static inline float rsqrt(float a) 
# 736
{ 
# 737
return rsqrtf(a); 
# 738
} 
# 740
static inline float rcbrt(float a) 
# 741
{ 
# 742
return rcbrtf(a); 
# 743
} 
# 745
static inline float sinpi(float a) 
# 746
{ 
# 747
return sinpif(a); 
# 748
} 
# 750
static inline float cospi(float a) 
# 751
{ 
# 752
return cospif(a); 
# 753
} 
# 755
static inline void sincospi(float a, float *sptr, float *cptr) 
# 756
{ 
# 757
sincospif(a, sptr, cptr); 
# 758
} 
# 760
static inline void sincos(float a, float *sptr, float *cptr) 
# 761
{ 
# 762
sincosf(a, sptr, cptr); 
# 763
} 
# 765
static inline float j0(float a) 
# 766
{ 
# 767
return j0f(a); 
# 768
} 
# 770
static inline float j1(float a) 
# 771
{ 
# 772
return j1f(a); 
# 773
} 
# 775
static inline float jn(int n, float a) 
# 776
{ 
# 777
return jnf(n, a); 
# 778
} 
# 780
static inline float y0(float a) 
# 781
{ 
# 782
return y0f(a); 
# 783
} 
# 785
static inline float y1(float a) 
# 786
{ 
# 787
return y1f(a); 
# 788
} 
# 790
static inline float yn(int n, float a) 
# 791
{ 
# 792
return ynf(n, a); 
# 793
} 
# 795
static inline float cyl_bessel_i0(float a) 
# 796
{ 
# 797
return cyl_bessel_i0f(a); 
# 798
} 
# 800
static inline float cyl_bessel_i1(float a) 
# 801
{ 
# 802
return cyl_bessel_i1f(a); 
# 803
} 
# 805
static inline float erfinv(float a) 
# 806
{ 
# 807
return erfinvf(a); 
# 808
} 
# 810
static inline float erfcinv(float a) 
# 811
{ 
# 812
return erfcinvf(a); 
# 813
} 
# 815
static inline float normcdfinv(float a) 
# 816
{ 
# 817
return normcdfinvf(a); 
# 818
} 
# 820
static inline float normcdf(float a) 
# 821
{ 
# 822
return normcdff(a); 
# 823
} 
# 825
static inline float erfcx(float a) 
# 826
{ 
# 827
return erfcxf(a); 
# 828
} 
# 830
static inline double copysign(double a, float b) 
# 831
{ 
# 832
return copysign(a, (double)b); 
# 833
} 
# 835
static inline double copysign(float a, double b) 
# 836
{ 
# 837
return copysign((double)a, b); 
# 838
} 
# 840
static inline unsigned min(unsigned a, unsigned b) 
# 841
{ 
# 842
return umin(a, b); 
# 843
} 
# 845
static inline unsigned min(int a, unsigned b) 
# 846
{ 
# 847
return umin((unsigned)a, b); 
# 848
} 
# 850
static inline unsigned min(unsigned a, int b) 
# 851
{ 
# 852
return umin(a, (unsigned)b); 
# 853
} 
# 855
static inline long min(long a, long b) 
# 856
{ 
# 862
if (sizeof(long) == sizeof(int)) { 
# 866
return (long)min((int)a, (int)b); 
# 867
} else { 
# 868
return (long)llmin((long long)a, (long long)b); 
# 869
}  
# 870
} 
# 872
static inline unsigned long min(unsigned long a, unsigned long b) 
# 873
{ 
# 877
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 881
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 882
} else { 
# 883
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 884
}  
# 885
} 
# 887
static inline unsigned long min(long a, unsigned long b) 
# 888
{ 
# 892
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 896
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 897
} else { 
# 898
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 899
}  
# 900
} 
# 902
static inline unsigned long min(unsigned long a, long b) 
# 903
{ 
# 907
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 911
return (unsigned long)umin((unsigned)a, (unsigned)b); 
# 912
} else { 
# 913
return (unsigned long)ullmin((unsigned long long)a, (unsigned long long)b); 
# 914
}  
# 915
} 
# 917
static inline long long min(long long a, long long b) 
# 918
{ 
# 919
return llmin(a, b); 
# 920
} 
# 922
static inline unsigned long long min(unsigned long long a, unsigned long long b) 
# 923
{ 
# 924
return ullmin(a, b); 
# 925
} 
# 927
static inline unsigned long long min(long long a, unsigned long long b) 
# 928
{ 
# 929
return ullmin((unsigned long long)a, b); 
# 930
} 
# 932
static inline unsigned long long min(unsigned long long a, long long b) 
# 933
{ 
# 934
return ullmin(a, (unsigned long long)b); 
# 935
} 
# 937
static inline float min(float a, float b) 
# 938
{ 
# 939
return fminf(a, b); 
# 940
} 
# 942
static inline double min(double a, double b) 
# 943
{ 
# 944
return fmin(a, b); 
# 945
} 
# 947
static inline double min(float a, double b) 
# 948
{ 
# 949
return fmin((double)a, b); 
# 950
} 
# 952
static inline double min(double a, float b) 
# 953
{ 
# 954
return fmin(a, (double)b); 
# 955
} 
# 957
static inline unsigned max(unsigned a, unsigned b) 
# 958
{ 
# 959
return umax(a, b); 
# 960
} 
# 962
static inline unsigned max(int a, unsigned b) 
# 963
{ 
# 964
return umax((unsigned)a, b); 
# 965
} 
# 967
static inline unsigned max(unsigned a, int b) 
# 968
{ 
# 969
return umax(a, (unsigned)b); 
# 970
} 
# 972
static inline long max(long a, long b) 
# 973
{ 
# 978
if (sizeof(long) == sizeof(int)) { 
# 982
return (long)max((int)a, (int)b); 
# 983
} else { 
# 984
return (long)llmax((long long)a, (long long)b); 
# 985
}  
# 986
} 
# 988
static inline unsigned long max(unsigned long a, unsigned long b) 
# 989
{ 
# 993
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 997
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 998
} else { 
# 999
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1000
}  
# 1001
} 
# 1003
static inline unsigned long max(long a, unsigned long b) 
# 1004
{ 
# 1008
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1012
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1013
} else { 
# 1014
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1015
}  
# 1016
} 
# 1018
static inline unsigned long max(unsigned long a, long b) 
# 1019
{ 
# 1023
if (sizeof(unsigned long) == sizeof(unsigned)) { 
# 1027
return (unsigned long)umax((unsigned)a, (unsigned)b); 
# 1028
} else { 
# 1029
return (unsigned long)ullmax((unsigned long long)a, (unsigned long long)b); 
# 1030
}  
# 1031
} 
# 1033
static inline long long max(long long a, long long b) 
# 1034
{ 
# 1035
return llmax(a, b); 
# 1036
} 
# 1038
static inline unsigned long long max(unsigned long long a, unsigned long long b) 
# 1039
{ 
# 1040
return ullmax(a, b); 
# 1041
} 
# 1043
static inline unsigned long long max(long long a, unsigned long long b) 
# 1044
{ 
# 1045
return ullmax((unsigned long long)a, b); 
# 1046
} 
# 1048
static inline unsigned long long max(unsigned long long a, long long b) 
# 1049
{ 
# 1050
return ullmax(a, (unsigned long long)b); 
# 1051
} 
# 1053
static inline float max(float a, float b) 
# 1054
{ 
# 1055
return fmaxf(a, b); 
# 1056
} 
# 1058
static inline double max(double a, double b) 
# 1059
{ 
# 1060
return fmax(a, b); 
# 1061
} 
# 1063
static inline double max(float a, double b) 
# 1064
{ 
# 1065
return fmax((double)a, b); 
# 1066
} 
# 1068
static inline double max(double a, float b) 
# 1069
{ 
# 1070
return fmax(a, (double)b); 
# 1071
} 
# 1082 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/math_functions.hpp"
extern "C" inline int min(int a, int b) 
# 1083
{ 
# 1084
return (a < b) ? a : b; 
# 1085
} 
# 1087
extern "C" inline unsigned umin(unsigned a, unsigned b) 
# 1088
{ 
# 1089
return (a < b) ? a : b; 
# 1090
} 
# 1092
extern "C" inline long long llmin(long long a, long long b) 
# 1093
{ 
# 1094
return (a < b) ? a : b; 
# 1095
} 
# 1097
extern "C" inline unsigned long long ullmin(unsigned long long a, unsigned long long 
# 1098
b) 
# 1099
{ 
# 1100
return (a < b) ? a : b; 
# 1101
} 
# 1103
extern "C" inline int max(int a, int b) 
# 1104
{ 
# 1105
return (a > b) ? a : b; 
# 1106
} 
# 1108
extern "C" inline unsigned umax(unsigned a, unsigned b) 
# 1109
{ 
# 1110
return (a > b) ? a : b; 
# 1111
} 
# 1113
extern "C" inline long long llmax(long long a, long long b) 
# 1114
{ 
# 1115
return (a > b) ? a : b; 
# 1116
} 
# 1118
extern "C" inline unsigned long long ullmax(unsigned long long a, unsigned long long 
# 1119
b) 
# 1120
{ 
# 1121
return (a > b) ? a : b; 
# 1122
} 
# 77 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_surface_types.h"
template< class T, int dim = 1> 
# 78
struct surface : public surfaceReference { 
# 81
surface() 
# 82
{ 
# 83
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 84
} 
# 86
surface(cudaChannelFormatDesc desc) 
# 87
{ 
# 88
(channelDesc) = desc; 
# 89
} 
# 91
}; 
# 93
template< int dim> 
# 94
struct surface< void, dim>  : public surfaceReference { 
# 97
surface() 
# 98
{ 
# 99
(channelDesc) = cudaCreateChannelDesc< void> (); 
# 100
} 
# 102
}; 
# 77 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_texture_types.h"
template< class T, int texType = 1, cudaTextureReadMode mode = cudaReadModeElementType> 
# 78
struct texture : public textureReference { 
# 81
texture(int norm = 0, cudaTextureFilterMode 
# 82
fMode = cudaFilterModePoint, cudaTextureAddressMode 
# 83
aMode = cudaAddressModeClamp) 
# 84
{ 
# 85
(normalized) = norm; 
# 86
(filterMode) = fMode; 
# 87
((addressMode)[0]) = aMode; 
# 88
((addressMode)[1]) = aMode; 
# 89
((addressMode)[2]) = aMode; 
# 90
(channelDesc) = cudaCreateChannelDesc< T> (); 
# 91
(sRGB) = 0; 
# 92
} 
# 94
texture(int norm, cudaTextureFilterMode 
# 95
fMode, cudaTextureAddressMode 
# 96
aMode, cudaChannelFormatDesc 
# 97
desc) 
# 98
{ 
# 99
(normalized) = norm; 
# 100
(filterMode) = fMode; 
# 101
((addressMode)[0]) = aMode; 
# 102
((addressMode)[1]) = aMode; 
# 103
((addressMode)[2]) = aMode; 
# 104
(channelDesc) = desc; 
# 105
(sRGB) = 0; 
# 106
} 
# 108
}; 
# 79 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.h"
extern "C" {
# 3201 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.h"
}
# 3209
__attribute__((unused)) static inline int mulhi(int a, int b); 
# 3211
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b); 
# 3213
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b); 
# 3215
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b); 
# 3217
__attribute__((unused)) static inline long long mul64hi(long long a, long long b); 
# 3219
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b); 
# 3221
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b); 
# 3223
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b); 
# 3225
__attribute__((unused)) static inline int float_as_int(float a); 
# 3227
__attribute__((unused)) static inline float int_as_float(int a); 
# 3229
__attribute__((unused)) static inline unsigned float_as_uint(float a); 
# 3231
__attribute__((unused)) static inline float uint_as_float(unsigned a); 
# 3233
__attribute__((unused)) static inline float saturate(float a); 
# 3235
__attribute__((unused)) static inline int mul24(int a, int b); 
# 3237
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b); 
# 3239
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode = cudaRoundZero); 
# 3241
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode = cudaRoundZero); 
# 3243
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode = cudaRoundNearest); 
# 3245
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 80 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mulhi(int a, int b) 
# 81
{int volatile ___ = 1;(void)a;(void)b;
# 83
::exit(___);}
#if 0
# 81
{ 
# 82
return __mulhi(a, b); 
# 83
} 
#endif
# 85 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, unsigned b) 
# 86
{int volatile ___ = 1;(void)a;(void)b;
# 88
::exit(___);}
#if 0
# 86
{ 
# 87
return __umulhi(a, b); 
# 88
} 
#endif
# 90 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(int a, unsigned b) 
# 91
{int volatile ___ = 1;(void)a;(void)b;
# 93
::exit(___);}
#if 0
# 91
{ 
# 92
return __umulhi((unsigned)a, b); 
# 93
} 
#endif
# 95 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned mulhi(unsigned a, int b) 
# 96
{int volatile ___ = 1;(void)a;(void)b;
# 98
::exit(___);}
#if 0
# 96
{ 
# 97
return __umulhi(a, (unsigned)b); 
# 98
} 
#endif
# 100 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline long long mul64hi(long long a, long long b) 
# 101
{int volatile ___ = 1;(void)a;(void)b;
# 103
::exit(___);}
#if 0
# 101
{ 
# 102
return __mul64hi(a, b); 
# 103
} 
#endif
# 105 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, unsigned long long b) 
# 106
{int volatile ___ = 1;(void)a;(void)b;
# 108
::exit(___);}
#if 0
# 106
{ 
# 107
return __umul64hi(a, b); 
# 108
} 
#endif
# 110 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(long long a, unsigned long long b) 
# 111
{int volatile ___ = 1;(void)a;(void)b;
# 113
::exit(___);}
#if 0
# 111
{ 
# 112
return __umul64hi((unsigned long long)a, b); 
# 113
} 
#endif
# 115 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned long long mul64hi(unsigned long long a, long long b) 
# 116
{int volatile ___ = 1;(void)a;(void)b;
# 118
::exit(___);}
#if 0
# 116
{ 
# 117
return __umul64hi(a, (unsigned long long)b); 
# 118
} 
#endif
# 120 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float_as_int(float a) 
# 121
{int volatile ___ = 1;(void)a;
# 123
::exit(___);}
#if 0
# 121
{ 
# 122
return __float_as_int(a); 
# 123
} 
#endif
# 125 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int_as_float(int a) 
# 126
{int volatile ___ = 1;(void)a;
# 128
::exit(___);}
#if 0
# 126
{ 
# 127
return __int_as_float(a); 
# 128
} 
#endif
# 130 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float_as_uint(float a) 
# 131
{int volatile ___ = 1;(void)a;
# 133
::exit(___);}
#if 0
# 131
{ 
# 132
return __float_as_uint(a); 
# 133
} 
#endif
# 135 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint_as_float(unsigned a) 
# 136
{int volatile ___ = 1;(void)a;
# 138
::exit(___);}
#if 0
# 136
{ 
# 137
return __uint_as_float(a); 
# 138
} 
#endif
# 139 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float saturate(float a) 
# 140
{int volatile ___ = 1;(void)a;
# 142
::exit(___);}
#if 0
# 140
{ 
# 141
return __saturatef(a); 
# 142
} 
#endif
# 144 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int mul24(int a, int b) 
# 145
{int volatile ___ = 1;(void)a;(void)b;
# 147
::exit(___);}
#if 0
# 145
{ 
# 146
return __mul24(a, b); 
# 147
} 
#endif
# 149 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned umul24(unsigned a, unsigned b) 
# 150
{int volatile ___ = 1;(void)a;(void)b;
# 152
::exit(___);}
#if 0
# 150
{ 
# 151
return __umul24(a, b); 
# 152
} 
#endif
# 154 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline int float2int(float a, cudaRoundMode mode) 
# 155
{int volatile ___ = 1;(void)a;(void)mode;
# 160
::exit(___);}
#if 0
# 155
{ 
# 156
return (mode == (cudaRoundNearest)) ? __float2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2int_rd(a) : __float2int_rz(a))); 
# 160
} 
#endif
# 162 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline unsigned float2uint(float a, cudaRoundMode mode) 
# 163
{int volatile ___ = 1;(void)a;(void)mode;
# 168
::exit(___);}
#if 0
# 163
{ 
# 164
return (mode == (cudaRoundNearest)) ? __float2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __float2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __float2uint_rd(a) : __float2uint_rz(a))); 
# 168
} 
#endif
# 170 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float int2float(int a, cudaRoundMode mode) 
# 171
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 171
{ 
# 172
return (mode == (cudaRoundZero)) ? __int2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __int2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __int2float_rd(a) : __int2float_rn(a))); 
# 176
} 
#endif
# 178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_functions.hpp"
__attribute__((unused)) static inline float uint2float(unsigned a, cudaRoundMode mode) 
# 179
{int volatile ___ = 1;(void)a;(void)mode;
# 184
::exit(___);}
#if 0
# 179
{ 
# 180
return (mode == (cudaRoundZero)) ? __uint2float_rz(a) : ((mode == (cudaRoundPosInf)) ? __uint2float_ru(a) : ((mode == (cudaRoundMinInf)) ? __uint2float_rd(a) : __uint2float_rn(a))); 
# 184
} 
#endif
# 107 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 117
{ } 
#endif
# 119 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 119
{ } 
#endif
# 121 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 121
{ } 
#endif
# 123 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 123
{ } 
#endif
# 125 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 125
{ } 
#endif
# 127 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 127
{ } 
#endif
# 129 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 129
{ } 
#endif
# 131 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 131
{ } 
#endif
# 133 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 133
{ } 
#endif
# 135 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 135
{ } 
#endif
# 137 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 137
{ } 
#endif
# 139 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 139
{ } 
#endif
# 141 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 141
{ } 
#endif
# 143 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 143
{ } 
#endif
# 145 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 145
{ } 
#endif
# 147 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS(unsigned *address, unsigned compare, unsigned val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 147
{ } 
#endif
# 173 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
extern "C" {
# 182
}
# 191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 191
{ } 
#endif
# 193 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 193
{ } 
#endif
# 195 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS(unsigned long long *address, unsigned long long compare, unsigned long long val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 195
{ } 
#endif
# 197 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__any() is deprecated in favor of __any_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool any(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 197
{ } 
#endif
# 199 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_atomic_functions.h"
__attribute((deprecated("__all() is deprecated in favor of __all_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppr" "ess this warning)."))) __attribute__((unused)) static inline bool all(bool cond) {int volatile ___ = 1;(void)cond;::exit(___);}
#if 0
# 199
{ } 
#endif
# 77 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
extern "C" {
# 1129 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.h"
}
# 1137
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode); 
# 1139
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1141
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1143
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode = cudaRoundNearest); 
# 1145
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode = cudaRoundZero); 
# 1147
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode = cudaRoundZero); 
# 1149
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode = cudaRoundZero); 
# 1151
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode = cudaRoundZero); 
# 1153
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1155
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode = cudaRoundNearest); 
# 1157
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode = cudaRoundNearest); 
# 1159
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode = cudaRoundNearest); 
# 1161
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode = cudaRoundNearest); 
# 83 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double fma(double a, double b, double c, cudaRoundMode mode) 
# 84
{int volatile ___ = 1;(void)a;(void)b;(void)c;(void)mode;
# 89
::exit(___);}
#if 0
# 84
{ 
# 85
return (mode == (cudaRoundZero)) ? __fma_rz(a, b, c) : ((mode == (cudaRoundPosInf)) ? __fma_ru(a, b, c) : ((mode == (cudaRoundMinInf)) ? __fma_rd(a, b, c) : __fma_rn(a, b, c))); 
# 89
} 
#endif
# 91 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dmul(double a, double b, cudaRoundMode mode) 
# 92
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 97
::exit(___);}
#if 0
# 92
{ 
# 93
return (mode == (cudaRoundZero)) ? __dmul_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dmul_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dmul_rd(a, b) : __dmul_rn(a, b))); 
# 97
} 
#endif
# 99 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dadd(double a, double b, cudaRoundMode mode) 
# 100
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 105
::exit(___);}
#if 0
# 100
{ 
# 101
return (mode == (cudaRoundZero)) ? __dadd_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dadd_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dadd_rd(a, b) : __dadd_rn(a, b))); 
# 105
} 
#endif
# 107 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double dsub(double a, double b, cudaRoundMode mode) 
# 108
{int volatile ___ = 1;(void)a;(void)b;(void)mode;
# 113
::exit(___);}
#if 0
# 108
{ 
# 109
return (mode == (cudaRoundZero)) ? __dsub_rz(a, b) : ((mode == (cudaRoundPosInf)) ? __dsub_ru(a, b) : ((mode == (cudaRoundMinInf)) ? __dsub_rd(a, b) : __dsub_rn(a, b))); 
# 113
} 
#endif
# 115 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline int double2int(double a, cudaRoundMode mode) 
# 116
{int volatile ___ = 1;(void)a;(void)mode;
# 121
::exit(___);}
#if 0
# 116
{ 
# 117
return (mode == (cudaRoundNearest)) ? __double2int_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2int_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2int_rd(a) : __double2int_rz(a))); 
# 121
} 
#endif
# 123 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned double2uint(double a, cudaRoundMode mode) 
# 124
{int volatile ___ = 1;(void)a;(void)mode;
# 129
::exit(___);}
#if 0
# 124
{ 
# 125
return (mode == (cudaRoundNearest)) ? __double2uint_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2uint_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2uint_rd(a) : __double2uint_rz(a))); 
# 129
} 
#endif
# 131 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline long long double2ll(double a, cudaRoundMode mode) 
# 132
{int volatile ___ = 1;(void)a;(void)mode;
# 137
::exit(___);}
#if 0
# 132
{ 
# 133
return (mode == (cudaRoundNearest)) ? __double2ll_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ll_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ll_rd(a) : __double2ll_rz(a))); 
# 137
} 
#endif
# 139 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline unsigned long long double2ull(double a, cudaRoundMode mode) 
# 140
{int volatile ___ = 1;(void)a;(void)mode;
# 145
::exit(___);}
#if 0
# 140
{ 
# 141
return (mode == (cudaRoundNearest)) ? __double2ull_rn(a) : ((mode == (cudaRoundPosInf)) ? __double2ull_ru(a) : ((mode == (cudaRoundMinInf)) ? __double2ull_rd(a) : __double2ull_rz(a))); 
# 145
} 
#endif
# 147 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ll2double(long long a, cudaRoundMode mode) 
# 148
{int volatile ___ = 1;(void)a;(void)mode;
# 153
::exit(___);}
#if 0
# 148
{ 
# 149
return (mode == (cudaRoundZero)) ? __ll2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ll2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ll2double_rd(a) : __ll2double_rn(a))); 
# 153
} 
#endif
# 155 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double ull2double(unsigned long long a, cudaRoundMode mode) 
# 156
{int volatile ___ = 1;(void)a;(void)mode;
# 161
::exit(___);}
#if 0
# 156
{ 
# 157
return (mode == (cudaRoundZero)) ? __ull2double_rz(a) : ((mode == (cudaRoundPosInf)) ? __ull2double_ru(a) : ((mode == (cudaRoundMinInf)) ? __ull2double_rd(a) : __ull2double_rn(a))); 
# 161
} 
#endif
# 163 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double int2double(int a, cudaRoundMode mode) 
# 164
{int volatile ___ = 1;(void)a;(void)mode;
# 166
::exit(___);}
#if 0
# 164
{ 
# 165
return (double)a; 
# 166
} 
#endif
# 168 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double uint2double(unsigned a, cudaRoundMode mode) 
# 169
{int volatile ___ = 1;(void)a;(void)mode;
# 171
::exit(___);}
#if 0
# 169
{ 
# 170
return (double)a; 
# 171
} 
#endif
# 173 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/device_double_functions.hpp"
__attribute__((unused)) static inline double float2double(float a, cudaRoundMode mode) 
# 174
{int volatile ___ = 1;(void)a;(void)mode;
# 176
::exit(___);}
#if 0
# 174
{ 
# 175
return (double)a; 
# 176
} 
#endif
# 90 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_20_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 90
{ } 
#endif
# 101 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 101
{ } 
#endif
# 103 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 103
{ } 
#endif
# 105 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 105
{ } 
#endif
# 107 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 107
{ } 
#endif
# 109 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 111
{ } 
#endif
# 113 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 113
{ } 
#endif
# 115 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 117
{ } 
#endif
# 119 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 119
{ } 
#endif
# 304 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 304
{ } 
#endif
# 307 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 307
{ } 
#endif
# 310 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAdd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 310
{ } 
#endif
# 313 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 313
{ } 
#endif
# 316 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAdd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 316
{ } 
#endif
# 319 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 319
{ } 
#endif
# 322 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAdd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 322
{ } 
#endif
# 325 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 325
{ } 
#endif
# 328 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicAdd_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 328
{ } 
#endif
# 331 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_block(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 331
{ } 
#endif
# 334 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline double atomicAdd_system(double *address, double val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 334
{ } 
#endif
# 337 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 337
{ } 
#endif
# 340 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicSub_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 340
{ } 
#endif
# 343 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 343
{ } 
#endif
# 346 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicSub_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 346
{ } 
#endif
# 349 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 349
{ } 
#endif
# 352 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicExch_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 352
{ } 
#endif
# 355 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 355
{ } 
#endif
# 358 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicExch_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 358
{ } 
#endif
# 361 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 361
{ } 
#endif
# 364 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicExch_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 364
{ } 
#endif
# 367 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_block(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 367
{ } 
#endif
# 370 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline float atomicExch_system(float *address, float val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 370
{ } 
#endif
# 373 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 373
{ } 
#endif
# 376 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMin_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 376
{ } 
#endif
# 379 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 379
{ } 
#endif
# 382 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMin_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 382
{ } 
#endif
# 385 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 385
{ } 
#endif
# 388 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMin_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 388
{ } 
#endif
# 391 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 391
{ } 
#endif
# 394 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMin_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 394
{ } 
#endif
# 397 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 397
{ } 
#endif
# 400 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicMax_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 400
{ } 
#endif
# 403 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 403
{ } 
#endif
# 406 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicMax_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 406
{ } 
#endif
# 409 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 409
{ } 
#endif
# 412 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicMax_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 412
{ } 
#endif
# 415 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 415
{ } 
#endif
# 418 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicMax_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 418
{ } 
#endif
# 421 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 421
{ } 
#endif
# 424 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicInc_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 424
{ } 
#endif
# 427 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 427
{ } 
#endif
# 430 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicDec_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 430
{ } 
#endif
# 433 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_block(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 433
{ } 
#endif
# 436 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicCAS_system(int *address, int compare, int val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 436
{ } 
#endif
# 439 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_block(unsigned *address, unsigned compare, unsigned 
# 440
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 440
{ } 
#endif
# 443 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicCAS_system(unsigned *address, unsigned compare, unsigned 
# 444
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 444
{ } 
#endif
# 447 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_block(unsigned long long *address, unsigned long long 
# 448
compare, unsigned long long 
# 449
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 449
{ } 
#endif
# 452 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicCAS_system(unsigned long long *address, unsigned long long 
# 453
compare, unsigned long long 
# 454
val) {int volatile ___ = 1;(void)address;(void)compare;(void)val;::exit(___);}
#if 0
# 454
{ } 
#endif
# 457 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 457
{ } 
#endif
# 460 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicAnd_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 460
{ } 
#endif
# 463 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 463
{ } 
#endif
# 466 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicAnd_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 466
{ } 
#endif
# 469 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 469
{ } 
#endif
# 472 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicAnd_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 472
{ } 
#endif
# 475 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 475
{ } 
#endif
# 478 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicAnd_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 478
{ } 
#endif
# 481 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 481
{ } 
#endif
# 484 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicOr_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 484
{ } 
#endif
# 487 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 487
{ } 
#endif
# 490 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicOr_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 490
{ } 
#endif
# 493 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 493
{ } 
#endif
# 496 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicOr_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 496
{ } 
#endif
# 499 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 499
{ } 
#endif
# 502 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicOr_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 502
{ } 
#endif
# 505 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_block(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 505
{ } 
#endif
# 508 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline int atomicXor_system(int *address, int val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 508
{ } 
#endif
# 511 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_block(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 511
{ } 
#endif
# 514 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline long long atomicXor_system(long long *address, long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 514
{ } 
#endif
# 517 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_block(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 517
{ } 
#endif
# 520 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned atomicXor_system(unsigned *address, unsigned val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 520
{ } 
#endif
# 523 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_block(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 523
{ } 
#endif
# 526 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_60_atomic_functions.h"
__attribute__((unused)) static inline unsigned long long atomicXor_system(unsigned long long *address, unsigned long long val) {int volatile ___ = 1;(void)address;(void)val;::exit(___);}
#if 0
# 526
{ } 
#endif
# 92 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
extern "C" {
# 1477 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
}
# 1484
__attribute((deprecated("__ballot() is deprecated in favor of __ballot_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to" " suppress this warning)."))) __attribute__((unused)) static inline unsigned ballot(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1484
{ } 
#endif
# 1486 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline int syncthreads_count(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1486
{ } 
#endif
# 1488 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_and(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1488
{ } 
#endif
# 1490 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline bool syncthreads_or(bool pred) {int volatile ___ = 1;(void)pred;::exit(___);}
#if 0
# 1490
{ } 
#endif
# 1497 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_20_intrinsics.h"
__attribute__((unused)) static inline unsigned __isGlobal(const void *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 1497
{ } 
#endif
# 107 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __fns(unsigned mask, unsigned base, int offset) {int volatile ___ = 1;(void)mask;(void)base;(void)offset;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync(unsigned id) {int volatile ___ = 1;(void)id;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __barrier_sync_count(unsigned id, unsigned cnt) {int volatile ___ = 1;(void)id;(void)cnt;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline void __syncwarp(unsigned mask = 4294967295U) {int volatile ___ = 1;(void)mask;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __all_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __any_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __uni_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __ballot_sync(unsigned mask, int pred) {int volatile ___ = 1;(void)mask;(void)pred;::exit(___);}
#if 0
# 114
{ } 
#endif
# 115 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __activemask() {int volatile ___ = 1;::exit(___);}
#if 0
# 115
{ } 
#endif
# 123 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline int __shfl(int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 123
{ } 
#endif
# 124 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_sync(unsigned mask, int var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 124
{ } 
#endif
# 126 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned __shfl(unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 126
{ } 
#endif
# 127 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_sync(unsigned mask, unsigned var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 127
{ } 
#endif
# 129 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_up(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_up_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 130
{ } 
#endif
# 132 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_up(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_up_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 133
{ } 
#endif
# 135 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_down(int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_down_sync(unsigned mask, int var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 136
{ } 
#endif
# 138 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_down(unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 138
{ } 
#endif
# 139 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_down_sync(unsigned mask, unsigned var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 139
{ } 
#endif
# 141 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline int __shfl_xor(int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline int __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 142
{ } 
#endif
# 144 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned __shfl_xor(unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned __shfl_xor_sync(unsigned mask, unsigned var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 145
{ } 
#endif
# 147 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline float __shfl(float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_sync(unsigned mask, float var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 148
{ } 
#endif
# 150 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_up(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_up_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 151
{ } 
#endif
# 153 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_down(float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_down_sync(unsigned mask, float var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 154
{ } 
#endif
# 156 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline float __shfl_xor(float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 157
{ } 
#endif
# 160 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long long __shfl(long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 160
{ } 
#endif
# 161 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_sync(unsigned mask, long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 161
{ } 
#endif
# 163 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl(unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 163
{ } 
#endif
# 164 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_sync(unsigned mask, unsigned long long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 164
{ } 
#endif
# 166 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_up(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_up_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 167
{ } 
#endif
# 169 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_up(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_up_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 170
{ } 
#endif
# 172 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_down(long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_down_sync(unsigned mask, long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 173
{ } 
#endif
# 175 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_down(unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 175
{ } 
#endif
# 176 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_down_sync(unsigned mask, unsigned long long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 176
{ } 
#endif
# 178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long long __shfl_xor(long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long long __shfl_xor_sync(unsigned mask, long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 179
{ } 
#endif
# 181 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long long __shfl_xor(unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __shfl_xor_sync(unsigned mask, unsigned long long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 182
{ } 
#endif
# 184 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline double __shfl(double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_sync(unsigned mask, double var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 185
{ } 
#endif
# 187 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_up(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 187
{ } 
#endif
# 188 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_up_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 188
{ } 
#endif
# 190 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_down(double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_down_sync(unsigned mask, double var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 191
{ } 
#endif
# 193 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline double __shfl_xor(double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 193
{ } 
#endif
# 194 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 194
{ } 
#endif
# 198 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline long __shfl(long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 198
{ } 
#endif
# 199 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_sync(unsigned mask, long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 199
{ } 
#endif
# 201 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl() is deprecated in favor of __shfl_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to sup" "press this warning)."))) __attribute__((unused)) static inline unsigned long __shfl(unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_sync(unsigned mask, unsigned long var, int srcLane, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)srcLane;(void)width;::exit(___);}
#if 0
# 202
{ } 
#endif
# 204 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_up(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_up_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 205
{ } 
#endif
# 207 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_up() is deprecated in favor of __shfl_up_sync() and may be removed in a future release (Use -Wno-deprecated-declarations " "to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_up(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_up_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 208
{ } 
#endif
# 210 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_down(long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 210
{ } 
#endif
# 211 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_down_sync(unsigned mask, long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 211
{ } 
#endif
# 213 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_down() is deprecated in favor of __shfl_down_sync() and may be removed in a future release (Use -Wno-deprecated-declarati" "ons to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_down(unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_down_sync(unsigned mask, unsigned long var, unsigned delta, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)delta;(void)width;::exit(___);}
#if 0
# 214
{ } 
#endif
# 216 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline long __shfl_xor(long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline long __shfl_xor_sync(unsigned mask, long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 217
{ } 
#endif
# 219 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute((deprecated("__shfl_xor() is deprecated in favor of __shfl_xor_sync() and may be removed in a future release (Use -Wno-deprecated-declaration" "s to suppress this warning)."))) __attribute__((unused)) static inline unsigned long __shfl_xor(unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_30_intrinsics.h"
__attribute__((unused)) static inline unsigned long __shfl_xor_sync(unsigned mask, unsigned long var, int laneMask, int width = 32) {int volatile ___ = 1;(void)mask;(void)var;(void)laneMask;(void)width;::exit(___);}
#if 0
# 220
{ } 
#endif
# 89 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 99
{ } 
#endif
# 100 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 101
{ } 
#endif
# 102 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 102
{ } 
#endif
# 103 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 103
{ } 
#endif
# 105 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 105
{ } 
#endif
# 106 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 106
{ } 
#endif
# 107 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 107
{ } 
#endif
# 108 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 109
{ } 
#endif
# 110 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 110
{ } 
#endif
# 111 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 112
{ } 
#endif
# 113 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 113
{ } 
#endif
# 114 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 114
{ } 
#endif
# 115 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 115
{ } 
#endif
# 117 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 117
{ } 
#endif
# 118 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 118
{ } 
#endif
# 119 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 119
{ } 
#endif
# 120 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 120
{ } 
#endif
# 121 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 121
{ } 
#endif
# 125 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcg(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 125
{ } 
#endif
# 126 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcg(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 126
{ } 
#endif
# 128 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcg(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 128
{ } 
#endif
# 129 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcg(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 129
{ } 
#endif
# 130 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcg(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 130
{ } 
#endif
# 131 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcg(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 131
{ } 
#endif
# 132 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcg(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 132
{ } 
#endif
# 133 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcg(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 133
{ } 
#endif
# 134 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcg(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 134
{ } 
#endif
# 135 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcg(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 135
{ } 
#endif
# 136 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcg(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 136
{ } 
#endif
# 137 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcg(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 137
{ } 
#endif
# 138 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcg(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 138
{ } 
#endif
# 139 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcg(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 139
{ } 
#endif
# 141 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcg(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 141
{ } 
#endif
# 142 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcg(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 142
{ } 
#endif
# 143 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcg(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 143
{ } 
#endif
# 144 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcg(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 144
{ } 
#endif
# 145 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcg(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 145
{ } 
#endif
# 146 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcg(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 146
{ } 
#endif
# 147 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcg(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 147
{ } 
#endif
# 148 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcg(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 148
{ } 
#endif
# 149 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcg(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 149
{ } 
#endif
# 150 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcg(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 150
{ } 
#endif
# 151 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcg(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 151
{ } 
#endif
# 153 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcg(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 153
{ } 
#endif
# 154 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcg(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 154
{ } 
#endif
# 155 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcg(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 155
{ } 
#endif
# 156 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcg(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 156
{ } 
#endif
# 157 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcg(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 157
{ } 
#endif
# 161 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldca(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 161
{ } 
#endif
# 162 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldca(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 162
{ } 
#endif
# 164 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldca(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 164
{ } 
#endif
# 165 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldca(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 165
{ } 
#endif
# 166 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldca(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 166
{ } 
#endif
# 167 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldca(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 167
{ } 
#endif
# 168 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldca(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 168
{ } 
#endif
# 169 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldca(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 169
{ } 
#endif
# 170 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldca(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 170
{ } 
#endif
# 171 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldca(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 171
{ } 
#endif
# 172 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldca(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 172
{ } 
#endif
# 173 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldca(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 173
{ } 
#endif
# 174 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldca(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 174
{ } 
#endif
# 175 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldca(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 175
{ } 
#endif
# 177 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldca(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 177
{ } 
#endif
# 178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldca(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 178
{ } 
#endif
# 179 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldca(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 179
{ } 
#endif
# 180 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldca(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 180
{ } 
#endif
# 181 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldca(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 181
{ } 
#endif
# 182 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldca(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 182
{ } 
#endif
# 183 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldca(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 183
{ } 
#endif
# 184 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldca(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 184
{ } 
#endif
# 185 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldca(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 185
{ } 
#endif
# 186 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldca(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 186
{ } 
#endif
# 187 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldca(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 187
{ } 
#endif
# 189 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldca(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 189
{ } 
#endif
# 190 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldca(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 190
{ } 
#endif
# 191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldca(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 191
{ } 
#endif
# 192 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldca(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 192
{ } 
#endif
# 193 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldca(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 193
{ } 
#endif
# 197 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long __ldcs(const long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 197
{ } 
#endif
# 198 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long __ldcs(const unsigned long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 198
{ } 
#endif
# 200 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char __ldcs(const char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 200
{ } 
#endif
# 201 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline signed char __ldcs(const signed char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 201
{ } 
#endif
# 202 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short __ldcs(const short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 202
{ } 
#endif
# 203 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int __ldcs(const int *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 203
{ } 
#endif
# 204 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline long long __ldcs(const long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 204
{ } 
#endif
# 205 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char2 __ldcs(const char2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 205
{ } 
#endif
# 206 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline char4 __ldcs(const char4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 206
{ } 
#endif
# 207 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short2 __ldcs(const short2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 207
{ } 
#endif
# 208 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline short4 __ldcs(const short4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 208
{ } 
#endif
# 209 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int2 __ldcs(const int2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 209
{ } 
#endif
# 210 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline int4 __ldcs(const int4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 210
{ } 
#endif
# 211 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline longlong2 __ldcs(const longlong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 211
{ } 
#endif
# 213 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned char __ldcs(const unsigned char *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 213
{ } 
#endif
# 214 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned short __ldcs(const unsigned short *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 214
{ } 
#endif
# 215 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __ldcs(const unsigned *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 215
{ } 
#endif
# 216 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned long long __ldcs(const unsigned long long *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 216
{ } 
#endif
# 217 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar2 __ldcs(const uchar2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 217
{ } 
#endif
# 218 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uchar4 __ldcs(const uchar4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 218
{ } 
#endif
# 219 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort2 __ldcs(const ushort2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 219
{ } 
#endif
# 220 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ushort4 __ldcs(const ushort4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 220
{ } 
#endif
# 221 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint2 __ldcs(const uint2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 221
{ } 
#endif
# 222 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline uint4 __ldcs(const uint4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 222
{ } 
#endif
# 223 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline ulonglong2 __ldcs(const ulonglong2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 223
{ } 
#endif
# 225 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float __ldcs(const float *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 225
{ } 
#endif
# 226 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double __ldcs(const double *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 226
{ } 
#endif
# 227 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float2 __ldcs(const float2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 227
{ } 
#endif
# 228 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline float4 __ldcs(const float4 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 228
{ } 
#endif
# 229 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline double2 __ldcs(const double2 *ptr) {int volatile ___ = 1;(void)ptr;::exit(___);}
#if 0
# 229
{ } 
#endif
# 236 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 236
{ } 
#endif
# 237 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_lc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 237
{ } 
#endif
# 240 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 240
{ } 
#endif
# 241 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_32_intrinsics.h"
__attribute__((unused)) static inline unsigned __funnelshift_rc(unsigned lo, unsigned hi, unsigned shift) {int volatile ___ = 1;(void)lo;(void)hi;(void)shift;::exit(___);}
#if 0
# 241
{ } 
#endif
# 91 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 91
{ } 
#endif
# 92 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 92
{ } 
#endif
# 94 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_lo(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 95
{ } 
#endif
# 97 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 98
{ } 
#endif
# 100 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp2a_hi(short2 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 100
{ } 
#endif
# 101 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 101
{ } 
#endif
# 108 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(int srcA, int srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 108
{ } 
#endif
# 109 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(unsigned srcA, unsigned srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 109
{ } 
#endif
# 111 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline int __dp4a(char4 srcA, char4 srcB, int c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 111
{ } 
#endif
# 112 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/sm_61_intrinsics.h"
__attribute__((unused)) static inline unsigned __dp4a(uchar4 srcA, uchar4 srcB, unsigned c) {int volatile ___ = 1;(void)srcA;(void)srcB;(void)c;::exit(___);}
#if 0
# 112
{ } 
#endif
# 83 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 83
{ } 
#endif
# 84 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, int value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 84
{ } 
#endif
# 85 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 85
{ } 
#endif
# 86 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 86
{ } 
#endif
# 87 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, unsigned long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 87
{ } 
#endif
# 88 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, long long value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 88
{ } 
#endif
# 89 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, float value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 89
{ } 
#endif
# 90 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_any_sync(unsigned mask, double value) {int volatile ___ = 1;(void)mask;(void)value;::exit(___);}
#if 0
# 90
{ } 
#endif
# 92 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 92
{ } 
#endif
# 93 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, int value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 93
{ } 
#endif
# 94 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 94
{ } 
#endif
# 95 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 95
{ } 
#endif
# 96 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, unsigned long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 96
{ } 
#endif
# 97 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, long long value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 97
{ } 
#endif
# 98 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, float value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 98
{ } 
#endif
# 99 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/crt/sm_70_rt.h"
__attribute__((unused)) static inline unsigned __match_all_sync(unsigned mask, double value, int *pred) {int volatile ___ = 1;(void)mask;(void)value;(void)pred;::exit(___);}
#if 0
# 99
{ } 
#endif
# 116 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 117
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 118
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)s;(void)mode;
# 122
::exit(___);}
#if 0
# 118
{ 
# 122
} 
#endif
# 124 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 125
__attribute((always_inline)) __attribute__((unused)) static inline T surf1Dread(surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 126
{int volatile ___ = 1;(void)surf;(void)x;(void)mode;
# 132
::exit(___);}
#if 0
# 126
{ 
# 132
} 
#endif
# 134 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 135
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dread(T *res, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 136
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)mode;
# 140
::exit(___);}
#if 0
# 136
{ 
# 140
} 
#endif
# 143 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 144
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 145
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 149
::exit(___);}
#if 0
# 145
{ 
# 149
} 
#endif
# 151 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 152
__attribute((always_inline)) __attribute__((unused)) static inline T surf2Dread(surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 153
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)mode;
# 159
::exit(___);}
#if 0
# 153
{ 
# 159
} 
#endif
# 161 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 162
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dread(T *res, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 163
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)mode;
# 167
::exit(___);}
#if 0
# 163
{ 
# 167
} 
#endif
# 170 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 171
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 172
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 176
::exit(___);}
#if 0
# 172
{ 
# 176
} 
#endif
# 178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 179
__attribute((always_inline)) __attribute__((unused)) static inline T surf3Dread(surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 180
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 186
::exit(___);}
#if 0
# 180
{ 
# 186
} 
#endif
# 188 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 189
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dread(T *res, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 190
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 194
::exit(___);}
#if 0
# 190
{ 
# 194
} 
#endif
# 198 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 199
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 200
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 204
::exit(___);}
#if 0
# 200
{ 
# 204
} 
#endif
# 206 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 207
__attribute((always_inline)) __attribute__((unused)) static inline T surf1DLayeredread(surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 208
{int volatile ___ = 1;(void)surf;(void)x;(void)layer;(void)mode;
# 214
::exit(___);}
#if 0
# 208
{ 
# 214
} 
#endif
# 217 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 218
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredread(T *res, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 219
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)layer;(void)mode;
# 223
::exit(___);}
#if 0
# 219
{ 
# 223
} 
#endif
# 226 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 227
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 228
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 232
::exit(___);}
#if 0
# 228
{ 
# 232
} 
#endif
# 234 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 235
__attribute((always_inline)) __attribute__((unused)) static inline T surf2DLayeredread(surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 236
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 242
::exit(___);}
#if 0
# 236
{ 
# 242
} 
#endif
# 245 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 246
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredread(T *res, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 247
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 251
::exit(___);}
#if 0
# 247
{ 
# 251
} 
#endif
# 254 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 255
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 256
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 260
::exit(___);}
#if 0
# 256
{ 
# 260
} 
#endif
# 262 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 263
__attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapread(surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 264
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 271
::exit(___);}
#if 0
# 264
{ 
# 271
} 
#endif
# 273 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 274
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapread(T *res, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 275
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 279
::exit(___);}
#if 0
# 275
{ 
# 279
} 
#endif
# 282 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 283
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 284
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 288
::exit(___);}
#if 0
# 284
{ 
# 288
} 
#endif
# 290 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 291
__attribute((always_inline)) __attribute__((unused)) static inline T surfCubemapLayeredread(surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 292
{int volatile ___ = 1;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 298
::exit(___);}
#if 0
# 292
{ 
# 298
} 
#endif
# 300 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 301
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredread(T *res, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 302
{int volatile ___ = 1;(void)res;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 306
::exit(___);}
#if 0
# 302
{ 
# 306
} 
#endif
# 309 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 310
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 311
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)s;(void)mode;
# 315
::exit(___);}
#if 0
# 311
{ 
# 315
} 
#endif
# 317 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 318
__attribute((always_inline)) __attribute__((unused)) static inline void surf1Dwrite(T val, surface< void, 1>  surf, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 319
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)mode;
# 323
::exit(___);}
#if 0
# 319
{ 
# 323
} 
#endif
# 327 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 328
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 329
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)s;(void)mode;
# 333
::exit(___);}
#if 0
# 329
{ 
# 333
} 
#endif
# 335 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 336
__attribute((always_inline)) __attribute__((unused)) static inline void surf2Dwrite(T val, surface< void, 2>  surf, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 337
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)mode;
# 341
::exit(___);}
#if 0
# 337
{ 
# 341
} 
#endif
# 344 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 345
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 346
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)s;(void)mode;
# 350
::exit(___);}
#if 0
# 346
{ 
# 350
} 
#endif
# 352 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 353
__attribute((always_inline)) __attribute__((unused)) static inline void surf3Dwrite(T val, surface< void, 3>  surf, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 354
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)z;(void)mode;
# 358
::exit(___);}
#if 0
# 354
{ 
# 358
} 
#endif
# 361 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 362
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 363
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)s;(void)mode;
# 367
::exit(___);}
#if 0
# 363
{ 
# 367
} 
#endif
# 369 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 370
__attribute((always_inline)) __attribute__((unused)) static inline void surf1DLayeredwrite(T val, surface< void, 241>  surf, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 371
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)layer;(void)mode;
# 375
::exit(___);}
#if 0
# 371
{ 
# 375
} 
#endif
# 378 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 379
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 380
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)s;(void)mode;
# 384
::exit(___);}
#if 0
# 380
{ 
# 384
} 
#endif
# 386 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 387
__attribute((always_inline)) __attribute__((unused)) static inline void surf2DLayeredwrite(T val, surface< void, 242>  surf, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 388
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layer;(void)mode;
# 392
::exit(___);}
#if 0
# 388
{ 
# 392
} 
#endif
# 395 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 396
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 397
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)s;(void)mode;
# 401
::exit(___);}
#if 0
# 397
{ 
# 401
} 
#endif
# 403 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 404
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapwrite(T val, surface< void, 12>  surf, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 405
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)face;(void)mode;
# 409
::exit(___);}
#if 0
# 405
{ 
# 409
} 
#endif
# 413 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 414
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, int s, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 415
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)s;(void)mode;
# 419
::exit(___);}
#if 0
# 415
{ 
# 419
} 
#endif
# 421 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_functions.h"
template< class T> 
# 422
__attribute((always_inline)) __attribute__((unused)) static inline void surfCubemapLayeredwrite(T val, surface< void, 252>  surf, int x, int y, int layerFace, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 423
{int volatile ___ = 1;(void)val;(void)surf;(void)x;(void)y;(void)layerFace;(void)mode;
# 427
::exit(___);}
#if 0
# 423
{ 
# 427
} 
#endif
# 68 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 69
struct __nv_tex_rmet_ret { }; 
# 71
template<> struct __nv_tex_rmet_ret< char>  { typedef char type; }; 
# 72
template<> struct __nv_tex_rmet_ret< signed char>  { typedef signed char type; }; 
# 73
template<> struct __nv_tex_rmet_ret< unsigned char>  { typedef unsigned char type; }; 
# 74
template<> struct __nv_tex_rmet_ret< char1>  { typedef char1 type; }; 
# 75
template<> struct __nv_tex_rmet_ret< uchar1>  { typedef uchar1 type; }; 
# 76
template<> struct __nv_tex_rmet_ret< char2>  { typedef char2 type; }; 
# 77
template<> struct __nv_tex_rmet_ret< uchar2>  { typedef uchar2 type; }; 
# 78
template<> struct __nv_tex_rmet_ret< char4>  { typedef char4 type; }; 
# 79
template<> struct __nv_tex_rmet_ret< uchar4>  { typedef uchar4 type; }; 
# 81
template<> struct __nv_tex_rmet_ret< short>  { typedef short type; }; 
# 82
template<> struct __nv_tex_rmet_ret< unsigned short>  { typedef unsigned short type; }; 
# 83
template<> struct __nv_tex_rmet_ret< short1>  { typedef short1 type; }; 
# 84
template<> struct __nv_tex_rmet_ret< ushort1>  { typedef ushort1 type; }; 
# 85
template<> struct __nv_tex_rmet_ret< short2>  { typedef short2 type; }; 
# 86
template<> struct __nv_tex_rmet_ret< ushort2>  { typedef ushort2 type; }; 
# 87
template<> struct __nv_tex_rmet_ret< short4>  { typedef short4 type; }; 
# 88
template<> struct __nv_tex_rmet_ret< ushort4>  { typedef ushort4 type; }; 
# 90
template<> struct __nv_tex_rmet_ret< int>  { typedef int type; }; 
# 91
template<> struct __nv_tex_rmet_ret< unsigned>  { typedef unsigned type; }; 
# 92
template<> struct __nv_tex_rmet_ret< int1>  { typedef int1 type; }; 
# 93
template<> struct __nv_tex_rmet_ret< uint1>  { typedef uint1 type; }; 
# 94
template<> struct __nv_tex_rmet_ret< int2>  { typedef int2 type; }; 
# 95
template<> struct __nv_tex_rmet_ret< uint2>  { typedef uint2 type; }; 
# 96
template<> struct __nv_tex_rmet_ret< int4>  { typedef int4 type; }; 
# 97
template<> struct __nv_tex_rmet_ret< uint4>  { typedef uint4 type; }; 
# 109 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template<> struct __nv_tex_rmet_ret< float>  { typedef float type; }; 
# 110
template<> struct __nv_tex_rmet_ret< float1>  { typedef float1 type; }; 
# 111
template<> struct __nv_tex_rmet_ret< float2>  { typedef float2 type; }; 
# 112
template<> struct __nv_tex_rmet_ret< float4>  { typedef float4 type; }; 
# 115
template< class T> struct __nv_tex_rmet_cast { typedef T *type; }; 
# 127 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 128
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeElementType>  t, int x) 
# 129
{int volatile ___ = 1;(void)t;(void)x;
# 135
::exit(___);}
#if 0
# 129
{ 
# 135
} 
#endif
# 137 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 138
struct __nv_tex_rmnf_ret { }; 
# 140
template<> struct __nv_tex_rmnf_ret< char>  { typedef float type; }; 
# 141
template<> struct __nv_tex_rmnf_ret< signed char>  { typedef float type; }; 
# 142
template<> struct __nv_tex_rmnf_ret< unsigned char>  { typedef float type; }; 
# 143
template<> struct __nv_tex_rmnf_ret< short>  { typedef float type; }; 
# 144
template<> struct __nv_tex_rmnf_ret< unsigned short>  { typedef float type; }; 
# 145
template<> struct __nv_tex_rmnf_ret< char1>  { typedef float1 type; }; 
# 146
template<> struct __nv_tex_rmnf_ret< uchar1>  { typedef float1 type; }; 
# 147
template<> struct __nv_tex_rmnf_ret< short1>  { typedef float1 type; }; 
# 148
template<> struct __nv_tex_rmnf_ret< ushort1>  { typedef float1 type; }; 
# 149
template<> struct __nv_tex_rmnf_ret< char2>  { typedef float2 type; }; 
# 150
template<> struct __nv_tex_rmnf_ret< uchar2>  { typedef float2 type; }; 
# 151
template<> struct __nv_tex_rmnf_ret< short2>  { typedef float2 type; }; 
# 152
template<> struct __nv_tex_rmnf_ret< ushort2>  { typedef float2 type; }; 
# 153
template<> struct __nv_tex_rmnf_ret< char4>  { typedef float4 type; }; 
# 154
template<> struct __nv_tex_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 155
template<> struct __nv_tex_rmnf_ret< short4>  { typedef float4 type; }; 
# 156
template<> struct __nv_tex_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 158
template< class T> 
# 159
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1Dfetch(texture< T, 1, cudaReadModeNormalizedFloat>  t, int x) 
# 160
{int volatile ___ = 1;(void)t;(void)x;
# 167
::exit(___);}
#if 0
# 160
{ 
# 167
} 
#endif
# 170 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 171
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1D(texture< T, 1, cudaReadModeElementType>  t, float x) 
# 172
{int volatile ___ = 1;(void)t;(void)x;
# 178
::exit(___);}
#if 0
# 172
{ 
# 178
} 
#endif
# 180 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 181
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1D(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x) 
# 182
{int volatile ___ = 1;(void)t;(void)x;
# 189
::exit(___);}
#if 0
# 182
{ 
# 189
} 
#endif
# 193 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 194
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2D(texture< T, 2, cudaReadModeElementType>  t, float x, float y) 
# 195
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 202
::exit(___);}
#if 0
# 195
{ 
# 202
} 
#endif
# 204 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 205
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2D(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y) 
# 206
{int volatile ___ = 1;(void)t;(void)x;(void)y;
# 213
::exit(___);}
#if 0
# 206
{ 
# 213
} 
#endif
# 217 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 218
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeElementType>  t, float x, int layer) 
# 219
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 225
::exit(___);}
#if 0
# 219
{ 
# 225
} 
#endif
# 227 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 228
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayered(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer) 
# 229
{int volatile ___ = 1;(void)t;(void)x;(void)layer;
# 236
::exit(___);}
#if 0
# 229
{ 
# 236
} 
#endif
# 240 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 241
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer) 
# 242
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 248
::exit(___);}
#if 0
# 242
{ 
# 248
} 
#endif
# 250 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 251
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayered(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer) 
# 252
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;
# 259
::exit(___);}
#if 0
# 252
{ 
# 259
} 
#endif
# 262 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 263
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3D(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z) 
# 264
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 270
::exit(___);}
#if 0
# 264
{ 
# 270
} 
#endif
# 272 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 273
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3D(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 274
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 281
::exit(___);}
#if 0
# 274
{ 
# 281
} 
#endif
# 284 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 285
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z) 
# 286
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 292
::exit(___);}
#if 0
# 286
{ 
# 292
} 
#endif
# 294 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 295
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemap(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z) 
# 296
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;
# 303
::exit(___);}
#if 0
# 296
{ 
# 303
} 
#endif
# 306 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 307
struct __nv_tex2dgather_ret { }; 
# 308
template<> struct __nv_tex2dgather_ret< char>  { typedef char4 type; }; 
# 309
template<> struct __nv_tex2dgather_ret< signed char>  { typedef char4 type; }; 
# 310
template<> struct __nv_tex2dgather_ret< char1>  { typedef char4 type; }; 
# 311
template<> struct __nv_tex2dgather_ret< char2>  { typedef char4 type; }; 
# 312
template<> struct __nv_tex2dgather_ret< char3>  { typedef char4 type; }; 
# 313
template<> struct __nv_tex2dgather_ret< char4>  { typedef char4 type; }; 
# 314
template<> struct __nv_tex2dgather_ret< unsigned char>  { typedef uchar4 type; }; 
# 315
template<> struct __nv_tex2dgather_ret< uchar1>  { typedef uchar4 type; }; 
# 316
template<> struct __nv_tex2dgather_ret< uchar2>  { typedef uchar4 type; }; 
# 317
template<> struct __nv_tex2dgather_ret< uchar3>  { typedef uchar4 type; }; 
# 318
template<> struct __nv_tex2dgather_ret< uchar4>  { typedef uchar4 type; }; 
# 320
template<> struct __nv_tex2dgather_ret< short>  { typedef short4 type; }; 
# 321
template<> struct __nv_tex2dgather_ret< short1>  { typedef short4 type; }; 
# 322
template<> struct __nv_tex2dgather_ret< short2>  { typedef short4 type; }; 
# 323
template<> struct __nv_tex2dgather_ret< short3>  { typedef short4 type; }; 
# 324
template<> struct __nv_tex2dgather_ret< short4>  { typedef short4 type; }; 
# 325
template<> struct __nv_tex2dgather_ret< unsigned short>  { typedef ushort4 type; }; 
# 326
template<> struct __nv_tex2dgather_ret< ushort1>  { typedef ushort4 type; }; 
# 327
template<> struct __nv_tex2dgather_ret< ushort2>  { typedef ushort4 type; }; 
# 328
template<> struct __nv_tex2dgather_ret< ushort3>  { typedef ushort4 type; }; 
# 329
template<> struct __nv_tex2dgather_ret< ushort4>  { typedef ushort4 type; }; 
# 331
template<> struct __nv_tex2dgather_ret< int>  { typedef int4 type; }; 
# 332
template<> struct __nv_tex2dgather_ret< int1>  { typedef int4 type; }; 
# 333
template<> struct __nv_tex2dgather_ret< int2>  { typedef int4 type; }; 
# 334
template<> struct __nv_tex2dgather_ret< int3>  { typedef int4 type; }; 
# 335
template<> struct __nv_tex2dgather_ret< int4>  { typedef int4 type; }; 
# 336
template<> struct __nv_tex2dgather_ret< unsigned>  { typedef uint4 type; }; 
# 337
template<> struct __nv_tex2dgather_ret< uint1>  { typedef uint4 type; }; 
# 338
template<> struct __nv_tex2dgather_ret< uint2>  { typedef uint4 type; }; 
# 339
template<> struct __nv_tex2dgather_ret< uint3>  { typedef uint4 type; }; 
# 340
template<> struct __nv_tex2dgather_ret< uint4>  { typedef uint4 type; }; 
# 342
template<> struct __nv_tex2dgather_ret< float>  { typedef float4 type; }; 
# 343
template<> struct __nv_tex2dgather_ret< float1>  { typedef float4 type; }; 
# 344
template<> struct __nv_tex2dgather_ret< float2>  { typedef float4 type; }; 
# 345
template<> struct __nv_tex2dgather_ret< float3>  { typedef float4 type; }; 
# 346
template<> struct __nv_tex2dgather_ret< float4>  { typedef float4 type; }; 
# 348
template< class T> 
# 349
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeElementType>  t, float x, float y, int comp = 0) 
# 350
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 357
::exit(___);}
#if 0
# 350
{ 
# 357
} 
#endif
# 360 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> struct __nv_tex2dgather_rmnf_ret { }; 
# 361
template<> struct __nv_tex2dgather_rmnf_ret< char>  { typedef float4 type; }; 
# 362
template<> struct __nv_tex2dgather_rmnf_ret< signed char>  { typedef float4 type; }; 
# 363
template<> struct __nv_tex2dgather_rmnf_ret< unsigned char>  { typedef float4 type; }; 
# 364
template<> struct __nv_tex2dgather_rmnf_ret< char1>  { typedef float4 type; }; 
# 365
template<> struct __nv_tex2dgather_rmnf_ret< uchar1>  { typedef float4 type; }; 
# 366
template<> struct __nv_tex2dgather_rmnf_ret< char2>  { typedef float4 type; }; 
# 367
template<> struct __nv_tex2dgather_rmnf_ret< uchar2>  { typedef float4 type; }; 
# 368
template<> struct __nv_tex2dgather_rmnf_ret< char3>  { typedef float4 type; }; 
# 369
template<> struct __nv_tex2dgather_rmnf_ret< uchar3>  { typedef float4 type; }; 
# 370
template<> struct __nv_tex2dgather_rmnf_ret< char4>  { typedef float4 type; }; 
# 371
template<> struct __nv_tex2dgather_rmnf_ret< uchar4>  { typedef float4 type; }; 
# 372
template<> struct __nv_tex2dgather_rmnf_ret< signed short>  { typedef float4 type; }; 
# 373
template<> struct __nv_tex2dgather_rmnf_ret< unsigned short>  { typedef float4 type; }; 
# 374
template<> struct __nv_tex2dgather_rmnf_ret< short1>  { typedef float4 type; }; 
# 375
template<> struct __nv_tex2dgather_rmnf_ret< ushort1>  { typedef float4 type; }; 
# 376
template<> struct __nv_tex2dgather_rmnf_ret< short2>  { typedef float4 type; }; 
# 377
template<> struct __nv_tex2dgather_rmnf_ret< ushort2>  { typedef float4 type; }; 
# 378
template<> struct __nv_tex2dgather_rmnf_ret< short3>  { typedef float4 type; }; 
# 379
template<> struct __nv_tex2dgather_rmnf_ret< ushort3>  { typedef float4 type; }; 
# 380
template<> struct __nv_tex2dgather_rmnf_ret< short4>  { typedef float4 type; }; 
# 381
template<> struct __nv_tex2dgather_rmnf_ret< ushort4>  { typedef float4 type; }; 
# 383
template< class T> 
# 384
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex2dgather_rmnf_ret< T> ::type tex2Dgather(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, int comp = 0) 
# 385
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)comp;
# 392
::exit(___);}
#if 0
# 385
{ 
# 392
} 
#endif
# 396 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 397
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeElementType>  t, float x, float level) 
# 398
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 404
::exit(___);}
#if 0
# 398
{ 
# 404
} 
#endif
# 406 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 407
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLod(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float level) 
# 408
{int volatile ___ = 1;(void)t;(void)x;(void)level;
# 415
::exit(___);}
#if 0
# 408
{ 
# 415
} 
#endif
# 418 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 419
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float level) 
# 420
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 426
::exit(___);}
#if 0
# 420
{ 
# 426
} 
#endif
# 428 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 429
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLod(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float level) 
# 430
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)level;
# 437
::exit(___);}
#if 0
# 430
{ 
# 437
} 
#endif
# 440 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 441
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float level) 
# 442
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 448
::exit(___);}
#if 0
# 442
{ 
# 448
} 
#endif
# 450 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 451
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredLod(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float level) 
# 452
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)level;
# 459
::exit(___);}
#if 0
# 452
{ 
# 459
} 
#endif
# 462 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 463
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float level) 
# 464
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 470
::exit(___);}
#if 0
# 464
{ 
# 470
} 
#endif
# 472 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 473
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredLod(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float level) 
# 474
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)level;
# 481
::exit(___);}
#if 0
# 474
{ 
# 481
} 
#endif
# 484 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 485
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 486
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 492
::exit(___);}
#if 0
# 486
{ 
# 492
} 
#endif
# 494 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 495
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DLod(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 496
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 503
::exit(___);}
#if 0
# 496
{ 
# 503
} 
#endif
# 506 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 507
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float level) 
# 508
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 514
::exit(___);}
#if 0
# 508
{ 
# 514
} 
#endif
# 516 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 517
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLod(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float level) 
# 518
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)level;
# 525
::exit(___);}
#if 0
# 518
{ 
# 525
} 
#endif
# 529 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 530
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer) 
# 531
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 537
::exit(___);}
#if 0
# 531
{ 
# 537
} 
#endif
# 539 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 540
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayered(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer) 
# 541
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;
# 548
::exit(___);}
#if 0
# 541
{ 
# 548
} 
#endif
# 552 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 553
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float level) 
# 554
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 560
::exit(___);}
#if 0
# 554
{ 
# 560
} 
#endif
# 562 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 563
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredLod(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float level) 
# 564
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 571
::exit(___);}
#if 0
# 564
{ 
# 571
} 
#endif
# 575 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 576
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 577
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 583
::exit(___);}
#if 0
# 577
{ 
# 583
} 
#endif
# 585 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 586
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapGrad(texture< T, 12, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 587
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 594
::exit(___);}
#if 0
# 587
{ 
# 594
} 
#endif
# 598 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 599
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeElementType>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 600
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 606
::exit(___);}
#if 0
# 600
{ 
# 606
} 
#endif
# 608 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 609
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type texCubemapLayeredGrad(texture< T, 252, cudaReadModeNormalizedFloat>  t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 610
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 617
::exit(___);}
#if 0
# 610
{ 
# 617
} 
#endif
# 621 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 622
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeElementType>  t, float x, float dPdx, float dPdy) 
# 623
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 629
::exit(___);}
#if 0
# 623
{ 
# 629
} 
#endif
# 631 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 632
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DGrad(texture< T, 1, cudaReadModeNormalizedFloat>  t, float x, float dPdx, float dPdy) 
# 633
{int volatile ___ = 1;(void)t;(void)x;(void)dPdx;(void)dPdy;
# 640
::exit(___);}
#if 0
# 633
{ 
# 640
} 
#endif
# 644 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 645
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeElementType>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 646
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 652
::exit(___);}
#if 0
# 646
{ 
# 652
} 
#endif
# 654 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 655
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DGrad(texture< T, 2, cudaReadModeNormalizedFloat>  t, float x, float y, float2 dPdx, float2 dPdy) 
# 656
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 663
::exit(___);}
#if 0
# 656
{ 
# 663
} 
#endif
# 666 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 667
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeElementType>  t, float x, int layer, float dPdx, float dPdy) 
# 668
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 674
::exit(___);}
#if 0
# 668
{ 
# 674
} 
#endif
# 676 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 677
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex1DLayeredGrad(texture< T, 241, cudaReadModeNormalizedFloat>  t, float x, int layer, float dPdx, float dPdy) 
# 678
{int volatile ___ = 1;(void)t;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 685
::exit(___);}
#if 0
# 678
{ 
# 685
} 
#endif
# 688 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 689
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeElementType>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 690
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 696
::exit(___);}
#if 0
# 690
{ 
# 696
} 
#endif
# 698 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 699
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex2DLayeredGrad(texture< T, 242, cudaReadModeNormalizedFloat>  t, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 700
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 707
::exit(___);}
#if 0
# 700
{ 
# 707
} 
#endif
# 710 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 711
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmet_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeElementType>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 712
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 718
::exit(___);}
#if 0
# 712
{ 
# 718
} 
#endif
# 720 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_fetch_functions.h"
template< class T> 
# 721
__attribute((always_inline)) __attribute__((unused)) static inline typename __nv_tex_rmnf_ret< T> ::type tex3DGrad(texture< T, 3, cudaReadModeNormalizedFloat>  t, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 722
{int volatile ___ = 1;(void)t;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 729
::exit(___);}
#if 0
# 722
{ 
# 729
} 
#endif
# 61 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> struct __nv_itex_trait { }; 
# 62
template<> struct __nv_itex_trait< char>  { typedef void type; }; 
# 63
template<> struct __nv_itex_trait< signed char>  { typedef void type; }; 
# 64
template<> struct __nv_itex_trait< char1>  { typedef void type; }; 
# 65
template<> struct __nv_itex_trait< char2>  { typedef void type; }; 
# 66
template<> struct __nv_itex_trait< char4>  { typedef void type; }; 
# 67
template<> struct __nv_itex_trait< unsigned char>  { typedef void type; }; 
# 68
template<> struct __nv_itex_trait< uchar1>  { typedef void type; }; 
# 69
template<> struct __nv_itex_trait< uchar2>  { typedef void type; }; 
# 70
template<> struct __nv_itex_trait< uchar4>  { typedef void type; }; 
# 71
template<> struct __nv_itex_trait< short>  { typedef void type; }; 
# 72
template<> struct __nv_itex_trait< short1>  { typedef void type; }; 
# 73
template<> struct __nv_itex_trait< short2>  { typedef void type; }; 
# 74
template<> struct __nv_itex_trait< short4>  { typedef void type; }; 
# 75
template<> struct __nv_itex_trait< unsigned short>  { typedef void type; }; 
# 76
template<> struct __nv_itex_trait< ushort1>  { typedef void type; }; 
# 77
template<> struct __nv_itex_trait< ushort2>  { typedef void type; }; 
# 78
template<> struct __nv_itex_trait< ushort4>  { typedef void type; }; 
# 79
template<> struct __nv_itex_trait< int>  { typedef void type; }; 
# 80
template<> struct __nv_itex_trait< int1>  { typedef void type; }; 
# 81
template<> struct __nv_itex_trait< int2>  { typedef void type; }; 
# 82
template<> struct __nv_itex_trait< int4>  { typedef void type; }; 
# 83
template<> struct __nv_itex_trait< unsigned>  { typedef void type; }; 
# 84
template<> struct __nv_itex_trait< uint1>  { typedef void type; }; 
# 85
template<> struct __nv_itex_trait< uint2>  { typedef void type; }; 
# 86
template<> struct __nv_itex_trait< uint4>  { typedef void type; }; 
# 97 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template<> struct __nv_itex_trait< float>  { typedef void type; }; 
# 98
template<> struct __nv_itex_trait< float1>  { typedef void type; }; 
# 99
template<> struct __nv_itex_trait< float2>  { typedef void type; }; 
# 100
template<> struct __nv_itex_trait< float4>  { typedef void type; }; 
# 104
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 105
tex1Dfetch(T *ptr, cudaTextureObject_t obj, int x) 
# 106
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 110
::exit(___);}
#if 0
# 106
{ 
# 110
} 
#endif
# 112 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 113
tex1Dfetch(cudaTextureObject_t texObject, int x) 
# 114
{int volatile ___ = 1;(void)texObject;(void)x;
# 120
::exit(___);}
#if 0
# 114
{ 
# 120
} 
#endif
# 122 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 123
tex1D(T *ptr, cudaTextureObject_t obj, float x) 
# 124
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;
# 128
::exit(___);}
#if 0
# 124
{ 
# 128
} 
#endif
# 131 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 132
tex1D(cudaTextureObject_t texObject, float x) 
# 133
{int volatile ___ = 1;(void)texObject;(void)x;
# 139
::exit(___);}
#if 0
# 133
{ 
# 139
} 
#endif
# 142 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 143
tex2D(T *ptr, cudaTextureObject_t obj, float x, float y) 
# 144
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;
# 148
::exit(___);}
#if 0
# 144
{ 
# 148
} 
#endif
# 150 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 151
tex2D(cudaTextureObject_t texObject, float x, float y) 
# 152
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;
# 158
::exit(___);}
#if 0
# 152
{ 
# 158
} 
#endif
# 160 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 161
tex3D(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 162
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 166
::exit(___);}
#if 0
# 162
{ 
# 166
} 
#endif
# 168 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 169
tex3D(cudaTextureObject_t texObject, float x, float y, float z) 
# 170
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 176
::exit(___);}
#if 0
# 170
{ 
# 176
} 
#endif
# 178 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 179
tex1DLayered(T *ptr, cudaTextureObject_t obj, float x, int layer) 
# 180
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;
# 184
::exit(___);}
#if 0
# 180
{ 
# 184
} 
#endif
# 186 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 187
tex1DLayered(cudaTextureObject_t texObject, float x, int layer) 
# 188
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;
# 194
::exit(___);}
#if 0
# 188
{ 
# 194
} 
#endif
# 196 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 197
tex2DLayered(T *ptr, cudaTextureObject_t obj, float x, float y, int layer) 
# 198
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;
# 202
::exit(___);}
#if 0
# 198
{ 
# 202
} 
#endif
# 204 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 205
tex2DLayered(cudaTextureObject_t texObject, float x, float y, int layer) 
# 206
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;
# 212
::exit(___);}
#if 0
# 206
{ 
# 212
} 
#endif
# 215 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 216
texCubemap(T *ptr, cudaTextureObject_t obj, float x, float y, float z) 
# 217
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;
# 221
::exit(___);}
#if 0
# 217
{ 
# 221
} 
#endif
# 224 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 225
texCubemap(cudaTextureObject_t texObject, float x, float y, float z) 
# 226
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;
# 232
::exit(___);}
#if 0
# 226
{ 
# 232
} 
#endif
# 235 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 236
texCubemapLayered(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer) 
# 237
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;
# 241
::exit(___);}
#if 0
# 237
{ 
# 241
} 
#endif
# 243 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 244
texCubemapLayered(cudaTextureObject_t texObject, float x, float y, float z, int layer) 
# 245
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;
# 251
::exit(___);}
#if 0
# 245
{ 
# 251
} 
#endif
# 253 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 254
tex2Dgather(T *ptr, cudaTextureObject_t obj, float x, float y, int comp = 0) 
# 255
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)comp;
# 259
::exit(___);}
#if 0
# 255
{ 
# 259
} 
#endif
# 261 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 262
tex2Dgather(cudaTextureObject_t to, float x, float y, int comp = 0) 
# 263
{int volatile ___ = 1;(void)to;(void)x;(void)y;(void)comp;
# 269
::exit(___);}
#if 0
# 263
{ 
# 269
} 
#endif
# 273 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 274
tex1DLod(T *ptr, cudaTextureObject_t obj, float x, float level) 
# 275
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)level;
# 279
::exit(___);}
#if 0
# 275
{ 
# 279
} 
#endif
# 281 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 282
tex1DLod(cudaTextureObject_t texObject, float x, float level) 
# 283
{int volatile ___ = 1;(void)texObject;(void)x;(void)level;
# 289
::exit(___);}
#if 0
# 283
{ 
# 289
} 
#endif
# 292 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 293
tex2DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float level) 
# 294
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)level;
# 298
::exit(___);}
#if 0
# 294
{ 
# 298
} 
#endif
# 300 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 301
tex2DLod(cudaTextureObject_t texObject, float x, float y, float level) 
# 302
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)level;
# 308
::exit(___);}
#if 0
# 302
{ 
# 308
} 
#endif
# 311 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 312
tex3DLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 313
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 317
::exit(___);}
#if 0
# 313
{ 
# 317
} 
#endif
# 319 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 320
tex3DLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 321
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 327
::exit(___);}
#if 0
# 321
{ 
# 327
} 
#endif
# 330 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 331
tex1DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, int layer, float level) 
# 332
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)level;
# 336
::exit(___);}
#if 0
# 332
{ 
# 336
} 
#endif
# 338 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 339
tex1DLayeredLod(cudaTextureObject_t texObject, float x, int layer, float level) 
# 340
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)level;
# 346
::exit(___);}
#if 0
# 340
{ 
# 346
} 
#endif
# 349 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 350
tex2DLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float level) 
# 351
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)level;
# 355
::exit(___);}
#if 0
# 351
{ 
# 355
} 
#endif
# 357 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 358
tex2DLayeredLod(cudaTextureObject_t texObject, float x, float y, int layer, float level) 
# 359
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)level;
# 365
::exit(___);}
#if 0
# 359
{ 
# 365
} 
#endif
# 368 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 369
texCubemapLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float level) 
# 370
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)level;
# 374
::exit(___);}
#if 0
# 370
{ 
# 374
} 
#endif
# 376 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 377
texCubemapLod(cudaTextureObject_t texObject, float x, float y, float z, float level) 
# 378
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)level;
# 384
::exit(___);}
#if 0
# 378
{ 
# 384
} 
#endif
# 387 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 388
texCubemapGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 389
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 393
::exit(___);}
#if 0
# 389
{ 
# 393
} 
#endif
# 395 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 396
texCubemapGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 397
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 403
::exit(___);}
#if 0
# 397
{ 
# 403
} 
#endif
# 405 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 406
texCubemapLayeredLod(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float level) 
# 407
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 411
::exit(___);}
#if 0
# 407
{ 
# 411
} 
#endif
# 413 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 414
texCubemapLayeredLod(cudaTextureObject_t texObject, float x, float y, float z, int layer, float level) 
# 415
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)level;
# 421
::exit(___);}
#if 0
# 415
{ 
# 421
} 
#endif
# 423 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 424
tex1DGrad(T *ptr, cudaTextureObject_t obj, float x, float dPdx, float dPdy) 
# 425
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)dPdx;(void)dPdy;
# 429
::exit(___);}
#if 0
# 425
{ 
# 429
} 
#endif
# 431 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 432
tex1DGrad(cudaTextureObject_t texObject, float x, float dPdx, float dPdy) 
# 433
{int volatile ___ = 1;(void)texObject;(void)x;(void)dPdx;(void)dPdy;
# 439
::exit(___);}
#if 0
# 433
{ 
# 439
} 
#endif
# 442 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 443
tex2DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float2 dPdx, float2 dPdy) 
# 444
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 449
::exit(___);}
#if 0
# 444
{ 
# 449
} 
#endif
# 451 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 452
tex2DGrad(cudaTextureObject_t texObject, float x, float y, float2 dPdx, float2 dPdy) 
# 453
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)dPdx;(void)dPdy;
# 459
::exit(___);}
#if 0
# 453
{ 
# 459
} 
#endif
# 462 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 463
tex3DGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 464
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 468
::exit(___);}
#if 0
# 464
{ 
# 468
} 
#endif
# 470 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 471
tex3DGrad(cudaTextureObject_t texObject, float x, float y, float z, float4 dPdx, float4 dPdy) 
# 472
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)dPdx;(void)dPdy;
# 478
::exit(___);}
#if 0
# 472
{ 
# 478
} 
#endif
# 481 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 482
tex1DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, int layer, float dPdx, float dPdy) 
# 483
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 487
::exit(___);}
#if 0
# 483
{ 
# 487
} 
#endif
# 489 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 490
tex1DLayeredGrad(cudaTextureObject_t texObject, float x, int layer, float dPdx, float dPdy) 
# 491
{int volatile ___ = 1;(void)texObject;(void)x;(void)layer;(void)dPdx;(void)dPdy;
# 497
::exit(___);}
#if 0
# 491
{ 
# 497
} 
#endif
# 500 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 501
tex2DLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 502
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 506
::exit(___);}
#if 0
# 502
{ 
# 506
} 
#endif
# 508 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 509
tex2DLayeredGrad(cudaTextureObject_t texObject, float x, float y, int layer, float2 dPdx, float2 dPdy) 
# 510
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)layer;(void)dPdx;(void)dPdy;
# 516
::exit(___);}
#if 0
# 510
{ 
# 516
} 
#endif
# 519 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_itex_trait< T> ::type 
# 520
texCubemapLayeredGrad(T *ptr, cudaTextureObject_t obj, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 521
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 525
::exit(___);}
#if 0
# 521
{ 
# 525
} 
#endif
# 527 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/texture_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 528
texCubemapLayeredGrad(cudaTextureObject_t texObject, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
# 529
{int volatile ___ = 1;(void)texObject;(void)x;(void)y;(void)z;(void)layer;(void)dPdx;(void)dPdy;
# 535
::exit(___);}
#if 0
# 529
{ 
# 535
} 
#endif
# 60 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> struct __nv_isurf_trait { }; 
# 61
template<> struct __nv_isurf_trait< char>  { typedef void type; }; 
# 62
template<> struct __nv_isurf_trait< signed char>  { typedef void type; }; 
# 63
template<> struct __nv_isurf_trait< char1>  { typedef void type; }; 
# 64
template<> struct __nv_isurf_trait< unsigned char>  { typedef void type; }; 
# 65
template<> struct __nv_isurf_trait< uchar1>  { typedef void type; }; 
# 66
template<> struct __nv_isurf_trait< short>  { typedef void type; }; 
# 67
template<> struct __nv_isurf_trait< short1>  { typedef void type; }; 
# 68
template<> struct __nv_isurf_trait< unsigned short>  { typedef void type; }; 
# 69
template<> struct __nv_isurf_trait< ushort1>  { typedef void type; }; 
# 70
template<> struct __nv_isurf_trait< int>  { typedef void type; }; 
# 71
template<> struct __nv_isurf_trait< int1>  { typedef void type; }; 
# 72
template<> struct __nv_isurf_trait< unsigned>  { typedef void type; }; 
# 73
template<> struct __nv_isurf_trait< uint1>  { typedef void type; }; 
# 74
template<> struct __nv_isurf_trait< long long>  { typedef void type; }; 
# 75
template<> struct __nv_isurf_trait< longlong1>  { typedef void type; }; 
# 76
template<> struct __nv_isurf_trait< unsigned long long>  { typedef void type; }; 
# 77
template<> struct __nv_isurf_trait< ulonglong1>  { typedef void type; }; 
# 78
template<> struct __nv_isurf_trait< float>  { typedef void type; }; 
# 79
template<> struct __nv_isurf_trait< float1>  { typedef void type; }; 
# 81
template<> struct __nv_isurf_trait< char2>  { typedef void type; }; 
# 82
template<> struct __nv_isurf_trait< uchar2>  { typedef void type; }; 
# 83
template<> struct __nv_isurf_trait< short2>  { typedef void type; }; 
# 84
template<> struct __nv_isurf_trait< ushort2>  { typedef void type; }; 
# 85
template<> struct __nv_isurf_trait< int2>  { typedef void type; }; 
# 86
template<> struct __nv_isurf_trait< uint2>  { typedef void type; }; 
# 87
template<> struct __nv_isurf_trait< longlong2>  { typedef void type; }; 
# 88
template<> struct __nv_isurf_trait< ulonglong2>  { typedef void type; }; 
# 89
template<> struct __nv_isurf_trait< float2>  { typedef void type; }; 
# 91
template<> struct __nv_isurf_trait< char4>  { typedef void type; }; 
# 92
template<> struct __nv_isurf_trait< uchar4>  { typedef void type; }; 
# 93
template<> struct __nv_isurf_trait< short4>  { typedef void type; }; 
# 94
template<> struct __nv_isurf_trait< ushort4>  { typedef void type; }; 
# 95
template<> struct __nv_isurf_trait< int4>  { typedef void type; }; 
# 96
template<> struct __nv_isurf_trait< uint4>  { typedef void type; }; 
# 97
template<> struct __nv_isurf_trait< float4>  { typedef void type; }; 
# 100
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 101
surf1Dread(T *ptr, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 102
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)mode;
# 106
::exit(___);}
#if 0
# 102
{ 
# 106
} 
#endif
# 108 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 109
surf1Dread(cudaSurfaceObject_t surfObject, int x, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 110
{int volatile ___ = 1;(void)surfObject;(void)x;(void)boundaryMode;
# 116
::exit(___);}
#if 0
# 110
{ 
# 116
} 
#endif
# 118 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 119
surf2Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 120
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)mode;
# 124
::exit(___);}
#if 0
# 120
{ 
# 124
} 
#endif
# 126 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 127
surf2Dread(cudaSurfaceObject_t surfObject, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 128
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)boundaryMode;
# 134
::exit(___);}
#if 0
# 128
{ 
# 134
} 
#endif
# 137 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 138
surf3Dread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 139
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 143
::exit(___);}
#if 0
# 139
{ 
# 143
} 
#endif
# 145 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 146
surf3Dread(cudaSurfaceObject_t surfObject, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 147
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)z;(void)boundaryMode;
# 153
::exit(___);}
#if 0
# 147
{ 
# 153
} 
#endif
# 155 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 156
surf1DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 157
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)layer;(void)mode;
# 161
::exit(___);}
#if 0
# 157
{ 
# 161
} 
#endif
# 163 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 164
surf1DLayeredread(cudaSurfaceObject_t surfObject, int x, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 165
{int volatile ___ = 1;(void)surfObject;(void)x;(void)layer;(void)boundaryMode;
# 171
::exit(___);}
#if 0
# 165
{ 
# 171
} 
#endif
# 173 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 174
surf2DLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 175
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 179
::exit(___);}
#if 0
# 175
{ 
# 179
} 
#endif
# 181 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 182
surf2DLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layer, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 183
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layer;(void)boundaryMode;
# 189
::exit(___);}
#if 0
# 183
{ 
# 189
} 
#endif
# 191 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 192
surfCubemapread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 193
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 197
::exit(___);}
#if 0
# 193
{ 
# 197
} 
#endif
# 199 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 200
surfCubemapread(cudaSurfaceObject_t surfObject, int x, int y, int face, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 201
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)face;(void)boundaryMode;
# 207
::exit(___);}
#if 0
# 201
{ 
# 207
} 
#endif
# 209 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 210
surfCubemapLayeredread(T *ptr, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 211
{int volatile ___ = 1;(void)ptr;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 215
::exit(___);}
#if 0
# 211
{ 
# 215
} 
#endif
# 217 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static T 
# 218
surfCubemapLayeredread(cudaSurfaceObject_t surfObject, int x, int y, int layerface, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap) 
# 219
{int volatile ___ = 1;(void)surfObject;(void)x;(void)y;(void)layerface;(void)boundaryMode;
# 225
::exit(___);}
#if 0
# 219
{ 
# 225
} 
#endif
# 227 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 228
surf1Dwrite(T val, cudaSurfaceObject_t obj, int x, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 229
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)mode;
# 233
::exit(___);}
#if 0
# 229
{ 
# 233
} 
#endif
# 235 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 236
surf2Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 237
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)mode;
# 241
::exit(___);}
#if 0
# 237
{ 
# 241
} 
#endif
# 243 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 244
surf3Dwrite(T val, cudaSurfaceObject_t obj, int x, int y, int z, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 245
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)z;(void)mode;
# 249
::exit(___);}
#if 0
# 245
{ 
# 249
} 
#endif
# 251 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 252
surf1DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 253
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)layer;(void)mode;
# 257
::exit(___);}
#if 0
# 253
{ 
# 257
} 
#endif
# 259 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 260
surf2DLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layer, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 261
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layer;(void)mode;
# 265
::exit(___);}
#if 0
# 261
{ 
# 265
} 
#endif
# 267 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 268
surfCubemapwrite(T val, cudaSurfaceObject_t obj, int x, int y, int face, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 269
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)face;(void)mode;
# 273
::exit(___);}
#if 0
# 269
{ 
# 273
} 
#endif
# 275 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/surface_indirect_functions.h"
template< class T> __attribute__((unused)) static typename __nv_isurf_trait< T> ::type 
# 276
surfCubemapLayeredwrite(T val, cudaSurfaceObject_t obj, int x, int y, int layerface, cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) 
# 277
{int volatile ___ = 1;(void)val;(void)obj;(void)x;(void)y;(void)layerface;(void)mode;
# 281
::exit(___);}
#if 0
# 277
{ 
# 281
} 
#endif
# 68 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/device_launch_parameters.h"
extern "C" {
# 71
extern const uint3 __device_builtin_variable_threadIdx; 
# 72
extern const uint3 __device_builtin_variable_blockIdx; 
# 73
extern const dim3 __device_builtin_variable_blockDim; 
# 74
extern const dim3 __device_builtin_variable_gridDim; 
# 75
extern const int __device_builtin_variable_warpSize; 
# 80
}
# 185 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 186
cudaLaunchKernel(const T *
# 187
func, dim3 
# 188
gridDim, dim3 
# 189
blockDim, void **
# 190
args, size_t 
# 191
sharedMem = 0, cudaStream_t 
# 192
stream = 0) 
# 194
{ 
# 195
return ::cudaLaunchKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 196
} 
# 245 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 246
cudaLaunchCooperativeKernel(const T *
# 247
func, dim3 
# 248
gridDim, dim3 
# 249
blockDim, void **
# 250
args, size_t 
# 251
sharedMem = 0, cudaStream_t 
# 252
stream = 0) 
# 254
{ 
# 255
return ::cudaLaunchCooperativeKernel((const void *)func, gridDim, blockDim, args, sharedMem, stream); 
# 256
} 
# 283 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 284
cudaSetupArgument(T 
# 285
arg, size_t 
# 286
offset) 
# 288
{ 
# 289
return ::cudaSetupArgument((const void *)(&arg), sizeof(T), offset); 
# 290
} 
# 322 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static inline cudaError_t cudaEventCreate(cudaEvent_t *
# 323
event, unsigned 
# 324
flags) 
# 326
{ 
# 327
return ::cudaEventCreateWithFlags(event, flags); 
# 328
} 
# 385 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
static inline cudaError_t cudaMallocHost(void **
# 386
ptr, size_t 
# 387
size, unsigned 
# 388
flags) 
# 390
{ 
# 391
return ::cudaHostAlloc(ptr, size, flags); 
# 392
} 
# 394
template< class T> static inline cudaError_t 
# 395
cudaHostAlloc(T **
# 396
ptr, size_t 
# 397
size, unsigned 
# 398
flags) 
# 400
{ 
# 401
return ::cudaHostAlloc((void **)((void *)ptr), size, flags); 
# 402
} 
# 404
template< class T> static inline cudaError_t 
# 405
cudaHostGetDevicePointer(T **
# 406
pDevice, void *
# 407
pHost, unsigned 
# 408
flags) 
# 410
{ 
# 411
return ::cudaHostGetDevicePointer((void **)((void *)pDevice), pHost, flags); 
# 412
} 
# 512 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 513
cudaMallocManaged(T **
# 514
devPtr, size_t 
# 515
size, unsigned 
# 516
flags = 1) 
# 518
{ 
# 519
return ::cudaMallocManaged((void **)((void *)devPtr), size, flags); 
# 520
} 
# 591 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 592
cudaStreamAttachMemAsync(cudaStream_t 
# 593
stream, T *
# 594
devPtr, size_t 
# 595
length = 0, unsigned 
# 596
flags = 4) 
# 598
{ 
# 599
return ::cudaStreamAttachMemAsync(stream, (void *)devPtr, length, flags); 
# 600
} 
# 602
template< class T> inline cudaError_t 
# 603
cudaMalloc(T **
# 604
devPtr, size_t 
# 605
size) 
# 607
{ 
# 608
return ::cudaMalloc((void **)((void *)devPtr), size); 
# 609
} 
# 611
template< class T> static inline cudaError_t 
# 612
cudaMallocHost(T **
# 613
ptr, size_t 
# 614
size, unsigned 
# 615
flags = 0) 
# 617
{ 
# 618
return cudaMallocHost((void **)((void *)ptr), size, flags); 
# 619
} 
# 621
template< class T> static inline cudaError_t 
# 622
cudaMallocPitch(T **
# 623
devPtr, size_t *
# 624
pitch, size_t 
# 625
width, size_t 
# 626
height) 
# 628
{ 
# 629
return ::cudaMallocPitch((void **)((void *)devPtr), pitch, width, height); 
# 630
} 
# 667 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 668
cudaMemcpyToSymbol(const T &
# 669
symbol, const void *
# 670
src, size_t 
# 671
count, size_t 
# 672
offset = 0, cudaMemcpyKind 
# 673
kind = cudaMemcpyHostToDevice) 
# 675
{ 
# 676
return ::cudaMemcpyToSymbol((const void *)(&symbol), src, count, offset, kind); 
# 677
} 
# 719 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 720
cudaMemcpyToSymbolAsync(const T &
# 721
symbol, const void *
# 722
src, size_t 
# 723
count, size_t 
# 724
offset = 0, cudaMemcpyKind 
# 725
kind = cudaMemcpyHostToDevice, cudaStream_t 
# 726
stream = 0) 
# 728
{ 
# 729
return ::cudaMemcpyToSymbolAsync((const void *)(&symbol), src, count, offset, kind, stream); 
# 730
} 
# 765 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 766
cudaMemcpyFromSymbol(void *
# 767
dst, const T &
# 768
symbol, size_t 
# 769
count, size_t 
# 770
offset = 0, cudaMemcpyKind 
# 771
kind = cudaMemcpyDeviceToHost) 
# 773
{ 
# 774
return ::cudaMemcpyFromSymbol(dst, (const void *)(&symbol), count, offset, kind); 
# 775
} 
# 817 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 818
cudaMemcpyFromSymbolAsync(void *
# 819
dst, const T &
# 820
symbol, size_t 
# 821
count, size_t 
# 822
offset = 0, cudaMemcpyKind 
# 823
kind = cudaMemcpyDeviceToHost, cudaStream_t 
# 824
stream = 0) 
# 826
{ 
# 827
return ::cudaMemcpyFromSymbolAsync(dst, (const void *)(&symbol), count, offset, kind, stream); 
# 828
} 
# 851 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 852
cudaGetSymbolAddress(void **
# 853
devPtr, const T &
# 854
symbol) 
# 856
{ 
# 857
return ::cudaGetSymbolAddress(devPtr, (const void *)(&symbol)); 
# 858
} 
# 881 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 882
cudaGetSymbolSize(size_t *
# 883
size, const T &
# 884
symbol) 
# 886
{ 
# 887
return ::cudaGetSymbolSize(size, (const void *)(&symbol)); 
# 888
} 
# 923 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 924
cudaBindTexture(size_t *
# 925
offset, const texture< T, dim, readMode>  &
# 926
tex, const void *
# 927
devPtr, const cudaChannelFormatDesc &
# 928
desc, size_t 
# 929
size = ((2147483647) * 2U) + 1U) 
# 931 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
{ 
# 932
return ::cudaBindTexture(offset, &tex, devPtr, &desc, size); 
# 933
} 
# 967 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 968
cudaBindTexture(size_t *
# 969
offset, const texture< T, dim, readMode>  &
# 970
tex, const void *
# 971
devPtr, size_t 
# 972
size = ((2147483647) * 2U) + 1U) 
# 974 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
{ 
# 975
return cudaBindTexture(offset, tex, devPtr, (tex.channelDesc), size); 
# 976
} 
# 1022 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1023
cudaBindTexture2D(size_t *
# 1024
offset, const texture< T, dim, readMode>  &
# 1025
tex, const void *
# 1026
devPtr, const cudaChannelFormatDesc &
# 1027
desc, size_t 
# 1028
width, size_t 
# 1029
height, size_t 
# 1030
pitch) 
# 1032
{ 
# 1033
return ::cudaBindTexture2D(offset, &tex, devPtr, &desc, width, height, pitch); 
# 1034
} 
# 1079 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1080
cudaBindTexture2D(size_t *
# 1081
offset, const texture< T, dim, readMode>  &
# 1082
tex, const void *
# 1083
devPtr, size_t 
# 1084
width, size_t 
# 1085
height, size_t 
# 1086
pitch) 
# 1088
{ 
# 1089
return ::cudaBindTexture2D(offset, &tex, devPtr, &(tex.channelDesc), width, height, pitch); 
# 1090
} 
# 1120 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1121
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1122
tex, cudaArray_const_t 
# 1123
array, const cudaChannelFormatDesc &
# 1124
desc) 
# 1126
{ 
# 1127
return ::cudaBindTextureToArray(&tex, array, &desc); 
# 1128
} 
# 1157 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1158
cudaBindTextureToArray(const texture< T, dim, readMode>  &
# 1159
tex, cudaArray_const_t 
# 1160
array) 
# 1162
{ 
# 1163
cudaChannelFormatDesc desc; 
# 1164
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 1166
return (err == (cudaSuccess)) ? cudaBindTextureToArray(tex, array, desc) : err; 
# 1167
} 
# 1197 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1198
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1199
tex, cudaMipmappedArray_const_t 
# 1200
mipmappedArray, const cudaChannelFormatDesc &
# 1201
desc) 
# 1203
{ 
# 1204
return ::cudaBindTextureToMipmappedArray(&tex, mipmappedArray, &desc); 
# 1205
} 
# 1234 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1235
cudaBindTextureToMipmappedArray(const texture< T, dim, readMode>  &
# 1236
tex, cudaMipmappedArray_const_t 
# 1237
mipmappedArray) 
# 1239
{ 
# 1240
cudaChannelFormatDesc desc; 
# 1241
cudaArray_t levelArray; 
# 1242
cudaError_t err = ::cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, 0); 
# 1244
if (err != (cudaSuccess)) { 
# 1245
return err; 
# 1246
}  
# 1247
err = ::cudaGetChannelDesc(&desc, levelArray); 
# 1249
return (err == (cudaSuccess)) ? cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc) : err; 
# 1250
} 
# 1273 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1274
cudaUnbindTexture(const texture< T, dim, readMode>  &
# 1275
tex) 
# 1277
{ 
# 1278
return ::cudaUnbindTexture(&tex); 
# 1279
} 
# 1307 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim, cudaTextureReadMode readMode> static inline cudaError_t 
# 1308
cudaGetTextureAlignmentOffset(size_t *
# 1309
offset, const texture< T, dim, readMode>  &
# 1310
tex) 
# 1312
{ 
# 1313
return ::cudaGetTextureAlignmentOffset(offset, &tex); 
# 1314
} 
# 1359 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1360
cudaFuncSetCacheConfig(T *
# 1361
func, cudaFuncCache 
# 1362
cacheConfig) 
# 1364
{ 
# 1365
return ::cudaFuncSetCacheConfig((const void *)func, cacheConfig); 
# 1366
} 
# 1368
template< class T> static inline cudaError_t 
# 1369
cudaFuncSetSharedMemConfig(T *
# 1370
func, cudaSharedMemConfig 
# 1371
config) 
# 1373
{ 
# 1374
return ::cudaFuncSetSharedMemConfig((const void *)func, config); 
# 1375
} 
# 1404 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1405
cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *
# 1406
numBlocks, T 
# 1407
func, int 
# 1408
blockSize, size_t 
# 1409
dynamicSMemSize) 
# 1410
{ 
# 1411
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, 0); 
# 1412
} 
# 1455 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1456
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *
# 1457
numBlocks, T 
# 1458
func, int 
# 1459
blockSize, size_t 
# 1460
dynamicSMemSize, unsigned 
# 1461
flags) 
# 1462
{ 
# 1463
return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void *)func, blockSize, dynamicSMemSize, flags); 
# 1464
} 
# 1469
class __cudaOccupancyB2DHelper { 
# 1470
size_t n; 
# 1472
public: __cudaOccupancyB2DHelper(size_t n_) : n(n_) { } 
# 1473
size_t operator()(int) 
# 1474
{ 
# 1475
return n; 
# 1476
} 
# 1477
}; 
# 1524 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 1525
cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int *
# 1526
minGridSize, int *
# 1527
blockSize, T 
# 1528
func, UnaryFunction 
# 1529
blockSizeToDynamicSMemSize, int 
# 1530
blockSizeLimit = 0, unsigned 
# 1531
flags = 0) 
# 1532
{ 
# 1533
cudaError_t status; 
# 1536
int device; 
# 1537
cudaFuncAttributes attr; 
# 1540
int maxThreadsPerMultiProcessor; 
# 1541
int warpSize; 
# 1542
int devMaxThreadsPerBlock; 
# 1543
int multiProcessorCount; 
# 1544
int funcMaxThreadsPerBlock; 
# 1545
int occupancyLimit; 
# 1546
int granularity; 
# 1549
int maxBlockSize = 0; 
# 1550
int numBlocks = 0; 
# 1551
int maxOccupancy = 0; 
# 1554
int blockSizeToTryAligned; 
# 1555
int blockSizeToTry; 
# 1556
int blockSizeLimitAligned; 
# 1557
int occupancyInBlocks; 
# 1558
int occupancyInThreads; 
# 1559
size_t dynamicSMemSize; 
# 1565
if (((!minGridSize) || (!blockSize)) || (!func)) { 
# 1566
return cudaErrorInvalidValue; 
# 1567
}  
# 1573
status = ::cudaGetDevice(&device); 
# 1574
if (status != (cudaSuccess)) { 
# 1575
return status; 
# 1576
}  
# 1578
status = cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device); 
# 1582
if (status != (cudaSuccess)) { 
# 1583
return status; 
# 1584
}  
# 1586
status = cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device); 
# 1590
if (status != (cudaSuccess)) { 
# 1591
return status; 
# 1592
}  
# 1594
status = cudaDeviceGetAttribute(&devMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device); 
# 1598
if (status != (cudaSuccess)) { 
# 1599
return status; 
# 1600
}  
# 1602
status = cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device); 
# 1606
if (status != (cudaSuccess)) { 
# 1607
return status; 
# 1608
}  
# 1610
status = cudaFuncGetAttributes(&attr, func); 
# 1611
if (status != (cudaSuccess)) { 
# 1612
return status; 
# 1613
}  
# 1615
funcMaxThreadsPerBlock = (attr.maxThreadsPerBlock); 
# 1621
occupancyLimit = maxThreadsPerMultiProcessor; 
# 1622
granularity = warpSize; 
# 1624
if (blockSizeLimit == 0) { 
# 1625
blockSizeLimit = devMaxThreadsPerBlock; 
# 1626
}  
# 1628
if (devMaxThreadsPerBlock < blockSizeLimit) { 
# 1629
blockSizeLimit = devMaxThreadsPerBlock; 
# 1630
}  
# 1632
if (funcMaxThreadsPerBlock < blockSizeLimit) { 
# 1633
blockSizeLimit = funcMaxThreadsPerBlock; 
# 1634
}  
# 1636
blockSizeLimitAligned = (((blockSizeLimit + (granularity - 1)) / granularity) * granularity); 
# 1638
for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) { 
# 1642
if (blockSizeLimit < blockSizeToTryAligned) { 
# 1643
blockSizeToTry = blockSizeLimit; 
# 1644
} else { 
# 1645
blockSizeToTry = blockSizeToTryAligned; 
# 1646
}  
# 1648
dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry); 
# 1650
status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&occupancyInBlocks, func, blockSizeToTry, dynamicSMemSize, flags); 
# 1657
if (status != (cudaSuccess)) { 
# 1658
return status; 
# 1659
}  
# 1661
occupancyInThreads = (blockSizeToTry * occupancyInBlocks); 
# 1663
if (occupancyInThreads > maxOccupancy) { 
# 1664
maxBlockSize = blockSizeToTry; 
# 1665
numBlocks = occupancyInBlocks; 
# 1666
maxOccupancy = occupancyInThreads; 
# 1667
}  
# 1671
if (occupancyLimit == maxOccupancy) { 
# 1672
break; 
# 1673
}  
# 1674
}  
# 1682
(*minGridSize) = (numBlocks * multiProcessorCount); 
# 1683
(*blockSize) = maxBlockSize; 
# 1685
return status; 
# 1686
} 
# 1719 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class UnaryFunction, class T> static inline cudaError_t 
# 1720
cudaOccupancyMaxPotentialBlockSizeVariableSMem(int *
# 1721
minGridSize, int *
# 1722
blockSize, T 
# 1723
func, UnaryFunction 
# 1724
blockSizeToDynamicSMemSize, int 
# 1725
blockSizeLimit = 0) 
# 1726
{ 
# 1727
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, blockSizeLimit, 0); 
# 1728
} 
# 1764 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1765
cudaOccupancyMaxPotentialBlockSize(int *
# 1766
minGridSize, int *
# 1767
blockSize, T 
# 1768
func, size_t 
# 1769
dynamicSMemSize = 0, int 
# 1770
blockSizeLimit = 0) 
# 1771
{ 
# 1772
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, 0); 
# 1773
} 
# 1823 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1824
cudaOccupancyMaxPotentialBlockSizeWithFlags(int *
# 1825
minGridSize, int *
# 1826
blockSize, T 
# 1827
func, size_t 
# 1828
dynamicSMemSize = 0, int 
# 1829
blockSizeLimit = 0, unsigned 
# 1830
flags = 0) 
# 1831
{ 
# 1832
return cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(minGridSize, blockSize, func, ((__cudaOccupancyB2DHelper)(dynamicSMemSize)), blockSizeLimit, flags); 
# 1833
} 
# 1874 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1875
cudaLaunch(T *
# 1876
func) 
# 1878
{ 
# 1879
return ::cudaLaunch((const void *)func); 
# 1880
} 
# 1911 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> inline cudaError_t 
# 1912
cudaFuncGetAttributes(cudaFuncAttributes *
# 1913
attr, T *
# 1914
entry) 
# 1916
{ 
# 1917
return ::cudaFuncGetAttributes(attr, (const void *)entry); 
# 1918
} 
# 1953 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T> static inline cudaError_t 
# 1954
cudaFuncSetAttribute(T *
# 1955
entry, cudaFuncAttribute 
# 1956
attr, int 
# 1957
value) 
# 1959
{ 
# 1960
return ::cudaFuncSetAttribute((const void *)entry, attr, value); 
# 1961
} 
# 1983 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim> static inline cudaError_t 
# 1984
cudaBindSurfaceToArray(const surface< T, dim>  &
# 1985
surf, cudaArray_const_t 
# 1986
array, const cudaChannelFormatDesc &
# 1987
desc) 
# 1989
{ 
# 1990
return ::cudaBindSurfaceToArray(&surf, array, &desc); 
# 1991
} 
# 2012 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
template< class T, int dim> static inline cudaError_t 
# 2013
cudaBindSurfaceToArray(const surface< T, dim>  &
# 2014
surf, cudaArray_const_t 
# 2015
array) 
# 2017
{ 
# 2018
cudaChannelFormatDesc desc; 
# 2019
cudaError_t err = ::cudaGetChannelDesc(&desc, array); 
# 2021
return (err == (cudaSuccess)) ? cudaBindSurfaceToArray(surf, array, desc) : err; 
# 2022
} 
# 2033 "/usr/local/cuda-9.0/bin/../targets/x86_64-linux/include/cuda_runtime.h"
#pragma GCC diagnostic pop
# 41 "CMakeCUDACompilerId.cu"
const char *info_compiler = ("INFO:compiler[NVIDIA]"); 
# 220 "CMakeCUDACompilerId.cu"
const char info_version[] = {'I', 'N', 'F', 'O', ':', 'c', 'o', 'm', 'p', 'i', 'l', 'e', 'r', '_', 'v', 'e', 'r', 's', 'i', 'o', 'n', '[', (('0') + ((9 / 10000000) % 10)), (('0') + ((9 / 1000000) % 10)), (('0') + ((9 / 100000) % 10)), (('0') + ((9 / 10000) % 10)), (('0') + ((9 / 1000) % 10)), (('0') + ((9 / 100) % 10)), (('0') + ((9 / 10) % 10)), (('0') + (9 % 10)), '.', (('0') + ((0 / 10000000) % 10)), (('0') + ((0 / 1000000) % 10)), (('0') + ((0 / 100000) % 10)), (('0') + ((0 / 10000) % 10)), (('0') + ((0 / 1000) % 10)), (('0') + ((0 / 100) % 10)), (('0') + ((0 / 10) % 10)), (('0') + (0 % 10)), '.', (('0') + ((176 / 10000000) % 10)), (('0') + ((176 / 1000000) % 10)), (('0') + ((176 / 100000) % 10)), (('0') + ((176 / 10000) % 10)), (('0') + ((176 / 1000) % 10)), (('0') + ((176 / 100) % 10)), (('0') + ((176 / 10) % 10)), (('0') + (176 % 10)), ']', '\000'}; 
# 258 "CMakeCUDACompilerId.cu"
const char *info_platform = ("INFO:platform[Linux]"); 
# 259
const char *info_arch = ("INFO:arch[]"); 
# 264
const char *info_language_dialect_default = ("INFO:dialect_default[98]"); 
# 278
int main(int argc, char *argv[]) 
# 279
{ 
# 280
int require = 0; 
# 281
require += (info_compiler[argc]); 
# 282
require += (info_platform[argc]); 
# 284
require += ((info_version)[argc]); 
# 292
require += (info_language_dialect_default[argc]); 
# 293
(void)argv; 
# 294
return require; 
# 295
} 

# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#define _NV_ANON_NAMESPACE _GLOBAL__N__27_CMakeCUDACompilerId_cpp1_ii_bd57c623
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#include "CMakeCUDACompilerId.cudafe1.stub.c"
# 1 "CMakeCUDACompilerId.cudafe1.stub.c"
#undef _NV_ANON_NAMESPACE
