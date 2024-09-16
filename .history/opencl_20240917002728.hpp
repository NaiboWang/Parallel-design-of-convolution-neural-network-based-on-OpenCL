#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include "mat.h"
#pragma warning( disable : 4996 )
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
using namespace std;
//������Щ����,ע��ֻ�ܶ���һ��
extern FILE *fpp;//����OPENCL����ʱ��
extern cl_context context;
extern cl_command_queue commandQueue;
extern cl_program program;
extern cl_device_id device;
extern cl_kernel kernel,kernel2;
extern cl_mem memObjects[3];//Ĭ��ûʲô��
extern cl_int errNum;
const int ARRAY_SIZE = 10000;
int initOpenCL(FILE *fp);
cl_context CreateContext(int type);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device, int type);
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b);
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem memObjects[3]);