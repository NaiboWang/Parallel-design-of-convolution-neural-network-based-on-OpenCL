#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#pragma warning( disable : 4996 )
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl.hpp"
//定义变量
using namespace std;
FILE *fpp;
cl_context context;
cl_command_queue commandQueue;
cl_program program;
cl_device_id device;
cl_kernel kernel, kernel2, kernel3;//三种卷积内核代表三种不同的情况
cl_mem memObjects[3];//默认没什么用
cl_int errNum;
///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext(int type)
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId, *platform;
	cl_context context = NULL;

	// First, select an OpenCL platform to run on.  For this example, we
	// simply choose the first available platform.  Normally, you would
	// query for all available platforms and select the most appropriate one.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	std::cout << "platformnum:" << numPlatforms << std::endl;
	platform = (cl_platform_id *)malloc(sizeof(cl_platform_id)*numPlatforms);

	errNum = clGetPlatformIDs(numPlatforms, platform, NULL);
	size_t size;
	char *Pname;
	for (int i = 0; i < numPlatforms; i++)
	{
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &size);
		Pname = (char *)malloc(size);
		clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, size, Pname, NULL);
		std::cout << Pname << std::endl;
		clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 0, NULL, &size);
		Pname = (char *)malloc(size);
		clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, size, Pname, NULL);

		if (type == 3)
			firstPlatformId = platform[2];//第一个是GPU,第二个第三个是CPU
		else
			firstPlatformId = platform[type];//第一个是GPU,第二个第三个是CPU

		std::cout << Pname << std::endl;
		cl_uint num;
		clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num);
		printf("devicenum:%d\n", num);
		cl_device_id *device;
		device = (cl_device_id *)malloc(sizeof(cl_device_id)*num);
		clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, num, device, NULL);
		//以下用来查询设备信息
		/*for (int j = 0; j < num; j++)
		{
		char buffer[1000];
		clGetDeviceInfo(device[j], CL_DEVICE_NAME, 100, buffer, NULL);
		printf("DeviceName:%s\n", buffer);
		cl_uint unum;
		cl_device_type the;
		size_t s[3] = { 0,0,0 };
		cl_device_svm_capabilities caps;
		cl_int err = clGetDeviceInfo(device[j], CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities),&caps, 0);
		cout << "SVm"<<err << " " << caps << endl;
		if (err == CL_INVALID_VALUE)
		cout << "NOSVM" << endl;
		else
		{
		if ((caps &CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) && (caps &CL_DEVICE_SVM_ATOMICS))
		cout << "细粒度原子系统" << endl;
		else if (caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
		cout << "细粒度系统" << endl;
		else if ((caps &CL_DEVICE_SVM_FINE_GRAIN_BUFFER) && (caps &CL_DEVICE_SVM_ATOMICS))
		cout << "细粒度原子缓冲" << endl;
		else if (caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
		cout << "细粒度缓冲" << endl;
		else if (caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
		cout << "粗粒度缓冲" << endl;
		else
		cout << caps << endl;
		}
		system("pause");
		clGetDeviceInfo(device[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*3,s, NULL);
		cout << "deviceITEMSIZE:" << s[0] <<" "<<s[1]<<" "<<s[2]<< endl << endl;
		clGetDeviceInfo(device[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &the, NULL);
		cout << "deviceTYPE:"<<the << endl<<endl;
		clGetDeviceInfo(device[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &unum, NULL);
		printf("DeviceCOMPUTENUM:%d\n", unum);
		clGetDeviceInfo(device[j], CL_DEVICE_VERSION, 100, buffer, NULL);
		printf("OPENCLVERSION:%s\n", buffer);
		clGetDeviceInfo(device[j], CL_DEVICE_OPENCL_C_VERSION, 100, buffer, NULL);
		printf("OPENCLCVERSION:%s\n", buffer);
		clGetDeviceInfo(device[j], CL_DEVICE_EXTENSIONS, 1000, buffer, NULL);
		printf("CL_DEVICE_EXTENSIONS:%s\n", buffer);
		clGetDeviceInfo(device[j], CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT, sizeof(cl_uint), &unum, NULL);
		printf("CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT:%d\n", unum);
		clGetDeviceInfo(device[j], CL_DEVICE_PIPE_MAX_PACKET_SIZE, sizeof(cl_uint), &unum, NULL);
		printf("CL_DEVICE_PIPE_MAX_PACKET_SIZE:%d\n", unum);
		}*/
	}

	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// Next, create an OpenCL context on the platform.  Attempt to
	// create a GPU-based context, and if that fails, try to create
	// a CPU-based context.
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	if (type == 4)
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
	else
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}
	/*clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
	cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id)*size);
	clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
	for (size_t t = 0; t < size / sizeof(cl_device_id); t++)
	{
	cl_device_type typ=0;
	clGetDeviceInfo(devices[t], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
	char buffer[1000];
	clGetDeviceInfo(devices[t], CL_DEVICE_NAME, 100, buffer, NULL);
	printf("DeviceName:%s\n", buffer);
	switch (typ)
	{
	case CL_DEVICE_TYPE_GPU:
	cout << endl << "TYPE:GPU" << endl;
	break;
	case CL_DEVICE_TYPE_CPU:
	cout << endl << "TYPE:CPU" << endl;
	break;
	default:
	cout << endl << "TYPE:"<<type << endl;
	break;
	}
	}*/
	return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device, int type)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer
	std::cout << "devices" << (deviceBufferSize / sizeof(cl_device_id)) << endl;
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		delete[] devices;
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a
	// real program, you would likely use all available devices or choose
	// the highest performance device based on OpenCL device queries
	cout << "device:" << devices[0] << endl;
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if (commandQueue == NULL)
	{
		delete[] devices;
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}
	const char *options = "-cl-std=CL2.0 -DBS=16";//2.0编译指定

	errNum = clBuildProgram(program, 0, NULL, options, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
	float *a, float *b)
{
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * ARRAY_SIZE, a, NULL);
	memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * ARRAY_SIZE, b, NULL);
	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
		sizeof(float) * ARRAY_SIZE, NULL, NULL);

	if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
	{
		std::cerr << "Error creating memory objects." << std::endl;
		return false;
	}

	return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
	cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
	for (int i = 0; i < 3; i++)
	{
		if (memObjects[i] != 0)
			clReleaseMemObject(memObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (context != 0)
		clReleaseContext(context);

}
int initOpenCL(FILE *fp)
{
	//OPENCL初始化部分
	fpp = fopen("E:\\CNNData\\testcl.txt", "w");
	int type = 3;//代表选择的OPENCL设备,0NAVIDIA 1,2INTEL 3AMD 4AMDINTEL
	int ttype = 0;//0,1,2代表了3种不同的内核函数
	if (type == 0)
		fprintf(fp, "类型:NVIDIAGPU\n");
	else if (type == 1 || type == 2 || type == 4)
		fprintf(fp, "类型:INTELCPU\n");
	else
		fprintf(fp, "类型:AMDGPU\n");

	context = CreateContext(type);//创建OPENCL上下文并选择设备
								  //pcon = &context;//指针指向context
	commandQueue = CreateCommandQueue(context, &device, type);
	//pcomq = &commandQueue;
	if (commandQueue == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}

	// Create OpenCL program from HelloWorld.cl kernel source
	program = CreateProgram(context, device, "HelloWorld.cl");
	//system("pause");
	if (program == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
	//ppro = &program;
	// Create OpenCL kernel
	if (ttype == 0)
	{
		kernel = clCreateKernel(program, "cnntest", NULL);//测试
		kernel2 = clCreateKernel(program, "cnntrain", NULL);//测试
	}

	if (kernel == NULL)
	{
		std::cerr << "Failed to create kernel" << std::endl;
		Cleanup(context, commandQueue, program, kernel, memObjects);
		return 1;
	}
	//pker = &kernel;
	std::cout << "OPENCL初始化完毕" << endl;
	return 0;
}
