	////////////////////////////////////////////////////////////////////
	//File: basic_environ.c
	//
	//Description: base file for environment exercises with openCL
	//
	// 
	////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#ifdef __APPLE__
	  #include <OpenCL/opencl.h>
#else
	  #include <CL/cl.h>
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
	  
	// check error, in such a case, it exits

	void cl_error(cl_int code, const char *string){
		if (code != CL_SUCCESS){
			printf("%d - %s\n", code, string);
			exit(-1);
		}
	}
	////////////////////////////////////////////////////////////////////////////////

	int main(int argc, char** argv)
	{
	size_t count = 128;
	int width=64;
	double temp[10];
	double temps[8];
	double time[10];
	double times[8];
	for (int indice=0;indice<6;indice++){
		width *= 2;
		count = (size_t)(width) * (size_t)(width);
	for (int repetition=0;repetition<5;repetition++){
	  clock_t starttt = clock();
	  int err;                            	// error code returned from api calls
	  size_t t_buf = 50;			// size of str_buffer
	  char str_buffer[t_buf];		// auxiliary buffer	
	  size_t e_buf;				// effective size of str_buffer in use
	  size_t program_Size;			// size of the opencl program
		
		  
	  size_t global_size;                      	// global domain size for our calculation
	  size_t local_size;                       	// local domain size for our calculation

	  const cl_uint num_platforms_ids = 10;				// max of allocatable platforms
	  cl_platform_id platforms_ids[num_platforms_ids];		// array of platforms
	  cl_uint n_platforms;						// effective number of platforms in use
	  const cl_uint num_devices_ids = 10;				// max of allocatable devices
	  cl_device_id devices_ids[num_platforms_ids][num_devices_ids];	// array of devices
	  cl_uint n_devices[num_platforms_ids];				// effective number of devices in use for each platform
		
	  cl_device_id device_id;             				// compute device id 
	  cl_context context;                 				// compute context
	  cl_command_queue command_queue;     				// compute command queue
	  cl_program program;                 				// compute program
	  cl_kernel kernel;                   				// compute kernel
		
	  //float in_host_object[count];
	  //float out_host_object[count];
	  float *in_host_object = (float*) malloc (sizeof(float) * count);
	  float *out_host_object = (float*) malloc (sizeof(float) * count);
	  if (!in_host_object || !out_host_object) { printf("malloc failed\n"); exit(1); }
	  
	  for (int i = 0; i < count; i++){
		in_host_object[i] = 10;  
		free(in_host_object);
		free(out_host_object)                    // (float) i a remplacer pour test si fonctionne meme si un peu compliquer de vérifier la coherance car en transformation comme si en deux D
	  }
	  
	
// valeur a donner

int gaussienSize = 3;
float sigma = 1.0f;
float gaussienFiltre[gaussienSize * gaussienSize];
float sum = 0.0f;

int half = gaussienSize / 2;

for (int i = - half; i <= half; i++) {
    for (int j = - half; j <= half; j++) {
        gaussienFiltre[(i + half )*gaussienSize + j + half] = 
            (1.0f / (2.0f * M_PI * sigma * sigma)) * exp(-(i*i + j*j) / (2.0f * sigma * sigma));
        sum += gaussienFiltre[(i + half) * gaussienSize + j + half];
    }
}


// Normalisation
for (int i = 0; i < gaussienSize * gaussienSize; i++) {
    gaussienFiltre[i] /= sum;
	//printf(" %f", gaussienFiltre[i]);
}

FILE *fileHandler = fopen("kernels/kernel.cl", "r");
if (!fileHandler) {
    printf("Cannot open kernel file\n");
    exit(1);
}

fseek(fileHandler, 0, SEEK_END);
size_t fileSize = ftell(fileHandler);
rewind(fileHandler);


// Read kernel source into buffer
char *sourceCode = (char*) malloc(fileSize + 1);
sourceCode[fileSize] = '\0';
fread(sourceCode, sizeof(char), fileSize, fileHandler);
fclose(fileHandler);




	  // 1. Scan the available platforms:
	  err = clGetPlatformIDs (num_platforms_ids, platforms_ids, &n_platforms);
	  cl_error(err, "Error: Failed to Scan for Platforms IDs");
	  //printf("Number of available platforms: %d\n\n", n_platforms);

	  for (int i = 0; i < n_platforms; i++ ){
		err= clGetPlatformInfo(platforms_ids[i], CL_PLATFORM_NAME, t_buf, str_buffer, &e_buf);
		cl_error (err, "Error: Failed to get info of the platform\n");
		//printf( "\t[%d]-Platform Name: %s\n", i, str_buffer);
	  }
	  //printf("\n");
	  // ***Task***: print on the screen the name, host_timer_resolution, vendor, versionm, ...
		
	  // 2. Scan for devices in each platform
	  for (int i = 0; i < n_platforms; i++) {
    	err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_GPU,
                         num_devices_ids, devices_ids[i], &n_devices[i]);

    if (err == CL_DEVICE_NOT_FOUND) {
        printf("No GPU on platform %d, trying CPU...\n", i);
        err = clGetDeviceIDs(platforms_ids[i], CL_DEVICE_TYPE_CPU,
                             num_devices_ids, devices_ids[i], &n_devices[i]);
    }
	cl_error(err, "Error: Failed to Scan for Devices IDs");
	//printf("[%d]-Platform. Number of available devices: %d\n", i, n_devices[i]);
}
	// 3. Create a context, with a device
          cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms_ids[0], 0};
          context = clCreateContext(properties, 1, &devices_ids[0][0], NULL, NULL, &err);
          cl_error(err, "Failed to create a compute context\n");

          // 4. Create a command queue
          cl_command_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
          command_queue = clCreateCommandQueueWithProperties(context, devices_ids[0][0], proprt, &err);
          cl_error(err, "Failed to create a command queue\n");


          // 5. Read an OpenCL program from the file kernel.cl
          // Calculate size of the file
    // Calculate size of the file



          // create program from buffer
          program = clCreateProgramWithSource(context, 1, (const char**) &sourceCode, &fileSize, &err);
          cl_error(err, "Failed to create program with source\n");
          free(sourceCode);

          // read kernel source back in from program to check
          size_t kernelSourceSize;
          clGetProgramInfo(program, CL_PROGRAM_SOURCE, 0, NULL, &kernelSourceSize);
          char *kernelSource = (char*) malloc(kernelSourceSize);
          clGetProgramInfo(program, CL_PROGRAM_SOURCE, kernelSourceSize, kernelSource, NULL);
          //printf("nKernel source:\n\n%s\n", kernelSource);
          free(kernelSource);

          // Build the executable and check errors
          err = clBuildProgram(program, 1, &devices_ids[0][0], NULL, NULL, NULL);
          if (err != CL_SUCCESS){
                size_t len;
                char buffer[2048];

                printf("Error: Some error at building process.\n");
                clGetProgramBuildInfo(program, devices_ids[0][0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                printf("%s\n", buffer);
                exit(-1);
          }

          // Create a compute kernel with the program we want to run
          kernel = clCreateKernel(program, "gaussienFiltre", &err);
          cl_error(err, "Failed to create kernel from the program\n");

	  // Create OpenCL buffer visible to the OpenCl runtime
	  cl_mem in_device_object  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * count, NULL, &err);
	  cl_error(err, "Failed to create memory buffer at device\n");
	  cl_mem out_device_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
	  cl_error(err, "Failed to create memory buffer at device\n");
    cl_mem bFiltregaussien  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * gaussienSize*gaussienSize, NULL, &err);
	  cl_error(err, "Failed to create memory buffer at device\n");

	  // Write date into the memory object 
	  err = clEnqueueWriteBuffer(command_queue, in_device_object, CL_TRUE, 0, sizeof(float) * count, in_host_object, 0, NULL, NULL);
	  cl_error(err, "Failed to enqueue a write command\n");
    err = clEnqueueWriteBuffer(command_queue,bFiltregaussien, CL_TRUE, 0, sizeof(float) * gaussienSize * gaussienSize, gaussienFiltre, 0, NULL, NULL);
	  cl_error(err, "Failed to enqueue a write command\n");

	  // Set the arguments to the kernel
		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_device_object);
		cl_error(err, "arg0");
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bFiltregaussien);
		cl_error(err, "arg1");
		err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_device_object);
		cl_error(err, "arg2");

		unsigned int gs = (unsigned int)gaussienSize;
		unsigned int ww = (unsigned int)width;
		unsigned int aa = (unsigned int)half;

		err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &gs);  // gaussienSize
		cl_error(err, "arg3");
		err = clSetKernelArg(kernel, 4, sizeof(unsigned int), &ww);  // width
		cl_error(err, "arg4");
		err = clSetKernelArg(kernel, 5, sizeof(unsigned int), &aa);  // a = half
		cl_error(err, "arg5");



	  // Launch Kernel
	  local_size = 256;
	  global_size = count;
	  cl_event event;
	  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
	  cl_error(err, "Failed to launch kernel");

	  clWaitForEvents(1, &event);
	  
	cl_ulong start, end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

	clReleaseEvent(event);
	temp[repetition]=(double)(end - start);


  // Read data form device memory back to host memory
  err = clEnqueueReadBuffer(command_queue, out_device_object, CL_TRUE, 0, sizeof(float) * count, out_host_object, 0, NULL, NULL);
  cl_error(err, "Failed to enqueue a read command\n");

  //for (int j=0;j<count;j++){
	//printf("Square of %f is: %f\n", in_host_object[j], out_host_object[j]);
 // }
  
	

  clReleaseMemObject(in_device_object);
  clReleaseMemObject(out_device_object);
  clReleaseMemObject(bFiltregaussien);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

	clock_t enddd = clock();
	time[repetition] = (double)(enddd - starttt) / CLOCKS_PER_SEC;
	}
	double summ=(double)0;
	double summs=(double)0;
	for(int rep=0;rep<5;rep++){
		summ +=temp[rep];
		summs +=time[rep];
	}
	temps[indice]=summ/(double)5.0;
	times[indice]=summs/(double)5.0;
	}
	for (int rep=0;rep<6;rep++){
	printf(" %.6f", times[rep]);
	//printf(" %.6f", temps[rep]);
	}
  return 0;
}
  
