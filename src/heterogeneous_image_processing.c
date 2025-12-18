////////////////////////////////////////////////////////////////////
// File: heterogeneous_image_processing.c
// Description: Heterogeneous image processing using CPU and GPU (C version)
// Lab 6 - PACS
////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_DEVICES 10
#define MAX_PLATFORMS 10

// Structure to hold device information
typedef struct {
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    char device_name[256];
    int device_index;
} DeviceContext;

// Structure to hold timing information
typedef struct {
    double kernel_time;
    double transfer_time_to_device;
    double transfer_time_from_device;
    double total_time;
} TimingInfo;

// Error checking function
void cl_error(cl_int code, const char *string) {
    if (code != CL_SUCCESS) {
        printf("%d - %s\n", code, string);
        exit(-1);
    }
}

// Get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Function to create Gaussian filter
void createGaussianFilter(float* filter, int size, float sigma) {
    float sum = 0.0f;
    int half = size / 2;
    
    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            filter[(i + half) * size + j + half] = 
                (1.0f / (2.0f * M_PI * sigma * sigma)) * 
                exp(-(i*i + j*j) / (2.0f * sigma * sigma));
            sum += filter[(i + half) * size + j + half];
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        filter[i] /= sum;
    }
}

// Function to setup OpenCL device
DeviceContext setupDevice(cl_platform_id platform, cl_device_id device, 
                          int device_idx, const char* kernel_source, size_t source_size) {
    DeviceContext dc;
    cl_int err;
    
    dc.device_id = device;
    dc.device_index = device_idx;
    
    // Get device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dc.device_name), dc.device_name, NULL);
    
    // Create context
    cl_context_properties properties[] = { 
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
    };
    dc.context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    cl_error(err, "Failed to create context");
    
    // Create command queue with profiling
    cl_command_queue_properties proprt[] = { 
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 
    };
    dc.command_queue = clCreateCommandQueueWithProperties(dc.context, device, proprt, &err);
    cl_error(err, "Failed to create command queue");
    
    // Create and build program
    dc.program = clCreateProgramWithSource(dc.context, 1, &kernel_source, &source_size, &err);
    cl_error(err, "Failed to create program");
    
    err = clBuildProgram(dc.program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char buffer[4096];
        clGetProgramBuildInfo(dc.program, device, CL_PROGRAM_BUILD_LOG, 
                             sizeof(buffer), buffer, NULL);
        printf("Build error on device %d:\n%s\n", device_idx, buffer);
        exit(-1);
    }
    
    // Create kernel
    dc.kernel = clCreateKernel(dc.program, "gaussienFiltre", &err);
    cl_error(err, "Failed to create kernel");
    
    return dc;
}

// Function to process images on a device
TimingInfo processImagesOnDevice(DeviceContext* dc, 
                                  float** images,
                                  int start_idx, int end_idx,
                                  int width, int height,
                                  float* gaussian_filter, int filter_size) {
    
    TimingInfo timing;
    timing.kernel_time = 0.0;
    timing.transfer_time_to_device = 0.0;
    timing.transfer_time_from_device = 0.0;
    
    cl_int err;
    double total_start = get_time();
    
    int half = filter_size / 2;
    unsigned int gs = (unsigned int)filter_size;
    unsigned int ww = (unsigned int)width;
    unsigned int aa = (unsigned int)half;
    
    size_t image_size = width * height;
    
    // Process each assigned image
    for (int img_idx = start_idx; img_idx < end_idx; img_idx++) {
        // Create buffers
        cl_mem in_buffer = clCreateBuffer(dc->context, CL_MEM_READ_ONLY, 
                                          sizeof(float) * image_size, NULL, &err);
        cl_error(err, "Failed to create input buffer");
        
        cl_mem filter_buffer = clCreateBuffer(dc->context, CL_MEM_READ_ONLY,
                                              sizeof(float) * filter_size * filter_size, 
                                              NULL, &err);
        cl_error(err, "Failed to create filter buffer");
        
        cl_mem out_buffer = clCreateBuffer(dc->context, CL_MEM_WRITE_ONLY,
                                           sizeof(float) * image_size, NULL, &err);
        cl_error(err, "Failed to create output buffer");
        
        // Transfer data to device
        double transfer_start = get_time();
        
        err = clEnqueueWriteBuffer(dc->command_queue, in_buffer, CL_TRUE, 0,
                                   sizeof(float) * image_size, images[img_idx], 
                                   0, NULL, NULL);
        cl_error(err, "Failed to write input buffer");
        
        err = clEnqueueWriteBuffer(dc->command_queue, filter_buffer, CL_TRUE, 0,
                                   sizeof(float) * filter_size * filter_size, 
                                   gaussian_filter, 0, NULL, NULL);
        cl_error(err, "Failed to write filter buffer");
        
        double transfer_end = get_time();
        timing.transfer_time_to_device += (transfer_end - transfer_start);
        
        // Set kernel arguments
        err = clSetKernelArg(dc->kernel, 0, sizeof(cl_mem), &in_buffer);
        err |= clSetKernelArg(dc->kernel, 1, sizeof(cl_mem), &filter_buffer);
        err |= clSetKernelArg(dc->kernel, 2, sizeof(cl_mem), &out_buffer);
        err |= clSetKernelArg(dc->kernel, 3, sizeof(unsigned int), &gs);
        err |= clSetKernelArg(dc->kernel, 4, sizeof(unsigned int), &ww);
        err |= clSetKernelArg(dc->kernel, 5, sizeof(unsigned int), &aa);
        cl_error(err, "Failed to set kernel arguments");
        
        // Execute kernel
        size_t local_size = 256;
        size_t global_size = ((image_size + local_size - 1) / local_size) * local_size;
        
        cl_event kernel_event;
        err = clEnqueueNDRangeKernel(dc->command_queue, dc->kernel, 1, NULL,
                                     &global_size, &local_size, 0, NULL, &kernel_event);
        cl_error(err, "Failed to execute kernel");
        
        clWaitForEvents(1, &kernel_event);
        
        // Get kernel execution time
        cl_ulong start, end;
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, 
                               sizeof(start), &start, NULL);
        clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, 
                               sizeof(end), &end, NULL);
        timing.kernel_time += (double)(end - start) / 1e9;
        
        clReleaseEvent(kernel_event);
        
        // Read back results
        double read_start = get_time();
        
        err = clEnqueueReadBuffer(dc->command_queue, out_buffer, CL_TRUE, 0,
                                  sizeof(float) * image_size, images[img_idx], 
                                  0, NULL, NULL);
        cl_error(err, "Failed to read output buffer");
        
        double read_end = get_time();
        timing.transfer_time_from_device += (read_end - read_start);
        
        // Cleanup
        clReleaseMemObject(in_buffer);
        clReleaseMemObject(filter_buffer);
        clReleaseMemObject(out_buffer);
    }
    
    double total_end = get_time();
    timing.total_time = total_end - total_start;
    
    return timing;
}

// Cleanup device context
void cleanupDevice(DeviceContext* dc) {
    clReleaseKernel(dc->kernel);
    clReleaseProgram(dc->program);
    clReleaseCommandQueue(dc->command_queue);
    clReleaseContext(dc->context);
}

int main(int argc, char** argv) {
    cl_int err;
    
    // Configuration
    const int NUM_IMAGES = 1000;
    const int FILTER_SIZE = 5;
    const float SIGMA = 1.5f;
    
    printf("=== Heterogeneous Image Processing - Lab 6 ===\n");
    printf("Number of images: %d\n", NUM_IMAGES);
    printf("Filter size: %dx%d\n", FILTER_SIZE, FILTER_SIZE);
    printf("Sigma: %.2f\n\n", SIGMA);
    
    // Image dimensions (using your montblanc image: 1920x1080)
    int width = 1920;
    int height = 1080;
    size_t image_size = width * height;
    
    printf("Image dimensions: %dx%d\n", width, height);
    printf("Image size: %.2f KB\n", (image_size * sizeof(float)) / 1024.0);
    printf("Total memory for stream: %.2f MB\n\n", 
           (NUM_IMAGES * image_size * sizeof(float)) / (1024.0 * 1024.0));
    
    // Create Gaussian filter
    float* gaussian_filter = (float*)malloc(FILTER_SIZE * FILTER_SIZE * sizeof(float));
    createGaussianFilter(gaussian_filter, FILTER_SIZE, SIGMA);
    
    // Create image stream (simulating with random data for now)
    float** image_stream = (float**)malloc(NUM_IMAGES * sizeof(float*));
    for (int i = 0; i < NUM_IMAGES; i++) {
        image_stream[i] = (float*)malloc(image_size * sizeof(float));
        // Initialize with some data (in real case, load from image)
        for (int j = 0; j < image_size; j++) {
            image_stream[i][j] = 128.0f; // Gray value
        }
    }
    
    // Load kernel source
    FILE* kernel_file = fopen("/mnt/user-data/uploads/kernels_kernel.cl", "r");
    if (!kernel_file) {
        kernel_file = fopen("kernels/kernel.cl", "r");
        if (!kernel_file) {
            printf("Cannot open kernel file\n");
            exit(1);
        }
    }
    fseek(kernel_file, 0, SEEK_END);
    size_t kernel_size = ftell(kernel_file);
    rewind(kernel_file);
    char* kernel_source = (char*)malloc(kernel_size + 1);
    fread(kernel_source, 1, kernel_size, kernel_file);
    kernel_source[kernel_size] = '\0';
    fclose(kernel_file);
    
    // Get platforms and devices
    cl_platform_id platforms[MAX_PLATFORMS];
    cl_uint num_platforms;
    err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
    cl_error(err, "Failed to get platforms");
    
    printf("Found %d platform(s)\n", num_platforms);
    
    DeviceContext devices[2];
    int num_devices_to_use = 0;
    
    for (int i = 0; i < num_platforms && num_devices_to_use < 2; i++) {
        char platform_name[256];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), 
                         platform_name, NULL);
        printf("\nPlatform %d: %s\n", i, platform_name);
        
        cl_device_id platform_devices[MAX_DEVICES];
        cl_uint num_devices;
        
        // Try GPU first
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, MAX_DEVICES, 
                            platform_devices, &num_devices);
        if (err == CL_DEVICE_NOT_FOUND) {
            // Try CPU if no GPU
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, MAX_DEVICES, 
                                platform_devices, &num_devices);
        }
        
        if (err == CL_SUCCESS) {
            for (int j = 0; j < num_devices && num_devices_to_use < 2; j++) {
                devices[num_devices_to_use] = setupDevice(platforms[i], 
                                                          platform_devices[j],
                                                          num_devices_to_use, 
                                                          kernel_source, kernel_size);
                printf("  Device %d: %s\n", num_devices_to_use, 
                       devices[num_devices_to_use].device_name);
                num_devices_to_use++;
            }
        }
    }
    
    if (num_devices_to_use == 0) {
        printf("No devices found!\n");
        exit(1);
    }
    
    printf("\n=== Processing Configuration ===\n");
    printf("Using %d device(s)\n", num_devices_to_use);
    
    // Split workload
    //int images_per_device = NUM_IMAGES / num_devices_to_use;
    //int remaining = NUM_IMAGES % num_devices_to_use;

    float device_weights[] = {0.75, 0.25}; // Example weights for 2 devices

    
    printf("\n=== Workload Distribution ===\n");
    TimingInfo timings[2];
    
    double heterogeneous_start = get_time();
    
    int start_idx = 0;
    for (int i = 0; i < num_devices_to_use; i++) {
        int num_images_device = (int)(NUM_IMAGES * device_weights[i]);
        int end_idx = start_idx + num_images_device;

        if (i == num_devices_to_use - 1) {
            end_idx = NUM_IMAGES; // Ensure all images are processed
        }
        
        
        printf("Device %d (%s): Processing images %d to %d (%d images)\n",
               i, devices[i].device_name, start_idx, end_idx - 1, num_images_device);
        
        // Process sequentially (C version without threads)
        timings[i] = processImagesOnDevice(&devices[i], image_stream, 
                                           start_idx, end_idx,
                                           width, height, 
                                           gaussian_filter, FILTER_SIZE);
        
        start_idx = end_idx;
    }
    
    double heterogeneous_end = get_time();
    double heterogeneous_total = heterogeneous_end - heterogeneous_start;
    
    // Print detailed timing results
    printf("\n=== Timing Results (Sequential Execution) ===\n");
    double total_kernel_time = 0;
    double total_transfer_to = 0;
    double total_transfer_from = 0;
    double total_device_time = 0;
    
    for (int i = 0; i < num_devices_to_use; i++) {
        printf("\nDevice %d (%s):\n", i, devices[i].device_name);
        printf("  Kernel execution time: %.6f s\n", timings[i].kernel_time);
        printf("  Transfer to device:    %.6f s\n", timings[i].transfer_time_to_device);
        printf("  Transfer from device:  %.6f s\n", timings[i].transfer_time_from_device);
        printf("  Total device time:     %.6f s\n", timings[i].total_time);
        
        total_kernel_time += timings[i].kernel_time;
        total_transfer_to += timings[i].transfer_time_to_device;
        total_transfer_from += timings[i].transfer_time_from_device;
        total_device_time += timings[i].total_time;
    }
    
    printf("\n=== Overall Results ===\n");
    printf("Total execution time (sequential): %.6f s\n", heterogeneous_total);
    printf("Total kernel time (sum):           %.6f s\n", total_kernel_time);
    printf("Total transfer to devices:         %.6f s\n", total_transfer_to);
    printf("Total transfer from devices:       %.6f s\n", total_transfer_from);
    printf("Images processed per second:       %.2f\n", NUM_IMAGES / heterogeneous_total);
    
    // Calculate bandwidth
    double data_to_device = NUM_IMAGES * (image_size + FILTER_SIZE * FILTER_SIZE) * sizeof(float);
    double data_from_device = NUM_IMAGES * image_size * sizeof(float);
    double bandwidth_to = data_to_device / (total_transfer_to * 1024 * 1024 * 1024);
    double bandwidth_from = data_from_device / (total_transfer_from * 1024 * 1024 * 1024);
    
    printf("\nBandwidth to devices:   %.2f GB/s\n", bandwidth_to);
    printf("Bandwidth from devices: %.2f GB/s\n", bandwidth_from);
    
    // Workload analysis (simulated - in real concurrent version would be different)
    printf("\n=== Workload Analysis (Sequential) ===\n");
    printf("NOTE: This is sequential execution. For true heterogeneous speedup,\n");
    printf("      you would need concurrent execution (requires pthreads or OpenMP).\n\n");
    
    double fastest_device_time = timings[0].total_time;
    double slowest_device_time = timings[0].total_time;
    int fastest_device = 0;
    int slowest_device = 0;
    
    for (int i = 1; i < num_devices_to_use; i++) {
        if (timings[i].total_time < fastest_device_time) {
            fastest_device_time = timings[i].total_time;
            fastest_device = i;
        }
        if (timings[i].total_time > slowest_device_time) {
            slowest_device_time = timings[i].total_time;
            slowest_device = i;
        }
    }
    
    printf("Fastest device: Device %d (%.6f s)\n", fastest_device, fastest_device_time);
    printf("Slowest device: Device %d (%.6f s)\n", slowest_device, slowest_device_time);
    
    if (num_devices_to_use > 1) {
        double imbalance = ((slowest_device_time - fastest_device_time) / slowest_device_time) * 100.0;
        printf("Workload imbalance: %.2f%%\n", imbalance);
    }
    
    // Bottleneck analysis per device
    printf("\n=== Bottleneck Analysis ===\n");
    for (int i = 0; i < num_devices_to_use; i++) {
        double comm_time = timings[i].transfer_time_to_device + 
                          timings[i].transfer_time_from_device;
        double comp_time = timings[i].kernel_time;
        double ratio = comm_time / comp_time;
        
        printf("\nDevice %d (%s):\n", i, devices[i].device_name);
        printf("  Communication time: %.6f s (%.1f%%)\n", 
               comm_time, (comm_time / timings[i].total_time) * 100);
        printf("  Computation time:   %.6f s (%.1f%%)\n",
               comp_time, (comp_time / timings[i].total_time) * 100);
        printf("  Comm/Comp ratio:    %.3f\n", ratio);
        
        if (ratio > 1.0) {
            printf(" BOTTLENECK: Communication-bound\n");
        } else {
            printf(" BOTTLENECK: Computation-bound\n");
        }
    }
    
    // Cleanup
    for (int i = 0; i < num_devices_to_use; i++) {
        cleanupDevice(&devices[i]);
    }
    
    for (int i = 0; i < NUM_IMAGES; i++) {
        free(image_stream[i]);
    }
    free(image_stream);
    free(gaussian_filter);
    free(kernel_source);
    
    printf("\n=== Processing Complete ===\n");
    printf("\nNOTE: This C version executes sequentially on multiple devices.\n");
    printf("For true concurrent heterogeneous execution, consider using:\n");
    printf("  - pthreads (POSIX threads)\n");
    printf("  - OpenMP (#pragma omp parallel)\n");
    printf("  - Or the C++ version with std::thread\n");
    
    return 0;
}
