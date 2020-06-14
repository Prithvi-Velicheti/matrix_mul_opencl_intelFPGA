//Matrix Multiplication
//Author: V.V.S.Prithvi
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "matrixMult.h"

using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
cl_device_id device; 


cl_context context = NULL;
cl_command_queue queue; // num_devices elements
cl_program program = NULL;
cl_kernel kernel; 

cl_mem input_a_buf; 
cl_mem input_b_buf; 
cl_mem output_buf; 

// Problem data.
unsigned A_height = 32 * BLOCK_SIZE;
unsigned A_width  = 16 * BLOCK_SIZE;
const unsigned &B_height = A_width;
unsigned B_width  = 16 * BLOCK_SIZE;
const unsigned &C_height = A_height;
const unsigned &C_width  = B_width;


float *input_a,*input_b,*output,*ref_output ;

// Function prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void compute_reference();
void verify();
void cleanup();

// Entry point.
int main(int argc, char **argv) {
	Options options(argc, argv);
	if(options.has("ah")) {
		A_height = options.get<unsigned>("ah");
	}
	if(options.has("aw")) {
		A_width = options.get<unsigned>("aw");
	}
	if(options.has("bw")) {
		B_width = options.get<unsigned>("bw");
	}

	printf("Matrix sizes:\n  A: %d x %d\n  B: %d x %d\n  C: %d x %d\n",
			A_height, A_width, B_height, B_width, C_height, C_width);

	// Spot check matrix sizes. They all must be a multiple of BLOCK_SIZE,
	// although it is relatively straightforward to handle non-multiples
	// by adding padding. For simplicity, this example does not pad.
	if((A_height % BLOCK_SIZE) != 0 || (A_width % BLOCK_SIZE) != 0 ||
			(B_height % BLOCK_SIZE) != 0 || (B_width % BLOCK_SIZE) != 0 ||
			(C_height % BLOCK_SIZE) != 0 || (C_width % BLOCK_SIZE) != 0) {
		printf("Matrix sizes must be a multiple of %d.\n", BLOCK_SIZE);
		return -1;
	}

	// Initialize OpenCL.
	if(!init_opencl()) {
		return -1;
	}

	// Initialize the problem data.
	// Requires the number of devices to be known.
	init_problem();

	// Run the kernel.
	run();

	// Free the resources allocated
	cleanup();

	return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
	return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

// Initializes the OpenCL objects.
bool init_opencl() {
	cl_int status;

	printf("Initializing OpenCL\n");

	if(!setCwdToExeDir()) {
		return false;
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
	if(platform == NULL) {
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
		return false;
	}


	// Query the available OpenCL device.
	scoped_array<cl_device_id> devices ;
	cl_uint  num_devices ;

	// device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	device = devices[0];   //We'll use only one device.



	printf("Platform: %s\n", getPlatformName(platform).c_str());

	/* printf("Using %d device(s)\n", num_devices);
	   for(unsigned i = 0; i < num_devices; ++i) {
	   printf("  %s\n", getDeviceName(device[i]).c_str());
	   }*/


	// Create the context.
	context = clCreateContext(NULL, 1,&device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the program for all device. Use the first device as the
	// representative device (assuming all device are of the same type).
	// std::string binary_file = getBoardBinaryFile("matrix_mult", device[0]);
	std::string binary_file = getBoardBinaryFile("matrix_mult", device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(),&device,1);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");


	// Command queue.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Kernel.
	const char *kernel_name = "matrixMult";   //Name inside the kernel
	kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel");



	input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_1_INTELFPGA,
			A_height * A_width * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for input A");

	input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_2_INTELFPGA,
			B_height * B_width * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for input B");

	// write to the output matrix.
	output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_1_INTELFPGA,
			C_height * C_width * sizeof(float), NULL, &status);
	checkError(status, "Failed to create buffer for output");

	return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {

	printf("Generating input matrices\n");


	
 	posix_memalign ((void **)(&input_a),64,sizeof(float)*A_height*A_width);
        posix_memalign ((void **)(&input_b),64,sizeof(float)*B_height*B_width);
        posix_memalign ((void **)(&output),64,sizeof(float)*A_height*B_width);
        posix_memalign ((void **)(&ref_output),64,sizeof(float)*A_height*B_width);
		

	for(unsigned j = 0; j < A_height*A_width; ++j) {
                input_a[j] = rand_float();
        }

	for(unsigned j = 0; j < B_height*B_width; ++j) {
                input_b[j] = rand_float();
        }


}

void run() {
	cl_int status;

		status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
				0, A_height* A_width * sizeof(float), input_a, 0, NULL, NULL);
		checkError(status, "Failed to transfer input A");

		status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
				0, B_width * B_height * sizeof(float), input_b, 0, NULL, NULL);
		checkError(status, "Failed to transfer input B");

	// Wait for all queues to finish.
		clFinish(queue);



	cl_event kernel_event;

	const double start_time = getCurrentTimestamp();
	unsigned argi = 0;



		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_buf);
		checkError(status, "Failed to set argument 0");

		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_a_buf);
		checkError(status, "Failed to set argument 1");

		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &input_b_buf);
		checkError(status, "Failed to set argument 2");





		status = clSetKernelArg(kernel,3, sizeof(A_width), &A_width);
		checkError(status, "Failed to set argument 3");

		status = clSetKernelArg(kernel,4, sizeof(B_width), &B_width);
		checkError(status, "Failed to set argument 4");

		// Enqueue kernel.
		// Use a global work size corresponding to the size of the output matrix.
		// Each work-item computes the result for one value of the output matrix,
		// so the global work size has the same dimensions as the output matrix.
		//
		// The local work size is one block, so BLOCK_SIZE x BLOCK_SIZE.
		//
		// Events are used to ensure that the kernel is not launched until
		// the writes to the input buffers have completed.
		const size_t global_work_size[2] = {C_width, A_height};
		const size_t local_work_size[2]  = {BLOCK_SIZE, BLOCK_SIZE};
		printf("Launching for device  (global size: %zd, %zd)\n", global_work_size[0], global_work_size[1]);

		status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
				global_work_size, local_work_size, 0, NULL, &kernel_event);
		checkError(status, "Failed to launch kernel");

	// Wait for all kernels to finish.
	clWaitForEvents(1, &kernel_event);

	const double end_time = getCurrentTimestamp();
	const double total_time = end_time - start_time;

	// Wall-clock time taken.
	printf("\nTime: %0.3f ms\n", total_time * 1e3);

		cl_ulong time_ns = getStartEndTime(kernel_event);
		printf("Kernel time (device ): %0.3f ms\n", double(time_ns) * 1e-6);

	// Compute the throughput (GFLOPS).
	// There are C_width * C_height output values, with each value
	// computed using A_width multiplies and adds.
	const float flops = (float)(2.0f * C_width * C_height * A_width / total_time);
	printf("\nThroughput: %0.2f GFLOPS\n\n", flops * 1e-9);

		clReleaseEvent(kernel_event);

		status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,
				0, C_height * C_width * sizeof(float), output, 0, NULL, NULL);
		checkError(status, "Failed to read output matrix");

	// Verify results.
	compute_reference();
	verify();
}


void compute_reference() {
	// Compute the reference output.
	printf("Computing reference output\n");

	for (unsigned y = 0 ; y < A_height ; ++y) {
		 for (unsigned x = 0; x < B_width ;++x) {
			//Compute the result for C(y,x)
			float sum = 0.0f ;
			  for (unsigned k = 0 ; k < A_width ; ++k) {
				  	 sum += input_a[y*A_width +k]* input_b[k*B_width +x] ;
			  }
			  ref_output[y*C_width +x] = sum;
		 }
	}
}




void verify() {
	printf("Verifying\n");

	// Compute the L^2-Norm of the difference between the output and reference
	// output matrices and compare it against the L^2-Norm of the reference.
	float diff = 0.0f;
	float ref = 0.0f;


	for(unsigned y = 0 ; y < C_height; ++y) {
			for(unsigned x = 0; x < C_width; ++x) {
				const float o = output[y* C_width + x];
				const float r = ref_output[y * C_width + x];
				const float d = o - r;
				diff += d * d;
				ref += r * r;
			}
		}

	const float diff_l2norm = sqrtf(diff);
	const float ref_l2norm = sqrtf(ref);
	const float error = diff_l2norm / ref_l2norm;
	const bool pass = error < 1e-6;
	printf("Verification: %s\n", pass ? "PASS" : "FAIL");
	if(!pass) {
		printf("Error (L^2-Norm): %0.3g\n", error);
	}
}




// Free the resources allocated during initialization
void cleanup() {
		if(kernel ) {
			clReleaseKernel(kernel);
		}
		if(queue ) {
			clReleaseCommandQueue(queue);
		}
		if(input_a_buf) {
			clReleaseMemObject(input_a_buf);
		}
		if(input_b_buf ) {
			clReleaseMemObject(input_b_buf);
		}
		if(output_buf ) {
			clReleaseMemObject(output_buf);
		}

	if(program) {
		clReleaseProgram(program);
	}
	if(context) {
		clReleaseContext(context);
	}
}

