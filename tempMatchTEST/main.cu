#include <stdio.h>
#include <cmath>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		printf("Error: %s:%d, ", __FILE__, __LINE__);\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(1);\
	}\
}

using namespace std;
using namespace cv;

/* functions */
double cpuSecond(void);
void cpuSAD(float*, float*, float*, const int, const int, const int, const int, const int);
void cpuMin(float*, int*, const int, const int);
void plotbox(Mat, int *, const int, const int);

__global__ void gpuSAD(float *, float *, float *, const int, const int, const int, const int, const int);

/* main routine */
int main(int argc, char** argv) {

    /* read target & template images (RGB) */
    Mat targetRGB;
    targetRGB = imread("image2Trg.jpg", CV_LOAD_IMAGE_COLOR);
    if (!targetRGB.data) {
        printf("No target image data\n");
        return -1;
    }

    Mat tempRGB;
    tempRGB = imread("image2Tmp.jpg", CV_LOAD_IMAGE_COLOR);
    if (!tempRGB.data) {
        printf("No template image data\n");
        return -1;
    }

    /* read target & template images (gray-scale) */
    Mat targetGRAY;
    targetGRAY = imread("image2Trg.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!targetGRAY.data) {
        printf("No target image data\n");
        return -1;
    }

    Mat tempGRAY;
    tempGRAY = imread("image2Tmp.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!tempGRAY.data) {
        printf("No template image data\n");
        return -1;
    }
/*
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("targetRGB", targetRGB);

    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("tempRGB", tempRGB);
    waitKey(0);
*/
    /* convert uchar to double */
    Mat target1;
    targetGRAY.convertTo(target1, CV_32FC1);

    Mat template1;
    tempGRAY.convertTo(template1, CV_32FC1);

    /* set rows & colums for each image and reference */
    const int rowsTrg = target1.rows, colsTrg = target1.cols;
    const int rowsTmp = template1.rows, colsTmp = template1.cols;
    const int rowsRef = rowsTrg - rowsTmp, colsRef = colsTrg - colsTmp;

    /* store gray image data into array */
    
    // configure size of vectors
    int nElemTrg = rowsTrg * colsTrg;
    int nElemTmp = rowsTmp * colsTmp;
    int nElemRef = (rowsTrg - rowsTmp) * (colsTrg - colsTmp);
    printf("target vector size = %d x %d\n", rowsTrg, colsTrg);
    printf("template vector size = %d x %d\n", rowsTmp, colsTmp);
    //printf("result vector size = %d\n", nElemRef);

    // allocate vectors in host memory
    size_t nBytesTrg = nElemTrg * sizeof(float);
    size_t nBytesTmp = nElemTmp * sizeof(float);
    size_t nBytesRef = nElemRef * sizeof(float);

    /*** CPU process start ***/
    float *h_target, *h_template, *hostRef, *gpuRef;

    h_target = (float *)malloc(nBytesTrg);
    h_template = (float *)malloc(nBytesTmp);
    hostRef = (float *)malloc(nBytesRef);
    gpuRef = (float *)malloc(nBytesRef);

    // store image data for each
    for (int j = 0; j < rowsTrg; j++) {
    	for (int i = 0; i < colsTrg; i++) {
            h_target[j * colsTrg + i] = target1.at<float>(j, i);
            //printf("%d %f %f\n", j * colsTrg + i, target1.at<float>(j, i), h_target[j * colsTrg + i]);
        }
    }

    for (int j = 0; j < rowsTmp; j++) {
    	for (int i = 0; i < colsTmp; i++) {
            h_template[j * colsTmp + i] = template1.at<float>(j, i);
            //printf("%d %f %f\n", j * colsTmp + i, template1.at<float>(j, i), h_template[j * colsTmp + i]);
        }
    }

    // initalize ref values
    memset(hostRef, 0, nBytesRef);

    /* template matching on CPU */
    // SAD
    double iStart = cpuSecond();
    cpuSAD(h_target, h_template, hostRef, colsTrg, rowsTmp, colsTmp, rowsRef, colsRef);
    double iElaps = cpuSecond() - iStart;
    printf("Time elapsed on CPU = %f ms\n", iElaps * 1000.0f);

    /* get position */

    // initialize value
    int *pos;
    size_t nBytesPos = 2 * sizeof(int);
    pos = (int *)malloc(nBytesPos);
    memset(pos, 0, nBytesPos);

    // use min function to obtain position
    //cpuMin(hostRef, pos, rowsRef, colsRef);

    /* display result image with box */
//    plotbox(targetRGB, pos, rowsTmp, colsTmp);

    /*** CPU process end ***/

    /*** GPU process start ***/
iStart = cpuSecond();
    // initalize ref values
    memset(gpuRef, 0, nBytesRef);

    /* allocate memory at device */
    float *d_target, *d_template, *d_ref;
    CHECK(cudaMalloc((float**)&d_target, nBytesTrg));
    CHECK(cudaMalloc((float**)&d_template, nBytesTmp));
    CHECK(cudaMalloc((float**)&d_ref, nBytesRef));

    /* transfer data from host to device */
    CHECK(cudaMemcpy(d_target, h_target, nBytesTrg, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_template, h_template, nBytesTmp, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ref, gpuRef, nBytesRef, cudaMemcpyHostToDevice));

    /* template matching on GPU */
    // declare nThreads & nBlocks
    dim3 nThreads(32, 32);
    dim3 nBlocks((colsRef / nThreads.x) + 1, (rowsRef / nThreads.y) + 1);
    printf("colsRef = %d nBlocks.x = %d \n", colsRef, nBlocks.x);

    // SAD
    //iStart = cpuSecond();
    gpuSAD<<<nBlocks, nThreads>>>(d_target, d_template, d_ref, colsTrg, rowsTmp, colsTmp, rowsRef, colsRef);
    CHECK(cudaDeviceSynchronize());
    //iElaps = cpuSecond() - iStart;
    //printf("Time elapsed on GPU = %f ms\n", iElaps * 1000.0f);

    // check kernel error
    CHECK(cudaGetLastError());

    // copy result from device to host
    CHECK(cudaMemcpy(gpuRef, d_ref, nBytesRef, cudaMemcpyDeviceToHost));

 iElaps = cpuSecond() - iStart;
 printf("Time elapsed on GPU = %f ms\n", iElaps * 1000.0f);  
 
    for (int j = 0; j < rowsRef; j++) {
    	for (int i = 0; i < colsRef; i++) {
            //printf("%d %d\n", j * (colsRef) + i, hostRef[j * (colsRef) + i] == gpuRef[j * (colsRef) + i]);
        }
    }

    /* get position */

    // initialize value
    

    // use min function to obtain position
    //cpuMin(gpuRef, pos, rowsRef, colsRef);

    /* display result image with box */
    //plotbox(targetRGB, pos, rowsTmp, colsTmp);

    // release cpu memory
    free(h_target);
    free(h_template);
    free(hostRef);
    free(pos);

    // release gpu memory
    cudaFree(d_target);
    cudaFree(d_template);
    cudaFree(d_ref);

    /*** GPU process end ***/


    return 0;
}

/*** 3 general functions ***/
/* 1/3 CPU Timer */
double cpuSecond(void) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/* 2/3 Plot Box */
// show result image with box
void plotbox(Mat target, int *pos, const int rowsTmp, const int colsTmp) {
     
     // set position
     int y = pos[0], x = pos[1];
     
     /* color */
     Vec3b p = Vec3b(255, 255, 255);
     
     // y-axis
     for (int i = x; i < x + colsTmp + 1; i++) {
         target.at<Vec3b>(y, i) = p;
         target.at<Vec3b>(y + rowsTmp, i) = p;
     }

     // x-axis
     for (int j = y; j < y + rowsTmp + 1; j++) {
         target.at<Vec3b>(j, x) = p;
         target.at<Vec3b>(j, x + colsTmp) = p;
     }

     /* display result image */
     namedWindow("Result", WINDOW_AUTOSIZE);
     imshow("Result", target);
     waitKey(0);

}

/* 3/3 CPU Minimum */
void cpuMin(float *ref, int *pos, const int rowsRef, const int colsRef) {
  
     // initialize value
     float minVal = 1.0e10f;

     for (int j = 0; j < rowsRef; j++) {
        for (int i = 0; i < colsRef; i++) {
            
           if (minVal > ref[j * colsRef + i]) {

               minVal = ref[j * colsRef + i];
               pos[0] = j;
	       pos[1] = i;

           } 

        }
     }

     printf("y = %d x = %d min = %f\n", pos[0], pos[1], minVal);

}


/*** 1 CPU functions ***/
/* 1/1 CPU SAD */
void cpuSAD(float *target, float *temp, float *ref, const int colsTrg, const int rowsTmp, const int colsTmp, const int rowsRef, const int colsRef) {

    // loop
    for (int j = 0; j < rowsRef; j++) {
        for (int i = 0; i < colsRef; i++) {
            
            // SAD algorithm
            for (int y = 0; y < rowsTmp; y++) {
                for (int x = 0; x < colsTmp; x++) {

                    ref[j * colsRef + i] += abs(target[(j + y) * colsTrg + (i + x)] - temp[y * colsTmp + x]);

                }
            }

        }
    }

}

/*** 1 GPU kernel ***/
__global__ void gpuSAD(float *target, float *temp, float *ref, const int colsTrg, const int rowsTmp, const int colsTmp, const int rowsRef, const int colsRef) {
	
        // prepare values
        int j = blockIdx.y * blockDim.y + threadIdx.y; 
        int i = blockIdx.x * blockDim.x + threadIdx.x; 

        // check the thread is in ref
        if (j < rowsRef && i < colsRef) {
            // SAD algorithm
            for (int y = 0; y < rowsTmp; y++) {
                for (int x = 0; x < colsTmp; x++) {

                    ref[j * colsRef + i] += abs(target[(j + y) * colsTrg + (i + x)] - temp[y * colsTmp + x]);

                }
            }
        }
}









