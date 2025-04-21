#include <stdio.h>
#include "cs1300bmp.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "Filter.h"
#include <omp.h>      // OpenMP
#include <immintrin.h> // SSE

using namespace std;

#include "rdtsc.h"

//
// Forward declare the functions
//
Filter * readFilter(string filename);
double applyFilter(Filter *filter, cs1300bmp *input, cs1300bmp *output);


int
main(int argc, char **argv)
{

  if ( argc < 2) {
    fprintf(stderr,"Usage: %s filter inputfile1 inputfile2 .... \n", argv[0]);
  }

  //
  // Convert to C++ strings to simplify manipulation
  //
  string filtername = argv[1];

  //
  // remove any ".filter" in the filtername
  //
  string filterOutputName = filtername;
  string::size_type loc = filterOutputName.find(".filter");
  if (loc != string::npos) {
    //
    // Remove the ".filter" name, which should occur on all the provided filters
    //
    filterOutputName = filtername.substr(0, loc);
  }

  Filter *filter = readFilter(filtername);

  double sum = 0.0;
  int samples = 0;

  for (int inNum = 2; inNum < argc; inNum++) {
    string inputFilename = argv[inNum];
    string outputFilename = "filtered-" + filterOutputName + "-" + inputFilename;
    struct cs1300bmp *input = new struct cs1300bmp;
    struct cs1300bmp *output = new struct cs1300bmp;
    int ok = cs1300bmp_readfile( (char *) inputFilename.c_str(), input);

    if ( ok ) {
      double sample = applyFilter(filter, input, output);
      sum += sample;
      samples++;
      cs1300bmp_writefile((char *) outputFilename.c_str(), output);
    }
    delete input;
    delete output;
  }
  fprintf(stdout, "Average cycles per sample is %f\n", sum / samples);

}

class Filter *
readFilter(string filename)
{
  ifstream input(filename.c_str());

  if ( ! input.bad() ) {
    int size = 0;
    input >> size;
    Filter *filter = new Filter(size);
    int div;
    input >> div;
    filter -> setDivisor(div);
    for (int i=0; i < size; i++) {
      for (int j=0; j < size; j++) {
	int value;
	input >> value;
	filter -> set(i,j,value);
      }
    }
    return filter;
  } else {
    cerr << "Bad input in readFilter:" << filename << endl;
    exit(-1);
  }
}


// // OpenMP
// double applyFilter(Filter *filter, cs1300bmp *input, cs1300bmp *output)
// {
//     long long cycStart, cycStop;
//     cycStart = rdtscll();

//     const int width = input->width;
//     const int height = input->height;
//     output->width = width;
//     output->height = height;

//     const int filterSize = filter->getSize();
//     const int filterDiv = filter->getDivisor();
//     const int half = filterSize / 2;

//     if (filterSize == 3) {
//         const int f00 = filter->get(0, 0);
//         const int f01 = filter->get(0, 1);
//         const int f02 = filter->get(0, 2);
//         const int f10 = filter->get(1, 0);
//         const int f11 = filter->get(1, 1);
//         const int f12 = filter->get(1, 2);
//         const int f20 = filter->get(2, 0);
//         const int f21 = filter->get(2, 1);
//         const int f22 = filter->get(2, 2);

//         #pragma omp parallel for collapse(3) schedule(static)
//         for (int plane = 0; plane < 3; plane++) {
//             for (int row = 1; row < height - 1; row++) {
//                 for (int col = 1; col < width - 1; col++) {
//                     int sum = 
//                         input->color[plane][row - 1][col - 1] * f00 +
//                         input->color[plane][row - 1][col    ] * f01 +
//                         input->color[plane][row - 1][col + 1] * f02 +
//                         input->color[plane][row    ][col - 1] * f10 +
//                         input->color[plane][row    ][col    ] * f11 +
//                         input->color[plane][row    ][col + 1] * f12 +
//                         input->color[plane][row + 1][col - 1] * f20 +
//                         input->color[plane][row + 1][col    ] * f21 +
//                         input->color[plane][row + 1][col + 1] * f22;

//                     sum /= filterDiv;

//                     sum = (sum < 0) ? 0 : ( (sum > 255) ? 255 : sum );
//                     output->color[plane][row][col] = sum;
//                 }
//             }
//         }
//     } else {

//         #pragma omp parallel for collapse(3) schedule(static)
//         for (int plane = 0; plane < 3; plane++) {
//             for (int row = half; row < height - half; row++) {
//                 for (int col = half; col < width - half; col++) {
//                     int sum = 0;
//                     for (int i = 0; i < filterSize; i++) {
//                         for (int j = 0; j < filterSize; j++) {
//                             sum += input->color[plane][row + i - half][col + j - half] * filter->get(i, j);
//                         }
//                     }
//                     sum /= filterDiv;
//                     sum = (sum < 0) ? 0 : ( (sum > 255) ? 255 : sum );
//                     output->color[plane][row][col] = sum;
//                 }
//             }
//         }
//     }

//     cycStop = rdtscll();
//     double diff = cycStop - cycStart;
//     double diffPerPixel = diff / (width * height);
//     fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n", diff, diffPerPixel);
//     return diffPerPixel;
// }



// Non OpenMP

double applyFilter(Filter *filter, cs1300bmp *input, cs1300bmp *output)
{
    long long cycStart, cycStop;
    cycStart = rdtscll();

    const int width  = input->width;
    const int height = input->height;
    output->width  = width;
    output->height = height;

    const int filterSize = filter->getSize();
    const int filterDiv  = filter->getDivisor();
    const int half = filterSize / 2;

    if (filterSize == 3) {
        const int f00 = filter->get(0, 0);
        const int f01 = filter->get(0, 1);
        const int f02 = filter->get(0, 2);
        const int f10 = filter->get(1, 0);
        const int f11 = filter->get(1, 1);
        const int f12 = filter->get(1, 2);
        const int f20 = filter->get(2, 0);
        const int f21 = filter->get(2, 1);
        const int f22 = filter->get(2, 2);

        for (int plane = 0; plane < 3; plane++) {
            for (int row = 1; row < height - 1; row++) {
                const int r0 = row - 1;
                const int r1 = row;
                const int r2 = row + 1;
                int col;

                for (col = 1; col <= width - 2 - 1; col += 2) {
                    const int c0 = col - 1;
                    const int c1 = col;
                    const int c2 = col + 1;
                    int sum1 =  input->color[plane][r0][c0] * f00 +
                                input->color[plane][r0][c1] * f01 +
                                input->color[plane][r0][c2] * f02 +
                                input->color[plane][r1][c0] * f10 +
                                input->color[plane][r1][c1] * f11 +
                                input->color[plane][r1][c2] * f12 +
                                input->color[plane][r2][c0] * f20 +
                                input->color[plane][r2][c1] * f21 +
                                input->color[plane][r2][c2] * f22;
                    sum1 /= filterDiv;
                    if (sum1 < 0)
                        sum1 = 0;
                    else if (sum1 > 255)
                        sum1 = 255;
                    output->color[plane][row][col] = sum1;

                    const int c0_2 = col;
                    const int c1_2 = col + 1;
                    const int c2_2 = col + 2;
                    int sum2 =  input->color[plane][r0][c0_2] * f00 +
                                input->color[plane][r0][c1_2] * f01 +
                                input->color[plane][r0][c2_2] * f02 +
                                input->color[plane][r1][c0_2] * f10 +
                                input->color[plane][r1][c1_2] * f11 +
                                input->color[plane][r1][c2_2] * f12 +
                                input->color[plane][r2][c0_2] * f20 +
                                input->color[plane][r2][c1_2] * f21 +
                                input->color[plane][r2][c2_2] * f22;
                    sum2 /= filterDiv;
                    if (sum2 < 0)
                        sum2 = 0;
                    else if (sum2 > 255)
                        sum2 = 255;
                    output->color[plane][row][col + 1] = sum2;
                }
                for (; col < width - 1; col++) {
                    const int c0 = col - 1;
                    const int c1 = col;
                    const int c2 = col + 1;
                    int sum =  input->color[plane][r0][c0] * f00 +
                               input->color[plane][r0][c1] * f01 +
                               input->color[plane][r0][c2] * f02 +
                               input->color[plane][r1][c0] * f10 +
                               input->color[plane][r1][c1] * f11 +
                               input->color[plane][r1][c2] * f12 +
                               input->color[plane][r2][c0] * f20 +
                               input->color[plane][r2][c1] * f21 +
                               input->color[plane][r2][c2] * f22;
                    sum /= filterDiv;
                    if (sum < 0)
                        sum = 0;
                    else if (sum > 255)
                        sum = 255;
                    output->color[plane][row][col] = sum;
                }
            }
        }
    } else {
        for (int plane = 0; plane < 3; plane++) {
            for (int row = half; row < height - half; row++) {
                for (int col = half; col < width - half; col++) {
                    int sum = 0;
                    for (int i = 0; i < filterSize; i++) {
                        for (int j = 0; j < filterSize; j++) {
                            sum += input->color[plane][row + i - half][col + j - half] * filter->get(i, j);
                        }
                    }
                    sum /= filterDiv;
                    if (sum < 0)
                        sum = 0;
                    else if (sum > 255)
                        sum = 255;
                    output->color[plane][row][col] = sum;
                }
            }
        }
    }

    cycStop = rdtscll();
    double diff = cycStop - cycStart;
    double diffPerPixel = diff / (width * height);
    fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n", diff, diffPerPixel);
    return diffPerPixel;
}


// #include <immintrin.h>  // AVX2 头文件
// #include <iostream>
// #include "cs1300bmp.h"
// #include "Filter.h"
// #include "rdtsc.h"

// double applyFilter(Filter *filter, cs1300bmp *input, cs1300bmp *output) {
//     long long cycStart, cycStop;
//     cycStart = rdtscll();

//     int width = input->width;
//     int height = input->height;
//     int filterSize = filter->getSize();
//     int filterDiv = filter->getDivisor();
//     int half = filterSize / 2;

//     output->width = width;
//     output->height = height;

//     if (filterSize == 3) {
//         // 使用 AVX2 优化 3x3 滤镜，每次处理 8 个像素
//         // 预加载 3x3 滤镜系数为 256 位向量
//         __m256i f00 = _mm256_set1_epi32(filter->get(0, 0));
//         __m256i f01 = _mm256_set1_epi32(filter->get(0, 1));
//         __m256i f02 = _mm256_set1_epi32(filter->get(0, 2));
//         __m256i f10 = _mm256_set1_epi32(filter->get(1, 0));
//         __m256i f11 = _mm256_set1_epi32(filter->get(1, 1));
//         __m256i f12 = _mm256_set1_epi32(filter->get(1, 2));
//         __m256i f20 = _mm256_set1_epi32(filter->get(2, 0));
//         __m256i f21 = _mm256_set1_epi32(filter->get(2, 1));
//         __m256i f22 = _mm256_set1_epi32(filter->get(2, 2));


//         __m256 divisor_f = _mm256_set1_ps((float)filterDiv);

//         for (int plane = 0; plane < 3; plane++) {
//             for (int row = 1; row < height - 1; row++) {
//                 int col;

//                 for (col = 1; col <= width - 9; col += 8) {
//                     __m256i sum = _mm256_setzero_si256();


//                     __m256i p00 = _mm256_loadu_si256((__m256i*)&input->color[plane][row - 1][col - 1]);
//                     __m256i p01 = _mm256_loadu_si256((__m256i*)&input->color[plane][row - 1][col    ]);
//                     __m256i p02 = _mm256_loadu_si256((__m256i*)&input->color[plane][row - 1][col + 1]);
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p00, f00));
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p01, f01));
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p02, f02));


//                     __m256i p10 = _mm256_loadu_si256((__m256i*)&input->color[plane][row][col - 1]);
//                     __m256i p11 = _mm256_loadu_si256((__m256i*)&input->color[plane][row][col    ]);
//                     __m256i p12 = _mm256_loadu_si256((__m256i*)&input->color[plane][row][col + 1]);
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p10, f10));
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p11, f11));
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p12, f12));


//                     __m256i p20 = _mm256_loadu_si256((__m256i*)&input->color[plane][row + 1][col - 1]);
//                     __m256i p21 = _mm256_loadu_si256((__m256i*)&input->color[plane][row + 1][col    ]);
//                     __m256i p22 = _mm256_loadu_si256((__m256i*)&input->color[plane][row + 1][col + 1]);
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p20, f20));
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p21, f21));
//                     sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(p22, f22));


//                     __m256 sum_f = _mm256_cvtepi32_ps(sum);
//                     sum_f = _mm256_div_ps(sum_f, divisor_f);
//                     sum_f = _mm256_round_ps(sum_f, _MM_FROUND_TRUNC);
//                     sum = _mm256_cvtps_epi32(sum_f);


//                     sum = _mm256_max_epi32(sum, _mm256_setzero_si256());
//                     sum = _mm256_min_epi32(sum, _mm256_set1_epi32(255));


//                     _mm256_storeu_si256((__m256i*)&output->color[plane][row][col], sum);
//                 }

//                 for (; col < width - 1; col++) {
//                     int sum = 0;
//                     sum += input->color[plane][row - 1][col - 1] * filter->get(0, 0);
//                     sum += input->color[plane][row - 1][col    ] * filter->get(0, 1);
//                     sum += input->color[plane][row - 1][col + 1] * filter->get(0, 2);
//                     sum += input->color[plane][row    ][col - 1] * filter->get(1, 0);
//                     sum += input->color[plane][row    ][col    ] * filter->get(1, 1);
//                     sum += input->color[plane][row    ][col + 1] * filter->get(1, 2);
//                     sum += input->color[plane][row + 1][col - 1] * filter->get(2, 0);
//                     sum += input->color[plane][row + 1][col    ] * filter->get(2, 1);
//                     sum += input->color[plane][row + 1][col + 1] * filter->get(2, 2);

//                     sum /= filterDiv;
//                     if(sum < 0) sum = 0;
//                     if(sum > 255) sum = 255;
//                     output->color[plane][row][col] = sum;
//                 }
//             }
//         }
//     } else {

//         for (int plane = 0; plane < 3; plane++) {
//             for (int row = half; row < height - half; row++) {
//                 for (int col = half; col < width - half; col++) {
//                     int sum = 0;
//                     for (int i = 0; i < filterSize; i++) {
//                         for (int j = 0; j < filterSize; j++) {
//                             sum += input->color[plane][row + i - half][col + j - half] * filter->get(i, j);
//                         }
//                     }
//                     sum /= filterDiv;
//                     if(sum < 0) sum = 0;
//                     if(sum > 255) sum = 255;
//                     output->color[plane][row][col] = sum;
//                 }
//             }
//         }
//     }

//     cycStop = rdtscll();
//     double diff = cycStop - cycStart;
//     fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n", diff, diff / (width * height));
//     return diff / (width * height);
// }

// OpenCV
// #include <opencv2/opencv.hpp>

// double applyFilter(Filter *filter, cs1300bmp *input, cs1300bmp *output) {
//     long long cycStart, cycStop;
//     cycStart = rdtscll();

//     int width = input->width;
//     int height = input->height;
//     int filterSize = filter->getSize();
//     int divisor = filter->getDivisor();
//     if (divisor == 0) divisor = 1;

//     cv::Mat image(height, width, CV_8UC3);
//     for (int row = 0; row < height; row++) {
//         for (int col = 0; col < width; col++) {
//             image.at<cv::Vec3b>(row, col)[0] = input->color[0][row][col];
//             image.at<cv::Vec3b>(row, col)[1] = input->color[1][row][col];
//             image.at<cv::Vec3b>(row, col)[2] = input->color[2][row][col];
//         }
//     }

//     cv::Mat kernel(filterSize, filterSize, CV_32F);
//     for (int i = 0; i < filterSize; i++) {
//         for (int j = 0; j < filterSize; j++) {
//             kernel.at<float>(i, j) = filter->get(i, j) / (float) divisor;
//         }
//     }

//     cv::Mat result;
//     cv::filter2D(image, result, -1, kernel);

//     for (int row = 0; row < height; row++) {
//         for (int col = 0; col < width; col++) {
//             output->color[0][row][col] = result.at<cv::Vec3b>(row, col)[0];
//             output->color[1][row][col] = result.at<cv::Vec3b>(row, col)[1];
//             output->color[2][row][col] = result.at<cv::Vec3b>(row, col)[2];
//         }
//     }

//     cycStop = rdtscll();
//     return (cycStop - cycStart) / (width * height);
// }

// Original Code
// double
// applyFilter(class Filter *filter, cs1300bmp *input, cs1300bmp *output)
// {

//   long long cycStart, cycStop;

//   cycStart = rdtscll();

//   output -> width = input -> width;
//   output -> height = input -> height;


//   for(int col = 1; col < (input -> width) - 1; col = col + 1) {
//     for(int row = 1; row < (input -> height) - 1 ; row = row + 1) {
//       for(int plane = 0; plane < 3; plane++) {

// 	output -> color[plane][row][col] = 0;

// 	for (int j = 0; j < filter -> getSize(); j++) {
// 	  for (int i = 0; i < filter -> getSize(); i++) {	
// 	    output -> color[plane][row][col]
// 	      = output -> color[plane][row][col]
// 	      + (input -> color[plane][row + i - 1][col + j - 1] 
// 		 * filter -> get(i, j) );
// 	  }
// 	}
	
// 	output -> color[plane][row][col] = 	
// 	  output -> color[plane][row][col] / filter -> getDivisor();

// 	if ( output -> color[plane][row][col]  < 0 ) {
// 	  output -> color[plane][row][col] = 0;
// 	}

// 	if ( output -> color[plane][row][col]  > 255 ) { 
// 	  output -> color[plane][row][col] = 255;
// 	}
//       }
//     }
//   }

//   cycStop = rdtscll();
//   double diff = cycStop - cycStart;
//   double diffPerPixel = diff / (output -> width * output -> height);
//   fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
// 	  diff, diff / (output -> width * output -> height));
//   return diffPerPixel;
// }
