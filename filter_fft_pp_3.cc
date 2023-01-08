#include "AudioFile.h"
// #include "FFT_pp.h"
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
// #include <cmath>
#include <ctime>
#include <windows.h>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <math.h>
using namespace std;

typedef struct fft_data {
   int id;
   int nthreads;
   int stage;
} fft_data;

void create_hann_window(vector<float>& v, int n);
void vector_mul(vector<float>& v1, vector<float>& v2);
void low_pass_filter(vector<float>& real, vector<float>& img, int sr, int nfft, int cut_off_freq);
void high_pass_filter(vector<float>& real, vector<float>& img, int sr, int nfft, int cut_off_freq);
void overlap_add(vector<float>& result, vector<float>& v, int frame_count, int hop_size);
void calculate_fft_twiddle_factor(float real[], float img[], int n);
void calculate_ifft_twiddle_factor(float real[], float img[], int n);

void swap (float &a, float &b);
void bitrp (vector<float>& xreal, vector<float>& ximag, int n);
void* FFT(void* input);
void* IFFT (void* input);
// void FFT(vector<float>& xreal, vector<float>& ximag, int n);
// void IFFT (vector<float>& xreal, vector<float>& ximag, int n);
const float PI = 3.1416;
// global variable
const int n_fft = pow(2, 16);
const int hop_size = n_fft/2;
float* wreal;
float* wimag;

float* wreal_i;
float* wimag_i;
pthread_barrier_t barr;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

AudioFile<float> audioFile;
AudioFile<float> audioFile_output;

vector<float> real_part(n_fft);
vector<float> img_part(n_fft);
vector<float> result;




int main(int argc, char* argv[]){
    // =============================================================================
    // load audio
    // =============================================================================
    bool successful = audioFile.load ("Tim_Henson_VS_Ichika_Nito.wav");
    int sampleRate = audioFile.getSampleRate();
    int bitDepth = audioFile.getBitDepth();
    int numSamples = audioFile.getNumSamplesPerChannel();
    double lengthInSeconds = audioFile.getLengthInSeconds();
    int numChannels = audioFile.getNumChannels();
    bool isMono = audioFile.isMono();
    bool isStereo = audioFile.isStereo();
    audioFile.printSummary();
    
    // =============================================================================
    // initialize output audio
    // =============================================================================
    audioFile_output.setNumChannels (1);
    audioFile_output.setSampleRate (sampleRate);
    audioFile_output.setBitDepth(bitDepth);

    
    // =============================================================================
    // parameters setting
    // =============================================================================
    int max_frame_size = (numSamples/ hop_size) - 1;
    // int max_frame_size = 1;
    int offset = 0;
    // create hann windows
    vector<float> hann_window(n_fft, 1);
    create_hann_window(hann_window, n_fft);
    result.resize(numSamples, 0);

    // =============================================================================
    // pthread parameters setting
    // =============================================================================
    int num_threads = atoi(argv[1]);
    pthread_t threads[num_threads];
    int rc;
    int ID[num_threads];
    pthread_barrier_init(&barr, NULL, num_threads);
    // =============================================================================
    // time
    // =============================================================================
    clock_t start_all, stop_all;
    // clock_t start_fft, stop_fft;
    // clock_t start_window, stop_window;
    // clock_t start_filter, stop_filter;
    // clock_t start_ifft, stop_ifft;

    start_all = clock(); //開始時間
    
    

    calculate_fft_twiddle_factor(wreal, wimag, n_fft);
    calculate_ifft_twiddle_factor(wreal_i, wimag_i, n_fft);

    // cout<<"twiddle done"<<endl;

    fft_data *data = (fft_data*)malloc(num_threads * sizeof(fft_data));
    for(int frame_count=0; frame_count<max_frame_size; frame_count++){
        // cout << "frame_count" << frame_count << endl;
        offset = frame_count*hop_size;
        real_part = {audioFile.samples[0].begin() + offset , audioFile.samples[0].begin() + offset + n_fft};

        vector_mul(real_part, hann_window);
        
        
        // bitrp (real_part, img_part, n_fft);

        // for(int stage=2; stage<=n_fft; stage*=2){
        for (int thread_count = 0; thread_count < num_threads; thread_count++) {
            data[thread_count].id = thread_count;
            data[thread_count].nthreads = num_threads;
            // data[thread_count].stage = stage;

            pthread_create(&threads[thread_count], NULL, FFT, (void*)&data[thread_count]);
            // pthread_join(threads[thread_count], NULL);
        }
        for(int cpu_index=0; cpu_index<num_threads; cpu_index++){
            pthread_join(threads[cpu_index], NULL);
        }
        // }
        // FFT(real_part, img_part, n_fft);
        // low_pass_filter(real_part, img_part, sampleRate, n_fft, 500);
        // bitrp (real_part, img_part, n_fft);

        // for(int stage=2; stage<=n_fft; stage*=2){
        // for (int thread_count = 0; thread_count < num_threads; thread_count++) {
        //     data[thread_count].id = thread_count;
        //     data[thread_count].nthreads = num_threads;
        //     // data[thread_count].stage = stage;

        //     pthread_create(&threads[thread_count], NULL, IFFT, (void*)&data[thread_count]);
        //     // pthread_join(threads[thread_count], NULL);
        // }
        // for(int cpu_index=0; cpu_index<num_threads; cpu_index++){
        //     pthread_join(threads[cpu_index], NULL);
        // }
        // }
        // for (int j=0; j < n_fft; j ++){
        //     real_part [j] /= n_fft;
        //     img_part [j] /= n_fft;
        // }
        // IFFT(real_part, img_part, n_fft);
        overlap_add(result, real_part, frame_count, hop_size);
    }
    // result = {audioFile.samples[0].begin() , audioFile.samples[0].begin() + numSamples};
    cout<<"size: "<<result.size()<<endl;
    audioFile_output.setNumSamplesPerChannel (result.size());
    audioFile_output.samples[0] = result;
    audioFile_output.save ("audioFile.wav", AudioFileFormat::Wave);
    

    // clock_t start, stop;

    

    // cout << typeid(audioFile.samples[0]).name()<<endl;
    stop_all = clock(); //結束時間
    cout << 1000 * double(stop_all - start_all) / CLOCKS_PER_SEC << " ms" <<endl;
}

void calculate_fft_twiddle_factor(float *real, float *img, int n){
    real = new float[n/2];
    img = new float[n/2];
    wreal = real;
    wimag = img;
    
    float w = - 2 * PI / n;
    float treal = cos (w);
    float timag = sin (w);
    real [0] = 1.0;
    img [0] = 0.0;
    for (int j = 1; j < n / 2; j ++){
        real [j] = real [j - 1] * treal - img [j - 1] * timag;
        img [j] = real [j - 1] * timag + img [j - 1] * treal;

        // cout << wreal[j]<<endl;
    }
}

void calculate_ifft_twiddle_factor(float *real, float *img, int n){
    real = new float[n/2];
    img = new float[n/2];
    wreal_i = real;
    wimag_i = img;
    
    float w = 2 * PI / n;
    float treal = cos (w);
    float timag = sin (w);
    real [0] = 1.0;
    img [0] = 0.0;
    for (int j = 1; j < n / 2; j ++)
        {
        real [j] = real [j - 1] * treal - img [j - 1] * timag;
        img [j] = real [j - 1] * timag + img [j - 1] * treal;
        }
}

inline void swap (float &a, float &b)
{
    float t;
    t = a;
    a = b;
    b = t;
}

void bitrp (vector<float>& xreal, vector<float>& ximag, int n)
{
    // 位反转置换 Bit-reversal Permutation
    int i, j, a, b, p;
 
    for (i = 1, p = 0; i < n; i *= 2){
        p ++;
    }
    for (i = 0; i < n; i ++){
        a = i;
        b = 0;
        for (j = 0; j < p; j ++){
            b = (b << 1) + (a & 1);     // b = b * 2 + a % 2;
            a >>= 1;                    // a = a / 2;  
        }
        if ( b > i){
            swap (xreal [i], xreal [b]);
            swap (ximag [i], ximag [b]);
        }
    }
}

void* FFT(void* input)
{
    fft_data *data=(fft_data*)input;
    float treal, timag, ureal, uimag, arg;
    int k, j, t, index1, index2;

    int id = data->id;
    int nthreads = data->nthreads;
    // int m = data->stage;
    // bit reverse
    int i, a, b, p;
 
    for (i = 1, p = 0; i < n_fft; i *= 2){
        p ++;
    }
    for (i = id; i < n_fft; i += nthreads){
        a = i;
        b = 0;
        for (j = 0; j < p; j ++){
            b = (b << 1) + (a & 1);     // b = b * 2 + a % 2;
            a >>= 1;                    // a = a / 2;  
        }
        if ( b > i){
            
            swap (real_part [i], real_part [b]);
            swap (img_part [i], img_part [b]);
        }
    }
    pthread_barrier_wait(&barr);
    // buterfly
    
    for (int m = 2; m <= n_fft; m *= 2){
        if(nthreads >= m){
            int start_pos = id * n_fft/nthreads;
            int end_pos = (id + 1) * n_fft/nthreads;
            for (k = start_pos; k < end_pos; k += m){
                for (j = 0; j < m / 2; j ++){
                    index1 = k + j;
                    index2 = index1 + m / 2;
                    t = n_fft * j / m; 
                    treal = wreal [t] * real_part [index2] - wimag [t] * img_part [index2];
                    timag = wreal [t] * img_part [index2] + wimag [t] * real_part [index2];
                    ureal = real_part [index1];
                    uimag = img_part [index1];
                    // pthread_mutex_lock(&mutex);
                    real_part [index1] = ureal + treal;
                    img_part [index1] = uimag + timag;
                    real_part [index2] = ureal - treal;
                    img_part [index2] = uimag - timag;
                    // pthread_mutex_unlock(&mutex);
                }
            }
        }else{
            int step = (m/(nthreads * 2));
            int round = n_fft / m;
            int start_pos = 0;
            int step_count = 0;
            for(int count = 0; count < round; count ++){
                start_pos = count*(n_fft/round) + id;
                step_count = 0;
                while(step_count < step){
                    index1 = start_pos + (step_count * nthreads);
                    index2 = index1 + m / 2;
                    t = n_fft * (step_count * nthreads + id) / m;
                    treal = wreal [t] * real_part [index2] - wimag [t] * img_part [index2];
                    timag = wreal [t] * img_part [index2] + wimag [t] * real_part [index2];
                    ureal = real_part [index1];
                    uimag = img_part [index1];
                    // pthread_mutex_lock(&mutex);
                    real_part [index1] = ureal + treal;
                    img_part [index1] = uimag + timag;
                    real_part [index2] = ureal - treal;
                    img_part [index2] = uimag - timag;
                    // pthread_mutex_unlock(&mutex);
                    step_count ++;
                }

            }
        }
        pthread_barrier_wait(&barr);
    }

    float freq_resolution = 44100 / ((float)n_fft);
    int bin = (int)(500 / freq_resolution);
    int v_size = real_part.size();
    // int start = (bin/id) * id;
    // if (start < bin){
    //     start += nthreads;
    // }
    // cout << start <<endl;
    // for(int i=start; i<(v_size); i+=nthreads){
    //     real_part[i] = 0;
    //     img_part[i] = 0;
    // }

    // ifft
    for (i = 1, p = 0; i < n_fft; i *= 2){
        p ++;
    }
    for (i = id; i < n_fft; i += nthreads){
        a = i;
        b = 0;
        for (j = 0; j < p; j ++){
            b = (b << 1) + (a & 1);     // b = b * 2 + a % 2;
            a >>= 1;                    // a = a / 2;  
        }
        if ( b > i){
            swap (real_part [i], real_part [b]);
            swap (img_part [i], img_part [b]);
        }
    }
    pthread_barrier_wait(&barr);
    // butterfly
    for (int m = 2; m <= n_fft; m *= 2){
        if(nthreads >= m){
            int start_pos = id * n_fft/nthreads;
            int end_pos = (id + 1) * n_fft/nthreads;
            for (k = start_pos; k < end_pos; k += m){
                for (j = 0; j < m / 2; j ++){
                    index1 = k + j;
                    index2 = index1 + m / 2;
                    t = n_fft * j / m; 
                    treal = wreal_i [t] * real_part [index2] - wimag_i [t] * img_part [index2];
                    timag = wreal_i [t] * img_part [index2] + wimag_i [t] * real_part [index2];
                    ureal = real_part [index1];
                    uimag = img_part [index1];
                    real_part [index1] = ureal + treal;
                    img_part [index1] = uimag + timag;
                    real_part [index2] = ureal - treal;
                    img_part [index2] = uimag - timag;

                }
            }
        }else{
            int step = (m/(nthreads * 2));
            int round = n_fft / m;
            int start_pos = 0;
            int step_count = 0;
            for(int count = 0; count < round; count ++){
                start_pos = count*(n_fft/round) + id;
                step_count = 0;
                while(step_count < step){
                    index1 = start_pos + (step_count * nthreads);
                    index2 = index1 + m / 2;
                    t = n_fft * (step_count * nthreads + id) / m;
                    // t = n_fft * (step_count) / m;
                    treal = wreal_i [t] * real_part [index2] - wimag_i [t] * img_part [index2];
                    timag = wreal_i [t] * img_part [index2] + wimag_i [t] * real_part [index2];
                    ureal = real_part [index1];
                    uimag = img_part [index1];
                    real_part [index1] = ureal + treal;
                    img_part [index1] = uimag + timag;
                    real_part [index2] = ureal - treal;
                    img_part [index2] = uimag - timag;
                    step_count ++;
                }

            }
            // cout<<"id: "<<id<<endl;
            // cout<<"m: "<<m<<endl;
        }
        pthread_barrier_wait(&barr);
    }

    for (int j=id; j < n_fft; j += nthreads){
        real_part [j] /= n_fft;
        img_part [j] /= n_fft;
    }
    
    
    pthread_exit(NULL);
    return NULL;
}




void* IFFT (void* input){
    fft_data *data=(fft_data*)input;
    float treal, timag, ureal, uimag, arg;
    int k, j, t, index1, index2;

    int id = data->id;
    int m = data->stage;
    int nthreads = data->nthreads;

    int i, a, b, p;
    
    //bit reverse
    for (i = 1, p = 0; i < n_fft; i *= 2){
        p ++;
    }
    for (i = id; i < n_fft; i += nthreads){
        a = i;
        b = 0;
        for (j = 0; j < p; j ++){
            b = (b << 1) + (a & 1);     // b = b * 2 + a % 2;
            a >>= 1;                    // a = a / 2;  
        }
        if ( b > i){
            swap (real_part [i], real_part [b]);
            swap (img_part [i], img_part [b]);
        }
    }
    pthread_barrier_wait(&barr);
    // butterfly
    for (int m = 2; m <= n_fft; m *= 2){
        if(nthreads >= m){
            int start_pos = id * n_fft/nthreads;
            int end_pos = (id + 1) * n_fft/nthreads;
            for (k = start_pos; k < end_pos; k += m){
                for (j = 0; j < m / 2; j ++){
                    index1 = k + j;
                    index2 = index1 + m / 2;
                    t = n_fft * j / m; 
                    treal = wreal_i [t] * real_part [index2] - wimag_i [t] * img_part [index2];
                    timag = wreal_i [t] * img_part [index2] + wimag_i [t] * real_part [index2];
                    ureal = real_part [index1];
                    uimag = img_part [index1];
                    real_part [index1] = ureal + treal;
                    img_part [index1] = uimag + timag;
                    real_part [index2] = ureal - treal;
                    img_part [index2] = uimag - timag;

                }
            }
        }else{
            int step = (m/(nthreads * 2));
            int round = n_fft / m;
            int start_pos = 0;
            int step_count = 0;
            for(int count = 0; count < round; count ++){
                start_pos = count*(n_fft/round) + id;
                step_count = 0;
                while(step_count < step){
                    index1 = start_pos + (step_count * nthreads);
                    index2 = index1 + m / 2;
                    t = n_fft * (step_count * nthreads + id) / m;
                    // t = n_fft * (step_count) / m;
                    treal = wreal_i [t] * real_part [index2] - wimag_i [t] * img_part [index2];
                    timag = wreal_i [t] * img_part [index2] + wimag_i [t] * real_part [index2];
                    ureal = real_part [index1];
                    uimag = img_part [index1];
                    real_part [index1] = ureal + treal;
                    img_part [index1] = uimag + timag;
                    real_part [index2] = ureal - treal;
                    img_part [index2] = uimag - timag;
                    step_count ++;
                }

            }
            // cout<<"id: "<<id<<endl;
            // cout<<"m: "<<m<<endl;
        }
        pthread_barrier_wait(&barr);
    }

    for (int j=id; j < n_fft; j += nthreads){
        real_part [j] /= n_fft;
        img_part [j] /= n_fft;
    }

    pthread_exit(NULL);
    return NULL;
}







void create_hann_window(vector<float>& v, int n){
    for (int i = 0; i < n; i++) {
        double multiplier = 0.5 * (1 - cos(2*PI*i/(n-1)));
        v[i] = multiplier;
    }
}

void vector_mul(vector<float>& v1, vector<float>& v2){
    for(int i=0; i<v1.size(); i++){
        v1[i] = v1[i] * v2[i];
    }
}

void low_pass_filter(vector<float>& real, vector<float>& img, int sr, int nfft, int cut_off_freq){
    float freq_resolution = sr / ((float)nfft);
    int bin = (int)(cut_off_freq / freq_resolution);
    int v_size = real.size();
    for(int i=bin; i<(v_size); i++){
        real[i] = 0;
        img[i] = 0;
    }
}

void high_pass_filter(vector<float>& real, vector<float>& img, int sr, int nfft, int cut_off_freq){
    float freq_resolution = sr / ((float)nfft);
    int bin = (int)(cut_off_freq / freq_resolution);
    for(int i=0; i<bin; i++){
        real[i] = 0;
        img[i] = 0;
    }
}

void overlap_add(vector<float>& result, vector<float>& v, int frame_count, int hop_size){
    int offset_index = frame_count * hop_size;
    for(int i=0; i<v.size(); i++){
        result[offset_index + i] = result[offset_index + i] + v[i];
    }
}

// void FFT(vector<float>& xreal, vector<float>& ximag, int n)
// {
//     float treal, timag, ureal, uimag, arg;
//     int m, k, j, t, index1, index2;
 
//     for (m = 2; m <= n; m *= 2){
//         for (k = 0; k < n; k += m){
//             for (j = 0; j < m / 2; j ++){
//                 index1 = k + j;
//                 index2 = index1 + m / 2;
//                 t = n * j / m;
//                 treal = wreal [t] * xreal [index2] - wimag [t] * ximag [index2];
//                 timag = wreal [t] * ximag [index2] + wimag [t] * xreal [index2];
//                 ureal = xreal [index1];
//                 uimag = ximag [index1];
//                 xreal [index1] = ureal + treal;
//                 ximag [index1] = uimag + timag;
//                 xreal [index2] = ureal - treal;
//                 ximag [index2] = uimag - timag;
//             }
//         }
//     }
//     // pthread_exit(NULL);
// }
// void  IFFT (vector<float>& xreal, vector<float>& ximag, int n)
// {
//     float treal, timag, ureal, uimag, arg;
//     int m, k, j, t, index1, index2;
 
//     for (m = 2; m <= n; m *= 2)
//         {
//         for (k = 0; k < n; k += m)
//             {
//             for (j = 0; j < m / 2; j ++)
//                 {
//                 index1 = k + j;
//                 index2 = index1 + m / 2;
//                 t = n * j / m;
//                 treal = wreal_i [t] * xreal [index2] - wimag_i [t] * ximag [index2];
//                 timag = wreal_i [t] * ximag [index2] + wimag_i [t] * xreal [index2];
//                 ureal = xreal [index1];
//                 uimag = ximag [index1];
//                 xreal [index1] = ureal + treal;
//                 ximag [index1] = uimag + timag;
//                 xreal [index2] = ureal - treal;
//                 ximag [index2] = uimag - timag;
//                 }
//             }
//         }
 
//     for (j=0; j < n; j ++)
//         {
//         xreal [j] /= n;
//         ximag [j] /= n;
//         }
// }

