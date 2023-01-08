#include "AudioFile.h"
#include "FFT.h"
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <ctime>
#include <windows.h>
#include <typeinfo>
#include <vector>
#include <algorithm>
using namespace std;

void create_hann_window(vector<float>& v, int n);
void vector_mul(vector<float>& v1, vector<float>& v2);
void low_pass_filter(vector<float>& real, vector<float>& img, int sr, int nfft, int cut_off_freq);
void high_pass_filter(vector<float>& real, vector<float>& img, int sr, int nfft, int cut_off_freq);
void overlap_add(vector<float>& result, vector<float>& v, int frame_count, int hop_size);
// or, just use this quick shortcut to print a summary to the console
int main(){
    AudioFile<float> audioFile;
    bool successful = audioFile.load ("Tim_Henson_VS_Ichika_Nito.wav");

    int sampleRate = audioFile.getSampleRate();
    int bitDepth = audioFile.getBitDepth();

    int numSamples = audioFile.getNumSamplesPerChannel();
    double lengthInSeconds = audioFile.getLengthInSeconds();

    int numChannels = audioFile.getNumChannels();
    bool isMono = audioFile.isMono();
    bool isStereo = audioFile.isStereo();

    AudioFile<float> audioFile_output;
    audioFile.setNumChannels (1);
    audioFile.setSampleRate (sampleRate);
    audioFile.setBitDepth(bitDepth);

    // printf("%d", lengthInSeconds);
    audioFile.printSummary();

    int n_fft = pow(2, 16);
    int hop_size = n_fft/2;
    // int max_frame_size = (numSamples/ hop_size) - 1;
    int max_frame_size = 1;
    int offset = 0;
    clock_t start_all, stop_all;
    clock_t start_fft, stop_fft;
    clock_t start_window, stop_window;
    clock_t start_filter, stop_filter;
    clock_t start_ifft, stop_ifft;

    start_all = clock(); //開始時間
    
    vector<float> real_part(n_fft);
    vector<float> img_part(n_fft);

    // create hann windows
    vector<float> hann_window(n_fft, 1);
    create_hann_window(hann_window, n_fft);
     
    vector<float> result(numSamples, 0);

    // vector<float> real_part

    cout << "frame size:" << max_frame_size << endl;
    for(int frame_count=0; frame_count<max_frame_size; frame_count++){
        offset = frame_count*hop_size;
        // cout << offset << endl;
        real_part = {audioFile.samples[0].begin() + offset , audioFile.samples[0].begin() + offset + n_fft};

        vector_mul(real_part, hann_window);
        
        FFT(real_part, img_part, n_fft);
        low_pass_filter(real_part, img_part, sampleRate, n_fft, 500);
        IFFT(real_part, img_part, n_fft);
        overlap_add(result, real_part, frame_count, hop_size);
        // for(int i=0; i<real_part.size(); i++){
        //     result[offset + i] = result[offset + i] + real_part[i];
        // }
    
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