#include <stdint.h>
#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

// #include"software_solutions_FPU.h"
// standard function to perform the neural feedforward path
// no activation on output :) gotta use it by yourself (imma regret this decision)
// weights matrix and biasses are in row major order 
// void forward_pass(const float* input,uint32_t input_size,const float* weights,
//                     const float* biases,float* output,uint32_t output_size)
// {
//     for (uint32_t i = 0; i < output_size; i++)
//     {
//         float accumulation=biases[i];
//         for (uint32_t j = 0; j < input_size; j++)
//         {
//             accumulation+=(input[j]*weights[(i*input_size)+j]);
//         }
//         output[i]=accumulation;
//     }
// }

void forward_pass(const float* input, uint32_t input_size,
                  const float* weights, const float* biases,
                  float* output, uint32_t output_size)
{
    for (uint32_t i = 0; i < output_size; i++) {
        float sum = biases[i];
        for (uint32_t j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + i];
        }
        output[i] = sum;
    }
}


// rectified linear unit :)
void ReLU(float* input,uint32_t size){
    for (uint32_t i = 0; i < size; i++)
    {
        if((int32_t)(input[i])<0){input[i]=0.0f;}
    }
}

void tanh_activation(float* input,uint32_t size){
    for(uint32_t i=0;i<size;i++){

        if ((int32_t)(input[i]) < -3){input[i]=-1.0f;continue;}
        if ((int32_t)(input[i]) >  3){input[i]=1.0f;continue;}
        
        float x2 = input[i] * input[i];
        float x3=input[i] * (27.0f + x2) / (27.0f + 9.0f * x2);
        input[i]=x3;
    }
}
// this version uses non zero uniform quantization 
void forward_pass_quantized(const float* input, uint32_t input_size,
                  const int8_t* weights,float scale_weights, const int8_t* biases,
                  float scale_biases,float* output, uint32_t output_size)
{
    for (uint32_t i = 0; i < output_size; i++) {
        float sum = biases[i]*scale_biases;
        for (uint32_t j = 0; j < input_size; j++) {
            sum += input[j] * weights[j * output_size + i]*scale_weights;
        }
        output[i] = sum;
    }
}

// this version uses 0 uniform quantization 
// Per layer symmetric uniform quantization
void forward_pass_quantized_symetric(const float* input, uint32_t input_size,
                  const int8_t* weights,float scale_weights, const int8_t* biases,
                  float scale_biases,float* output, uint32_t output_size)
{
    for (uint32_t i = 0; i < output_size; i++) {
        float sum = biases[i]/scale_biases;
        for (uint32_t j = 0; j < input_size; j++) {
            float w= weights[j * output_size + i]/scale_weights;
            sum += input[j] * w;
        }
        output[i] = sum;
    }
}

void tanh_activation_quantized(float* input,uint32_t size){
    for(uint32_t i=0;i<size;i++){
        
        // if (float_lt(input[i],-3.0f)){input[i]=-1.0f;continue;}
        // if (float_gt(input[i],3.0f)){input[i]=1.0f;continue;}
        if (input[i]<-3.0f){input[i]=-1.0f;continue;}
        if (input[i]>3.0f){input[i]=1.0f;continue;}
        
        float x2 = input[i] * input[i];
        float x3=input[i] * (27.0f + x2) / (27.0f + 9.0f * x2);
        input[i]=x3;
    }
}


#endif
