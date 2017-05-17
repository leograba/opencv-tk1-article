#!/bin/sh

OCV_SAMPLES_D="/home/ubuntu/opencv/samples"
# 3483x2642 pixels
echo "Dog 1"
ln -fs ${PWD}/dog1.jpg ${OCV_SAMPLES_D}/dog.jpg
echo "Starting GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-gpu -1 500 > ${OCV_SAMPLES_D}/gpu_dog1.log
echo "Starting CPU-GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-cpu-gpu -1 500 > ${OCV_SAMPLES_D}/cpu_gpu_dog1.log
echo "Starting CPU"
${OCV_SAMPLES_D}/cpp/cpp-example-SobelDerivatives-cpu -1 500 > ${OCV_SAMPLES_D}/cpu_dog1.log

# 2122x1415 pixels
echo "Dog 2"
ln -fs ${PWD}/dog2.jpg ${OCV_SAMPLES_D}/dog.jpg
echo "Starting GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-gpu -1 500 > ${OCV_SAMPLES_D}/gpu_dog2.log
echo "Starting CPU-GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-cpu-gpu -1 500 > ${OCV_SAMPLES_D}/cpu_gpu_dog2.log
echo "Starting CPU"
${OCV_SAMPLES_D}/cpp/cpp-example-SobelDerivatives-cpu -1 500 > ${OCV_SAMPLES_D}/cpu_dog2.log

# 845x450 pixels
echo "Dog 3"
ln -fs ${PWD}/dog3.jpg ${OCV_SAMPLES_D}/dog.jpg
echo "Starting GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-gpu -1 500 > ${OCV_SAMPLES_D}/gpu_dog3.log
echo "Starting CPU-GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-cpu-gpu -1 500 > ${OCV_SAMPLES_D}/cpu_gpu_dog3.log
echo "Starting CPU"
${OCV_SAMPLES_D}/cpp/cpp-example-SobelDerivatives-cpu -1 500 > ${OCV_SAMPLES_D}/cpu_dog3.log

# 460x290 pixels
echo "Dog 4"
ln -fs ${PWD}/dog4.jpg ${OCV_SAMPLES_D}/dog.jpg
echo "Starting GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-gpu -1 500 > ${OCV_SAMPLES_D}/gpu_dog4.log
echo "Starting CPU-GPU"
${OCV_SAMPLES_D}/gpu/gpu-example-SobelDerivatives-cpu-gpu -1 500 > ${OCV_SAMPLES_D}/cpu_gpu_dog4.log
echo "Starting CPU"
${OCV_SAMPLES_D}/cpp/cpp-example-SobelDerivatives-cpu -1 500 > ${OCV_SAMPLES_D}/cpu_dog4.log
