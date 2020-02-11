gpu:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math 3DAB.cu -o gpu

cpu:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math 3DAB.cu -D USE_HOST -o cpu

clean:
	rm -f gpu
	rm -f cpu