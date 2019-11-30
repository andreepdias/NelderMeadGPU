all:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math main.cu -o gpu

clean:
	rm -f gpu