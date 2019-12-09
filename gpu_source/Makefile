all:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math main.cu -rdc=true -lcudadevrt -lcublas_device -o gpu

clean:
	rm -f gpu