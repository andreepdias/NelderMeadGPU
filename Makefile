all:
	g++ src/cpu/main.cpp -std=c++11 -O3 -o sequential
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math src/gpu/main.cu -rdc=true -lcudadevrt -o parallel
	scp parallel server:NelderMeadGPU/

cpu:
	g++ src/cpu/main.cpp -std=c++11 -O3 -o cpu

gpu:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math src/gpu/main.cu -rdc=true -lcudadevrt -o gpu

copy_gpu:
	scp parallel server:NelderMeadGPU/

test:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math test.cu -rdc=true -lcudadevrt -o test

clean:
	rm -f cpu
	rm -f gpu
	rm -f test
