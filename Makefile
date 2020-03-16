all:
	make clean
	g++ src/cpu/main.cpp -std=c++11 -O3 -o cpu
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math src/gpu/main.cu -rdc=true -lcudadevrt -o gpu

server:
	make clean
	g++ src/cpu/main.cpp -std=c++11 -O3 -o cpu
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math src/gpu/main.cu -rdc=true -lcudadevrt -o gpu
	scp gpu server:NelderMeadGPU/
	scp resources/inputs/input.txt server:NelderMeadGPU/resources/inputs/

cpu:
	g++ src/cpu/main.cpp -std=c++11 -O3 -o cpu

fcpu:
	g++ src/fast_cpu/main.cpp -std=c++11 -O3 -o fcpu

fgpu:
	nvcc -arch=sm_60 -std=c++11 -O3 -use_fast_math src/fast_gpu/main.cu -o fgpu

gpu:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math src/gpu/main.cu -rdc=true -lcudadevrt -o gpu

copy_gpu:
	scp gpu server:NelderMeadGPU/

copy_input:
	scp resources/inputs/input.txt server:NelderMeadGPU/resources/inputs/

test:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math test.cu -o test

clean:
	rm -f cpu
	rm -f gpu
	rm -f test
