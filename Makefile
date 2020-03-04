all:
	g++ src/cpu/main.cpp -std=c++11 -O3 -o sequential
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math src/gpu/main.cu -rdc=true -lcudadevrt -o parallel

sequential:
	g++ src/cpu/main.cpp -std=c++11 -O3 -o sequential

parallel:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math src/gpu/main.cu -rdc=true -lcudadevrt -o parallel

test:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math test.cu -rdc=true -lcudadevrt -o test

clean:
	rm -f sequential
	rm -f parallel
	rm -f test
