sequential:
	g++ src/cpu/main.cpp -std=c++11 -O3 -o sequential

parallel:
	nvcc -arch=sm_60 -std=c++11 -O3  -use_fast_math /home/andre/Music/NelderMeadGpu/sequential/src/gpu/main.cu -rdc=true -lcudadevrt -o /home/andre/Music/NelderMeadGpu/sequential/parallel


clean:
	rm -f sequential
	rm -f parallel