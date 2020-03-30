c:
	g++ src/cpu/main.cpp -std=c++11 -O3 -o c

g:
	nvcc -arch=sm_60 -rdc=true -std=c++11 -O3 -use_fast_math src/gpu/main.cu -o g

t:
	g++ src/test.cpp -std=c++11 -O3 -o t

run:
	sh run.sh
