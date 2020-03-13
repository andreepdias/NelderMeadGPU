#include <thrust/count.h>
#include <thrust/device_vector.h>
struct is_odd
{
  __host__ __device__
  bool operator()(int &x)
  {
    return x & 1;
  }
};

int main(){
    thrust::device_vector<int> vec(5);
    vec[0] = 0;
    vec[1] = 1;
    vec[2] = 2;
    vec[3] = 3;
    vec[4] = 4;
    // count the odd elements in vec
    int result = thrust::count_if(vec.begin(), vec.end(), is_odd());

    printf("%d\n", result);
}