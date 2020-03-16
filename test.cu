#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

struct is_odd
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return x & 1;
  }
};

int main(){
  // fill a device_vector with even & odd numbers
  thrust::device_vector<int> vec(5);
  vec[0] = 0;
  vec[1] = 1;
  vec[2] = 2;
  vec[3] = 3;
  vec[4] = 4;
  // count the odd elements in vec
  
  int result = thrust::count_if(thrust::device, vec.begin(), vec.end(), is_odd());
  // result == 2
  std::cout << result << std::endl;

}