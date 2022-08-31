
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

void usage(const char* filename) {
  printf("Sort the random key-value data set of the given length by key.\n");
  printf("Usage: %s <n>\n", filename);
}

constexpr int n = 1024 * 1024;


void random_vector(uint64_t* h_vec, size_t N) {
  static thrust::default_random_engine rng;
  static thrust::uniform_int_distribution<uint64_t> dist;

  for (size_t i = 0; i < N; i++) h_vec[i] = dist(rng);
}

using namespace thrust;

#define THRUST_FWD(x) ::std::forward<decltype(x)>(x)

#define THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(func, impl)                  \
  template <>                                                                  \
  struct func<void>                                                            \
  {                                                                            \
    template <typename T1, typename T2>                                        \
    __host__ __device__                                                        \
    constexpr auto operator()(T1&& t1, T2&& t2) const                          \
    {                                                                          \
      return impl;                                                             \
    }                                                                          \
  }

#define THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(func, op)                 \
  THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION(                                   \
    func, THRUST_FWD(t1) op THRUST_FWD(t2))

template<typename T = void>
struct masked_less
{
  /*! \typedef first_argument_type
   *  \brief The type of the function object's first argument.
   */
  typedef T first_argument_type;

  /*! \typedef second_argument_type
   *  \brief The type of the function object's second argument.
   */
  typedef T second_argument_type;

  /*! \typedef result_type
   *  \brief The type of the function object's result;
   */
  typedef bool result_type;

  /*! Function call operator. The return value is <tt>lhs < rhs</tt>.
   */
  __thrust_exec_check_disable__
  __host__ __device__
  constexpr bool operator()(const T &lhs, const T &rhs) const
  {
    return (lhs & 0xFFFFFFFFFFFF) < (rhs & 0xFFFFFFFFFFFF);
  }
}; // end less

THRUST_BINARY_FUNCTOR_VOID_SPECIALIZATION_OP(masked_less, <);

//template<typename T = void>
//struct masked_less
//{
//  /*! \typedef first_argument_type
//   *  \brief The type of the function object's first argument.
//   */
//  typedef T first_argument_type;
//
//  /*! \typedef second_argument_type
//   *  \brief The type of the function object's second argument.
//   */
//  typedef T second_argument_type;
//
//  /*! \typedef result_type
//   *  \brief The type of the function object's result;
//   */
//  typedef bool result_type;
//
//  /*! Function call operator. The return value is <tt>lhs < rhs</tt>.
//   */
//  __thrust_exec_check_disable__
//  __host__ __device__
//  constexpr bool operator()(const T &lhs, const T &rhs) const
//  {
//    return (lhs & 0xFFFFFFFFFFFF) < (rhs & & 0xFFFFFFFFFFFF);
//  }
//}; // end less

int main() {
  constexpr int TEST_TIMES = 1;
  const int printable_n = 16;

  // TODO: Generate random keys and values on host
  // host_vector<int> ...
  // generate( ...
  uint64_t* h_keys;
  uint64_t* h_vals;
  cudaMallocHost((void**)&h_keys, n * sizeof(uint64_t));
  cudaMallocHost((void**)&h_vals, n * sizeof(uint64_t));
  uint64_t* d_keys;
  uint64_t* d_vals;
  cudaMalloc((void**)&d_keys, n * sizeof(uint64_t));
  cudaMalloc((void**)&d_vals, n * sizeof(uint64_t));
  random_vector(h_keys, n);
  random_vector(h_vals, n);

  // Print out the input data if n is small.
  printf("Input data:\n");
  for (int i = 0; i < printable_n; i++)
    printf("(%d, %d)\n", h_keys[i], h_vals[i]);
  printf("\n");

  // TODO: Transfer data to the device.
  // device_vector<int> ...
  cudaMemcpy(d_keys, h_keys, sizeof(uint64_t) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vals, h_vals, sizeof(uint64_t) * n, cudaMemcpyHostToDevice);

  std::chrono::time_point<std::chrono::steady_clock> start_test;
  std::chrono::duration<double> diff_test;
  // TODO: Use sort_by_key or stable_sort_by_key to sort
  // pairs by key.
  // sort_by_key( ...
  start_test = std::chrono::steady_clock::now();
  thrust::device_ptr<uint64_t> d_keys_ptr(d_keys);
  thrust::device_ptr<uint64_t> d_vals_ptr(d_vals);
  thrust::sort_by_key(d_keys_ptr, d_keys_ptr + n, d_vals_ptr,
                      masked_less<uint64_t>());
  //   cudaDeviceSynchronize();
  diff_test = std::chrono::steady_clock::now() - start_test;
  printf("[timing] sort d2h=%.2fms\n", diff_test.count() * 1000 / TEST_TIMES);

  // TODO: Transfer data back to host.
  cudaMemcpy(h_keys, d_keys, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vals, d_vals, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);

  // Print out the output data if n is small.
  printf("Output data:\n");
  for (int i = 0; i < printable_n; i++)
    printf("(%d, %d)\n", h_keys[i], h_vals[i]);
  printf("\n");

  cudaFreeHost(h_keys);
  cudaFreeHost(h_vals);
  cudaFree(d_keys);
  cudaFree(d_vals);

  return 0;
}
