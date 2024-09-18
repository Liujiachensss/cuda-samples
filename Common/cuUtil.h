#include "helper_cuda.h"
#include "cuda_runtime.h"

#include <tuple>
#include <iostream>
#include <type_traits>

class gpuTimer
{
public:
    gpuTimer(std::string &&name = "", cudaStream_t streamId = 0) : _name(name), _streamId(streamId)
    {
        checkCudaErrors(cudaEventCreate(&_start));
        checkCudaErrors(cudaEventCreate(&_stop));

        // checkCudaErrors(cudaEventRecord(_start, _streamId));
    }

    ~gpuTimer()
    {
        checkCudaErrors(cudaEventDestroy(_start));
        checkCudaErrors(cudaEventDestroy(_stop));
    }

    void start()
    {
        checkCudaErrors(cudaEventRecord(_start, _streamId));
    }

    void stop()
    {
        checkCudaErrors(cudaEventRecord(_stop, _streamId));
        checkCudaErrors(cudaEventSynchronize(_stop));
        checkCudaErrors(cudaEventElapsedTime(&_time, _start, _stop));
    }

    float time() const { return _time; }
    std::string name() const { return _name; }

private:
    std::string _name;
    float _time;
    cudaStream_t _streamId;
    cudaEvent_t _start;
    cudaEvent_t _stop;
};

template <class Base, class... Timers>
struct gpuTimerShow
{
    gpuTimerShow(Base &base, Timers &...timers)
    {
        float baseline = base.time();
        auto tup = std::forward_as_tuple(timers...);

        std::cout << std::to_string(baseline) + "ms for baseline\n";

        std::apply([baseline](const Timers &...timers)
                   { std::cout << ((std::to_string(timers.time()) + "ms for " + timers.name() + ", ups = " + std::to_string(baseline / timers.time())) + ...) << "\n"; }, tup);
    }
};
