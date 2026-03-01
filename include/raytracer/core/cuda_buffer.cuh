/**
 * CUDA Buffer - RAII Wrapper for GPU Memory
 */

#ifndef RAYTRACER_CORE_CUDA_BUFFER_CUH
#define RAYTRACER_CORE_CUDA_BUFFER_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

namespace rt {

/**
 * RAII wrapper for CUDA device memory
 * Automatically frees memory when object goes out of scope
 * Move-only (no copy semantics)
 */
template<typename T>
class CudaBuffer {
    T* ptr_ = nullptr;
    size_t count_ = 0;

public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t count) : count_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CudaBuffer: cudaMalloc failed: ") + cudaGetErrorString(err));
            }
        }
    }

    ~CudaBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Move constructor
    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    // Move assignment
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // Disable copy
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Accessors
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return count_; }
    size_t bytes() const noexcept { return count_ * sizeof(T); }
    bool empty() const noexcept { return count_ == 0; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    // Copy data to device
    void copyFrom(const T* host_data, size_t count) {
        if (count > count_) {
            throw std::runtime_error("CudaBuffer: copy count exceeds buffer size");
        }
        cudaError_t err = cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CudaBuffer: cudaMemcpy H2D failed: ") + cudaGetErrorString(err));
        }
    }

    void copyFrom(const T* host_data) {
        copyFrom(host_data, count_);
    }

    // Copy data from device
    void copyTo(T* host_data, size_t count) const {
        if (count > count_) {
            throw std::runtime_error("CudaBuffer: copy count exceeds buffer size");
        }
        cudaError_t err = cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("CudaBuffer: cudaMemcpy D2H failed: ") + cudaGetErrorString(err));
        }
    }

    void copyTo(T* host_data) const {
        copyTo(host_data, count_);
    }

    // Release ownership (for manual management when needed)
    T* release() noexcept {
        T* tmp = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return tmp;
    }

    // Reset buffer (free and optionally reallocate)
    void reset(size_t new_count = 0) {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = new_count;
        if (new_count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, new_count * sizeof(T));
            if (err != cudaSuccess) {
                count_ = 0;
                throw std::runtime_error(std::string("CudaBuffer: cudaMalloc failed: ") + cudaGetErrorString(err));
            }
        }
    }
};

/**
 * Convenience function to create CudaBuffer
 */
template<typename T>
CudaBuffer<T> make_cuda_buffer(size_t count) {
    return CudaBuffer<T>(count);
}

/**
 * Create CudaBuffer and copy data from host
 */
template<typename T>
CudaBuffer<T> make_cuda_buffer_from(const T* host_data, size_t count) {
    CudaBuffer<T> buffer(count);
    buffer.copyFrom(host_data, count);
    return buffer;
}

} // namespace rt

#endif // RAYTRACER_CORE_CUDA_BUFFER_CUH
