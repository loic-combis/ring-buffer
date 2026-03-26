# Ring Buffer
A high-performance, zero-copy ring buffer implementation for Python. This library provides a seamless way to handle streaming data using standard NumPy arrays or Shared Memory for ultra-fast inter-process communication (IPC).

## Key Features
- Zero-Copy Design: Read operations return a view of the buffer rather than a copy, minimizing overhead.

- Multiprocessing Ready: Built-in support for SharedMemory to move data between processes at lightning speed.

- NumPy Integration: Native support for NumPy arrays, allowing for direct mathematical operations on buffer data.

- Lock-Free Performance: Optimized for Single-Producer Single-Reader (SPSR) scenarios.

## 🚀 Get Started
### Prerequisites
Python: 3.11 or higher

*Optional*: numpy (required for NumPy-specific buffer classes)

### Installation
#### Using pip:
```bash
# Basic installation
pip install shm-ring-buffer

# With NumPy support
pip install shm-ring-buffer[numpy]
```

#### Using poetry:
```bash
# Basic installation
poetry add shm-ring-buffer

# With NumPy support
poetry add shm-ring-buffer -E numpy
```

## 🛠 Usage & Architecture

> [!WARNING]
> **Single-Producer Single-Reader (SPSR):** This implementation is lock-free to maximize throughput. It is not thread-safe for multiple concurrent writers or multiple concurrent readers. Ensure your architecture follows the SPSR pattern.

### Shared Memory IPC
Shared memory is the fastest method for IPC in Python. This library implements two primary shared memory buffers:

`BytesShmRingBuffer`: Optimized for raw byte streams.

`NumpyShmRingBuffer`: Optimized for structured numerical data.

#### Implementation Note:
One process must initialize the buffer with create=True, while the consumer/other side should set create=False. Always call `.close()` to properly release system resources.

### The Zero-Copy Workflow
To maintain high performance, the buffer returns a view of the data.

Read: Access the current item.

Process: Use the data (or copy it if you need it to persist).

Release: You must call `.release()` after processing to signal that the slot is available for the producer to overwrite.

## 🗺️ Roadmap
- [x] High-performance multiprocessing.shared_memory implementation.

- [x] CuPy implementation for hardware acceleration via NVIDIA GPUs (CUDA).

## 👥 Authors
- Loïc Combis - [GitHub](https://github.com/loic-combis) | [LinkedIn](https://www.linkedin.com/in/lo%C3%AFc-combis-a211a813a/)