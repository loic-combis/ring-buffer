try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "\n\n"
        "Missing dependency: 'cupy'\n"
        "To use CupyShmRingBuffer, you must install the cupy extra:\n"
        "    pip install ring-buffer[cupy]\n"
    ) from None

from typing import Optional

from ring_buffer.abstract_ring_buffer import RingBuffer


# cudaIpcMemHandle_t is 64 bytes
_IPC_HANDLE_SIZE = 64


def _cuda_alloc(size: int) -> int:
    """Allocate via cudaMalloc (bypasses cupy's pool, required for IPC)."""
    return cp.cuda.runtime.malloc(size)


def _ipc_handle(ptr: int) -> bytes:
    """Get an IPC memory handle for a cudaMalloc'd pointer."""
    return cp.cuda.runtime.ipcGetMemHandle(ptr)


def _ipc_open(handle: bytes) -> int:
    """Open a device pointer from an IPC handle."""
    return cp.cuda.runtime.ipcOpenMemHandle(handle, 1)  # cudaIpcMemLazyEnablePeerAccess


def _wrap_ptr(ptr: int, size: int, owner: object, shape, dtype: cp.dtype) -> cp.ndarray:
    """Wrap a raw device pointer as a cupy ndarray view."""
    mem = cp.cuda.UnownedMemory(ptr, size, owner=owner)
    memptr = cp.cuda.MemoryPointer(mem, 0)
    return cp.ndarray(shape=shape, dtype=dtype, memptr=memptr)


class CupyRingBuffer(RingBuffer[cp.ndarray]):
    """
    GPU ring buffer with cross-process support.

    Everything lives in CUDA memory — both data slots and read/write
    indices are allocated via cudaMalloc, producing stable IPC handles
    that other processes can open.

    Usage:
        # Single process
        buf = CupyRingBuffer(5, cp.float32, (224, 224, 3), create=True)

        # Cross-process: producer allocates, consumer opens via IPC handles
        producer = CupyRingBuffer(5, cp.float32, (224, 224, 3), create=True)
        handles = producer.ipc_handles  # bytes — send to consumer

        consumer = CupyRingBuffer(5, cp.float32, (224, 224, 3),
                                  create=False, ipc_handles=handles)
    """

    def __init__(
        self,
        slots: int,
        dtype: cp.dtype,
        shape: Optional[tuple[int, ...]] = None,
        *,
        create: bool,
        ipc_handles: Optional[bytes] = None,
    ) -> None:

        super().__init__(slots=slots)

        self._create = create
        self._dtype = cp.dtype(dtype)
        self._shape = shape or ()

        if shape is None:
            self._item_size = self._dtype.itemsize
        else:
            self._item_size = int(cp.prod(cp.array(shape)).item() * self._dtype.itemsize)

        # Raw device pointers we own / opened (for cleanup)
        self._idx_ptrs: list[int] = []
        self._slot_ptrs: list[int] = []

        self._elts: list[cp.ndarray] = []

        if create:
            if ipc_handles is not None:
                raise ValueError("ipc_handles must be None when create=True")
            self._init_producer()
        else:
            if ipc_handles is None:
                raise ValueError("ipc_handles is required when create=False")
            self._init_consumer(ipc_handles)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_producer(self) -> None:
        # Allocate index storage (two int64 scalars) via cudaMalloc
        r_ptr = _cuda_alloc(8)
        w_ptr = _cuda_alloc(8)
        cp.cuda.runtime.memset(r_ptr, 0, 8)
        cp.cuda.runtime.memset(w_ptr, 0, 8)
        self._idx_ptrs = [r_ptr, w_ptr]

        self._r_idx_arr = _wrap_ptr(r_ptr, 8, self, (), cp.int64)
        self._w_idx_arr = _wrap_ptr(w_ptr, 8, self, (), cp.int64)

        # Allocate data slots via cudaMalloc
        for _ in range(self._slots):
            ptr = _cuda_alloc(self._item_size)
            self._slot_ptrs.append(ptr)
            self._elts.append(
                _wrap_ptr(ptr, self._item_size, self, self._shape, self._dtype)
            )

        # Build the serialised handle blob
        parts: list[bytes] = []
        for ptr in self._idx_ptrs:
            parts.append(_ipc_handle(ptr))
        for ptr in self._slot_ptrs:
            parts.append(_ipc_handle(ptr))
        self._ipc_handles = b"".join(parts)

    def _init_consumer(self, ipc_handles: bytes) -> None:
        n_handles = 2 + self._slots  # 2 index handles + N slot handles
        expected = n_handles * _IPC_HANDLE_SIZE
        if len(ipc_handles) != expected:
            raise ValueError(
                f"Expected {expected} bytes of IPC handles "
                f"({n_handles} handles), got {len(ipc_handles)}"
            )

        def _get(i: int) -> bytes:
            off = i * _IPC_HANDLE_SIZE
            return ipc_handles[off : off + _IPC_HANDLE_SIZE]

        # Open index handles
        r_ptr = _ipc_open(_get(0))
        w_ptr = _ipc_open(_get(1))
        self._idx_ptrs = [r_ptr, w_ptr]

        self._r_idx_arr = _wrap_ptr(r_ptr, 8, self, (), cp.int64)
        self._w_idx_arr = _wrap_ptr(w_ptr, 8, self, (), cp.int64)

        # Open data slot handles
        for i in range(self._slots):
            ptr = _ipc_open(_get(2 + i))
            self._slot_ptrs.append(ptr)
            self._elts.append(
                _wrap_ptr(ptr, self._item_size, self, self._shape, self._dtype)
            )

    # ------------------------------------------------------------------
    # IPC handles (pass these to the consumer process)
    # ------------------------------------------------------------------

    @property
    def ipc_handles(self) -> bytes:
        """Serialised IPC handles: [r_idx, w_idx, slot_0, …, slot_N]."""
        if not self._create:
            raise RuntimeError("Only the producer (create=True) owns the IPC handles")
        return self._ipc_handles

    # ------------------------------------------------------------------
    # Index properties (CUDA-memory backed)
    # ------------------------------------------------------------------

    @property
    def _r_idx(self) -> int:
        return int(self._r_idx_arr)

    @_r_idx.setter
    def _r_idx(self, value: int) -> None:
        self._r_idx_arr[()] = value

    @property
    def _w_idx(self) -> int:
        return int(self._w_idx_arr)

    @_w_idx.setter
    def _w_idx(self, value: int) -> None:
        self._w_idx_arr[()] = value

    # ------------------------------------------------------------------
    # Storage methods
    # ------------------------------------------------------------------

    def _write(self, index: int, data: cp.ndarray) -> None:
        self._elts[index][()] = data

    def _read(self, index: int) -> cp.ndarray:
        return self._elts[index]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._elts.clear()

        if self._create:
            for ptr in self._slot_ptrs:
                cp.cuda.runtime.free(ptr)
            for ptr in self._idx_ptrs:
                cp.cuda.runtime.free(ptr)
        else:
            for ptr in self._slot_ptrs:
                cp.cuda.runtime.ipcCloseMemHandle(ptr)
            for ptr in self._idx_ptrs:
                cp.cuda.runtime.ipcCloseMemHandle(ptr)

        self._slot_ptrs.clear()
        self._idx_ptrs.clear()
