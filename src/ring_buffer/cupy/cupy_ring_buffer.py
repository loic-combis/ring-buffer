try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "\n\n"
        "Missing dependency: 'cupy'\n"
        "To use CupyRingBuffer, you must install the cupy extra:\n"
        "    pip install ring-buffer[cupy]\n"
    ) from None


from typing import Optional

from ring_buffer.base_ring_buffer import BaseRingBuffer


class CupyRingBuffer(BaseRingBuffer[cp.ndarray]):
    def __init__(self,
                slots: int,
                dtype: cp.dtype,
                shape: Optional[tuple[int, ...]] = None) -> None:

        super().__init__(slots=slots)

        dtype = cp.dtype(dtype)
        item_size: int

        if shape is None:
            item_size = dtype.itemsize
        else:
            item_size = int(cp.prod(cp.array(shape)).item() * dtype.itemsize)

        # Allocate a single contiguous GPU memory block
        self._mem = cp.cuda.alloc(item_size * self._slots)

        self._elts: list[cp.ndarray] = []

        # Pre-allocate cupy views into the GPU buffer
        for i in range(self._slots):
            memptr = self._mem + (i * item_size)
            view = cp.ndarray(
                shape=shape or (),
                dtype=dtype,
                memptr=memptr,
            )

            self._elts.append(view)


    def _write(self, index: int, data: cp.ndarray) -> None:
        """
        Store 'data' into the buffer at the given 'index'.
        """
        view = self._elts[index]
        view[()] = data


    def _read(self, index: int) -> cp.ndarray:
        """
        Retrieve data from the buffer at the given 'index'.
        """
        return self._elts[index]
