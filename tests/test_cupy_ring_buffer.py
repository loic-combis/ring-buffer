import cupy as cp
from cupy import testing as cpt

from ring_buffer.cupy import CupyRingBuffer


def test_initialization():
    """Verify capacity and initial empty state."""
    buf = CupyRingBuffer(slots=5, dtype=cp.float64, create=True)
    try:
        assert buf.capacity == 5
        assert buf.can_read() is False
        assert buf.can_write() is True
        assert buf.read() is None
    finally:
        buf.close()


def test_basic_write_read_scalar():
    """Test write/read with scalar (0-d) arrays."""
    buf = CupyRingBuffer(slots=3, dtype=cp.float64, create=True)
    try:
        buf.write(cp.float64(42.0))
        result = buf.read()
        assert float(result) == 42.0
        buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_basic_write_read_1d():
    """Test write/read with 1-d arrays."""
    buf = CupyRingBuffer(slots=3, dtype=cp.float32, shape=(4,), create=True)
    try:
        data = cp.array([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
        buf.write(data)
        cpt.assert_array_equal(buf.read(), data)
        buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_basic_write_read_2d():
    """Test write/read with 2-d arrays."""
    buf = CupyRingBuffer(slots=3, dtype=cp.int32, shape=(2, 3), create=True)
    try:
        data = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.int32)
        buf.write(data)
        cpt.assert_array_equal(buf.read(), data)
    finally:
        buf.close()


def test_fill_to_capacity():
    """Verify we can fill exactly 'capacity' items, then no more."""
    capacity = 3
    buf = CupyRingBuffer(slots=capacity, dtype=cp.float64, shape=(2,), create=True)
    try:
        for i in range(capacity):
            assert buf.write(cp.array([float(i), float(i + 10)], dtype=cp.float64)) is True
        assert buf.can_write() is False
        assert buf.write(cp.array([99.0, 99.0], dtype=cp.float64)) is False
    finally:
        buf.close()


def test_fill_and_drain():
    """Fill buffer to capacity, then read all items in FIFO order."""
    capacity = 4
    buf = CupyRingBuffer(slots=capacity, dtype=cp.float64, shape=(2,), create=True)
    try:
        items = [cp.array([float(i), float(i * 10)], dtype=cp.float64) for i in range(capacity)]
        for item in items:
            buf.write(item)
        for item in items:
            cpt.assert_array_equal(buf.read(), item)
            buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_circular_wrap_around():
    """Ensure indices wrap around correctly."""
    capacity = 3
    buf = CupyRingBuffer(slots=capacity, dtype=cp.int32, shape=(2,), create=True)
    try:
        for i in range(capacity):
            buf.write(cp.array([i, i + 10], dtype=cp.int32))
        for _ in range(capacity):
            buf.release()

        a = cp.array([100, 200], dtype=cp.int32)
        b = cp.array([300, 400], dtype=cp.int32)
        buf.write(a)
        buf.write(b)
        cpt.assert_array_equal(buf.read(), a)
        buf.release()
        cpt.assert_array_equal(buf.read(), b)
        buf.release()
        assert buf.can_read() is False
    finally:
        buf.close()


def test_interleaved_write_read():
    """Interleave writes and reads to simulate streaming usage."""
    buf = CupyRingBuffer(slots=2, dtype=cp.float64, shape=(3,), create=True)
    try:
        for i in range(10):
            data = cp.array([float(i), float(i * 2), float(i * 3)], dtype=cp.float64)
            assert buf.write(data) is True
            cpt.assert_array_equal(buf.read(), data)
            buf.release()
    finally:
        buf.close()


def test_release_on_empty():
    """Releasing an empty buffer should return False."""
    buf = CupyRingBuffer(slots=2, dtype=cp.float64, create=True)
    try:
        assert buf.release() is False
    finally:
        buf.close()


def test_read_on_empty():
    """Reading an empty buffer should return None."""
    buf = CupyRingBuffer(slots=2, dtype=cp.float64, create=True)
    try:
        assert buf.read() is None
    finally:
        buf.close()


def test_read_without_release():
    """Read (peek) should not advance the read pointer."""
    buf = CupyRingBuffer(slots=2, dtype=cp.float64, shape=(2,), create=True)
    try:
        data = cp.array([1.0, 2.0], dtype=cp.float64)
        buf.write(data)
        cpt.assert_array_equal(buf.read(), data)
        cpt.assert_array_equal(buf.read(), data)
        assert buf.can_read() is True
    finally:
        buf.close()


def test_zero_copy_read():
    """Read returns a view into the buffer, not a copy."""
    buf = CupyRingBuffer(slots=2, dtype=cp.float64, shape=(2,), create=True)
    try:
        data = cp.array([1.0, 2.0], dtype=cp.float64)
        buf.write(data)

        view1 = buf.read()
        view2 = buf.read()
        assert view1 is view2  # Same object, not a copy
    finally:
        buf.close()


def test_different_dtypes():
    """Verify support for various cupy dtypes."""
    for dtype in [cp.int8, cp.int16, cp.int32, cp.int64,
                  cp.float32, cp.float64, cp.uint8, cp.uint16]:
        buf = CupyRingBuffer(slots=2, dtype=dtype, shape=(3,), create=True)
        try:
            data = cp.array([1, 2, 3], dtype=dtype)
            buf.write(data)
            cpt.assert_array_equal(buf.read(), data)
        finally:
            buf.close()


def test_multiple_wrap_cycles():
    """Run multiple full fill-drain cycles to stress wrap-around logic."""
    capacity = 3
    buf = CupyRingBuffer(slots=capacity, dtype=cp.int32, create=True)
    try:
        for cycle in range(5):
            for i in range(capacity):
                val = cp.int32(cycle * capacity + i)
                assert buf.write(val) is True
            for i in range(capacity):
                assert int(buf.read()) == cycle * capacity + i
                buf.release()
    finally:
        buf.close()


def test_capacity_one():
    """Edge case: single-slot buffer."""
    buf = CupyRingBuffer(slots=1, dtype=cp.float64, shape=(2,), create=True)
    try:
        assert buf.capacity == 1

        data = cp.array([1.0, 2.0], dtype=cp.float64)
        assert buf.write(data) is True
        assert buf.can_write() is False
        cpt.assert_array_equal(buf.read(), data)
        buf.release()

        data2 = cp.array([3.0, 4.0], dtype=cp.float64)
        assert buf.write(data2) is True
        cpt.assert_array_equal(buf.read(), data2)
    finally:
        buf.close()


def test_complex_dtype():
    """Test with complex number dtype."""
    buf = CupyRingBuffer(slots=3, dtype=cp.complex128, shape=(2,), create=True)
    try:
        data = cp.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=cp.complex128)
        buf.write(data)
        cpt.assert_array_equal(buf.read(), data)
    finally:
        buf.close()


def test_bool_dtype():
    """Test with boolean dtype."""
    buf = CupyRingBuffer(slots=3, dtype=cp.bool_, shape=(4,), create=True)
    try:
        data = cp.array([True, False, True, False], dtype=cp.bool_)
        buf.write(data)
        cpt.assert_array_equal(buf.read(), data)
    finally:
        buf.close()


def test_write_from_numpy():
    """Test writing numpy arrays - they should be transferred to GPU."""
    import numpy as np
    buf = CupyRingBuffer(slots=3, dtype=cp.float32, shape=(4,), create=True)
    try:
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf.write(cp.asarray(data))
        result = buf.read()
        assert isinstance(result, cp.ndarray)
        cpt.assert_array_equal(result, cp.asarray(data))
    finally:
        buf.close()


def test_result_stays_on_gpu():
    """Verify that read returns GPU arrays, not CPU arrays."""
    buf = CupyRingBuffer(slots=2, dtype=cp.float64, shape=(3,), create=True)
    try:
        data = cp.array([1.0, 2.0, 3.0], dtype=cp.float64)
        buf.write(data)
        result = buf.read()
        assert isinstance(result, cp.ndarray)
        assert result.device == cp.cuda.Device(0)
    finally:
        buf.close()


# --- IPC / cross-process tests ---


def test_ipc_handles_consumer():
    """A consumer opening IPC handles should see writes from the producer."""
    producer = CupyRingBuffer(4, dtype=cp.float64, shape=(2,), create=True)
    consumer = CupyRingBuffer(
        4, dtype=cp.float64, shape=(2,),
        create=False, ipc_handles=producer.ipc_handles,
    )
    try:
        data1 = cp.array([1.0, 2.0], dtype=cp.float64)
        producer.write(data1)
        assert consumer.can_read() is True
        cpt.assert_array_equal(consumer.read(), data1)
        consumer.release()

        data2 = cp.array([3.0, 4.0], dtype=cp.float64)
        producer.write(data2)
        cpt.assert_array_equal(consumer.read(), data2)
    finally:
        consumer.close()
        producer.close()


def test_ipc_handles_blob_size():
    """IPC handles blob should be (2 + slots+1) * 64 bytes."""
    slots = 5
    buf = CupyRingBuffer(slots, dtype=cp.float64, create=True)
    try:
        # _slots = slots + 1 (sentinel), plus 2 index handles
        expected = (2 + slots + 1) * 64
        assert len(buf.ipc_handles) == expected
    finally:
        buf.close()


def test_consumer_rejects_bad_handles():
    """Consumer should reject wrong-length handle blobs."""
    import pytest
    with pytest.raises(ValueError, match="Expected .* bytes"):
        CupyRingBuffer(
            3, dtype=cp.float64,
            create=False, ipc_handles=b"\x00" * 10,
        )


def test_producer_rejects_handles_arg():
    """Producer should reject ipc_handles being passed."""
    import pytest
    with pytest.raises(ValueError, match="ipc_handles must be None"):
        CupyRingBuffer(
            3, dtype=cp.float64,
            create=True, ipc_handles=b"\x00" * 64,
        )
