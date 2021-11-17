import unittest

from replay_buffer import PrevReplayBuffer, BufferFieldSpec
from prioritized_buffer import PrioritizedReplayBuffer

import numpy as np


class TestBuffer(unittest.TestCase):
    def _init_data(self, n=2):
        buffer = PrevReplayBuffer(n=n)(max_size=10, action_spec=BufferFieldSpec((1,)), obs_spec=BufferFieldSpec((1,)))
        
        buffer._actions = np.arange(0, 10).reshape(-1, 1)
        buffer._obs = np.arange(0, 10).reshape(-1, 1)
        buffer._obs_next = np.arange(0, 10).reshape(-1, 1)
        buffer._rewards = np.arange(0, 10)
        buffer._policies = np.arange(0, 10).reshape(-1, 1)
        buffer._dones = np.zeros(10)
        buffer._ends = np.zeros(10)

        return buffer

    def test_vector(self):
        buffer = self._init_data()
        buffer._pointer = 9
        buffer._current_size = 9
        buffer._sample_random_index = lambda n: np.array([2, 3, 4, 5], dtype=np.int32)

        batch, l, pl = buffer.get_vec(4, 3)

        self.assertTrue((l == [3, 3, 3, 3]).all())
        self.assertTrue((pl == [2, 2, 2, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(0, 5),
            np.arange(1, 6),
            np.arange(2, 7),
            np.arange(3, 8),
        ], -1)).all()
        )

    def test_vector_pointer_ovf(self):
        buffer = self._init_data()
        buffer._pointer = 5
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([2, 3], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 2]).all())
        self.assertTrue((pl == [2, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(0, 5),
            np.arange(1, 6),
        ], -1)).all()
        )

    def test_vector_size_ovf(self):
        buffer = self._init_data()
        buffer._pointer = 5
        buffer._current_size = 5
        buffer._sample_random_index = lambda n: np.array([2, 3], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 2]).all())
        self.assertTrue((pl == [2, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(0, 5),
            np.arange(1, 6),
        ], -1)).all()
        )

    def test_vector_wrap(self):
        buffer = self._init_data()
        buffer._pointer = 5
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([7, 8], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 3]).all())
        self.assertTrue((pl == [2, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(5, 10),
            np.arange(6, 11) % 10,
        ], -1)).all()
        )

    def test_vector_wrap_ovf(self):
        buffer = self._init_data()
        buffer._pointer = 1
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([8, 9], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 2]).all())
        self.assertTrue((pl == [2, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(6, 11) % 10,
            np.arange(7, 12) % 10,
        ], -1)).all()
        )

    def test_vector_prev_pointer_undf(self):
        buffer = self._init_data()
        buffer._pointer = 3
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([4, 5], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 3]).all())
        self.assertTrue((pl == [1, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(2, 7) % 10,
            np.arange(3, 8) % 10,
        ], -1)).all()
        )

    def test_vector_prev_wrap(self):
        buffer = self._init_data()
        buffer._pointer = 7
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([0, 1], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 3]).all())
        self.assertTrue((pl == [2, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(-2, 3) % 10,
            np.arange(-1, 4) % 10,
        ], -1)).all()
        )

    def test_vector_prev_wrap_undf(self):
        buffer = self._init_data()
        buffer._pointer = 7
        buffer._current_size = 7
        buffer._sample_random_index = lambda n: np.array([0, 1], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 3]).all())
        self.assertTrue((pl == [0, 1]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(-2, 3) % 10,
            np.arange(-1, 4) % 10,
        ], -1)).all()
        )

    def test_vector_prev_pointer_wrap_undf(self):
        buffer = self._init_data()
        buffer._pointer = 9
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([0, 1], dtype=np.int32)

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 3]).all())
        self.assertTrue((pl == [1, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(-2, 3) % 10,
            np.arange(-1, 4) % 10,
        ], -1)).all()
        )

    def test_vector_ends_ovf(self):
        buffer = self._init_data()
        buffer._pointer = 10
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([2, 3], dtype=np.int32)
        buffer._ends = np.arange(0, 10) == 4

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 2]).all())
        self.assertTrue((pl == [2, 2]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(0, 5) % 10,
            np.arange(1, 6) % 10 * (np.arange(1, 6) < 5),
        ], -1)).all()
        )

    def test_vector_prev_ends_ovf(self):
        buffer = self._init_data()
        buffer._pointer = 10
        buffer._current_size = 10
        buffer._sample_random_index = lambda n: np.array([3, 4], dtype=np.int32)
        buffer._ends = np.arange(0, 10) == 2

        batch, l, pl = buffer.get_vec(2, 3)

        self.assertTrue((l == [3, 3]).all())
        self.assertTrue((pl == [0, 1]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(1, 6) % 10 * (np.arange(1, 6) > 2),
            np.arange(2, 7) % 10 * (np.arange(2, 7) > 2),
        ], -1)).all()
        )

    def test_vector_no_n(self):
        buffer = self._init_data(None)
        buffer._pointer = 9
        buffer._current_size = 9
        buffer._sample_random_index = lambda n: np.array([2, 3, 4, 5], dtype=np.int32)

        batch, l = buffer.get_vec(4, 3)

        self.assertTrue((l == [3, 3, 3, 3]).all())

        self.assertTrue((batch['actions'] == np.expand_dims([
            np.arange(2, 5),
            np.arange(3, 6),
            np.arange(4, 7),
            np.arange(5, 8),
        ], -1)).all()
        )

class TestPrioritizedBuffer(unittest.TestCase):
    def test_inital_values_not_full(self):
        buffer = PrioritizedReplayBuffer(
            20, BufferFieldSpec((1,), np.float32), BufferFieldSpec((1,), np.float32), BufferFieldSpec((1,), np.float32),
            4, 2
        )

        buffer.put(0, 1, 2, 0, 0, False, False)
        buffer.put(0, 2, 3, 0, 0, False, False)
        buffer.put(0, 3, 4, 0, 0, False, False)
        buffer.put(0, 4, 5, 0, 0, False, False)
        buffer.put(0, 5, 6, 0, 0, False, False)

        self.assertEqual(buffer._total_priorities, 5)

        self.assertTrue((buffer._priorities[0] == [1] * 5 + [0] * 15).all())
        self.assertTrue((buffer._priorities[1] == [4, 1, 0, 0, 0]).all())
        self.assertTrue((buffer._priorities[2] == [5, 0]).all())

        self.assertTrue((buffer._priorities_cumsums[0] == list(range(1, 5)) + [1] * 4 + [0] * 12).all())
        self.assertTrue((buffer._priorities_cumsums[1] == [4, 5, 5, 0, 0]).all())
        self.assertTrue((buffer._priorities_cumsums[2] == [5, 5]).all())

    def test_update_priorities(self):
        buffer = PrioritizedReplayBuffer(
            20, BufferFieldSpec((1,), np.float32), BufferFieldSpec((1,), np.float32), BufferFieldSpec((1,), np.float32),
            4, 2
        )

        for i in range(1, 10):
            buffer.put(0, i, i + 1, 0, 0, False, False)
            if i < 4:
                self.assertFalse(buffer.should_update_block())
            else:
                self.assertTrue(buffer.should_update_block())


        self.assertTrue((buffer._priorities[0] == [1] * 9 + [0] * 11).all())
        self.assertTrue((buffer._priorities[1] == [4, 4, 1, 0, 0]).all())
        self.assertTrue((buffer._priorities[2] == [9, 0]).all())

        self.assertTrue((buffer._priorities_cumsums[0] == list(range(1, 5)) + list(range(1, 5)) + [1] * 4 + [0] * 8).all())
        self.assertTrue((buffer._priorities_cumsums[1] == [4, 8, 9, 0, 0]).all())
        self.assertTrue((buffer._priorities_cumsums[2] == [9, 9]).all())

        buffer.update_block([0.5, 2, 2, 1])

        self.assertTrue((buffer._priorities[0] == [0.5, 2, 2, 1] + [1] * 5 + [0] * 11).all())
        self.assertTrue((buffer._priorities[1] == [5.5, 4, 1, 0, 0]).all())
        self.assertTrue((buffer._priorities[2] == [10.5, 0]).all())

        self.assertTrue((buffer._priorities_cumsums[0] == [0.5, 2.5, 4.5, 5.5] + list(range(1, 5)) + [1] * 4 + [0] * 8).all())
        self.assertTrue((buffer._priorities_cumsums[1] == [5.5, 9.5, 10.5, 0, 0]).all())
        self.assertTrue((buffer._priorities_cumsums[2] == [10.5, 10.5]).all())

        buffer.update_block([1.5, 1, 0.5, 2])

        self.assertTrue((buffer._priorities[0] == [0.5, 2, 2, 1, 1.5, 1, 0.5, 2, 1] + [0] * 11).all())
        self.assertTrue((buffer._priorities[1] == [5.5, 5, 1, 0, 0]).all())
        self.assertTrue((buffer._priorities[2] == [11.5, 0]).all())

        self.assertTrue((buffer._priorities_cumsums[0] == [0.5, 2.5, 4.5, 5.5, 1.5, 2.5, 3, 5] + [1] * 4 + [0] * 8).all())
        self.assertTrue((buffer._priorities_cumsums[1] == [5.5, 10.5, 11.5, 0, 0]).all())
        self.assertTrue((buffer._priorities_cumsums[2] == [11.5, 11.5]).all())

        buffer.update_block([1, 0.5, 1, 0.5])

        self.assertTrue((buffer._priorities[0] == [1, 0.5, 1, 0.5, 1.5, 1, 0.5, 2, 1] + [0] * 11).all())
        self.assertTrue((buffer._priorities[1] == [3, 5, 1, 0, 0]).all())
        self.assertTrue((buffer._priorities[2] == [9, 0]).all())

        self.assertTrue((buffer._priorities_cumsums[0] == [1, 1.5, 2.5, 3, 1.5, 2.5, 3, 5] + [1] * 4 + [0] * 8).all())
        self.assertTrue((buffer._priorities_cumsums[1] == [3, 8, 9, 0, 0]).all())
        self.assertTrue((buffer._priorities_cumsums[2] == [9, 9]).all())

    def test_draw_samples(self):
        buffer = PrioritizedReplayBuffer(
            20, BufferFieldSpec((1,), np.float32), BufferFieldSpec((1,), np.float32), BufferFieldSpec((1,), np.float32),
            4, 2
        )

        for i in range(1, 10):
            buffer.put(0, i, i + 1, 0, 0, False, False)

        buffer.update_block([0.5, 2, 2, 1])
        buffer.update_block([1.5, 1, 0.5, 2])

        indices = buffer._sample_indices_from_rands([1, 5, 6.5, 11, 10])

        self.assertSequenceEqual(list(indices), [1, 3, 4, 8, 7])

        buffer.get_vec(4, 2)
