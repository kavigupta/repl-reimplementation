import unittest
import torch
from sketch_hypergraph.architecture.tools.sequence import (
    flat_to_batched,
    sequence_mask_for,
)


class TestDeserializer(unittest.TestCase):
    def test_flat_to_batched_jagged(self):
        values = torch.randn(10, 5, 4, 3)
        lengths = [2, 5, 3]
        result = flat_to_batched(values, lengths)
        self.assertEqual(result.shape, (3, 5, 5, 4, 3))
        self.assertTrue(torch.allclose(result[0, :2], values[:2]))
        self.assertTrue(torch.allclose(result[1, :5], values[2:7]))
        self.assertTrue(torch.allclose(result[2, :3], values[7:]))
        self.assertTrue(torch.allclose(result.sum(), values.sum()))

    def test_flat_to_batched_non_jagged(self):
        values = torch.randn(15, 5, 4, 3)
        lengths = [5, 5, 5]
        result = flat_to_batched(values, lengths)
        self.assertEqual(result.shape, (3, 5, 5, 4, 3))
        self.assertTrue(torch.allclose(result[0, :5], values[:5]))
        self.assertTrue(torch.allclose(result[1, :5], values[5:10]))
        self.assertTrue(torch.allclose(result[2, :5], values[10:]))
        self.assertTrue(torch.allclose(result.sum(), values.sum()))

    def test_sequence_mask(self):
        lengths = [5, 3, 8]
        result = sequence_mask_for(lengths)
        self.assertEqual(result.shape, (3, 8))
        self.assertTrue(torch.all(result[0, :5]))
        self.assertTrue(torch.all(result[1, :3]))
        self.assertTrue(torch.all(result[2, :8]))
        self.assertEqual(result.sum(), sum(lengths))
