import unittest
from src.stream_simulation import simulate_data_stream

class TestStreamSimulation(unittest.TestCase):
    def test_stream_length(self):
        stream = simulate_data_stream(length=100)
        data = list(stream)
        self.assertEqual(len(data), 100)

    def test_stream_noise(self):
        stream = simulate_data_stream(length=100, noise=0.5)
        data = list(stream)
        self.assertTrue(all(isinstance(value, float) for value in data))

if __name__ == "__main__":
    unittest.main()
