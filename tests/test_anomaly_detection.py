import unittest
from src.anomaly_detection import AnomalyDetector
import numpy as np

class TestAnomalyDetection(unittest.TestCase):

    def setUp(self):
        """Initialize the AnomalyDetector with sample data."""
        self.detector = AnomalyDetector(contamination=0.01, n_estimators=100)
        # Generate normal data and one outlier
        self.normal_data = np.array([0.5, 0.1, -0.2, 0.7, -0.5, 1.0, 0.3, -0.1, 0.6, 0.2])
        self.anomaly_data = [100]  # An extreme value

    def test_isolation_forest_detection(self):
        """Test Isolation Forest detects anomalies correctly."""
        all_data = list(self.normal_data) + self.anomaly_data
        self.detector.fit(self.normal_data)  # Fit on normal data

        # Test on normal data - should not detect anomaly
        is_anomaly = self.detector.detect_isolation_forest(np.mean(self.normal_data))
        self.assertFalse(is_anomaly)

        # Test on anomaly data - should detect anomaly
        is_anomaly = self.detector.detect_isolation_forest(130)
        self.assertTrue(is_anomaly)

    def test_z_score_detection(self):
        """Test Z-Score detects anomalies correctly."""
        all_data = list(self.normal_data) + self.anomaly_data
        self.detector.fit(self.normal_data)  # Fit on normal data

        # Test on normal data - should not detect anomaly
        is_anomaly = self.detector.detect_z_score(np.mean(self.normal_data))
        self.assertFalse(is_anomaly)

        # Test on anomaly data - should detect anomaly
        for value in self.anomaly_data:
            is_anomaly = self.detector.detect_z_score(value)
            self.assertTrue(is_anomaly)

    def test_z_score_zero_variance(self):
        """Test Z-Score with zero variance."""
        constant_data = [5] * 100
        self.detector.fit(constant_data)

        # Since variance is zero, there should be no anomalies
        for value in constant_data:
            is_anomaly = self.detector.detect_z_score(value)
            self.assertFalse(is_anomaly)

    def test_isolation_forest_without_fit(self):
        """Test error raised when using Isolation Forest without fitting."""
        with self.assertRaises(RuntimeError):
            self.detector.detect_isolation_forest(5)

    def test_z_score_without_fit(self):
        """Test error raised when using Z-Score without fitting."""
        with self.assertRaises(RuntimeError):
            self.detector.detect_z_score(5)


if __name__ == "__main__":
    unittest.main()
