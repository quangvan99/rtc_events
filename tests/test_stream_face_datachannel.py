"""Unit tests for DataChannel integration in stream_face.py"""
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent dir to path
sys.path.insert(0, '/home/mq/disk2T/quangnv/face')


class TestTrackerManager(unittest.TestCase):
    """Test TrackerManager.cleanup() returns removed IDs"""

    def test_cleanup_returns_removed_ids(self):
        # Import locally to avoid GStreamer init
        from stream_face import TrackerManager, Tracker

        mgr = TrackerManager(max_age=2)
        # Add trackers
        trk1 = mgr.update(1, 0, [0, 0, 10, 10])
        trk2 = mgr.update(2, 0, [0, 0, 10, 10])

        # Age them past max_age
        trk1.age = 3
        trk2.age = 1

        removed = mgr.cleanup()

        self.assertEqual(removed, [1])
        self.assertEqual(len(mgr.trackers), 1)
        self.assertEqual(mgr.trackers[0].object_id, 2)


class TestSendFaceEvent(unittest.TestCase):
    """Test _send_face_event behavior"""

    def setUp(self):
        # Mock GStreamer and pyds
        self.gst_mock = MagicMock()
        self.pyds_mock = MagicMock()

        sys.modules['gi'] = MagicMock()
        sys.modules['gi.repository'] = MagicMock()
        sys.modules['pyds'] = self.pyds_mock

    def test_send_face_event_skip_if_already_sent(self):
        """Should not send if object_id already in sent_faces"""
        # Create mock client
        client = MagicMock()
        client.sent_faces = {1}
        client.data_channel = MagicMock()

        # Import the method logic
        from stream_face import WebRTCFaceClient

        # Test: object_id 1 already sent
        WebRTCFaceClient._send_face_event(client, 1, "Test")

        # Should not call emit
        client.data_channel.emit.assert_not_called()

    def test_send_face_event_skip_if_no_channel(self):
        """Should not send if data_channel is None"""
        client = MagicMock()
        client.sent_faces = set()
        client.data_channel = None

        from stream_face import WebRTCFaceClient

        # Should not raise, just return
        WebRTCFaceClient._send_face_event(client, 1, "Test")

        # sent_faces should be unchanged
        self.assertEqual(len(client.sent_faces), 0)

    def test_send_face_event_adds_to_sent_faces(self):
        """Should add object_id to sent_faces on send"""
        client = MagicMock()
        client.sent_faces = set()
        client.data_channel = MagicMock()

        from stream_face import WebRTCFaceClient

        WebRTCFaceClient._send_face_event(client, 1, "Test")

        self.assertIn(1, client.sent_faces)
        client.data_channel.emit.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
