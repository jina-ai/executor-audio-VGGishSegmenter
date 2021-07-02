__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina import Document, DocumentArray
from jina.executors import BaseExecutor
# from jinahub.segmenter.vggish_audio_segmenter import VGGishSegmenter

import sys
sys.path.insert(1, '../..')

from vggish_audio_segmenter import VGGishSegmenter

def test_exec():
    ex = BaseExecutor.load_config('../../config.yml')
    assert ex.frame_length==2048




def test_sliding_window_mono():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(frame_length * n_frames)

    segmenter = VGGishSegmenter(frame_length, frame_length // 2)
    segmented_chunks_per_doc = segmenter.segment(DocumentArray([Document(blob=np.stack([signal_orig, signal_orig]))]))
    assert len(segmented_chunks_per_doc) == 2
    for segmented_chunk in segmented_chunks_per_doc:
        assert len(segmented_chunk) == n_frames * 2 - 1


def test_sliding_window_stereo():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(2, frame_length * n_frames)

    segmenter = VGGishSegmenter(frame_length)  #frame_length // 2)
    segmented_chunks_per_doc = segmenter.segment(DocumentArray([Document(blob=np.stack([signal_orig, signal_orig]))]))
    assert len(segmented_chunks_per_doc) == 2
    for segmented_chunk in segmented_chunks_per_doc:
        assert len(segmented_chunk) == (n_frames * 2 - 1) * 2


def test_location_mono():
    frame_length = 10
    frame_overlap_length = frame_length // 2
    hop_length = frame_length - frame_overlap_length
    n_frames = 5
    num_docs = 3

    signal_orig = np.random.randn(frame_length * n_frames)
    expected_n_frames = (signal_orig.shape[0] - frame_length) / hop_length + 1
    expected_locations = [[i * hop_length, i * hop_length + frame_length] for i in range(int(expected_n_frames))]
    expected_channel = 'mono'

    segmenter = VGGishSegmenter(frame_length=frame_length, hop_length=hop_length)
    docs = DocumentArray([Document(blob=np.stack([signal_orig])) for i in range(num_docs)])
    segmenter.segment(docs)

    assert len(docs) == num_docs
    for d in docs:
        assert len(d.chunks) == expected_n_frames
        for i, chunk in enumerate(d.chunks):
            assert chunk['location'] == expected_locations[int(i % expected_n_frames)]
            assert chunk['tags']['channel'] == expected_channel


def test_location_stereo():
    frame_length = 10
    frame_overlap_length = frame_length // 2
    hop_length = frame_length - frame_overlap_length
    n_frames = 5
    num_docs = 3
    num_channels = 2

    signal_orig = np.random.randn(num_channels, frame_length * n_frames)
    expected_n_frames = (signal_orig.shape[1] - frame_length) / hop_length + 1
    expected_locations = [[i * hop_length, i * hop_length + frame_length] for i in range(int(expected_n_frames))]

    segmenter = VGGishSegmenter(frame_length=frame_length, hop_length=hop_length)
    docs = DocumentArray([Document(blob=np.stack([signal_orig])) for i in range(num_docs)])
    segmenter.segment(docs)
    assert len(docs) == num_docs
    for d in docs:
        assert len(d.chunks) == expected_n_frames * num_channels
        for i, chunk in enumerate(d.chunks):
            assert chunk['location'] == expected_locations[int(i % expected_n_frames)]
            expected_channel = 'left' if i // expected_n_frames == 0 else 'right'
            assert chunk['tags']['channel'] == expected_channel