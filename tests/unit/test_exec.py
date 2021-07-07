__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina import Document, DocumentArray
from jina.executors import BaseExecutor
from vggish_audio_segmenter import VGGishSegmenter


def test_exec():
    ex = BaseExecutor.load_config('../../config.yml')
    assert ex.chunk_duration == 10


def test_mono():
    n_frames = 100
    frame_length = 2048 # the number of samples in each frame
    signal_orig = np.random.randn(frame_length * n_frames)

    segmenter = VGGishSegmenter()
    docs = DocumentArray([Document(blob=signal_orig,
                                   tags={'n_frames': 100, 'frame_length': 2048, 'sampling_rate': 44100})])
    segmenter.segment(docs, {})
    assert len(docs) == 1

    chunks = docs.traverse_flat(['c'])
    assert len(chunks) == 0


def test_stereo():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(2, frame_length * n_frames)

    segmenter = VGGishSegmenter()
    docs = DocumentArray([Document(blob=signal_orig,
                                   tags={'n_frames': 100, 'frame_length': 2048, 'sampling_rate': 44100})])
    segmenter.segment(docs, {})
    assert len(docs) == 1
    chunks = docs.traverse_flat(['c'])
    assert len(chunks) == 0


def test_location_mono():
    frame_length = 2048
    n_frames = 100
    num_docs = 3
    chunk_duration = 1
    sample_rate = 44100
    sampling_factor = 2
    chunk_size = chunk_duration * sample_rate

    signal_orig = np.random.randn(frame_length * n_frames)
    expected_n_frames = int(signal_orig.shape[0] / chunk_size)
    expected_channel = 'mono'

    segmenter = VGGishSegmenter(chunk_duration=chunk_duration)
    docs = DocumentArray([Document(blob=signal_orig,
                            tags={'n_frames': 100, 'frame_length': 2048, 'sampling_rate': 44100}) for i in range(num_docs)])
    segmenter.segment(docs, {})

    assert len(docs) == num_docs
    for d in docs:
        assert len(d.chunks) == expected_n_frames * sampling_factor
        for i, chunk in enumerate(d.chunks):
            expected_start = int(i % (expected_n_frames*sampling_factor)*chunk_size/sampling_factor)
            assert chunk.location == [expected_start, expected_start + chunk_size]
            assert chunk.tags['channel'] == expected_channel

def test_location_stereo():
    frame_length = 2048
    n_frames = 100
    num_docs = 3
    num_channels = 2
    chunk_duration = 1
    sampling_factor = 2
    sample_rate = 44100
    chunk_size = chunk_duration*sample_rate

    signal_orig = np.random.randn(num_channels, frame_length * n_frames)
    expected_n_frames = int(signal_orig.shape[1] / chunk_size)

    segmenter = VGGishSegmenter(chunk_duration=chunk_duration)
    docs = DocumentArray([Document(blob=signal_orig,
                                   tags={'n_frames': 100, 'frame_length': 2048, 'sampling_rate': 44100}) for i in range(num_docs)])
    segmenter.segment(docs, {})
    assert len(docs) == num_docs
    for d in docs:
        assert len(d.chunks) == expected_n_frames * num_channels * sampling_factor
        for i, chunk in enumerate(d.chunks):
            expected_start = int(i % (expected_n_frames*sampling_factor)*chunk_size/sampling_factor)
            assert chunk.location == [expected_start, expected_start + chunk_size]
            expected_channel = 'left' if i // (expected_n_frames*sampling_factor) == 0 else 'right'
            assert chunk.tags['channel'] == expected_channel