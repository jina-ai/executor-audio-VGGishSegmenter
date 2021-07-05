__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina import Document, DocumentArray
from jina.executors import BaseExecutor
#from jinahub.segmenter.vggish_audio_segmenter import VGGishSegmenter

import sys
sys.path.insert(1, '../..')

from vggish_audio_segmenter import VGGishSegmenter

def test_exec():
    ex = BaseExecutor.load_config('../../config.yml')
    assert ex.sample_rate==44100




def test_mono():
    n_frames = 100
    frame_length = 2048 # the number of samples in each frame
    signal_orig = np.random.randn(frame_length * n_frames)

    segmenter = VGGishSegmenter()
    docs = DocumentArray([Document(blob=signal_orig, tags={'n_frames': 100, 'frame_length': 2048})])
    segmenter.segment(docs, {})
    assert len(docs) == 1
    chunks = docs.get_attributes('chunks')
    assert len(chunks) == 1
    assert len(chunks[0]) == 0
    # for segmented_chunk in docs.get_attributes('chunks'):
    #     assert len(segmented_chunk) == n_frames * 2 - 1


def test_stereo():
    n_frames = 100
    frame_length = 2048
    signal_orig = np.random.randn(2, frame_length * n_frames)

    segmenter = VGGishSegmenter()
    docs = DocumentArray([Document(blob=np.stack([signal_orig, signal_orig]), tags={'n_frames': 100, 'frame_length': 2048})])
    segmenter.segment(docs, {})
    assert len(docs) == 1
    chunks = docs.get_attributes('chunks')
    assert len(chunks) == 1
    assert len(chunks[0]) == 0
    # for segmented_chunk in docs.get_attributes('chunks'):
    #     assert len(segmented_chunk) == (n_frames * 2 - 1) * 2

def test_location_stereo():
    frame_length = 10
    n_frames = 5
    num_docs = 3
    num_channels = 2
    chunk_duration = 1
    sample_rate = 44100
    chunk_size = chunk_duration*sample_rate

    signal_orig = np.random.randn(num_channels, frame_length * n_frames)
    expected_n_frames = int(signal_orig.shape[1] / chunk_size)
    expected_locations = [[i, i + frame_length] for i in range(int(expected_n_frames))]

    segmenter = VGGishSegmenter(chunk_duration=chunk_duration)
    docs = DocumentArray([Document(blob=signal_orig, tags={'n_frames': 100, 'frame_length': 2048}) for i in range(num_docs)])
    segmenter.segment(docs, {})
    assert len(docs) == num_docs
    for d in docs:
        assert len(d.chunks) == expected_n_frames * num_channels
        for i, chunk in enumerate(d.chunks):
            assert chunk['location'] == expected_locations[int(i % expected_n_frames)]
            expected_channel = 'left' if i // expected_n_frames == 0 else 'right'
            assert chunk['tags']['channel'] == expected_channel