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




def test_mono():
    n_frames = 100
    frame_length = 2048 # the number of samples in each frame
    signal_orig = np.random.randn(frame_length * n_frames)

    segmenter = VGGishSegmenter()
    docs = DocumentArray([Document(blob=signal_orig)])
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
    docs = DocumentArray([Document(blob=np.stack([signal_orig, signal_orig]))])
    segmenter.segment(docs, {})
    assert len(docs) == 1
    chunks = docs.get_attributes('chunks')
    assert len(chunks) == 1
    assert len(chunks[0]) == 0
    # for segmented_chunk in docs.get_attributes('chunks'):
    #     assert len(segmented_chunk) == (n_frames * 2 - 1) * 2
