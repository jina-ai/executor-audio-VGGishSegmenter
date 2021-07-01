__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Flow, Document
from jinahub.segmenter.vggish_audio_segmenter import VGGishSegmenter

def test_exec():
    f = Flow().add(uses=VGGishSegmenter)
    with f:
        resp = f.post(on='/test', inputs=Document(), return_results=True)
        assert resp is not None
