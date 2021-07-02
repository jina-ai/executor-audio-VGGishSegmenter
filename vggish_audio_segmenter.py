__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable
import numpy as np

from jina import Executor, DocumentArray, requests
from vggish import vggish_input, vggish_params

class VGGishSegmenter(Executor):
    """
    Segment the Document blob into audio regions.

    :param embedding_dim: the output dimensionality of the embedding
    """

    def __init__(self, frame_length: int = 2048, hop_length: int = 512, default_traversal_paths: Iterable[str] = ['r'], *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.default_traversal_paths = default_traversal_paths

        # def _segment(self, signal):
        #     if signal.ndim == 1:  # mono
        #
        #     elif signal.ndim == 2:  # stereo


    @requests
    def segment(self, docs: Optional[DocumentArray], **kwargs):
        """
        Encode all docs with audio and store the segmented regions in the chunks attribute of the docs.
        :param docs: documents sent to the segmenter. The docs must have `blob` of the shape `256`.
        """
        if not docs:
            return

        docs = self._get_input_data(docs, {})
        for doc in docs:
            channel_frames = vggish_input.waveform_to_examples(doc.blob, self.frame_length)

            chunks = []

            channel_tags = ('mono',) if len(channel_frames) == 1 else ('left', 'right')

            for frames, tag in zip(channel_frames, channel_tags):
                start = 0
                for idx, frame in enumerate(frames):
                    chunks.append(dict(offset=idx, weight=1.0, blob=frame, location=[start, start + len(frame)],
                                       tags={'channel': tag}))
                    start += self.hop_length
            doc.chunks = chunks

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        """Create a filtered set of Documents to iterate over."""

        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without audio
        filtered_docs = DocumentArray([doc for doc in flat_docs if doc.blob is not None])

        return filtered_docs