__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable
import numpy as np

from jina import Executor, Document, DocumentArray, requests
from vggish import vggish_input, vggish_params

class VGGishSegmenter(Executor):
    """
    Segment the Document blob into audio regions.

    :param embedding_dim: the output dimensionality of the embedding
    """

    def __init__(self, chunk_duration: int = 10,
                sample_rate: int = 44100, default_traversal_paths: Iterable[str] = ['r'], *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.default_traversal_paths = default_traversal_paths


    @requests
    def segment(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Encode all docs with audio and store the segmented regions in the chunks attribute of the docs.
        :param docs: documents sent to the segmenter.
        """
        if not docs:
            return

        filtered_docs = self._get_input_data(docs, parameters)

        for idx, doc in enumerate(filtered_docs):
            # a chunk consists of samples collected during chunk_duration
            chunk_size = int(self.chunk_duration * self.sample_rate) # number of samples
            strip = int(2 * self.sample_rate)
            # print(doc.blob.shape[0])
            # print(chunk_size)
            # print(strip)
            # if doc.blob.shape[0] < chunk_size:
            #     doc.chunks.append(
            #         Document(
            #             blob=doc.blob[:chunk_size],
            #             offset=idx,
            #             location=[0, self.chunk_duration],
            #             tags=doc.tags))
            # else:
            # num_chunks = int((doc.blob.shape[0] - chunk_size) / strip)
            num_chunks = int(doc.blob.shape[0] / chunk_size)
            for chunk_id in range(num_chunks):
                beg = chunk_id * strip
                end = beg + chunk_size
                if end > doc.blob.shape[0]:
                    continue
                doc.chunks.append(
                    Document(
                        blob=doc.blob[beg:end],
                        offset=idx,
                        location=[chunk_id * 2, chunk_id * 2 + self.chunk_duration],
                        tags=doc.tags))

        for doc in filtered_docs:
            result_chunk = []
            for chunk in doc.chunks:
                print(chunk.blob)
                mel_data = vggish_input.waveform_to_examples(chunk.blob, self.sample_rate)
                if mel_data.ndim != 3:
                    print(
                        f'failed to convert from wave to mel, chunk.blob: {chunk.blob.shape}, sample_rate: {SAMPLE_RATE}')
                    continue
                if mel_data.shape[0] <= 0:
                    print(f'chunk between {chunk.location} is skipped due to the duration is too short')
                if mel_data.ndim == 2:
                    mel_data = np.atleast_3d(mel_data)
                    mel_data = mel_data.reshape(1, mel_data.shape[0], mel_data.shape[1])
                chunk.blob = mel_data
                result_chunk.append(chunk)
            doc.chunks = result_chunk


    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        """Create a filtered set of Documents to iterate over."""

        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without audio
        filtered_docs = DocumentArray([doc for doc in flat_docs if doc.blob is not None])

        return filtered_docs