__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable
import numpy as np

from jina import Executor, Document, DocumentArray, requests
from jina.logging.logger import JinaLogger
from vggish import vggish_input, vggish_params

class VGGishSegmenter(Executor):
    """
    Segment the Document blob into audio regions.

    :param sampling_factor: the n in sampling notation s(nT),
        used to multiply the chunks' sampling rate based on that of original audio docs
            e.g.
            With sampling factor of 2, sampling rate of 44.1kHz, and doc blob of length 204800,
                the beginning and end sequence of chunks should be :

                beg=0,    end=44100
                beg=22050,    end=66150
                beg=44100,    end=88200
                beg=66150,    end=110250
                beg=88200,    end=132300
                beg=110250,    end=154350
                beg=132300,    end=176400
                beg=154350,    end=198450
                beg=176400,    end=220500

    :param chunk_duration: sets the chunks' time length in seconds
    :param default_traversal_paths: sets the travseral path(s) used to select parts within documents
    """

    def __init__(self, sampling_factor: int=2, chunk_duration: int = 10, default_traversal_paths: Iterable[str] = ['r'], *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.sampling_factor = sampling_factor
        self.chunk_duration = chunk_duration
        self.default_traversal_paths = default_traversal_paths
        self.logger = JinaLogger(self.__class__.__name__)

    @requests
    def segment(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Encode all docs with audio and store the segmented regions in the chunks attribute of the docs.

        :param docs: documents sent to the segmenter, with tags containing their sampling rate.
        :param parameters: dictionary to define the sampling factor and traversal_path. For example,
            `parameters={'traversal_paths': 'c', 'sampling_factor': 10}`
            will set the parameters for traversal_paths, sampling_factor and that are actually used
        """
        if not docs:
            return

        def _get_num_chunks_per_channel(doc_blob, sampling_factor):
            num_channels = doc_blob.ndim
            if num_channels == 1:
                num_chunks_per_channel = int(doc.blob.shape[0] / chunk_size) * sampling_factor + 1
                channel_tags = ('mono',)
            else:
                num_chunks_per_channel = int(doc.blob.shape[1] / chunk_size) * sampling_factor + 1
                channel_tags = ('left', 'right')

            return num_chunks_per_channel, channel_tags

        filtered_docs = self._get_input_data(docs, parameters)

        for doc in filtered_docs:
            # a chunk consists of samples collected during chunk_duration
            doc_sampling_rate = doc.tags['sampling_rate']
            chunk_size = int(self.chunk_duration * doc_sampling_rate) # number of samples
            sampling_factor = parameters.get('sampling_factor', self.sampling_factor)
            num_chunks_per_channel, channel_tags = _get_num_chunks_per_channel(doc.blob, sampling_factor)

            for channel_idx, (chunks, tag) in enumerate(zip(doc.blob, channel_tags)): # traverse through channels
                for chunk_id in range(num_chunks_per_channel):
                    beg = int(chunk_id * chunk_size / sampling_factor)
                    end = beg + chunk_size
                    segment = doc.blob if doc.blob.ndim == 1 else chunks
                    segment_size = len(segment)
                    if end > segment_size:
                        continue
                    doc.chunks.append(
                        Document(
                            blob=segment[beg:end],
                            location=[beg, end],
                            tags={'channel': tag, 'sampling_rate': doc_sampling_rate*sampling_factor}))

        for doc in filtered_docs:
            result_chunk = []
            for chunk in doc.chunks:
                mel_data = vggish_input.waveform_to_examples(chunk.blob, chunk.tags['sampling_rate'])
                if mel_data.ndim != 3:
                    self.logger.error(
                        f'failed to convert from wave to mel, chunk.blob: {chunk.blob.shape}, sample_rate: {SAMPLE_RATE}')
                    continue
                if mel_data.shape[0] <= 0:
                    self.logger.info(f'chunk between {chunk.location} is skipped due to the duration is too short')
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