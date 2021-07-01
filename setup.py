__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-VGGishSegmenter',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='VGGishSegmenter segments the audio signal on the doc-level into frames on the chunk-level.',
    url='https://github.com/jina-ai/executor-audio-VGGishSegmenter',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=['jinahub.segmenter.vggish_audio_segmenter'],
    package_dir={'jinahub.segmenter': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)
