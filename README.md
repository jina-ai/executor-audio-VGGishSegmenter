# 📝 PLEASE READ [THE GUIDELINES](.github/GUIDELINES.md) BEFORE STARTING.

# 🏗️ PLEASE CHECK OUT [STEP-BY-STEP](.github/STEP_BY_STEP.md)

----

# ✨ VGGishSegmenter

**VGGishSegmenter** is a class that segments the audio signal on the doc-level into frames on the chunk-level.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

None
## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://VGGishSegmenter')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://VGGishSegmenter'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://VGGishSegmenter')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://VGGishSegmenter'
```


### 📦️ Via Pypi

1. Install the `jinahub-VGGishSegmenter` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-audio-VGGishSegmenter.git
	```

1. Use `jinahub-VGGishSegmenter` in your code

	```python
	from jina import Flow
	from jinahub.segmenter.vggish_audio_segmenter import VGGishSegmenter
	
	f = Flow().add(uses=VGGishSegmenter)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-audio-VGGishSegmenter.git
	cd executor-audio-VGGishSegmenter
	docker build -t executor-audio-VGGishSegmenter-image .
	```

1. Use `executor-audio-VGGishSegmenter-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-audio-VGGishSegmenter-image:latest')
	```
	

## 🎉️ Example 

Here we **MUST** show a **MINIMAL WORKING EXAMPLE**. We recommend to use `jinahub+docker://MyDummyExecutor` for the purpose of boosting the usage of Jina Hub. 

It not necessary to demonstrate the usages of every inputs. It will be demonstrate in the next section.

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://VGGishSegmenter')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
    print(f'{resp}')
```

### `on=/index` (Optional)

When there are multiple APIs, we need to list the inputs and outputs for each one. If there is only one universal API, we only demonstrate the inputs and outputs for it.

#### Inputs 

`Document` with `blob` of the shape `256`.

#### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.

### `on=/update` (Optional)

When there are multiple APIs, we need to list the inputs and outputs for each on

#### Inputs 

`Document` with `blob` of the shape `256`.

#### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (=128, by default) with `dtype=nfloat32`.

## 🔍️ Reference
- https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md

