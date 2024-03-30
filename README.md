# Generative Models by Stability AI

## The codebase

## Install

```
pip install git+https://github.com/StableDraw/stablediffusionxl
``` 
or for locale build:
```
python setup.py build
python bdist_wheel
``` 
then your wheel will be in the dist directory, then you may install it, for example, by
```
cd dist
pip install stablediffusionxl-3.0.1-py3-none-any.whl
``` 
or with another name of your package that will be in dist directory

## Packaging

This repository uses PEP 517 compliant packaging using [Hatch](https://hatch.pypa.io/latest/).

To build a distributable wheel, install `hatch` and run `hatch build`
(specifying `-t wheel` will skip building a sdist, which is not necessary).

```
pip install hatch
hatch build -t wheel
```

You will find the built package in `dist/`. You can install the wheel with `pip install dist/*.whl`.

Note that the package does **not** currently specify dependencies; you will need to install the required packages,
depending on your use case and PyTorch version, manually.

#### Conditioner

The `GeneralConditioner` is configured through the `conditioner_config`. Its only attribute is `emb_models`, a list of
different embedders (all inherited from `AbstractEmbModel`) that are used to condition the generative model.
All embedders should define whether or not they are trainable (`is_trainable`, default `False`), a classifier-free
guidance dropout rate is used (`ucg_rate`, default `0`), and an input key (`input_key`), for example, `txt` for
text-conditioning or `cls` for class-conditioning.
When computing conditionings, the embedder will get `batch[input_key]` as input.
We currently support two to four dimensional conditionings and conditionings of different embedders are concatenated
appropriately.
Note that the order of the embedders in the `conditioner_config` is important.

#### Network

The neural network is set through the `network_config`. This used to be called `unet_config`, which is not general
enough as we plan to experiment with transformer-based diffusion backbones.

#### Loss

The loss is configured through `loss_config`. For standard diffusion model training, you will have to
set `sigma_sampler_config`.

#### Sampler config

As discussed above, the sampler is independent of the model. In the `sampler_config`, we set the type of numerical
solver, number of steps, type of discretization, as well as, for example, guidance wrappers for classifier-free
guidance.

### Dataset Handling

For large scale training we recommend using the data pipelines from
our [data pipelines](https://github.com/Stability-AI/datapipelines) project. The project is contained in the requirement
and automatically included when following the steps from the [Installation section](#installation).
Small map-style datasets should be defined here in the repository (e.g., MNIST, CIFAR-10, ...), and return a dict of
data keys/values,
e.g.,

```python
example = {"jpg": x,  # this is a tensor -1...1 chw
           "txt": "a beautiful image"}
```

where we expect images in -1...1, channel-first format.