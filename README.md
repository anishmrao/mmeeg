# mmeeg
An opensource toolbox for EEG decoding tasks.

MMEEG is a framework built on top of [mmengine](https://github.com/open-mmlab/mmengine) that facilitates easy and fast experiments for EEG decoding tasks.

## Installation

Install mmengine first by following their [installation instructions](https://github.com/open-mmlab/mmengine#installation).

Next, install the following dependencies:
```
scipy
numpy
einops
```

## Usage

The current implementation supports standard EEGDatasets and networks for Motor Imagery Decoding. Custom dataset classes can be implemented and registered in a similar fashion. Please see the datasets folder for an exaple.

EEGConformer and EEGNet implementations are provided in the models folder. However, all models writted in PyTorch can be registered and used with the framework with minimal effort, following the same pattern.

The abstractions provided by this config-driven approach allows for quick, easy and reproducible experiments with minimum guesswork.

Please see the MMEEG_Tutorial.ipynb for an example of how to use this framework to train almost any model on any dataset using just a few commands.