# Neural Super Radiance Field 
**Goal:** Neural radiance field model comprised of a super-resolution network 

# Table of Contents
- [Introduction](#introduction)
- [Strategy](#strategy)
- [Methodology](#methodology)
  - [Topology](#topology)
  - [Papers](#papers)
- [Results](#results)
  - [Experiment 1](#experiment_1)
    - [Description](#description)
    - [Images](#images)
    - [Metrics & Loss Curves](#metric-&-loss-curves)
    - [Issues & Problems](#issues-&-problems)
- [Contribution](#contributions)
- [Documentation](#documentation)
- [Citation](#citation)

# Introduction

The neural radiance field encodes a scene representation in its weights to synthesize novel views by learning to optimize the pixel loss between the volume renderered colour and density of a NeRF model. The dataset is transformed into cropped image pairs of (256,256) - (64,64) for effiecient training. Our strategy is to forward propagate the input image into the neural radiance field, and concatenate the result and its depth map into the super-resolution network and back propagate the content loss. The super-resolution network follows an Enhanced SRGAN

# Strategy

This branch crops the input image with a factor of 8 and concatenates the result of the NeRF with the convolution of a Super-Resolution network.

# Methodolgy
The neural super radiance field is trained using a multi-task learning framework where the network trains to learn two tasks
- encode weights of a scene representation in its by creating a novel view
- upsample a low-resolution to match a specific resolution of 1024x768

### How It Works
The Neural Super Radiance Field is one model such that we add upsampling convolutions to a NeRF model

- Given an image input of a 100x100, we would like to produce a 1024x768 image
  - The 1024x768 specification serves as an input to the Virtual Try-On Network
- The super-resolution model attaches itself to the result of the pixel colour and density prediction
  - i.e NeRF outputs a 100x100 and a SR generator enhances it to 1024x768
- Fdf

## Results
### Experiment 1 / Run 1



#### Images 
The NeSuRF model appears to converge at the 250th epoch such that it appears to learn to super-resolve a synthesized novel view
![convergence](https://github.com/momolefe24/Fashion-SuperNeRF/blob/nerf_sr_combined_model/Convergence.png?raw=true)

However, the model seems to diverge drastically after more training. This may have been caused by the discriminator
![divergence-1](https://github.com/momolefe24/Fashion-SuperNeRF/blob/nerf_sr_combined_model/Divergence.png?raw=true)
and after further training, the model completely forgets how to do its job
![divergence-2](https://github.com/momolefe24/Fashion-SuperNeRF/blob/nerf_sr_combined_model/Beginning%20of%20divergence.png?raw=true)

## Contribution
Contributions to the LNN codebase are welcome!

Please have a look at the [contribution guide](https://github.com/IBM/LNN/blob/master/CONTRIBUTING.md) for more information on how to set up the LNN for contributing and how to follow our development standards.

## Documentation
| [Read the Docs][Docs] | [Academic Papers][Papers]	| [Educational Resources][Education] | [Neuro-Symbolic AI][Neuro-Symbolic AI] | [API Overview][API] | [Python Module][Module] |
|:-----------------------:|:---------------------------:|:-----------------:|:----------:|:-------:|:-------:|
| [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/doc.png alt="Docs" width="60"/>][Docs] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/academic.png alt="Academic Papers" width="60"/>][Papers] |  [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/help.png alt="Getting Started" width="60"/>][Education] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/nsai.png alt="Neuro-Symbolic AI" width="60"/>][Neuro-Symbolic AI] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/api.png alt="API" width="60"/>][API] | [<img src=https://raw.githubusercontent.com/IBM/LNN/master/docsrc/images/icons/python.png alt="Python Module" width="60"/>][Module] |

## Citation
If you use Logical Neural Networks for research, please consider citing the
reference paper:
```raw
@article{riegel2020logical,
  title={Logical neural networks},
  author={Riegel, Ryan and Gray, Alexander and Luus, Francois and Khan, Naweed and Makondo, Ndivhuwo and Akhalwaya, Ismail Yunus and Qian, Haifeng and Fagin, Ronald and Barahona, Francisco and Sharma, Udit and others},
  journal={arXiv preprint arXiv:2006.13155},
  year={2020}
}
```

