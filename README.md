# Enhanced Super-Resolution Generative Adversarial Network Without Penalty 
**Goal:** Enahc 



## Table of Contents
## Contents
- [Methodology]()
  - [Introduction](introduction)
  - [Topology]()
  - [Papers]()
- [Results](#results)
  - [Experiment 1]()
    - [Description]()
    - [Images](#images)
    - [Metrics & Loss Curves]()
    - [Issues & Problems]()
- [Contribution]()
- [Documentation]()
- [Citation]()

## Methodolgy
### Introduction 
GANs are a zero-sum network architecture that learns to mimic data distributions. For the task of image super-resolution, we use the _ESRGAN_ that introduces the residual-in residual dense blocks without normalization to perform the task of super-resolution.
### How It Works

The method uses a super-resolution resnet with residual blocks while removing all batch normalization layers to enhance the stability of training and consistent performance.


## Results

### Experiment 1 / Run 1
#### Images

#### Scalars


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

[Docs]: https://ibm.github.io/LNN/introduction.html
[Papers]: https://ibm.github.io/LNN/papers.html
[Education]: https://ibm.github.io/LNN/education/education.html
[API]: https://ibm.github.io/LNN/usage.html
[Module]: https://ibm.github.io/LNN/lnn/LNN.html
[Neuro-Symbolic AI]: https://research.ibm.com/teams/neuro-symbolic-ai
