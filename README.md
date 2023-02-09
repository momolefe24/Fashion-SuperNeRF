# Fashion Neural Super-Resolution Radiance Field for Virtual Try-On 
Fashion-NeRF attempts to reconstruct virtual try-on with a novel view using neural radiance field to emulate a smart mirror. The paper's contributions are 

- Novel model to synthesize novel virtual try-on views
- Using super-resolution as a preprocessing step to 
  - Speed up process of synthesizing a novel view using a neural radiance field by decreasing the amount of rays marched through scene
  - Prepare outputs of a novel view to synthesize clothing
- Novel dataset to accomplish the task of synthesizing novel fashion views 

## Table of Contents
## Contents
- [Tasks]()
  - [Neural Radiance Fields]()
  - [Super-Resolution]()
  - [Virtual Try-On Network]()
- [Methodology]()
  - [Super-Resolution]()
    - [Pretrained SRCNN]()
    - [Pretrained SRGAN]()
    - [Pretrained ESRGAN]()
  - [Neural Radiance Fields]()
    - [Simple NeRF]()
  - [One-Stage Network]()
  - [Two-Stage Network]()
- [Progress]()
- [Contributions]()
- [Results]()
  
## Tasks 
### Neural Radiance Field
Neural radiance field synthesize novel views of an object by encoding the geometry and pose of an object. It marches rays through an image plane and predicts the pixels in the world coordinate system. Given the ground truth, the network is trained through a photometric loss to train the network to predict the color and density of a ray.
### Super-Resolution
Image super-resolution describes the task of enhancing the spatial dimensions of an image while maintaining the perceptual quality.
In this paper, super-resolution serves as a preprocessing task for the virtual try-on network such that it prepares the output of the NeRF to fit as the input of the virtual try-on network


### Virtual Try-On Network

## Methodology
Different approaches of solving the problem 
### Super-Resolution

```bash
python3 sr.py -f experiment/experiment_01_run_01
```
#### Pretrained SRCNN

> **branch:** <a href="#">pretrain_srcnn</a>
 
**Goal:** 

#### Pretrained SRGAN
> **branch:** <a href="#">pretrain_srgan</a>
 
**Goal:** 
#### Pretrained ESRGAN
> **branch:** <a href="#">pretrain_sr</a>


#### Pretrained ESRGAN With Gradient Accumulation
> **branch:** <a href="#">pretrain_sr_grad_acc</a>

#### Pretrained ESRGAN With Bicubic Downsampling
> **branch:** <a href="#">pretrain_sr_bicubic</a>


**Goal:** Decrease number of convolutions by replacing them with bicubic downsampling
### Neural Radiance Fields
#### NeRF
> **branch:** <a href="#">nerf_kwea123</a>
#### Simple NeRF
> **branch:** <a href="#">simple_nerf</a>
#### Neural Radiance Field: Pytorch3D
> **branch:** <a href="#">simple_nerf_pytorch</a>
#### Neural Radiance Field: Pretrained Super-Resolution
> **branch:** <a href="#">nerf_pretrain_sr</a>
#### NerF++
> **branch:** <a href="#">nerf++</a>
#### Plenoxels: Radiance Fields Without Neural Networks
> **branch:** <a href="#">plenoxels</a>




### One-Stage Network:
#### Neural Radiance Super-Resolution Field
> **branch:** <a href="#">ne-surf</a>
**Goal:** 

### Two-Stage Network:
#### Neural Super-Resolution Field
> **branch:** <a href="#">ne-surf-two-stage</a>
**Goal:** 
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
