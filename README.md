## Stable Diffusion in pure MLX

Features that you might not find at other places and were reasonably hard to implement:
- Loading ckpt files from civitai models
- LORA loading and strength adjustment
- Text to Image and Image to Image

### Motivation
Basically a faster and more customizable way to run SD 1.5 models on M1Pro Macbook. Rather than run it on Metal accelrated Pytorch (this was inefficient) I used some references listed below to roll my own in MLX. The official repo [mlx-examples](https://github.com/ml-explore/mlx-examples) does not have an implementation of SD1.5 which is a shame since it is quite literally the most flexible diffusion model till date.

This was also a learning opportunity for me since I am a total noob to diffusion models in general, coming from a GAN/ConvNet background it is extremely counter intuitive to how I'm used to doing things.

### Gotchas
A few gotchas that might not be immediately evident are listed below:
1. `nn.GroupNorm(groups, dims, pytorch_compatible=True)` must be marked pytorch_compatible else they work on some totally weird implementation
2. Channels last (this effed me up so many times) for all Convolution layers means weights and input tensors need to be permuted. This also results in a non 1 to 1 translation of pytorch code, which is annoying to say the least, but is still workable for anyone that's tried interoprating between TF/Keras and Pytorch
3. Random Number generation does not follow the pytorch sense, and needs a couple of different methods to set a global PRNG seed and then work from there. I like Pytorch's api better for this, even though MLX seems to be lower level.

### Speed
Here are my __non scientific__ benchmarks. In a sort of practical application I think there's a _rule of tenths_ that applies. CPU = 10x MLX = 10x CUDA roughly in terms of inference times. However, having 

### Text to Image
| Device               | Steps | Inference Time |
|----------------------|-------|----------------|
| PyTorch CPU (M1 Pro) | 50    | 645.2s         |
| PyTorch CUDA (T4)    | 50    | 8.8s           |
| MLX (M1 Pro)         | 50    | 150.1s         |

### Image to Image
| Device               | Steps | Inference Time |
|----------------------|-------|----------------|
| PyTorch CPU (M1 Pro) | 10    | 145.3s         |
| PyTorch CUDA (T4)    | 10    | 2.8s           |
| MLX (M1 Pro)         | 10    | 30.1s          |


### Details of License
```
While the code provided is MIT licensed the models and the original Stable Diffusion 1.5 are licensed under CreativeML OpenRAIL-M license. Also the test images provided are a combination of my own artwork and AI generated refinements. I do not claim any responsibility for how you are to use this code for any harmful or libelous image generation, and it is simply provided in good faith for research and learning purposes. The intended uses of this code are listed below:
+ The code is intended for research purposes only.
+ To understand safe deployment of models which have the potential to generate harmful content.
+ For probing and understanding the limitations and biases of generative models.
+ For generation of artworks and use in design and other artistic processes.
+ For applications in educational or creative tools.
+ Aiding research on generative models.

```

### References
+ https://github.com/hkproj/pytorch-stable-diffusion
+ https://github.com/kjsman/stable-diffusion-pytorch
