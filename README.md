# Introduction 

This is an implmention of the 2015 Whitepaper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597) on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) using PyTorch. 


## Depedencies 

1. Python 3.12, I would suggest using [pyenv](https://github.com/pyenv/pyenv) to manage your python versions
2. [UV](https://docs.astral.sh/uv/) is used as our package manager. 
3. A [Hugging Face Hub](https://huggingface.co/docs/hub/en/index) account is a nice to have if you plan up uploading models, but I don't know why would. 

## Set up

The only set up that is required is running 
```
uv sync
source .venv/bin/activate
```
if you are on Mac / Linux 
or 
```
uv sync
./venv/bin/activate.exe
```
if you are on mac. 
Then you should be able to run any of the commands / files in this repository. 

## Work Done
- A 1:1 Unet model train on black and white images with Pixel Accuracy of 88.90% and a Mean IoU of 0.6940 is available [here](https://huggingface.co/taidgh-robinson/UNet-Whitepaper/blob/main/full_model_adam)
- An almost 1:1 UNet model trained on color images with Pixel Accuracy of 89.56% and a Mean IoU of 0.7103 is available [here](https://huggingface.co/taidgh-robinson/UNet-Whitepaper/blob/main/full_model_adam_color_50_epoch)

## TODO

2. Some sort of CLI interface to load / test models instead of having to modify main.py directly. 
3. Gather presentation data like loss graphs.
4. Update model evaluation loop to get the image with got the best IoU score on and the worst IoU one. 