# Neural Doodle

The aim of this project is to implement a generative neural network to turn doodles into fine artworks. This is done by implementing a Semantic Style Transfer. A much more in-depth discussion of this project can be found in `pytorch-neural-doodle/docs`.

The original paper about the Semantic Style Transfer can be found at https://arxiv.org/abs/1603.01768. It is based on the Neural Patches algorithm which can be found at https://arxiv.org/abs/1601.04589.

This project is a research project created in fulfillment of requirements for the course "Advanced Machine Learning" at Heidelberg University in the summer semester 2018.

## Project Structure

```
.
├─ data/
│  └─ samples/                  <- sample images
├─ docs/                        <- project documentation
├─ models/                      <- pre-trained weights and frozen models
│  └─ vgg_19_imagenet/          <- pre-trained VGG 19 weights
├─ run/                         <- run configurations and saved checkpoints
│  └─ run_*/                       created by src/train.py
├─ src/
│  ├─ models/                   <- model implementation
│  ├─ utils/                    <- utility functions and classes
│  │  └─ common/
│  ├─ extract_vgg_19_weights.py <- extract pre-trained VGG 19 weights
│  └─ generate.py               <- generate a new image
├─ LICENSE.md
└─ README.md
```

## Getting started

To get started, download the pre-trained [VGG 19 weights](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) and extract the file `vgg_19.ckpt` to `pytorch-neural-doodle/models/vgg_19_imagenet`. To extract the minimum weights and biases needed for this project, run the Python script `extract_vgg_19_weights.py`. The compatible file `pytorch-neural-doodle/models/vgg_19_imagenet/vgg_19.minimum.pkl` will be created.

## Dependencies

The project was compiled using the following packages:
- *tbd.*

## LICENSE

All Python code in this repository is available under the MIT license.
