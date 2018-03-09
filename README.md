# Project 3: Neuron Finding

### Team Elders

## Neuron Finding

The Project can best be described as object-finding or image segmentation, where your goal is to design a model whose output is the coordinates to regions of interest in an image.There’s no discrete label; rather, your model needs to learn segments in a continuous two-dimensional plane; relevant information to learning these segments, however, may be strewn over a third dimension of time. This makes for a very high-dimensional, largescale

## The Problem
The data are height-by-width-by-time, and your model needs to learn a height-by-width mapping of pixels, where each pixel is either part of a neuron, or isn’t. Each folder of training and testing images is a single plane, and the images are numbered according to their temporal ordering. The neurons in the images will “flicker” on and off, as calcium (Ca2+) is added, activating the action potential gates. You’ll have to use this information in order to locate the neurons and segment them out from the surrounding

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

#### Data
Each folder contains a variable number of images; sample 00.00 contains 3,024 images, while sample 00.01 contains 3,048. The image files themselves are numbered, e.g. image00000.tiff, but all the images in a single folder represent the same sample, just taken at different times with different calcium levels. The training labels exist at the sample level, so you’ll use all the images in a single folder to learn the locations of the neurons. Each folder will have a unique sample with unique numbers and positions of neurons. However, while time is a dimension to the data, you may not need to explicitly model it; we are just interested in finding the active neurons in space

The image on left represents more or less what you’ll receive in the training and testing data. The image on right is the goal of your learner i.e draw circles around the regions that contain neurons.

<img src="https://github.com/dsp-uga/Elders/blob/master/Examples/Train.JPG" width="400"> <img src="https://github.com/dsp-uga/Elders/blob/master/Examples/output.JPG" width="400">

### Installing

Creating an environment from an ```environment.yml``` file:

Use the Terminal or an Anaconda Prompt for the following steps. Create the environment from the environment.yml file:
```
   conda env create -f environment.yml
```
Activate the new environment:
     
 * Windows: ```activate myenv```
     
 * macOS and Linux: ```source activate myenv```

NOTE: Replace myenv with the name of the environment.

Verify that the new environment was installed correctly:
```
   conda list
```
Below packages need to be installed on GCP:

`apt-get install python-pip`

`sudo pip install pyspark`

`sudo pip install thunder`

`sudo pip install thunder-extraction`

`sudo pip install thunder-factorization`

## Built With

* Pycharm 
* GCP 
* Atom

## Contributing

Please read [CONTRIBUTING.md](https://github.com/dsp-uga/Elders/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Nihal Soans** - [Nihalsoans91](https://github.com/nihalsoans91)
* **Ankit Vaghela** - [ankit-vaghela30](https://github.com/ankit-vaghela30)
* **Maulik Shah** - [mauliknshah](https://github.com/mauliknshah)


See also the list of [contributors](https://github.com/dsp-uga/Elders/blob/master/CONTRIBUTORS.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* 'houchaoqun' for frcn model
[link](https://github.com/Houchaoqun/keras_frcnn)
* 'Zizhao Zhang' for UNet architecture model
[link](https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py)
* 'Alex Klibisz' for 'Download dataset module' inspiration
[link](https://github.com/alexklibisz/deep-calcium)
