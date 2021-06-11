# lumbosacral segmentation
Pipeline for training new models with deepseg_sc

- [Pre-processing](#pre-processing)
- [Fine-tuning](#fine-tuning)
- [Prediction](#prediction)
- [Contributors](#contributors)
- [License](#license)

The pipeline presented here has been created in the context of a Master thesis at Ecole Polytechnique de Louvain. The subject of the thesis is "Automatic segmentation of the lumbosacral spinal cord". The pipeline is greatly inspired by an [existing github repository](https://github.com/sct-pipeline/deepseg-training) treating a similar subject (retraining for lesion detection). 

`sct_deepseg_sc` is a deep learning based function that automatically segments spinal cord. For more information please see the [article](https://arxiv.org/pdf/1805.06349.pdf). The models distributed with Spinal Cord Toolbox (SCT) were trained on a large dataset (more than 1000 MR images). Unfortunately the majority of those images does not cover the entire spinal cord. The models may thus have difficulties to correctly segment the end of the spinal cord (the lumbosacral part).

The role of this pipeline is to fine-tune the existing deepseg segmentation model with lumbosacral MR images such that we obtain a more robust model for the lumbosacral spinal cord segmentation. The pipeline could be used for other kind of fine-tuning with appropriate training data.

Our pipeline uses a modified version of the Spinal Cord Toolbox such that it work on Python 3.7 with Tensorflow 2.2 and Keras 2.4.3. In addition there are modifications in `sct_deepseg_sc` such that it can use our fine-tuned model. The modification done to the SCT concerning the adaptation to Python 3.7 can be found [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3361/files) with very slightly modification in `requirements.txt` to use to good versions of Keras and Tensorflow.
The modification concerning the segmentation consists in the addition of a new boolean parameter `custom` which indicates wheter or not the segmentation need to use the fine-tuned model. Do not use the custom option until you have fine-tuned a model on your own it will not work.

This adapted version of the SCT can be installed via the folder `sct_custom`. Note that the README file of the SCT has NOT been adapted. Installation instruction remains exactly the same (run the command `./install_sct` when you are in your `sct_custom` folder). Errors or warnings may occurs  during installation but it will not be an issue for the good behavior of the fine-tuning pipeline. Note that this version of SCT is only designed for this pipeline and should not be used for anything else. 

To get started, you need to have data that consists of input image and its corresponding segmentation masks, both in NIFTI format.

Before the preprocessing the data (MR images and masks) needs to be organized in as shown below :

~~~
data_mask/nii/Sub001_contrast001.nii
data_mask/nii/Sub002_contrast002.nii
.
.
.
data_mask/nii/Sub00N_contrast00N.nii

data_mask/mask/Sub001_contrast001_seg.nii
data_mask/mask/Sub002_contrast002_seg.nii
.
.
.
data_mask/mask/Sub00N_contrast00N_seg.nii

~~~


## Pre-processing

It is believed that instead of trying to segment in the whole image, narrowing our search area in small zone the spinal cord (SC) will improve the segmentation. To do that, in simple words, first we need to find the centerline of the spinal cord from input image using SC detection algorithm and crop the input image around SC then using segmentation algorithm to detect the spinal cord.

So, we need 1) SC detection model 2) cropping the image around SC 3) segmentation model.

This is explained in the paper Gros et al, 2018 (https://arxiv.org/pdf/1805.06349.pdf).

The step-by-step procedure is described in [Preprocessing_script.ipynb](https://github.com/nidebroux/lumbosacral_segmentation/tree/master/scripts/Preprocessing_script.ipynb).

The following points resume the procedure.

Since, the orientation of the images could vary across datasets/centers, we need to systematically set the orientation of the input image and mask to Right-Left, Posterior-Anterior, Inferior-Superior (RPI):

~~~
sct_image -i IMAGE -set-orient RPI
~~~

Then, resolution should be set to 0.5mm isotropic for all images masks:
~~~
sct_resample -i IMAGE -mm 0.5x0.5
~~~

In order to have the best centerline detection we use the segmentation mask. By computing the center of mass on each slice of the segmentation we obtain the best centerline detection possible.

~~~
sct_get_centerline -i MASK
~~~

Next step consists in cropping the resampled image and mask around the spinal cord centerline.


Later we standardize the intensities of the cropped image such that similar intensities will have similar tissue meaning.


The prepocessing will finally arrange the data such that it can be used in the fine-tuning pipeline.

## Fine-tuning
Files that are necessary for fine-tuning are:
- [config_file.py](https://github.com/nidebroux/lumbosacral_segmentation/tree/master/scripts/config_file.py): Global parameters. They need to be changed according to the need.
- [generator.py](https://github.com/nidebroux/lumbosacral_segmentation/tree/master/scripts/generator.py): Define data augmentation (e.g., flipping, distorting).
- [utils.py](https://github.com/nidebroux/lumbosacral_segmentation/tree/master/scripts/utils.py): Collection of functions that are called in the main script. E.g.: extracting 3D patches.
- [Main_file.ipynb](https://github.com/nidebroux/lumbosacral_segmentation/tree/master/scripts/Main_file.ipynb): Notebook to fine-tune the network with new dataset. It contains also a serie of tests in order to find the best parameters possible for the fine-tuning.


## Prediction

The last script stands simply for the segmentation of data not used during the training with our new fine-tuned model.

The script can be found in [Custom_deepseg.ipynb](https://github.com/nidebroux/lumbosacral_segmentation/tree/master/scripts/Custom_deepseg.ipynb)

## Contributors
This project has been developed by Nikita de Broux during his Master Thesis.

It is greatly inspired from an existing github repository. The contributors of it can be found [here](https://github.com/sct-pipeline/deepseg-training/graphs/contributors).



## License

The MIT License (MIT)

Copyright (c) 2018 École Polytechnique, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
