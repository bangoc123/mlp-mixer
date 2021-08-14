## MLP Mixer

Implementation for paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf). Give us a star if you like this repo.

Run it on colab:

<a href="https://colab.research.google.com/drive/1CfUq7vGV_ZHvn28qSfoOnAFqUbZe8mCR?usp=sharing"><img src="https://storage.googleapis.com/protonx-cloud-storage/colab_favicon_256px.png" width=80> </a>

Author:
- Github: bangoc123
- Email: protonxai@gmail.com

This library belongs to our project: [Papers-Videos-Code](https://docs.google.com/document/d/1bjmwsYFafizRXlZyJFazd5Jcr3tqpWSiHLvfllWRQBc/edit?usp=sharing) where we will implement AI SOTA papers and publish all source code. Additionally, videos to explain these models will be uploaded to ProtonX Youtube channels.

![image](https://storage.googleapis.com/protonx-cloud-storage/Capture3.PNG)

<!-- <img src=./images/architecture.png width=400> -->

**[Note] You can use your data to train this model.**

### I. Set up environment

1. Make sure you have installed Miniconda. If not yet, see the setup document [here](https://conda.io/en/latest/user-guide/install/index.html#regular-installation).

2. `cd` into `mlp-mixer` and use command line `conda env create -f environment.yml` to setup the environment

3. Run conda environment using the command `conda activate mlp-mixer`

### II. Set up your dataset.

Create 2 folders `train` and `validation` in the `data` folder (which was created already). Then `Please copy` your images with the corresponding names into these folders.

- `train` folder was used for the training process
- `validation` folder was used for validating training result after each epoch 

This library use `image_dataset_from_directory` API from `Tensorflow 2.0` to load images. Make sure you have some understanding of how it works via [its document](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory).

Structure of these folders.

```
train/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

```
validation/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
...class_c/
......c_image_1.jpg
......c_image_2.jpg
```

### III. Train your model by running this command line

```bash
python train.py --epochs ${epochs} --num-classes ${num_classes}
```

You want to train a model in 10 epochs for binary classification problems (with 2 classes)

Example: 

```bash
python train.py --epochs 10 --num-classes 2
```

There are some `important` arguments for the script you should consider when running it:

-  `train-folder`: The folder of training images
-  `valid-folder`: The folder of validation images
-  `model-folder`: Where the model after training saved
-  `num-classes`: The number of your problem classes.  
- `batch-size`: The batch size of the dataset
- `c`: Patch Projection Dimension
- `ds`: Token-mixing units. It was mentioned in the paper on [page 3](https://arxiv.org/pdf/2105.01601.pdf)
- `dc`: Channel-mixing units. It was mentioned in the paper on [page 3](https://arxiv.org/pdf/2105.01601.pdf)
- `num-of-mlp-blocks`: The number of MLP Blocks
- `learning-rate`: The learning rate of Adam Optimizer

After training successfully, your model will be saved to `model-folder` defined before

### IV. Testing model with a new image

We offer a script for testing a model using a new image via a command line:

```bash
python predict.py --test-file-path ${test_file_path}
```

where `test_file_path` is the path of your test image.

Example:

```bash
python predict.py --test-file-path ./data/test/cat.2000.jpg
```

### V. Feedback

If you meet any issues  when using this library, please let us know via the issues submission tab.


