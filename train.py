import os
from argparse import ArgumentParser
import tensorflow as tf
from model import MLPMixer
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--train-folder", default='{}/data/train'.format(home_dir), type=str)
    parser.add_argument("--valid-folder", default='{}/data/validation'.format(home_dir), type=str)
    parser.add_argument("--model-folder", default='{}/model/mlp/'.format(home_dir), type=str)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--dc", default=2048, type=int, help='Token-mixing units')
    parser.add_argument("--ds", default=256, type=int, help='Channel-mixing units')
    parser.add_argument("--c", default=512, type=int, help='Projection units')
    parser.add_argument("--image-size", default=150, type=int)
    parser.add_argument("--patch-size", default=5, type=int)
    parser.add_argument("--num-of-mlp-blocks", default=8, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--validation-split", default=0.2, type=float)
    parser.add_argument("--image-channels", default=3, type=int)

    args = parser.parse_args()

    print('---------------------Welcome to ProtonX MLP Mixer-------------------')
    print('Github: bangoc123')
    print('Email: protonxai@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training MLP-Mixer model with hyper-params:')
    print('===========================')
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('===========================')



    train_folder = args.train_folder
    valid_folder = args.valid_folder

    # Load train images from folder
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_folder,
        subset="training",
        seed=123,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        validation_split = args.validation_split,
        batch_size=args.batch_size,
    )

    # Load Validation images from folder
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_folder,
        subset="validation",
        seed=123,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        validation_split = args.validation_split,
        batch_size= args.batch_size,
    )

    assert args.image_size * args.image_size % ( args.patch_size * args.patch_size) == 0, 'Make sure that image-size is divisible by patch-size'
    assert args.image_channels == 3, 'Unfortunately, model accepts jpg images with 3 channels so far'
    
    S = (args.image_size * args.image_size) // (args.patch_size * args.patch_size)
    # C = args.patch_size * args.patch_size * args.image_channels

    
    
    # Initializing model
    mlpmixer = MLPMixer(args.patch_size, S, args.c, args.ds, args.dc, args.num_of_mlp_blocks, args.image_size, args.batch_size, args.num_classes)

    # Set up loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Optimizer Definition
    adam = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Compile optimizer and loss function into model
    mlpmixer.compile(optimizer=adam, loss=loss_object, metrics=['acc'])


    # Do Training model
    mlpmixer.fit(train_ds, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        validation_data=val_ds,
    )

    # Saving model
    mlpmixer.save(args.model_folder)
    

