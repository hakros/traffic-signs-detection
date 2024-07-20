import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    # Store np_images and labels in a list
    images = []
    labels = []

    # Path to the data directory
    path = os.path.join(data_dir)

    # Go through each sub directory of data_dir
    # Each sub directory will be assumed as a category
    for category in os.scandir(path):
        category_path = os.path.join(data_dir, category.name)

        if not os.path.isdir(category_path):
            continue

        # Go through each file in the sub directory
        # Each file will be assumed as an image
        for file in os.scandir(category_path):
            # Retrieve real path to the file
            file_path = os.path.join(data_dir, category.name, file.name)

            # Makes keeping track of where we are easier
            print(f"Reading {file_path}")

            # Read the file using cv2
            # Assume that it is an image
            image = cv2.imread(file_path)

            # Resize the image to the following dimensions:
            # IMG_WIDTH x IMG_HEIGHT x 3
            resized = cv2.resize(
                src=image,
                dsize=[IMG_WIDTH, IMG_HEIGHT]
            )

            # Convert the resized image into a numpy ndarray
            np_image = np.array(
                object=resized
            )

            # Add the file and the category it belongs to our tuple (images, labels)
            images.append(np_image)
            labels.append(category.name)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Input Image -> [Convolutional Layer] -> [Activation Layer] -> [Pooling Layer] -> [Fully Connected or Dense Layer] -> [Output]

    keras = tf.keras

    input_layer = keras.layers.Input(
        shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )

    x = input_layer

    # Keep count of how many convulution layers we added
    convulutionNum = 0

    # It seems 3 convulution layers, 3 activation layers,
    # and 2 pooling layers gets us closest to 100% with
    # an efficient 6ms/step
    convulutionLimit = 3

    while convulutionNum < convulutionLimit:
        # We only add a pooling layer if the previous layer is a convulution/activation layer
        if convulutionNum > 0:
            # We use Pooling to reduce the dimension of the image
            # Thus making the model more efficient but less accurate
            # Max Value Pooling using a pool size of 2x2
            # seems to give the best result for efficiency and accuracy
            x = keras.layers.MaxPooling2D(
                pool_size=(2, 2)
            )(x)

        # Add convolutional layers
        # We use Conv2D since Conv3d requires the input _layer to have 5 units
        # We add an activation layer, "ReLU"
        # For a NUM_CATEGORIES = 43, it seems 32 to 64 filters
        # in a 3x3 kernel is the closest I got to 100% accuracy
        filterNum = 64 if convulutionNum > 0 else 32
        x = keras.layers.Conv2D(filterNum, (3, 3), activation="relu")(x)

        convulutionNum += 1

    # Flatten the result so we can pass it to a Dense Layer
    x = keras.layers.Flatten()(x)

    # For a classification task such as identifying whitch traffic sign
    # appears in a picture, a Dense layer using the activation softmax
    # is a good choice. It allows each of the 43 units to output a value
    # representing the score or probability for one of the 43 categories.
    # It then normalizes those scores into probabilities that sums up to 1
    # Can also be called Fully Connected Layer
    output_layer = keras.layers.Dense(NUM_CATEGORIES, activation="softmax")(x)

    # Create a model instance using the defined input and output layer
    model = keras.models.Model(
        inputs=input_layer,
        outputs=output_layer
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # 0.001

        # We use CategoricalCrossentropy since we are classifying 43 categories
        # Making this a multi-class classification problem
        loss=keras.losses.CategoricalCrossentropy(),

        # For multi-class classification, we use CategoricalAccuracy
        metrics=[
            keras.metrics.CategoricalAccuracy(),
        ],
    )

    return model


if __name__ == "__main__":
    main()
