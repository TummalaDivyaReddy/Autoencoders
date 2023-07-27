# Autoencoders
code file: [Autoencoders](Autoencoders.ipynb)

An autoencoder is a type of neural network that learns to reconstruct its input. It does this by first encoding the input into a compressed representation, and then decoding the compressed representation back into the original input. The compressed representation is typically much smaller than the original input, which allows the autoencoder to learn the most important features of the data.

## Role of Autoencoders in Unsupervised Learning
Autoencoders are a powerful tool for unsupervised learning. They can be used to learn the latent features of data, which can then be used for a variety of tasks, such as:

Dimensionality reduction: Autoencoders can be used to reduce the dimensionality of data while preserving the most important features. This can be useful for visualization, compression, and other tasks.

Feature extraction: Autoencoders can be used to extract features from data. These features can then be used for other tasks, such as classification or clustering.

Denoising: Autoencoders can be used to denoise data. This can be useful for removing noise from images or audio recordings.

## Types of Autoencoders
There are two main types of autoencoders:

Simple autoencoders: Simple autoencoders have a single hidden layer. They are the most basic type of autoencoder, but they can still be very effective for learning the latent features of data.

Stacked autoencoders: Stacked autoencoders have multiple hidden layers. They are more complex than simple autoencoders, but they can also learn more complex features.


## Add one more hidden layer to the autoencoder and  Do the prediction on the test data and then visualize one of the reconstructed versions of that test data. Also, visualize the same test data before reconstruction using Matplotlib

The first few lines import the necessary libraries for this code and The next few lines define the size of the encoded representations and the input placeholder.The encoding_dim variable specifies the number of features in the encoded representation. The input_img variable creates a placeholder for the input image, which is a 784-dimensional vector and then defines the encoded and decoded representations.The encoded variable creates a 32-dimensional vector from the input image. The decoded variable then reconstructs the input image from the encoded representation and we define the autoencoder model and compile it and The autoencoder variable creates a model that maps an input image to its reconstruction. The model is compiled using the adadelta optimizer and the binary_crossentropy loss function.
The next few lines load the MNIST dataset and prepare it for training.
The fashion_mnist.load_data() function loads the MNIST dataset, which contains 60,000 training images and 10,000 test images. The images are then normalized to be between 0 and 1. The training and test data are then reshaped to be 784-dimensional vectors.
The next few lines train the autoencoder model.
The autoencoder.fit() function trains the autoencoder model for 5 epochs, using a batch size of 256. The validation data is used to evaluate the model's performance.
The next few lines make a prediction and plot the input and reconstructed images,the prediction variable makes a prediction on the 6th test image. The plt.imshow() function then plots the input image and the reconstructed image.

```ruby
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

from keras.datasets import mnist, fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
# prediction
prediction = autoencoder.predict(x_test[[6],:])

from matplotlib import pyplot as plt
# Input Image
plt.imshow(x_test[1].reshape(28,28))
plt.show()
# Reconstructed Image
plt.imshow(prediction[0].reshape(28,28))
plt.show()

```


## Do the prediction on the test data and then visualize one of the reconstructed versions of that test data. Also, visualize the same test data before reconstruction using Matplotlib on the denoising autoencoder and plot loss and accuracy using the history object.

It imports the required libraries and modules and Defines the architecture of the autoencoder with an input layer, encoded layer, and decoded layer using the functional API of Keras and then Compiles the autoencoder model with 'adadelta' optimizer and 'binary_crossentropy' loss. Loads and preprocesses the Fashion MNIST dataset and then Trains the autoencoder on clean data for 5 epochs and stores the training history and Adds noise to the training data to create noisy versions and Retrains the autoencoder on the noisy data for 10 epochs and stores the training history for comparison and Visualizes the input image, noisy image, and the reconstructed image from the autoencoder and at last Plots the training history of accuracy and loss for both training processes (clean and noisy data).

```ruby
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import fashion_mnist
import numpy as np

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Fit the model on clean data
history_clean = autoencoder.fit(x_train, x_train,
                                epochs=5,
                                batch_size=256,
                                shuffle=True,
                                validation_data=(x_test, x_test))

# Introducing noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Fit the model on noisy data
history_noisy = autoencoder.fit(x_train_noisy, x_train,
                                epochs=10,
                                batch_size=256,
                                shuffle=True,
                                validation_data=(x_test_noisy, x_test_noisy))

# Prediction
prediction1 = autoencoder.predict(x_test_noisy[[15], :])

# Input Image
plt.imshow(x_test[9].reshape(28, 28))
plt.show()
# After applying noise to data
plt.imshow(x_test_noisy[15].reshape(28, 28))
plt.show()

# Reconstructed Image
plt.imshow(prediction1[0].reshape(28, 28))
plt.show()

# Plot the loss and accuracy for both training processes
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(history_clean.history['accuracy'])
plt.plot(history_noisy.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.legend(['Clean Data', 'Noisy Data'], loc='upper left')

plt.subplot(122)
plt.plot(history_clean.history['loss'])
plt.plot(history_noisy.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.legend(['Clean Data', 'Noisy Data'], loc='upper left')

plt.tight_layout()
plt.show()


```

## Using stacked autoencoders

It imports the required libraries and modules and then the Fashion MNIST dataset is loaded and normalized to values between 0 and 1.
and the data is reshaped to a 1D array to be used as input to the neural network.
and we define the stacked autoencoder is defined using Keras' functional API,the Two encoder layers with 128 and 64 units are created, respectively, to compress the input data and the encoded representation is a layer with 32 units, resulting in data compression and two decoder layers with 64 and 128 units are created to reconstruct the data and the final decoder layer has 784 units, representing the output of the autoencoder.

The autoencoder model is compiled with the 'adadelta' optimizer and 'binary_crossentropy' loss function and the 'accuracy' metric is included, though not typically used for autoencoders.

The autoencoder is trained on the clean training data for 5 epochs and the training history is stored for further analysis and then Gaussian noise is added to the training data to create noisy versions.The autoencoder is trained on the noisy data for 10 epochs.The training history for noisy data is also stored.

Several images are displayed using plt.imshow() to visualize the input, noisy, and reconstructed images and the Two subplots are created to compare the model's accuracy and loss on both clean and noisy data and the plots of accuracy and loss are displayed using plt.show() and the code demonstrates how the stacked autoencoder can reconstruct clean and noisy images and provides insights into the model's performance during training.

```ruby
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import fashion_mnist
import numpy as np

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# Create the first encoder layer
input_img = Input(shape=(784,))
encoder_layer_1 = Dense(128, activation='relu')(input_img)  # 128 units in the first encoder layer
# Create the second encoder layer
encoder_layer_2 = Dense(64, activation='relu')(encoder_layer_1)  # 64 units in the second encoder layer
# Create the encoded representation
encoded = Dense(encoding_dim, activation='relu')(encoder_layer_2)

# Create the first decoder layer
decoder_layer_1 = Dense(64, activation='relu')(encoded)  # Decoder with 64 units
# Create the second decoder layer
decoder_layer_2 = Dense(128, activation='relu')(decoder_layer_1)  # Decoder with 128 units
# Create the final decoder layer
decoded = Dense(784, activation='sigmoid')(decoder_layer_2)  # Output layer with 784 units

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Fit the model on clean data
history_clean = autoencoder.fit(x_train, x_train,
                                epochs=5,
                                batch_size=256,
                                shuffle=True,
                                validation_data=(x_test, x_test))

# Introducing noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Fit the model on noisy data
history_noisy = autoencoder.fit(x_train_noisy, x_train,
                                epochs=10,
                                batch_size=256,
                                shuffle=True,
                                validation_data=(x_test_noisy, x_test_noisy))

# Prediction
prediction1 = autoencoder.predict(x_test_noisy[[15], :])

# Input Image
plt.imshow(x_test[9].reshape(28, 28))
plt.show()
# After applying noise to data
plt.imshow(x_test_noisy[15].reshape(28, 28))
plt.show()

# Reconstructed Image
plt.imshow(prediction1[0].reshape(28, 28))
plt.show()

# Plot the loss and accuracy for both training processes
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(history_clean.history['accuracy'])
plt.plot(history_noisy.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.legend(['Clean Data', 'Noisy Data'], loc='upper left')

plt.subplot(122)
plt.plot(history_clean.history['loss'])
plt.plot(history_noisy.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.legend(['Clean Data', 'Noisy Data'], loc='upper left')

plt.tight_layout()
plt.show()


```

Youtube video: [Autoencoders](https://youtu.be/4myRr3yvz9E)









