import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.datasets import cifar10  # Możesz również użyć innego zbioru danych

# Załaduj dane
(train_images, _), (test_images, _) = cifar10.load_data()


train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
# Przygotuj dane
train_images_gray = tf.image.rgb_to_grayscale(train_images).numpy() 
test_images_gray = tf.image.rgb_to_grayscale(test_images).numpy() 



# Model Autoenkodera
def build_autoencoder(input_shape):
    model = models.Sequential()

    # Koder
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    # Dekoder
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    return model

# Skonfiguruj model
input_shape = train_images_gray.shape[1:]
autoencoder = build_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mse')

# Wytrenuj model
autoencoder.fit(train_images_gray, train_images, epochs=10, batch_size=128, shuffle=True, validation_data=(test_images_gray, test_images))

# Przetestuj model
decoded_images = autoencoder.predict(test_images_gray)

# Wyświetl przykładowe wyniki
import matplotlib.pyplot as plt

n = 10  # liczba przykładów do wyświetlenia
plt.figure(figsize=(20, 4))
for i in range(n):
    # Wyświetl oryginalny obraz
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_images[i])
    plt.title("Oryginalny")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Wyświetl zdekodowany obraz
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i])
    plt.title("Zdekodowany")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
