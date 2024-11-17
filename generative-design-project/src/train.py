import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.optimizers import Adam
from gan_model import build_generator, build_discriminator, build_gan  # Import from gan_model.py

def load_images(image_directory, target_size=(128, 128)):
    images = []
    for filename in os.listdir(image_directory):
        img_path = os.path.join(image_directory, filename)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img = image.load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
    images = np.array(images)
    images = (images - 127.5) / 127.5
    return images

image_directory = 'C:/Users/Ahmadee/PycharmProjects/generative-design-project/data/raw_images'
images = load_images(image_directory)

latent_dim = 100
BATCH_SIZE = 32
EPOCHS = 10000
real_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')

def train_gan(generator, discriminator, gan, images, latent_dim, batch_size, epochs):
    for epoch in range(epochs):
        # Get a batch of real images
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images = images[idx]

        # Generate a batch of fake images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        # Labels for real and fake images
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # Average the losses

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)  # We want to fool the discriminator

        # Print the progress
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")