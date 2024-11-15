import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from gan_model import build_generator, build_discriminator

# Load preprocessed images
data_dir = 'data/raw_images/'
images = preprocess_data(data_dir)

# Set hyperparameters
latent_dim = 100
batch_size = 64
epochs = 10000

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Compile the discriminator
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy",
                      metrics=["accuracy"])

# Compile the GAN (combining generator and discriminator)
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss="binary_crossentropy")

# Training loop
for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, images.shape[0], batch_size)
    real_images = images[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generated_images = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

    # Save generated images periodically
    if epoch % 1000 == 0:
        noise = np.random.normal(0, 1, (10, latent_dim))
        generated_images = generator.predict(noise)
        for i, img in enumerate(generated_images):
            img = (img * 255).astype(np.uint8)
            file_path = f"output/generated_images/gen_{epoch}_{i}.png"
            tf.keras.preprocessing.image.save_img(file_path, img)