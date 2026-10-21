# -*- coding: utf-8 -*-
#In this Model i am using CaIIK and added feature machating losses
#so this is Pix2PixHD model

import tensorflow as tf
import h5py
import pathlib
import time
import os
import datetime
import numpy as np
import random
from astropy.io import fits
from matplotlib import pyplot as plt
from IPython import display


input_dir = "/scratch/rohanbmandrai.phy24.itbhu/rohan/train_new/input"
target_dir = "/scratch/rohanbmandrai.phy24.itbhu/rohan/train_new/target"
loss_dir = "/scratch/rohanbmandrai.phy24.itbhu/rohan/result_1/Loss_plot/PPM_HD_Cak_tar/"
os.makedirs(loss_dir, exist_ok=True)


# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 2
seed = 877
np.random.seed(seed)
tf.random.set_seed(seed)


# (cycle_number, start_year, start_month, end_year, end_month)
SOLAR_CYCLES = [
    (18, 1944,  2, 1954,  3),
    (19, 1954,  4, 1964,  9),
    (20, 1964, 10, 1976,  2),
    (21, 1976,  3, 1986,  8),
    (22, 1986,  9, 1996,  7),
    (23, 1996,  8, 2008, 11),
]


def get_solar_cycle(year, month):
    for cycle, sy, sm, ey, em in SOLAR_CYCLES:
        if (year > sy or (year == sy and month >= sm)) and (year < ey or (year == ey and month <= em)):
            return cycle
    return None


epochs= 20 #250
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256


all_images = sorted(os.listdir(input_dir))




train_files = []
val_files = []
test_files = []
'''
For training jan to june months are used 
For val: aug 
For testing  oct,nov
'''

train_months=[1,2,3,4,5]
val_months=[7,8]
test_months=[10,11]

for image in all_images:
    month = int(image[4:6])  # x represent month in image filename assuming YYYYMMDD...

    if month in train_months:
        train_files.append(image)

    elif month in val_months:
        val_files.append(image)

    elif month in test_months:
        test_files.append(image)


print("Train:", len(train_files))
print("Val  :", len(val_files))
print("Test :", len(test_files))
print("Total_used_data:", len(train_files) + len(val_files) + len(test_files))
print("All   :", len(all_images))





def augmentation(input1,target):

# --------Random North-south flip--------
  do_flip = tf.random.uniform([]) > 0.5

  input1 = tf.cond(do_flip,lambda: tf.image.flip_up_down(input1),lambda: input1)
  target = tf.cond(do_flip,lambda: -tf.image.flip_up_down(target),lambda: target)

  return input1, target


# Normalizing the images to [-1, 1]
def normalize(input1, target):

    # Replace NaNs with 0 temporarily
    input1 = tf.where(tf.math.is_nan(input1), tf.fill(tf.shape(input1), 0.0), input1)
    target  = tf.where(tf.math.is_nan(target),  tf.zeros_like(target),  target)

    # Input1 normalization
    input1 = tf.clip_by_value(input1, -1.0, 1.0)

    # Target normalization
    target = target / 750
    target = tf.clip_by_value(target, -1.0, 1.0)

    return input1, target



@tf.function()


def load(image_file):
    def _load_numpy(filename):
        filename = filename.numpy().decode("utf-8")  # convert from bytes
        
        
        input_path1 = os.path.join(input_dir, filename)           # input path for cak
        with fits.open(input_path1) as hdul:                      # file read for input path-cak
            input1_img = hdul[0].data   
    
        
        target_path = os.path.join(target_dir, filename)        # target path- magnetogram
        with fits.open(target_path) as hdul:                    # file read for target path-magnetogram
            target_img = hdul[0].data
            
        year  = int(filename[:4])
        month = int(filename[4:6]) 
        decimal_year = year + (month/12) 
        
        cycle = get_solar_cycle(year, month) 
        # --- Hale cycle correction ---
        if cycle is not None and cycle % 2 == 0:
            target_img = -target_img                 
            
        input_img1 = input1_img.astype(np.float32)[..., np.newaxis]
        target_img = target_img.astype(np.float32)[..., np.newaxis]

        return input_img1, target_img

    input_img1, target_img = tf.py_function(
        func=_load_numpy,
        inp=[image_file],
        Tout=[tf.float32, tf.float32]
    )

    input_img1.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
    target_img.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])

    return input_img1, target_img


def load_image_train(image_file):
  input_image1, target = load(image_file)
  input_image1, target = augmentation(input_image1, target)
  input_image1, target = normalize(input_image1, target)
  
  return input_image1, target

def load_image_val(image_file):
  input_image1, target = load(image_file)
  input_image1, target = normalize(input_image1, target)
  
  return input_image1, target

def load_image_test(image_file):
  input_image1, target = load(image_file)
  input_image1, target = normalize(input_image1, target)
 
  return input_image1, target

"""## Build an input pipeline with `tf.data`"""


train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)


test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
test_dataset = test_dataset.map(load_image_test,
                                  num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)


val_dataset = tf.data.Dataset.from_tensor_slices(val_files)
val_dataset = val_dataset.map(load_image_val,
                                  num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)



"""## Build the generator

"""

OUTPUT_CHANNELS = 1

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


"""Define the upsampler (decoder):"""

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

"""Define the generator with the downsampler and the upsampler:"""

def Generator():
  inputs = tf.keras.layers.Input(shape=[256,256, 1])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



generator = Generator()
  
  
### Define the generator loss

LAMBDA = 100.0
LAMBDA_FM = 10.0
#loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_object = tf.keras.losses.MeanSquaredError()

def generator_loss(disc_generated_output):
    return tf.reduce_mean(tf.square(disc_generated_output - 1.0))



"""The training procedure for the generator is as follows:
## Build the discriminator_A
Let's define the discriminator_A:
"""

def Discriminator_A():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  down4 = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 33, 33, 512)

  out = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=[out,down1,down2,down3,down4])

"""Visualize the discriminator_A model architecture:"""

discriminator_A = Discriminator_A()
"""
## Build the discriminator_B
Let's define the discriminator_B:
Discriminator_B reduce 
"""
def Discriminator_B():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
   
  x = tf.keras.layers.AveragePooling2D(
      pool_size=3,
      strides=2,
      padding='same')(x)                     # (batch_size, 128, 128, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 64, 64, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 32, 32, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 16, 16, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 18, 18, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 15, 15, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  down4 = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 17, 17, 512)

  out = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 14, 14, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=[out,down1,down2,down3,down4])

"""Visualize the discriminator_B model architecture:"""

discriminator_B = Discriminator_B()

"""### Define the discriminator loss
"""

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def feature_matching_loss(real_features, fake_features):
    fm_loss = 0
    for real_f, fake_f in zip(real_features, fake_features):
        fm_loss += tf.reduce_mean(tf.abs(real_f - fake_f))
    return fm_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = '/scratch/rohanbmandrai.phy24.itbhu/rohan/result_1/training_checkpoints/PPM_HD_Cak_tar'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator_A=discriminator_A,
                                 discriminator_B=discriminator_B)

"""## Generate images
"""

def generate_images(model,input1, test_input, tar, save_path):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [input1[0], tar[0], prediction[0]]
  title = ['Input Image','Ground Truth', 'Predicted Image']
  vmin = [-0.5,-0.5,-0.5]	
  vmax = [0.5,0.5,0.5]
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    
    #to plot.
    plt.imshow(display_list[i], cmap='gray', vmin=vmin[i],vmax=vmax[i],origin="lower")
     #plt.imshow(display_list[i], cmap='gray',origin="lower")
    plt.axis('off')
    
  # Save the output image to the specified path
  plt.tight_layout()
  plt.savefig(save_path)
  plt.close()  # Close the plot to release memory


log_dir="/scratch/rohanbmandrai.phy24.itbhu/rohan/result_1/logs/PPM_HD_Cak_tar/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_epoch(input1, target):
  
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    combined_input = input1
    gen_output = generator(combined_input, training=True)
    
    
	# -------- Discriminator A --------
    disc_real_output_A = discriminator_A([combined_input, target], training=True)
    disc_generated_output_A = discriminator_A([combined_input, gen_output], training=True)
    
    
	# -------- Discriminator B --------
    disc_real_output_B = discriminator_B([combined_input, target], training=True)
    disc_generated_output_B = discriminator_B([combined_input, gen_output], training=True)



	# -------- Generator loss --------
    # --- Generator adversarial loss (per discriminator) ---
    gen_gan_loss_A = generator_loss(disc_generated_output_A[0])
    gen_gan_loss_B = generator_loss(disc_generated_output_B[0])
    gen_gan_loss = gen_gan_loss_A + gen_gan_loss_B

    # --- L1 loss ---
    gen_l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    
    
    fm_loss_A = feature_matching_loss(disc_real_output_A[1:], disc_generated_output_A[1:])
    fm_loss_B = feature_matching_loss(disc_real_output_B[1:], disc_generated_output_B[1:])
    
  

    # --- Total generator loss ---
    gen_total_loss = (gen_gan_loss + LAMBDA * gen_l1_loss + LAMBDA_FM * (fm_loss_A + fm_loss_B) )

    
    
    	# -------- Discriminator loss --------
    disc_loss_A = discriminator_loss(disc_real_output_A[0], disc_generated_output_A[0])
    disc_loss_B = discriminator_loss(disc_real_output_B[0], disc_generated_output_B[0])
    disc_loss = disc_loss_A + disc_loss_B
    
    
  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
                                          
  disc_vars = (discriminator_A.trainable_variables +discriminator_B.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               disc_vars)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              disc_vars))
                                              
  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def fit_epochs(train_ds, test_ds,val_ds, epochs,model,early_stop_patience):

  save_dir="/scratch/rohanbmandrai.phy24.itbhu/rohan/result_1/checkpoints/PPM_HD_Cak_tar"
  output_dir = "/scratch/rohanbmandrai.phy24.itbhu/rohan/result_1/trained_output_imgs/PPM_HD_Cak_tar/"
  model_dir = "/scratch/rohanbmandrai.phy24.itbhu/rohan/result_1/Models/PPM_HD_Cak_tar/"
  
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(model_dir, exist_ok=True)
  
  
  example_input1, example_target = next(iter(test_ds.take(1)))
  example_input = example_input1
  
  
  # Initialize lists to track loss per epoch
  epoch_list = []
  gen_total_losses = []
  disc_losses = []
  mae_values = []
  global_step = 0
  
  # Validation history
  val_mae_history = []

  best_val_mae = np.inf
  best_model_path = os.path.join(model_dir, f"best_generator_{model}.keras")
  no_improve_count = 0


  # Helper: compute MAE on val_ds
  def val_compute(val_dataset):
    total_mae = 0.0
    batches = 0
    for v_inp1, v_tar in val_dataset:
      combined_val_inp = v_inp1
                  
      v_pred = generator(combined_val_inp, training=False) # predictions

      mae = tf.reduce_mean(tf.abs(v_tar - v_pred)).numpy()  # L1 / MAE;  
      total_mae += mae
      batches += 1
    if batches == 0:
      return None
    return (total_mae / batches)
  
  for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}", flush=True)
    epoch_start_time = time.time()
    temp_gen_losses = []
    temp_disc_losses = []
    temp_mae = []
  


    for input1, target in train_ds:
    # Train and get losses
      gen_total_loss, _, gen_l1_loss, disc_total_loss = train_epoch(input1, target)
      temp_gen_losses.append(gen_total_loss.numpy())
      temp_disc_losses.append(disc_total_loss.numpy())
      temp_mae.append(gen_l1_loss.numpy())
    
      global_step += 1
      

    # ---------- Epoch averages ----------
    avg_gen_loss = np.mean(temp_gen_losses) 
    avg_disc_loss = np.mean(temp_disc_losses) 
    avg_mae = np.mean(temp_mae) 
    
      
    # Save losses
    epoch_list.append(epoch+1)
    gen_total_losses.append(avg_gen_loss)
    disc_losses.append(avg_disc_loss)
    mae_values.append(avg_mae)
      
    # Generate and save a sample image
    if (epoch + 1) % 10 == 0:
      sample_op_path = os.path.join(output_dir, f"sample_output_step_{epoch+1}.png")
      example_input = input1
      generate_images(generator,input1, example_input, target, save_path=sample_op_path)
      # Save (checkpoint) the model
      checkpoint.save(file_prefix=checkpoint_prefix)  
    print(f"Train → Gen: {avg_gen_loss:.4f}, Disc: {avg_disc_loss:.4f}, MAE: {avg_mae:.5f}", flush=True)
    # Log to summary writer
    with summary_writer.as_default():
      tf.summary.scalar('Gen_loss', avg_gen_loss, step=epoch+1)
      tf.summary.scalar('disc_loss', avg_disc_loss, step=epoch+1)
      tf.summary.scalar('mae', avg_mae, step=epoch+1)
    summary_writer.flush()

      

    # ---------------VALIDATION ---------
    val_mae = val_compute(val_ds)
      
    if val_mae is not None:
      val_mae_history.append(val_mae)
      
      print(f"Val → MAE: {val_mae:.5f}", flush=True)

      # Log to summary writer
      with summary_writer.as_default():
        tf.summary.scalar('val_mae', val_mae, step=epoch+1)
      summary_writer.flush()
      # Save best model by val_mae
      if val_mae < best_val_mae:
        best_val_mae = val_mae
        generator.save(best_model_path)
        print(f"Saved new best generator to {best_model_path} (val_mae={best_val_mae:.6f}) ", flush=True)
        no_improve_count = 0
      else:
        no_improve_count += 1

      # Save model every 10 epochs
      if (epoch + 1) % 10 == 0:
        every_10_model_path = os.path.join(model_dir,f"generator_epoch_{epoch+1}.keras")
        generator.save(every_10_model_path)
        print(f"Saved generator at epoch {epoch+1}", flush=True)

        
          # Early stopping (optional)
      if early_stop_patience is not None and no_improve_count >= early_stop_patience:
        print(f"Early stopping triggered", flush=True)
        break
        
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} minutes", flush=True)
      
    if epoch != 0 and epoch % 50 == 0:
      print("This epoch is a multiple of 50")
      plt.figure(figsize=(10,6))
      ax1 = plt.gca()
      ax2 = ax1.twinx()

      #  Plot Gen Total Loss
      ax1.plot(epoch_list, gen_total_losses, color='blue', label='Generator Total Loss')
      ax1.set_ylabel("Generator Total Loss", color='blue')
      ax1.tick_params(axis='y', labelcolor='blue')
      ax1.grid(True)

      # --- Plot Disc Loss on second y-axis ---
      ax2.plot(epoch_list, disc_losses, color='red', label='Discriminator Loss')
      ax2.set_ylabel("Discriminator Loss", color='red')
      ax2.tick_params(axis='y', labelcolor='red')
  
      # X-axis label & Title
      plt.xlabel("Training Epochs")
      plt.title("Generator vs Discriminator Loss")

      # --- Combined legend ---
      lines1, labels1 = ax1.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

      plt.tight_layout()  
      plot_path = os.path.join(loss_dir, f"Dis_Gen_Loss.png")
      plt.savefig(plot_path)
      plt.close()
  

  # Save final model (and best model is saved earlier)
  final_model_path = os.path.join(model_dir, f"model_{model}.keras")
  generator.save(final_model_path)
  print("Saved final generator to:", final_model_path ,flush=True)
  

  
  # Return histories
  histories = {
      "epoch_list": epoch_list,
      "gen_total_losses": gen_total_losses,
      "disc_losses": disc_losses,
      "mae_values": mae_values,
      "val_mae_history": val_mae_history,
      "best_model_path": best_model_path
  }
  return histories

  
  



histories=fit_epochs(train_dataset, test_dataset,val_dataset, epochs=epochs,model=f"PPM_HD_Cak_tar_B{BATCH_SIZE}_E{epochs}", early_stop_patience=250)


  #  ----------------- PLOTS ----------------------
  
plt.figure(figsize=(10,6))
ax1 = plt.gca()
ax2 = ax1.twinx()

#  Plot Gen Total Loss
ax1.plot(histories["epoch_list"], histories["gen_total_losses"], color='blue', label='Generator Total Loss')
ax1.set_ylabel("Generator Total Loss", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# --- Plot Disc Loss on second y-axis ---
ax2.plot(histories["epoch_list"], histories["disc_losses"], color='red', label='Discriminator Loss')
ax2.set_ylabel("Discriminator Loss", color='red')
ax2.tick_params(axis='y', labelcolor='red')
  
# X-axis label & Title
plt.xlabel("Training Epochs")
plt.title("Generator vs Discriminator Loss")

# --- Combined legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()  
plot_path = os.path.join(loss_dir, f"Dis_Gen_Loss_PPM_HD_Cak_tar_B{BATCH_SIZE}_E{epochs}.png")
plt.savefig(plot_path)
plt.close()
 
 
 
 
plt.figure(figsize=(10,6))
ax1 = plt.gca()
ax2 = ax1.twinx()

#  Plot MAE Loss
ax1.plot(histories["epoch_list"], histories["mae_values"], color='blue', label='MAE Loss')
ax1.set_ylabel("Training MAE Loss", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)
  
# --- Plot validation MAE LOSS on second y-axis ---
ax2.plot(histories["epoch_list"], histories["val_mae_history"], color='red', label='validation MAE Loss')
ax2.set_ylabel("validation MAE Loss", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# X-axis label & Title
plt.xlabel("Training epochs")
plt.title("Training MAE & validation MAE Loss")

# --- Combined legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()  
plot_path = os.path.join(loss_dir, f"MAE_Loss_PPM_HD_Cak_tar_B{BATCH_SIZE}_E{epochs}.png")
plt.savefig(plot_path)
plt.close()



