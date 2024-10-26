# Super-Resolution using ESRGAN

## Discreption
This project is about implementing a super-resolution approach using the Enhanced Super Resolution Generative Adversarial Network (ESRGAN). It takes low-resolution images as input and generates high-resolution images, improving image clarity and detail. The project involves preprocessing input data, training the ESRGAN model on high-resolution data, evaluating the model's performance, and finally using the trained model to produce high-resolution outputs. It can be applied in fields like medical imaging, satellite imagery, and any area where image quality enhancement is needed.

### Program
```
import os
import time
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
     

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
     

IMAGE_PATH = "original.png"
     

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
     

def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

     

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)
     

%matplotlib inline
# Declaring the user-defined function for the ploting the image

def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)
     

hr_image = preprocess_image(IMAGE_PATH)
     

plot_image(tf.squeeze(hr_image), title="Original Image")
save_image(tf.squeeze(hr_image), filename="Original Image")
     
Saved as Original Image.jpg


original_image = cv2.imread("original.png")
     

original_image.shape
     
(168, 299, 3)

model = hub.load(SAVED_MODEL_PATH)
     

start = time.time()
fake_image = model(hr_image)
fake_image = tf.squeeze(fake_image)
print("Time Taken: %f" % (time.time() - start))
     
Time Taken: 13.495102

plot_image(tf.squeeze(fake_image), title="Super Resolution")
save_image(tf.squeeze(fake_image), filename="Super Resolution")
     
Saved as Super Resolution.jpg


upscaled_image = cv2.imread('/content/Super Resolution.jpg')
     

upscaled_image.shape
     
(672, 1184, 3)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
     

sharpened_image = unsharp_mask(upscaled_image)
     

cv2.imwrite('original_final_image.jpg', sharpened_image)
     
True

IMAGE_PATH = "/content/test_image.png"
     

test_image=cv2.imread("/content/test_image.png")
     

test_image.shape
     
(177, 284, 3)

def downscale_image(image):
  """
      Scales down images using bicubic downsampling.
      Args:
          image: 3D or 4D tensor of preprocessed image
  """
  image_size = []
  if len(image.shape) == 3:
    image_size = [image.shape[1], image.shape[0]]
  else:
    raise ValueError("Dimension mismatch. Can work only on single image.")

  image = tf.squeeze(
      tf.cast(
          tf.clip_by_value(image, 0, 255), tf.uint8))

  lr_image = np.asarray(
    Image.fromarray(image.numpy())
    .resize([image_size[0] // 4, image_size[1] // 4],
              Image.BICUBIC))

  lr_image = tf.expand_dims(lr_image, 0)
  lr_image = tf.cast(lr_image, tf.float32)
  return lr_image
     

hr_image = preprocess_image(IMAGE_PATH)
     

lr_image = downscale_image(tf.squeeze(hr_image))
     

plot_image(tf.squeeze(lr_image), title="Low Resolution")
save_image(tf.squeeze(lr_image), filename="Low Resolution")
     
Saved as Low Resolution.jpg


test_low_res_image=cv2.imread("/content/Low Resolution.jpg")
     

test_low_res_image.shape
     
(44, 71, 3)

model = hub.load(SAVED_MODEL_PATH)
     

start = time.time()
fake_image = model(lr_image)
fake_image = tf.squeeze(fake_image)
print("Time Taken: %f" % (time.time() - start))
     
Time Taken: 2.654266

plot_image(tf.squeeze(fake_image), title="Super Resolution")
# Calculating PSNR wrt Original Image
psnr = tf.image.psnr(
    tf.clip_by_value(fake_image, 0, 255),
    tf.clip_by_value(hr_image, 0, 255), max_val=255)
print("PSNR Achieved: %f" % psnr)
save_image(tf.squeeze(fake_image), filename="downscaled Resolution01")
     
PSNR Achieved: 17.945190
Saved as downscaled Resolution01.jpg


plt.rcParams['figure.figsize'] = [15, 10]
fig, axes = plt.subplots(1,3)
fig.tight_layout()
plt.subplot(131)
plot_image(tf.squeeze(hr_image), title="Original")   # Original image

plt.subplot(132)
fig.tight_layout()
# Downscaled test image by resize value as 4
plot_image(tf.squeeze(lr_image), "Downscaled Image by 4")

plt.subplot(133)
fig.tight_layout()
plot_image(tf.squeeze(fake_image), "Super Resolution")  # Super Resolution test image
plt.savefig("ESRGAN_DIV2K.jpg", bbox_inches="tight")

     


test_upscaled_image=cv2.imread("/content/downscaled Resolution01.jpg")
     

test_upscaled_image.shape
     
(176, 284, 3)

sharpened_edge_image = unsharp_mask(test_upscaled_image)
     

cv2.imwrite('sharpened_edge_image.jpg', sharpened_edge_image)
     
True

img_path = '/content/sharpened_edge_image.jpg'
img = mpimg.imread(img_path)
     

figsize = (6, 4)

# Create a figure and axis
fig, ax = plt.subplots(figsize=figsize)

# Display the image
imgplot = ax.imshow(img)

# Show the plot
plt.show()
print("PSNR: %f" % psnr)
