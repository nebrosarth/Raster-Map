from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img


def split_image(img, patch_size):
    patches = []
    x_count = img.shape[0] // patch_size
    y_count = img.shape[1] // patch_size
    for i in range(x_count):
        for j in range(y_count):
            patch = img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch)
    result = np.array(patches)
    return result


# %%
def reconstruct_image(patches, img_shape):
    x_count = img_shape[0] // 256
    y_count = img_shape[1] // 256
    img = np.zeros(img_shape)
    for i in range(x_count):
        for j in range(y_count):
            img[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = patches[i * y_count + j]
    return img


# %%
# make X from array of patches
def make_X(patches):
    patches = resize(patches, (len(patches), 256, 256, 1), mode='constant', preserve_range=True)
    X = np.zeros((len(patches), 256, 256, 1), dtype=np.float32)
    for i in range(len(patches)):
        X[i, ..., 0] = patches[i].squeeze() / 255
    return X


def predict_image_and_plot(img_path, model, title=None):
    global reconstructed_img
    patch_size = 256
    img = load_img(img_path)

    original_size = img.size

    img = ImageOps.expand(img, (0, 0, patch_size - img.size[0] % patch_size, patch_size - img.size[1] % patch_size),
                          fill="white")

    img = img_to_array(img)

    patches = split_image(img, patch_size)

    X = make_X(patches)

    pred_patches = model.predict(X)

    reconstructed_img = reconstruct_image(pred_patches, img.shape)
    # Rescale it to 0-255
    reconstructed_img = (reconstructed_img - np.min(reconstructed_img)) / (
            np.max(reconstructed_img) - np.min(reconstructed_img)) * 255

    reconstructed_img = reconstructed_img[:original_size[1], :original_size[0]]
    img = img[:original_size[1], :original_size[0]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 15))
    fig.set_facecolor("white")
    ax[0].imshow(img.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Input")
    ax[1].imshow(reconstructed_img.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Prediction")

    if title:
        plt.title(title)

    for a in ax:
        a.axis("off")


def predict_image(img_path, model):
    global reconstructed_img
    patch_size = 256
    img = load_img(img_path)

    original_size = img.size

    img = ImageOps.expand(img, (0, 0, patch_size - img.size[0] % patch_size, patch_size - img.size[1] % patch_size),
                          fill="white")

    img = img_to_array(img)

    patches = split_image(img, patch_size)

    X = make_X(patches)

    pred_patches = model.predict(X)

    reconstructed_img = reconstruct_image(pred_patches, img.shape)
    # Rescale it to 0-255
    reconstructed_img = (reconstructed_img - np.min(reconstructed_img)) / (
            np.max(reconstructed_img) - np.min(reconstructed_img)) * 255

    reconstructed_img = reconstructed_img[:original_size[1], :original_size[0]]

    return reconstructed_img