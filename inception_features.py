import numpy as np
import os.path
from data import DataSet
from tqdm import tqdm

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input

# Creates feature vectors for each of the frames and stores them in data/sequences

# Get the dataset.
data = DataSet()

# Get model with pretrained weights.
base_model = InceptionV3(
    weights='imagenet',
    include_top=True
)

# We'll extract features at the final pool layer.
model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)

def extract(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the prediction.
    features = model.predict(x)
    features = features[0]

    return features


# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:
    print video[0]
    # Get the path to the sequence for this video.
    path = './data/sequences/' + video[0] + '--features.txt'

    # Check if we already have it.
    #if os.path.isfile(path):
    #    pbar.update(1)
    #    continue

    frames = data.get_frames_for_sample(video)
    frames = data.rescale_list(frames)


    sequence = []
    for image_path in frames:
        features = extract(image_path)
        sequence.append(features)

    print sequence

    # Save the sequence.
    np.savetxt(path, sequence)

    pbar.update(1)

pbar.close()
