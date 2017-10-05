import csv
import numpy as np
import random
import glob
import os.path
import pandas as pd
import sys
import operator
from keras.utils import np_utils

from keras.preprocessing.image import img_to_array, load_img

class DataSet():

    def __init__(self, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.class_limit = class_limit
        self.sequence_path = './data/sequences/'
        self.max_frames = 8000  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        self.image_shape = image_shape
    
    def get_data(self):
        """Load our data from file."""
        with open('./data/data_file.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = np_utils.to_categorical(label_encoded, len(self.classes))
        label_hot = label_hot[0]  # just get a single row

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def frame_generator(self, batch_size, train_test, data_type="features", concat=False):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames)
                else:
                    # Get the sequence from disk.
                    dummy = np.asarray([0 for el in range(2048)])
                    sequence = self.get_extracted_sequence(data_type, sample)
                    if len(sequence) > 130:
                        sequence = sequence[:130]

                    if len(sequence) < 130:
                        deficit = 130 - len(sequence)
                        for det in range(deficit):
                            sequence = np.vstack((sequence,dummy))

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()  # TODO this should raise

                if concat:
                    # We want to pass the sequence back as a single array. This
                    # is used to pass into a fully connected neural network rather than an RNN.
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [self.process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = self.sequence_path + sample[0] + '--' + sample[1] + '--' + filename + '--features.txt'
        if os.path.isfile(path):
            # Use a dataframe/read_csv for speed increase over numpy.
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    def get_frames_for_sample(self, sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = './data/' + sample[0] + '/' + sample[1] + '/'
        filename = sample[2]
        images = sorted(glob.glob(path + filename + '*jpg'))
        return images

    def get_filename_from_image(self, filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')

    def rescale_list(self, input_list):
        """
        Limit the number of frames.
        Assuming a trailer is around 3 minutes long, it has around 4000 frames
        Every 300 frames, consider 10 frames skipping 4 frames each.
        ex: Select frame 300, 305, 310, 315, 320, 325, 330, 335, 340, 345 and then 600, 605, 610 ....
        """

        # Build our new output.
        output = []

        start_frame = 300
        while start_frame < len(input_list):
            for j in range(10):
                cur_frame = start_frame + 5 * j
                if cur_frame > len(input_list):
                    break
                output.append(input_list[cur_frame])
            start_frame += 300

        # Cut off the last one if needed.
        return output

    def print_class_from_prediction(self, predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))

    def process_image(self, image, target_shape):
        """Given an image, process it and return the array."""
        # Load the image.
        h, w, _ = target_shape
        image = load_img(image, target_size=(h, w))

        # Turn it into numpy, normalize and return.
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)

        return x