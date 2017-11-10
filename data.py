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

genre_dict = {"knitting": 0, "shotput": 1}

class DataSet():

    def __init__(self, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.class_limit = class_limit
        self.sequence_path = './data/sequences/'
        self.max_frames = 720  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Dictionary created with the entire train and test summaries together
        self.dictionary = {}
        self.summary_pad_length = 100

        # Contain intermediate info for the entire dataset
        self.score_classes = self.get_score_classes()
        self.genre_classes = self.get_genre_classes()
        self.summaries = self.get_summaries()

        # Contain the final data split into train and test
        self.train = {}
        self.test = {}

        self.split_train_test()

        print 'Loaded data and split into train/test'

        self.image_shape = image_shape

    def get_data(self):
        """Load our data from file."""
        with open('./data/data_file.csv', 'r') as fin:
            reader = csv.reader(fin, delimiter=';')
            data = list(reader)

        return data

    def get_score_classes(self):
        """
        Extract the score classes from our data.
        """
        score_classes = []
        for item in self.data:
            critic_score = item[2]
            if '-' in critic_score or 'tbd' in critic_score:
                critic_score = "50"
            critic_score = float(critic_score)
            user_score = item[3]
            if '-' in user_score or 'tbd' in user_score:
                user_score = "5"
            user_score = float(user_score) * 10
            final_score = critic_score * 0.5 + user_score * 0.5
            final_score = int(round(final_score, -1))/10
            final_score_one_hot = [0.0] * 10
            final_score_one_hot[final_score-1] = 1.0
            score_classes.append(final_score_one_hot)
        return score_classes


    def get_genre_classes(self):
        """
        Extract the genre classes from our data.
        """
        genre_class_dict = {
            "action": 0,
            "role-playing": 1,
            "adventure": 1,
            "third-person": 1,
            "first-person": 1,
            "strategy": 2,
            "turn-based": 2,
            "wargame": 2,
            "puzzle": 2,
            "platformer": 2,
            "sports": 3,
            "fighting": 3,
            "racing": 3,
            "wrestling": 3,
            "simulation": 4,
            "flight": 4,
            "party": 4,
            "real-time": 4
        }
        genre_classes = []
        for item in self.data:
            genre = item[7]
            final_genre_one_hot = [0.0] * 5
            final_genre_one_hot[genre_class_dict[genre]] = 1.0
            genre_classes.append(final_genre_one_hot)
        return genre_classes


    def get_summaries(self):
        summaries = []
        cnt = 1
        for item in self.data:
            summary = item[4]
            words = summary.strip().split(' ')
            for word in words:
                if word not in self.dictionary:
                    self.dictionary[word] = cnt
                    cnt += 1
        for item in self.data:
            summary = item[4]
            current_summary = []
            words = summary.strip().split(' ')
            for word in words:
                current_summary.append(self.dictionary[word])
            current_summary = current_summary[:self.summary_pad_length]
            while len(current_summary) < self.summary_pad_length:
                current_summary.append(0)
            summaries.append(current_summary)
        print 'Dictionary size:', len(self.dictionary)
        return summaries


    def split_train_test(self):
        """Split the data into train and test groups."""
        train_percent = 0.8
        train_elems = {"score":[], "genre":[], "features":[], "summary":[], "original_data_index":[]}
        test_elems = {"score":[], "genre":[], "features":[], "summary":[], "original_data_index":[]}
        for item_idx in range(len(self.data)):
            if random.uniform(0, 1) <= train_percent:
                train_elems["score"].append(self.score_classes[item_idx])
                train_elems["genre"].append(self.genre_classes[item_idx])
                train_elems["summary"].append(self.summaries[item_idx])
                train_elems["original_data_index"].append(item_idx)
            else:
                test_elems["score"].append(self.score_classes[item_idx])
                test_elems["genre"].append(self.genre_classes[item_idx])
                test_elems["summary"].append(self.summaries[item_idx])
                test_elems["original_data_index"].append(item_idx)

        self.train = train_elems
        self.test = test_elems


    def get_features_for_datapoint(self, data_idx):
        dummy = np.asarray([0 for el in range(2048)]) # Inception feature layer output size
        sequence = self.get_extracted_sequence("features", self.data[data_idx])
        if len(sequence) > 360:
            sequence = sequence[:360]
        if len(sequence) < 360:
            deficit = 360 - len(sequence)
            for det in range(deficit):
                sequence = np.vstack((sequence,dummy))
        return sequence


    def frame_generator_features(self, batch_size, train_test):
        data = self.train if train_test == 'train' else self.test
        print("Creating %s generator with %d samples." % (train_test, len(data["score"])))

        while 1:
            X, X1, y, y1 = [], [], [], []
            for _ in range(batch_size):
                sample = random.randint(0, len(data["score"])-1)
                X.append(self.get_features_for_datapoint(data["original_data_index"][sample]))
                X1.append(data["summary"][sample])
                y.append(data["score"][sample])
                y1.append(data["genre"][sample])

            # Frame features, descriptions, score class and genre class
            #yield [np.array(X), np.array(X1)], [np.array(y), np.array(y1)]
            yield [np.array(X), np.array(X1)], np.array(y)


    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[0]
        path = self.sequence_path + filename + '--features.txt'
        if os.path.isfile(path):
            # Use a dataframe/read_csv for speed increase over numpy.
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    def get_frames_for_sample(self, sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = './data/frames/'
        filename = sample[0]
        images = sorted(glob.glob(path + filename + '*jpg'))
        return images

    def get_filename_from_image(self, filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')

    def rescale_list(self, input_list):
        """
        Limit the number of frames.
        Assuming a trailer is around 3 minutes long, it has around 720 frames (4 frames per second)
        Every 150 frames, consider 10 frames.
        ex: Select frame 150, 151, 152, 153, 154, 155, 156, 157, 158, 159 and then 300, 301, 302 ....
        """

        # Build our new output.
        output = []

        start_frame = 50
        while start_frame < len(input_list):
            for j in range(10):
                cur_frame = start_frame + j
                if cur_frame >= len(input_list):
                    break
                output.append(input_list[cur_frame])
            start_frame += 150

        # Cut off the last one if needed.
        return output

    def process_image(self, image, target_shape):
        """Given an image, process it and return the array."""
        # Load the image.
        h, w, _ = target_shape
        image = load_img(image, target_size=(h, w))

        # Turn it into numpy, normalize and return.
        img_arr = img_to_array(image)
        x = (img_arr / 255.).astype(np.float32)

        return x
