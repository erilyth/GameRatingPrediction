from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.models import Sequential, Model, load_model
from keras.layers import TimeDistributed, Activation, Input, Dense, Flatten, Dropout, Convolution1D, MaxPooling1D
from keras.layers.merge import Concatenate, Multiply, Add
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization

from data import DataSet

import time

def train(saved_model=None):

    nb_epoch = 100
    batch_size = 32

    data = DataSet()

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.8) // batch_size
    val_steps_per_epoch = (len(data.data) * 0.2) // batch_size

    train_generator = data.frame_generator_features(batch_size, 'train')
    val_generator = data.frame_generator_features(batch_size, 'test')


    """
    # Model 1
    # 65 percent test acc - 95 percent train acc
    # Input: Frame features and Output: Score
    input_shape = (360, 2048)
    model = Sequential()
    model.add(LSTM(2048, return_sequences=True, input_shape=input_shape, dropout=0.4))
    model.add(LSTM(512, return_sequences=True, dropout=0.4))
    model.add(LSTM(64, return_sequences=True, dropout=0.4))
    model.add(Flatten())
    model.add(Dense(512, activation="tanh"))
    model.add(Dropout(0.6))
    model.add(Dense(10, activation='softmax'))
    """

    input_shape1 = (360, 2048)
    model_input1 = Input(shape=input_shape1)
    x1 = LSTM(2048, return_sequences=True, input_shape=input_shape1, dropout=0.4)(model_input1)
    #x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = LSTM(512, return_sequences=True, dropout=0.4)(x1)
    #x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = LSTM(64, return_sequences=True, dropout=0.4)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(512, activation="tanh")(x1)
    x1 = Dropout(0.7)(x1)

    input_shape2 = (100,)
    model_input2 = Input(shape=input_shape2)
    x2 = Embedding(14000, 300, input_length=100)(model_input2)
    x2 = Dropout(0.5)(x2)
    x2 = Convolution1D(filters=512, kernel_size=3, padding="same", activation="tanh", strides=1)(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    x2 = Convolution1D(filters=256, kernel_size=3, padding="same", activation="tanh", strides=1)(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(512, activation="tanh")(x2)
    x2 = Dropout(0.5)(x2)

    x3 = Concatenate()([x1, x2])
    model_output1 = Dense(10, activation='softmax')(x3)

    model = Model([model_input1, model_input2], model_output1)

    # Get the appropriate model.
    if saved_model is not None:
        print("Loading model %s" % saved_model)
        model = load_model(saved_model)

    # Now compile the network.
    optimizer = Adam(lr=1e-4, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(model.summary())

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./finalmodels/checkpoints/' + 'lstm' + '-final' + '.hdf5',
        verbose=1,
        save_best_only=True)

    # Helper: Save results.
    csv_logger = CSVLogger('./finalmodels/logs/' + 'lstm' + '-final-' + 'training.log')

    for _ in range(nb_epoch):
        # Use fit generator.
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            verbose=1,
            callbacks=[csv_logger, checkpointer])

        print model.evaluate_generator(
            generator=val_generator,
            steps=val_steps_per_epoch
        )

def main():
    """These are the main training settings. Set each before running
    this file."""
    saved_model = None  # None or weights file

    train(saved_model=saved_model)

if __name__ == '__main__':
    main()
