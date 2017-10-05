from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from data import DataSet
import time

def train(data_type, saved_model=None):

    nb_epoch = 1000000
    batch_size = 1

    data = DataSet()

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    train_generator = data.frame_generator(batch_size, 'train', data_type)
    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    input_shape = (130, 2048)
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False, input_shape=input_shape, dropout=0.5))
    #model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

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
        filepath='./data/checkpoints/' + 'lstm' + '-' + data_type + '.hdf5',
        verbose=1,
        save_best_only=True)

    # Helper: Save results.
    csv_logger = CSVLogger('./data/logs/' + 'lstm' + '-' + 'training.log')

    # Use fit generator.
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=10)

def main():
    """These are the main training settings. Set each before running
    this file."""
    saved_model = None  # None or weights file
    data_type = 'features'

    train(data_type, saved_model=saved_model)

if __name__ == '__main__':
    main()
