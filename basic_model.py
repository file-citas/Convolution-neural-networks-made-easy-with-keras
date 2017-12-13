import sys
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import imageio
import random
import os

# global static variables
dtype_mult = 255.0 # unit8
epoch = 200
#test_train_split = 0.8
# TODO: add your error and valid directories to this list
ok_dir = ["./ok", ]
nok_dir = ["./nok", ]

def get_dataset():
    # read input directories and store filenames in list for generator
    data_list = []
    for od in ok_dir:
        for f in os.listdir(od):
            data_list.append([os.path.join(od, f), 0])
    for nod in nok_dir:
        for f in os.listdir(nod):
            data_list.append([os.path.join(nod, f), 1])

    # shuffle data list
    random.shuffle(data_list)
    # read one image to get shape
    image = imageio.imread(data_list[0][0])
    return data_list, image.shape, 2

def generate_optimizer():
    return keras.optimizers.Adam()

def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer=generate_optimizer(),
                  metrics=['accuracy'])

def generate_model(in_shape, num_classes):
    # check if model exists if exists then load model from saved state
    #if Path('./models/convnet_model.json').is_file():
    #    sys.stdout.write('Loading existing model\n\n')
    #    sys.stdout.flush()

    #    with open('./models/convnet_model.json') as file:
    #        model = keras.models.model_from_json(json.load(file))
    #        file.close()

    #    # likewise for model weight, if exists load from saved state
    #    if Path('./models/convnet_weights.h5').is_file():
    #        model.load_weights('./models/convnet_weights.h5')

    #    compile_model(model)

    #    return model

    sys.stdout.write('Loading new model\n\n')
    sys.stdout.flush()

    model = Sequential()

    # Conv1 32 32 (3) => 30 30 (32)
    #model.add(Conv2D(32, (3, 3), input_shape=X_shape[1:]))
    model.add(Conv2D(32, (3, 3), input_shape=in_shape))
    model.add(Activation('relu'))
    # Conv2 30 30 (32) => 28 28 (32)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    # Pool1 28 28 (32) => 14 14 (32)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Conv3 14 14 (32) => 12 12 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # Conv4 12 12 (64) => 6 6 (64)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    # Pool2 6 6 (64) => 3 3 (64)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # FC layers 3 3 (64) => 576
    model.add(Flatten())
    # Dense1 576 => 256
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Dense2 256 => 10
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # compile has to be done impurely
    compile_model(model)

    #with open('./models/convnet_model.json', 'w') as outfile:
    #    json.dump(model.to_json(), outfile)
    #    outfile.close()

    return model

# TODO: adjust batch size according to ram
def DataGenerator(data_list, num_classes, batch_size=8):
    start = 0
    while start < len(data_list):
        X = []
        Y = []
        end = min(start+batch_size, len(data_list) -1)
        for data, label in data_list[start:end]:
            image = imageio.imread(data)
            image = image.astype('float32') / dtype_mult
            X.append(image)
            Y.append([label,])
        yield np.asarray(X), keras.utils.to_categorical(np.asarray(Y), num_classes)
        start += batch_size

def train(model, gen):
    sys.stdout.write('Training model\n\n')
    sys.stdout.flush()

    # train each iteration individually to back up current state
    # safety measure against potential crashes
    epoch_count = 0
    while epoch_count < epoch:
        epoch_count += 1
        sys.stdout.write('Epoch count: ' + str(epoch_count) + '\n')
        sys.stdout.flush()
        model.fit_generator(gen,
                            samples_per_epoch=4,
                            nb_epoch=1, validation_data=None)
        sys.stdout.write('Epoch {} done, saving model to file\n\n'.format(epoch_count))
        sys.stdout.flush()
        model.save_weights('./models/convnet_weights.h5')

    return model

def get_accuracy(pred, real):
    # reward algorithm
    result = pred.argmax(axis=1) == real.argmax(axis=1)
    return np.sum(result) / len(result)

def main():
    data_list, in_shape, num_classes = get_dataset()
    print("Read data list %d files" % len(data_list))
    print("Image shape:")
    print(in_shape)
    print("Number of classes:")
    print(num_classes)
    model = generate_model(in_shape, num_classes)
    model = train(model, DataGenerator(data_list, num_classes))

if __name__ == "__main__":
    # execute only if run as a script
    main()
