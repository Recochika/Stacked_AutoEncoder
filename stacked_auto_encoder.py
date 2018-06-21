# pre training
# python 3.6
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout

class EncoderStack01(object):
    def __init__(self):
        inputNum = Input(shape=(586,))
        layer1 = Dense(400, activation="relu")(inputNum)
        layer1 = Dropout(0.5)(layer1)
        layer2 = Dense(250, activation="relu")(layer1)
        layer2 = Dropout(0.5)(layer2)
        
        encoded = layer2
        
        deLayer2 = Dense(250, activation="relu")(layer2)
        deLayer1 = Dense(400, activation="relu")(deLayer2)
        decoded = Dense(586, activation="relu")(deLayer1)
        
        self.encoder = Model(input=inputNum, output=encoded)
        self.autoencoder = Model(input=inputNum, output=decoded)
        
    def compile(self, optimizer="adam", loss="mean_squared_error"):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)

    def train(self, xTrain=None, xTest=None, epoch=10, batchSize=None, shuffle=True):
        self.autoencoder.fit(xTrain, xTrain,
                             nb_epoch=epoch,
                             batch_size=batchSize,
                             shuffle=shuffle,
                             validation_data=(xTest, xTest))

        self.encoder.save("./save_models/stack01_encoder.h5")
        self.autoencoder.save("./save_models/stack01_autoencoder.h5")

class EncoderStack02(object):
    def __init__(self):
        inputNum = Input(shape=(250,))
        layer1 = Dense(100, activation="relu")(inputNum)
        layer1 = Dropout(0.5)(layer1)
        layer2 = Dense(70, activation="relu")(layer1)
        layer2 = Dropout(0.5)(layer2)
        
        encoded = layer2
        
        deLayer2 = Dense(70, activation="relu")(layer2)
        deLayer1 = Dense(100, activation="relu")(deLayer2)
        decoded = Dense(250, activation="relu")(deLayer1)
        
        self.encoder = Model(input=inputNum, output=encoded)
        self.autoencoder = Model(input=inputNum, output=decoded)
        
    def compile(self, optimizer="adam", loss="mean_squared_error"):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)

    def train(self, xTrain=None, xTest=None, epoch=10, batchSize=None, shuffle=True):
        self.autoencoder.fit(xTrain, xTrain,
                             nb_epoch=epoch,
                             batch_size=batchSize,
                             shuffle=shuffle,
                             validation_data=(xTest, xTest))

        self.encoder.save("./save_models/stack02_encoder.h5")
        self.autoencoder.save("./save_models/stack02_autoencoder.h5")

class EncoderStack03(object):
    def __init__(self):
        inputNum = Input(shape=(70,))
        layer1 = Dense(50, activation="relu")(inputNum)
        layer1 = Dropout(0.5)(layer1)
        layer2 = Dense(30, activation="relu")(layer1)
        layer2 = Dropout(0.5)(layer2)
        
        encoded = layer2
        
        deLayer2 = Dense(30, activation="relu")(layer2)
        deLayer1 = Dense(50, activation="relu")(deLayer2)
        decoded = Dense(70, activation="relu")(deLayer1)
        
        self.encoder = Model(input=inputNum, output=encoded)
        self.autoencoder = Model(input=inputNum, output=decoded)
        
    def compile(self, optimizer="adam", loss="mean_squared_error"):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)

    def train(self, xTrain=None, xTest=None, epoch=10, batchSize=None, shuffle=True):
        self.autoencoder.fit(xTrain, xTrain,
                             nb_epoch=epoch,
                             batch_size=batchSize,
                             shuffle=shuffle,
                             validation_data=(xTest, xTest))

        self.encoder.save("./save_models/stack03_encoder.h5")
        self.autoencoder.save("./save_models/stack03_autoencoder.h5")

class EncoderStack04(object):
    def __init__(self):
        inputNum = Input(shape=(30,))
        layer1 = Dense(17, activation="relu")(inputNum)
        layer1 = Dropout(0.5)(layer1)
        layer2 = Dense(7, activation="relu")(layer1)
        layer2 = Dropout(0.5)(layer2)
        
        encoded = layer2
        
        deLayer2 = Dense(7, activation="relu")(layer2)
        deLayer1 = Dense(17, activation="relu")(deLayer2)
        decoded = Dense(30, activation="relu")(deLayer1)
        
        self.encoder = Model(input=inputNum, output=encoded)
        self.autoencoder = Model(input=inputNum, output=decoded)
        
    def compile(self, optimizer="adam", loss="mean_squared_error"):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)

    def train(self, xTrain=None, xTest=None, epoch=10, batchSize=None, shuffle=True):
        self.autoencoder.fit(xTrain, xTrain,
                             nb_epoch=epoch,
                             batch_size=batchSize,
                             shuffle=shuffle,
                             validation_data=(xTest, xTest))

        self.encoder.save("./save_models/stack04_encoder.h5")
        self.autoencoder.save("./save_models/stack04_autoencoder.h5")

class EncoderStack05(object):
    def __init__(self):
        inputNum = Input(shape=(7,))
        layer1 = Dense(1, activation="relu")(inputNum)
        layer1 = Dropout(0.2)(layer1)
        
        encoded = layer1
        
        deLayer1 = Dense(1, activation="relu")(layer1)
        decoded = Dense(7, activation="relu")(deLayer1)
        
        self.encoder = Model(input=inputNum, output=encoded)
        self.autoencoder = Model(input=inputNum, output=decoded)
        
    def compile(self, optimizer="adam", loss="mean_squared_error"):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)

    def train(self, xTrain=None, xTest=None, epoch=10, batchSize=None, shuffle=True):
        self.autoencoder.fit(xTrain, xTrain,
                             nb_epoch=epoch,
                             batch_size=batchSize,
                             shuffle=shuffle,
                             validation_data=(xTest, xTest))

        self.encoder.save("./save_models/stack05_encoder.h5")
        self.autoencoder.save("./save_models/stack05_autoencoder.h5")