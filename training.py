from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, Dropout
from keras import regularizers
from keras.callbacks import History
from sklearn.model_selection import train_test_split
import seaborn as sns

# plot setting
%pylab inline --no-import-all
sns.set_style("whitegrid")

class NitolexTrain(object):
    def __init__(self):
        inputNum = Input(shape=(586,))
        layer = Dense(400, activation="relu", \
                     activity_regularizer=regularizers.l2(0.0005))(inputNum)
        layer = Dropout(0.5)(layer)
        
        units = [250, 100, 70, 50, 30, 17, 7, 1]
        
        for unit in units:
            layer = Dense(unit, activation="relu", \
                          activity_regularizer=regularizers.l2(0.0005))(layer)
            layer = Dropout(0.5)(layer)
        
        self.model = Model(input=inputNum, output=layer)
        
    def compile(self, optimizer="adam", loss="mean_squared_error"):
        adam = Adam(lr=0.001, decay=0.005)
        self.model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

    def train(self, xTrain=None, yTrain=None, xTest=None, yTest=None, \
              epoch=10, batchSize=None, shuffle=True):
        hi = History()
        history = self.model.fit(xTrain, yTrain, \
                                       nb_epoch=epoch, \
                                       batch_size=batchSize, \
                                       shuffle=shuffle, \
                                       validation_data=(xTest, yTest), callbacks=[hi])

        self.model.save("./save_models/nitolex_model.h5")
        
        return history
    
    def loadWeight(self):
        enPath = ["stack01_encoder.h5", "stack02_encoder.h5", \
                  "stack03_encoder.h5", "stack04_encoder.h5"]
        weightList = []
        
        # load weight
        for path in enPath:
            path = "./save_models/" + path
            en = load_model(path)
            weightList.append(en.layers[1].get_weights())
            weightList.append(en.layers[3].get_weights())
        en = load_model("./save_models/stack05_encoder.h5")
        weightList.append(en.layers[1].get_weights())
        
        # set weight
        for i in range(1, 18, 2):
            self.model.layers[i].set_weights(weightList.pop(0))
        en = load_model("./save_models/stack05_encoder.h5")

# Train
def normalize(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result

def plot_history(history):
    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

def main():
    df = pd.read_csv("path/data.csv")#[:200]
    y = df["y"].values
    X = df.as_matrix()
    # split data
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, \
                    test_size=0.1, random_state=180613)
    nitolexModel = NitolexTrain()
    nitolexModel.compile()
    
    # model summary
    nitolexModel.summary()
    
    # weights load
    nitolexModel.loadWeight()
    
    # train
    history = nitolexModel.train(xTrain, yTrain, xTest, yTest, 100, 128)
    
    # plot error function
    plot_history(history)
    
if __name__ == "__main__":
    main()