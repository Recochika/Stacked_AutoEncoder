import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from stacked_auto_encoder import EncoderStack01, EncoderStack02, \
    EncoderStack03,EncoderStack04, EncoderStack05

def preTraining(autoEncoder, xTrain, xTest):
    autoEncoder.compile()
    autoEncoder.train(xTrain, xTest, 100, 128)
    
    preXTrain = autoEncoder.encoder.predict(xTrain)
    preXTest = autoEncoder.encoder.predict(xTest)
    
    del xTrain, xTest
    
    return preXtrain, preXTest

def main():
    df = pd.read_csv("data/path.csv")#[:200]
    X = df.as_matrix()
    # data split
    xTrain, xTest = train_test_split(X, test_size=0.1, random_state=180608)
    
    # step 1
    en01 = EncoderStack01()
    xTrain02, xTest02 = preTraining(en01, xTrain, xTest)
    
    # step 2
    en02 = EncoderStack02()
    xTrain03, xTest03 = preTraining(en02, xTrain02, xTest02)
    
    # step 3
    en03 = EncoderStack03()
    xTrain04, xTest04 = preTraining(en03, xTrain03, xTest03)
    
    # step 4
    en04 = EncoderStack04()
    xTrain05, xTest05 = preTraining(en04, xTrain04, xTest04)
    
    # step 5
    en05 = EncoderStack05()
    xTrain06, xTest06 = preTraining(en05, xTrain05, xTest05)
    
if __name__ == "__main__":
    main()