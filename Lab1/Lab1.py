
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.utils import to_categorical


random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
#ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

# ================================ < START: Parameters For loading Saved Models Here > ================================ #
load_model = True
save_model = True
#plot = True 
plot = False
# ================================ < END:  Parameters For loading Saved Models Here > ================================ #

DATASET = "mnist_d" #GOAL: 99%
#DATASET = "mnist_f" #GOAL: 92%
#DATASET = "cifar_10" #GOAL: 70%
#DATASET = "cifar_100_f" # GOAL: 35%
#DATASET = "cifar_100_c" # GOAL: 50% 

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28 # height
    IW = 28 # width
    IZ = 1 # depth
    IS = 784
    # ANN Hyperparameters - Acc == 95.350000%
    tf_eps = 25 
    tfNeuronsPerLayer = 512
    # CNN Hyperparameters - Acc == 99.06%
    cnn_eps = 25
    layer1_k = (5, 5)
    layer2_k = (4, 4)
    pool_size = (2, 2)
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
    # ANN Hyperparameters - Acc == 77.540000%
    tf_eps = 70
    tfNeuronsPerLayer = 512
    # CNN Hyperparameters - Acc == AIM 92
    cnn_eps = 48 # 25 == 90.85, 30 == 90.77, 35 == 90.810000%, 40 eps == 91.26 acc, 45 == 90.630000%, 50 == 91.030000%
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    # ANN Hyperparameters - Acc == 10%
    tf_eps = 70
    tfNeuronsPerLayer = 512
    # CNN Hyperparameters
    cnn_eps = 35
    # layer1: (6,6), layer2: (5, 5), pool: (2, 2) == 51.12%
elif DATASET == "cifar_100_f": # fine class
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    # ANN Hyperparameters - Acc == 1%
    tf_eps = 70
    tfNeuronsPerLayer = 512
    # CNN Hyperparameters
    cnn_eps = 100
elif DATASET == "cifar_100_c": # coarse class
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    # ANN Hyperparameters - Acc == 5.050000%
    tf_eps = 70
    tfNeuronsPerLayer = 512
    # CNN Hyperparameters
    cnn_eps = 77

# ================================ < Classifier Functions > ================================ #
# Guesser Classifier
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

# Standard ANN - using Keras
def buildTFNeuralNet(x, y, eps = 70):
    model = keras.Sequential()
    lossType = keras.losses.mean_squared_error
    optimizer = tf.train.AdamOptimizer()
    inShape = (x.shape[1], ) 
    
		# Load a saved model
    if load_model:
        return loadModel(DATASET + '_model_ann.h5')

    model.add(keras.layers.Dense(tfNeuronsPerLayer, input_shape=inShape, activation = 'sigmoid')) 
    model.add(keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))
    model.compile(optimizer=optimizer, loss=lossType)
    model.fit(x, y, epochs=eps)

    # Save Model
    if save_model:
        model.save(DATASET + '_model_ann.h5')
        print('Saving model to ' + DATASET + '_model_ann.h5')
    return model

# Convolutional Network - using Keras with dropouts
def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
    model = keras.Sequential()
    inShape = (IH, IW, IZ) # Input height, width, depth
    lossType = keras.losses.categorical_crossentropy
    opt = tf.train.AdamOptimizer()
		
		# Load a saved model
    if load_model:
        return loadModel(DATASET + '_model.h5')

    if DATASET == 'mnist_d':
        model = mnist_d_model_layers(model, dropout, inShape)
    elif DATASET == 'mnist_f':
        model = mnist_f_model_layers(model, dropout, inShape)
    elif DATASET == "cifar_10":
        model = cifar_10_model_layers(model, dropout, inShape)
    elif DATASET == "cifar_100_f":
        model = cifar_100_f_model_layers(model, dropout, inShape)
    elif DATASET == "cifar_100_c":
        model = cifar_100_c_model_layers(model, dropout, inShape)

    print('Saving model to ' + DATASET + '_model.h5')
    model.compile(optimizer=opt, loss=lossType)
    model.fit(x, y, epochs = eps)
    model.save(DATASET + '_model.h5')
    return model

# Add layers for mnist digit data set
def mnist_d_model_layers(model, dropout, inShape):
    model.add(keras.layers.Conv2D(32, kernel_size = layer1_k, activation="relu", input_shape=inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = layer2_k, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = pool_size))
    if dropout:
        model.add(keras.layers.Dropout(0.25)) # 0.25

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation = 'relu')) # 128
    if dropout:
        model.add(keras.layers.Dropout(0.5)) # 0.5
    model.add(keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))

    # Save Model
    if save_model:
        model.save(DATASET + '_model.h5')
    return model

# Add layers for mnist fashion data set
def mnist_f_model_layers(model, dropout, inShape):
    model.add(keras.layers.Conv2D(32, kernel_size = (2, 2), activation="relu", input_shape=inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = (1, 1), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.25)) # 0.25

    model.add(keras.layers.Conv2D(32, kernel_size = (2, 2), activation="relu"))
    model.add(keras.layers.Conv2D(64, kernel_size = (1, 1), activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.25)) # 0.25

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation = 'relu')) # 128
    if dropout:
        model.add(keras.layers.Dropout(0.5)) # 0.5
    model.add(keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))

    # Save Model
    if save_model:
        model.save(DATASET + '_model.h5')
    return model

# Add layers for cifar 10 data set
def cifar_10_model_layers(model, dropout, inShape):
    l_1 = (3, 3)
    l_2 = (2, 2)
    print("32 ==", l_1, ', 64 ==', l_2)
    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", input_shape=inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.2)) # 0.25

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu"))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.2)) # 0.25

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", padding='SAME'))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu", padding='SAME'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), padding='SAME'))
    if dropout:
        model.add(keras.layers.Dropout(0.2)) # 0.25

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(384, activation = 'relu')) # 128
    if dropout:
        model.add(keras.layers.Dropout(0.2)) # 0.5
    model.add(keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))

    # Save Model
    if save_model:
        model.save(DATASET + '_model.h5')
    return model

# Add layers for cifar 100 fine data set
def cifar_100_f_model_layers(model, dropout, inShape):
    l_1 = (2, 2)
    l_2 = (1, 1)
    print("32 ==", l_1, ', 64 ==', l_2)
    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", input_shape=inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.25

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu"))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.25

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", padding='SAME'))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu", padding='SAME'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), padding='SAME'))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.25

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", padding='SAME'))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu", padding='SAME'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), padding='SAME'))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.25

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(384, activation = 'relu')) # 128
    if dropout:
        model.add(keras.layers.Dropout(0.2)) # 0.5
    model.add(keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))

    # Save Model
    if save_model:
        model.save(DATASET + '_model.h5')
    return model

# Add layers for cifar 100 coarse data set aiming for an accuracy of 50%
def cifar_100_c_model_layers(model, dropout, inShape):
    l_1 = (2, 2)
    l_2 = (1, 1)
    print("32 ==", l_1, ', 64 ==', l_2)
    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", input_shape=inShape))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.15 ---> 0.1

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu"))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.15 ---> 0.1

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", padding='SAME'))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu", padding='SAME'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), padding='SAME'))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.15 ---> 0.1

    model.add(keras.layers.Conv2D(32, kernel_size = l_1, activation="relu", padding='SAME'))
    model.add(keras.layers.Conv2D(64, kernel_size = l_2, activation="relu", padding='SAME'))
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2), padding='SAME'))
    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.15 ---> 0.1

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(384, activation = 'relu')) # 128 ---> 256
    print("526 nodes")

    if dropout:
        model.add(keras.layers.Dropout(0.1)) # 0.2
    model.add(keras.layers.Dense(NUM_CLASSES, activation = 'softmax'))

    # Save Model
    if save_model:
        model.save(DATASET + '_model.h5')
    return model

# Load Model
def loadModel(file):
		print('Loading a saved model instead of training')
#model = keras.models.load_model(DATASET + '_model.h5')
		model = keras.models.load_model(file)
		return model

# ================================== < Pipeline Functions > ================================== #
# Load Data
def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        cifar_10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar_10.load_data()
    elif DATASET == "cifar_100_f":
        cifar_100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100.load_data(label_mode='fine')
    elif DATASET == "cifar_100_c":
        cifar_100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar_100.load_data(label_mode='coarse')
    else:
        raise ValueError("Dataset not recognized.")
    
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

# Process Data
def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))

# Train Model
def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain, eps=tf_eps)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain, eps=cnn_eps)
    else:
        raise ValueError("Algorithm not recognized.")

# Run model
def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)  
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds    
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")

# Evalute results
def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()
    return accuracy * 100

# Function to plot accuracy of ANN and CNN data set
def makePlots():
	global ALGORITHM
	global DATASET
	dataSets = ["mnist_d", "mnist_f", "cifar_10", "cifar_100_f", "cifar_100_c"]
	index = np.arange(len(dataSets))
	
	# Make plot for tf_net
	fileName = 'ANN_Accuracy_Plot.pdf'
	accuracy_percent = []
	ALGORITHM = 'tf_net'
	for dataS in dataSets:
	  DATASET = dataS
	  setGlobals()
	  raw = getRawData()
	  data = preprocessData(raw)
	  model = trainModel(data[0])
	  preds = runModel(data[1][0], model)
	  accuracy_percent.append(evalResults(data[1], preds))

	print(accuracy_percent)
	plt.bar(index, np.array(accuracy_percent))
	
	plt.xlabel('Dataset')
	plt.ylabel('Accuracy')
	plt.xticks(index, dataSets, fontsize=5, rotation=30)
	plt.title('Accuracy in % for ANN')

	# Make plot for tf_conv
	fileName = 'CNN_Accuracy_Plot.pdf'
	accuracy_percent = []
	ALGORITHM = 'tf_conv'
	for dataS in dataSets:
		# add to plot
		a = 1+1

	plt.bar(index, accuracy_percent)
	plt.xlabel('Dataset')
	plt.ylabel('Accuracy')
	plt.xticks(index, dataSets, fontsize=5, rotation=30)
	plt.title('Accuracy in % for CNN')

	return

def setGlobals():
	global NUM_CLASSES, IH, IW, IZ, IS, load_model
	load_model = True
	if DATASET == "mnist_d":
	    NUM_CLASSES = 10
	    IH = 28 # height
	    IW = 28 # width
	    IZ = 1 # depth
	    IS = 784
	elif DATASET == "mnist_f":
	    NUM_CLASSES = 10
	    IH = 28
	    IW = 28
	    IZ = 1
	    IS = 784
	elif DATASET == "cifar_10":
	    NUM_CLASSES = 10
	    IH = 32
	    IW = 32
	    IZ = 3
	    IS = 3072
	elif DATASET == "cifar_100_f": # fine class
	    NUM_CLASSES = 100
	    IH = 32
	    IW = 32
	    IZ = 3
	    IS = 3072
	elif DATASET == "cifar_100_c": # coarse class
	    NUM_CLASSES = 20
	    IH = 32
	    IW = 32
	    IZ = 3
	    IS = 3072

# ================================================ < Main > ================================================ #
def main():
		# Generate Bar graphs instead of running 
    if plot:
      makePlots()
      return

    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

if __name__ == '__main__':
    main()
