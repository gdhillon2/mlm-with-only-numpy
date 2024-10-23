import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
import the data and normalize it

normalizing helps the model learn by improving neural networks
ability to converge on values that minimize the loss function

this is because the gradients can oscillate wildly if not normalized
"""
data = pd.read_csv("mlp_regression_data.csv")

x = np.array(data["x"]).reshape(-1, 1)
y = np.array(data["y"]).reshape(-1, 1)


xMean, xStd = x.mean(), x.std()
yMean, yStd = y.mean(), y.std()

xNorm = (x - xMean) / xStd
yNorm = (y - yMean) / yStd

def ReLU(Z):
    return np.maximum(Z, 0)

def derivReLU(Z):
    return Z > 0

class MLP:
    """
    my model was struggling to learn so i tried xavier initialization to help
    if i don't use this initialization method my model doesn't work very well
    i still don't really understand why this method is good but it works
    """
    def __init__(self, inputSize, hidden1Size, hidden2Size, outputSize, batchSize, learningRate):
        self.inputSize = inputSize
        self.hidden1Size = hidden1Size
        self.hidden2Size = hidden2Size
        self.outputSize = outputSize
        self.batchSize = batchSize
        self.learningRate = learningRate

        self.hidden1Weights = np.random.randn(hidden1Size, inputSize) * np.sqrt(2. / (inputSize + hidden1Size))
        self.hidden1Bias = np.zeros((hidden1Size, 1))

        self.hidden2Weights = np.random.randn(hidden2Size, hidden1Size) * np.sqrt(2. / (hidden1Size + hidden2Size))
        self.hidden2Bias = np.zeros((hidden2Size, 1))

        self.outputWeights = np.random.randn(outputSize, hidden2Size) * np.sqrt(2. / (hidden2Size + outputSize))
        self.outputBias = np.zeros((outputSize, 1))

    """
    standard forward propagation
    take previous activated layer, multiply weights, add bias, apply activation function unless output layer
    repeat for hidden layers
    output layer doesn't even require softmax since we're doing regression and not classification
    softmax would convert the raw output value to a probability, which is wrong here
    """
    def forwardProp(self, inputLayer):
        self.inputLayer = inputLayer

        self.Z1 = self.hidden1Weights @ self.inputLayer + self.hidden1Bias
        self.A1 = ReLU(self.Z1)

        self.Z2 = self.hidden2Weights @ self.A1 + self.hidden2Bias
        self.A2 = ReLU(self.Z2)

        self.Z3 = self.outputWeights @ self.A2 + self.outputBias
        self.outputLayer = self.Z3

        return self.outputLayer
        
    """
    standard backpropagation but still the hardest part for sure
    dZ3 is the gradient of the loss with respect to the output layer (Z3), calculated by subtracting the actual values from the predictions
    dW3, db3 are the gradients of the output weights and bias, derived from dZ3
    dZ2 is the gradient of the loss with respect to second hidden layer (Z2), computed by propagating dZ3 backwards and using the output layer weights and ReLU derivative
    dW2, db2 are the gradients of the second hidden layer's weights and biases, derived from dZ2
    dZ1 is the gradient of the loss with respect to first hidden layer (Z1), calculated by propagating dZ2 backwards using the second hidden layer weights and ReLU derivative
    dW1, db1 are gradients of first hidden layer's weights and bias, derived from dZ1

    this process helps us propagate the loss backward, adjusting the weights to minimize the loss, this video helps explain backwards propagation a lot:
    https://www.youtube.com/watch?v=SmZmBKc7Lrs
    """
    def backwardProp(self, actualValues):
        m = self.batchSize
        self.dZ3 = self.outputLayer - actualValues
        self.dW3 = 1 / m * self.dZ3 @ self.A2.T
        self.db3 = 1 / m * np.sum(self.dZ3, axis=1, keepdims=True)
        
        self.dZ2 = self.outputWeights.T @ self.dZ3 * derivReLU(self.A2)
        self.dW2 = 1 / m * self.dZ2 @ self.A1.T
        self.db2 = 1 / m * np.sum(self.dZ2, axis=1, keepdims=True)

        self.dZ1 = self.hidden2Weights.T @ self.dZ2 * derivReLU(self.A1)
        self.dW1 = 1 / m * self.dZ1 @ self.inputLayer.T
        self.db1 = 1 / m * np.sum(self.dZ1, axis=1, keepdims=True)

    """
    update the weights and biases with the gradients calculated from backpropagation
    """
    def updateModel(self):
        self.outputWeights -= (self.learningRate * self.dW3)
        self.outputBias -= (self.learningRate * self.db3)

        self.hidden2Weights -= (self.learningRate * self.dW2)
        self.hidden2Bias -= (self.learningRate * self.db2)

        self.hidden1Weights -= (self.learningRate * self.dW1)
        self.hidden1Bias -= (self.learningRate * self.db1)

    """
    TRAIN!
    """
    def train(self, xNorm, yNorm, epochs):
        dataSize = len(xNorm)
        lossHistory = []

        for epoch in range(epochs):
            indices = np.arange(dataSize)
            np.random.shuffle(indices)
            xNorm, yNorm = xNorm[indices], yNorm[indices]

            for i in range(0, dataSize, self.batchSize):
                xBatch = xNorm[i:i + self.batchSize]
                yBatch = yNorm[i:i + self.batchSize]

                self.forwardProp(xBatch.T)
                self.backwardProp(yBatch.T)
                self.updateModel()

            self.forwardProp(xNorm.T)
            loss = np.mean((self.outputLayer - yNorm.T) ** 2)
            lossHistory.append(loss)
            if epoch % 100 == 0:
                print(f"epoch {epoch + 1}, loss: {loss}")

        return lossHistory

model = MLP(inputSize=1, hidden1Size=64, hidden2Size=32, outputSize=1, batchSize=16, learningRate=0.01)

lossHistory = model.train(xNorm, yNorm, epochs=10)

print(model.outputWeights.shape)
print(model.dZ3.shape)

plt.plot(lossHistory)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()

xMin = x.min()
xMax = x.max()
xRange = np.arange(xMin, xMax, 0.01).reshape(-1, 1)

xRangeNorm = (xRange - xMean) / xStd

model.forwardProp(xRangeNorm.T)

yPred = model.outputLayer.flatten() * yStd + yMean

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Actual Data', color='blue')
plt.plot(xRange, yPred, label='Predicted Line', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Actual Data vs Predicted Line')
plt.legend()
plt.show()
