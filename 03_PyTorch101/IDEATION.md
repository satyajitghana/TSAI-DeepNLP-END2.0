# MNIST Adder

Read about loss functions here: https://pytorch.org/docs/stable/nn.html#loss-functions and https://neptune.ai/blog/pytorch-loss-functions and https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7

Use a FNet architecture: https://github.com/rishikksh20/FNet-pytorch

I'll use pytorch lightning with some logging backend

## Training

The model will have 2 outputs, both one hot encoded, first one will be the one hot for the mnist image, and the second one will be the sum

BUT i will first backprop the mnist classifier output for first 20 epochs or so, because that is more important. followed by 20 epochs with backprop of the sum of the numbers also.

The loss will be categorical cross entropy loss for each of the output, followed by a sum or not.
