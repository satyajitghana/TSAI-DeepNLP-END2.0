# 01 Very Basics

## Assignment:

Rewrite the Colab file and:
- remove the last activation function
- make sure there are in total 44 parameters
- run it for 2001 epochs

You must upload your assignment to a public GitHub Repository and share the link as the submission to this assignment 

- Add a readme file to your project and describe these things:
- What is a neural network neuron?
- What is the use of the learning rate?
- How are weights initialized?
- What is "loss" in a neural network?
- What is the "chain rule" in gradient flow?
- This assignment is worth 300pts (150 for Code, and 150 for your readme file (directly proportional to your description).

## Solution

![network](network.png)

1. What is a "neuron" in a neural network

A neuron is the fundamental block of any deep learning network, they can be thought of being analogous to a human brain neuron.

> `input <-> dendrites` : responsible to receive the input from a previous neuron or from a sensor like that of an image sensor.

> `calculation/activation <-> soma` : this is where the actual computation happends, all the inputs are put together and a signal (output/spike) is generated. In an ANN this can be an activation function like Sigmoid, ReLU, TanH etc.

> `output <-> axon` : the output from the activation is carries onto the next neuron layer.

2. What is the use of Learning Rate ?

Let us try to imagine this scenario, you are going on a hike in a hilly area, there are mountains all around, and you are standing somewhere random.
Now close your eyes, and you are given the task to go to the top of a hill, what do you do? you try to take a step, and try to feel, did that step go low, or did that step go high? and after you make that step, you have to make another until you reach the top of a hill.

This "step" that you make is the learning rate, let's say you can small small steps, it will take you a long time to reach the hill top, let's say you can long long steps, now you can easily miss the top of the hill and go forward, now you come down from the top, now you try to climb again, you miss it! again! so you need to take exactly the right "size" of step to reach the top of thhe hill and stay on it!

There are various learning rate schedulers to help with the step size, like `StepLR`, `MultiStepLR`, `ExponentialLR`, `CyclicLR`, `OneCycleLR`, etc.

3. How are weights initialized ?

There are two ways to initialize a network

- Zero Initialization
- Random Initialization

NOTE: Let's not consider biases, since they become zero anyways dure to Batch Normalization

If we initialize all of the weights to zero, then during backprop, the derivative w.r.t to the loss function is the same for every weight, this makes the hidden units symmetric, thus there is no non-linearity in the deep learning model, this makes it no different than a linear model, hence zero initialization is never really used.

In Random Initialization, quite often we sample from the Normal Distrubution, why ?

```
Usually the data distribution in Nature follows a Normal distribution ( few examples like - age, income, height, weight etc., ) . So its the best approximation when we are not aware of the underlying distribution pattern.

Most often the goal in ML/ AI is to strive to make the data linearly separable even if it means projecting the data into higher dimensional space so as to find a fitting "hyperplane" (for example - SVM kernels, Neural net layers, Softmax etc.,). The reason for this being "Linear boundaries always help in reducing variance and is the most simplistic, natural and interpret-able" besides reducing mathematical / computational complexities. And, when we aim for linear separability, its always good to reduce the effect of outliers, influencing points and leverage points. Why? Because the hyperplane is very sensitive to the influencing points and leverage points (aka outliers) - To undertstand this - Lets shift to a 2D space where we have one predictor (X) and one target(y) and assume there exists a good positive correlation between X and y. Given this, if our X is normally distributed and y is also normally distributed, you are most likely to fit a straight line that has many points centered in the middle of the line rather than the end-points (aka outliers, leverage / influencing points). So the predicted regression line will most likely suffer little variance when predicting on unseen data.
```
~ [stackoverflow](https://stats.stackexchange.com/a/303640https://stats.stackexchange.com/a/303640)

Xavier and Kaiming initialization are some of the most commonly used methods for weight initialization, more on it [https://pouannes.github.io/blog/initialization/](https://pouannes.github.io/blog/initialization/)

This is how weights are initialized in `Conv2D` and `Linear` module of PyTorch

```python
def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
```

4. What is "loss" in a neural network ?

`Inputs (x) -> Neural Network -> Output (y^)`, this is a typical flow, now the outputs obtained is `y^` and the expected output for the input `x` is `y`, the parameter of how wrong we are, i.e. `y^` from `y` is the loss. There are various loss functions options we have, namely, `Mean Square Error Loss`, `Binary CrossEntropy Loss`, `CrossEntropy Loss`, `KL Divergence Loss`, `L1 Loss`, `Huber Loss`, etc.

All of these loss function have their purpose (based on the type on inputs to the network, based on the learning rate you want), and affects how the neural network learns.

5. What is the "chain rule" in gradient flow ?

Before trying to understand the chain rule, let's first try to understand what is a "gradient" in a gradient flow.

So basically our main goal is to redule the "loss" of the output, to do that we use gradient. Do you remember gradient, divergence, and curl from Engineering Maths ? :') should have payed more attention then, these terms are really really important.

The gradient of a vector can be interpreted as the "direction and rate of the fastest increase". If the gradient of a function is non-zero at a point p, the direction of the gradient in which the function increases most quickly from p, and the magnitude of the gradient is the rate of increase in that direction, the greatest absolute directional derivative.

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\large&space;\nabla&space;f&space;(x,&space;y,&space;z)&space;=&space;\left&space;(&space;\dfrac{\partial&space;f}{&space;\partial&space;x}&space;,&space;\dfrac{\partial&space;f}{&space;\partial&space;y}&space;,&space;\dfrac{\partial&space;f}{\partial&space;z}&space;\right&space;)&space;=&space;\dfrac{\partial&space;f}{\partial&space;x}\textbf{i}&space;&plus;&space;\dfrac{\partial&space;f}{&space;\partial&space;y}\textbf{j}&space;&plus;&space;\dfrac{\partial&space;f}{&space;\partial&space;z}\textbf{k}" title="\large \nabla f (x, y, z) = \left ( \dfrac{\partial f}{ \partial x} , \dfrac{\partial f}{ \partial y} , \dfrac{\partial f}{\partial z} \right ) = \dfrac{\partial f}{\partial x}\textbf{i} + \dfrac{\partial f}{ \partial y}\textbf{j} + \dfrac{\partial f}{ \partial z}\textbf{k}" />
</div>

And also you must be wondering how the partial derivatives came in, this `grad` was the reason so,

Here would be `grad L` (gradient of Loss)

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\large&space;\nabla&space;L=\begin{bmatrix}&space;\frac{\partial&space;L}{\partial&space;\theta_{0}&space;}&space;\\&space;\frac{\partial&space;L}{\partial&space;\theta_{1}&space;}&space;\\&space;...&space;\\&space;\frac{\partial&space;L}{\partial&space;\theta_{n-1}&space;}&space;\\&space;\end{bmatrix}" title="\large \nabla L=\begin{bmatrix} \frac{\partial L}{\partial \theta_{0} } \\ \frac{\partial L}{\partial \theta_{1} } \\ ... \\ \frac{\partial L}{\partial \theta_{n-1} } \\ \end{bmatrix}" />
</div>

Now does it make all sense?  Since the `grad F` gives the direction of highest increase, we multiple this `grad F` by `-ve 1`, thus now we have the direction of highest decrease in the loss value, and we thus move towards that steepest decreasing loss! amazing right?

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\bg_white&space;\large&space;W_{new}&space;=&space;W-\eta&space;\nabla&space;L" title="\large W_{new} = W-\eta \nabla L" />
</div>

But what is this chain rule?

The problem is that <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;\theta_{0}}" title="\frac{\partial L}{\partial \theta_{0}}" /> may not be directly calculatable.

`w0 -> i_0 -> activation -> o_0`

`w0`: weight 0
`i_0`: multiply the weights with output of previous layer, this is input to this neuron
`activation`: a activation function is applied to `i_0` producing `o_0`

This is the chain rule from maths

<div align="center">
<img src="chain_rule.png">
</div>

see how <img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;z}{\partial&space;s}&space;=&space;\frac{\partial&space;z}{\partial&space;x}\frac{\partial&space;x}{\partial&space;s}" title="\frac{\partial z}{\partial s} = \frac{\partial z}{\partial x}\frac{\partial x}{\partial s}" />

Similarly,

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;\theta_{0}}&space;=&space;\frac{\partial&space;L_{0}}{\partial&space;o_{0}}\frac{\partial&space;o_{0}}{\partial&space;i_{0}}&space;\frac{\partial&space;o_{0}}{\partial&space;\theta_{0}}" title="\frac{\partial L}{\partial \theta_{0}} = \frac{\partial L_{0}}{\partial o_{0}}\frac{\partial o_{0}}{\partial i_{0}} \frac{\partial o_{0}}{\partial \theta_{0}}" />
</div>

That's the chain rule, plain and simple
