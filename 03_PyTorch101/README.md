# PyTorch 101

1.  Write a neural network that can:
    1.  take 2 inputs:
        1.  an image from MNIST dataset, and
        2.  a random number between 0 and 9
    2.  and gives two outputs:
        1.  the "number" that was represented by the MNIST image, and
        2.  the "sum" of this number with the random number that was generated and sent as the input to the network  
            ![assign.png](https://canvas.instructure.com/courses/2734517/files/136727252/preview) 
    3.  you can mix fully connected layers and convolution layers
    4.  you can use one-hot encoding to represent the random number input as well as the "summed" output.
2.  Your code MUST be:
    1.  well documented (via readme file on github and comments in the code)
    2.  must mention the data representation
    3.  must mention your data generation strategy
    4.  must mention how you have combined the two inputs
    5.  must mention how you are evaluating your results
    6.  must mention "what" results you finally got and how did you evaluate your results
    7.  must mention what loss function you picked and why!
    8.  training MUST happen on the GPU
3.  Once done, upload the code with shot  **training logs in the readme file**  from colab to GitHub, and share the GitHub link (public repository)
4.  Please note that the deadline for your quiz is: May 21 6am

## Solution

Notebook: [https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/03_PyTorch101/MNIST_Adder.ipynb](https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/03_PyTorch101/MNIST_Adder.ipynb)

Colab Link: [https://githubtocolab.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/03_PyTorch101/MNIST_Adder.ipynb](https://githubtocolab.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/03_PyTorch101/MNIST_Adder.ipynb)

### Data Representation

The Data sample is always represented as `(input, target)` where both the input and target have to be `torch.Tensor`, conversion to Tensor is taken care by PyTorch's `DataLoader` for primitive types, for MNIST Image we need to use the torchvision transforms' `ToTensor` method.

My Representation for `(input, target` is `(img, random_number),  (target, target + random_number)`

### Data Generation Strategy

```python
class  MNISTAdder(MNIST):
	"""
	A modified version of MNIST
	It adds a random number along with the mnist image, and the target now is
	the mnist image's target + the random number
	For example:
	[MNIST Image for 1], 5 => 1, 6
	Usage:
	>>> mnistadder = MNISTAdder(root='data')
	>>> dloader = DataLoader(mnistadder)
	>>> batch = next(iter(dloader))
	"""
	def  __init__(
		self,
		*args,
		**kwargs
		) -> None:
		super(MNISTAdder,  self).__init__(*args, **kwargs)

	def  __getitem__(self, index: int) -> Tuple[Any, Any]:
		img, target = super(self.__class__,  self).__getitem__(index)
		random_number = np.random.randint(low=0, high=10)

		return  (img, random_number),  (target, target + random_number)

```

### The Model

The model's primary task is to figure out the MNIST Image Classification, for that I simply copy-paste one of my past MNIST Classification Model and only use the backbone layers, which is upto the GAP Layer + Conv1D,

Now after this i got 20 Channels, there 20 channels are flattened out, and concatenated with the one-hot representation of the "random" number, so 30 features in total, Now from here on i use Linear Layers which will do the addition, and out of that I will extract 10 features of my MNIST classification, and 19 features of my Addition classification.

Why 19 features ? because 0-9 (MNIST) + 0-9 (Random) will be 0-18 numbers, or 19 possible numbers in total.

```python
    def forward(self, mnist_img, rand_num):
        rand_num = F.one_hot(rand_num, num_classes=self.num_classes)
        
        # mnist embedding: 1x20
        mnist_embed = self.mnist_base(mnist_img)

        # concat the mnist embedding and the random number = 30 features
        ccat = torch.cat([mnist_embed, rand_num], dim=-1)

        pre_out = self.prefinal_layer1(ccat)
        pre_out = self.prefinal_layer2(pre_out)

        mnist_out = self.mnist_final_layer(pre_out)
        adder_out = self.adder_final_layer(pre_out)

        return mnist_out, adder_out
```

### Loss Functions

The first task is to classify the mnist image, so obvious choice of loss function for that is cross entropy loss, which is basically `log_softmax` + `nll_loss` (negative log likelihood).

But for the adder, since i used a one hot representation for the output, cross entropy seems like a good choice for it too. So i went with that.

Now both these losses are combined by simple addition. We can also give more weightage to the MNIST Loss, because without the correct prediction for MNIST we cannot give the correct output for the adder.

```python
	...
	self.loss = nn.CrossEntropyLoss()
	...
	...
	# both mnist and adder use cross entropy loss
	mnist_loss = self.loss(mnist_pred, mnist_y)
	adder_loss = self.loss(adder_pred, adder_y)
	
	# final loss is sum of the two loss
	loss = mnist_loss + adder_loss
```

### Results Evaluation

The Output obtained from the model is evaluated against the target using accuracy functions, which basically does.

$$\text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)$$

Where  $y$ is a tensor of target values, and $\hat{y}$ is a tensor of predictions. [Source Code](https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/functional/classification/accuracy.py)

```python
        mnist_pred = torch.argmax(F.log_softmax(mnist_pred, dim=1), dim=1)
        adder_pred = torch.argmax(F.log_softmax(adder_pred, dim=1), dim=1)

        mnist_acc = accuracy(mnist_pred, mnist_y)
        adder_acc = accuracy(adder_pred, adder_y)
```

The total accuracy is the product of `mnist_acc` and `adder_acc`, because end of the day we need both the mnist output AND the added value output to be correct.

## Training Logs

The Model was trained for `20` epochs with `SWE: Stochastic Weight Averaging` with `OneCycleLR`

**Model Parameters**

```
  | Name              | Type             | Params
-------------------------------------------------------
0 | mnist_base        | MNISTModel       | 9.9 K 
1 | prefinal_layer1   | Sequential       | 1.9 K 
2 | prefinal_layer2   | Sequential       | 3.7 K 
3 | mnist_final_layer | Sequential       | 600   
4 | adder_final_layer | Sequential       | 1.1 K 
5 | loss              | CrossEntropyLoss | 0     
-------------------------------------------------------
17.3 K    Trainable params
0         Non-trainable params
17.3 K    Total params
0.069     Total estimated model params size (MB)
```

**Model Testing**

```
>>> trainer.test()
>>>
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Testing: 100%

78/78 [00:01<00:00, 55.57it/s]

--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'adder_acc': 0.9921875,
 'mnist_acc': 0.9931890964508057,
 'total_acc': 0.9855143427848816,
 'val_loss': 0.057633303105831146}
```

Achieved a Total Accuracy of `98.5%` in `20` epochs


| Overall Accuracy |  |
|--|--|
| ![](https://raw.githubusercontent.com/extensive-nlp/TSAI-DeepNLP-END2.0/a6ee4fe0a15be3447cd5a839672b072be9edff4f/03_PyTorch101/training_logs/total_acc.svg =400x400) |  |


| MNIST Accuracy | Adder Accuracy |
|--|--|
| ![](https://raw.githubusercontent.com/extensive-nlp/TSAI-DeepNLP-END2.0/a6ee4fe0a15be3447cd5a839672b072be9edff4f/03_PyTorch101/training_logs/mnist_acc.svg =400x400) | ![](https://raw.githubusercontent.com/extensive-nlp/TSAI-DeepNLP-END2.0/a6ee4fe0a15be3447cd5a839672b072be9edff4f/03_PyTorch101/training_logs/adder_acc.svg =400x400) |

| Learning Rate | Validation Loss |
|--|--|
| ![](https://raw.githubusercontent.com/extensive-nlp/TSAI-DeepNLP-END2.0/a6ee4fe0a15be3447cd5a839672b072be9edff4f/03_PyTorch101/training_logs/learning_rate.svg =400x400) | ![](https://raw.githubusercontent.com/extensive-nlp/TSAI-DeepNLP-END2.0/a6ee4fe0a15be3447cd5a839672b072be9edff4f/03_PyTorch101/training_logs/val_loss.svg =400x400) |

**LR Finder**

![lr finder](https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/03_PyTorch101/training_logs/lr_finder.png?raw=true)

```
>>> lr finder suggested lr: 0.16601414902764572
```

**Sample Test Outputs**

![enter image description here](https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/03_PyTorch101/training_logs/test_output1.png?raw=true)

![enter image description here](https://github.com/extensive-nlp/TSAI-DeepNLP-END2.0/blob/main/03_PyTorch101/training_logs/test_output2.png?raw=true)
