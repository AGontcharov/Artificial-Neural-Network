# Artificial-Neural-Network
> A working artificial brain

An Artifical Neural Network implemented from scratch based on the neural structure of the human brain using Python and the Numpy library. Features (set of input) from a trainning data file are sent to the input nodes of the neural network. The neural network is then trained over a number of epochs adjusting the weights for its connection based on the actual output through forward and back propagation. Finally, after being trained the testing data is then used as input to the neural network and an output is spit back on a scale of 0 to 1.

## Installation

Make sure you have the [SciPY stack](http://scipy.org/install.html) installed on your system.

Linux & OS X:

```sh
git clone https://github.com/AGontcharov/Artificial-Neural-Network.git
cd Artificial-Neural-Network/
chmod u+x NeuralNetwork.py
```

Windows:

```sh
Not yet available
```
##Running

The Artificial Neural Network accepts a number of arguments that must be supplied to it and are listed in the following order:

| Argument |          Description          |
|----------|-------------------------------|
| arg1     | Path to the training data set |
| arg2     | Number of input nodes         |
| arg3     | Number of hidden nodes        |
| arg4     | Number of output nodes        |
| arg5     | Path to the testingt data set |

The number of elements (columns) in the feature must match the number of input nodes.
Likewise, the number of elements in the output must match the number of output nodes.

Linux & OS X:

```sh
./NeuralNetwork.py [arg1] [arg2] [arg3] [arg4] [arg5]
```

## Meta

Alexander Gontcharov – alexander.goncharov@gmail.com

[https://github.com/AGontcharov/](https://github.com/AGontcharov/)
