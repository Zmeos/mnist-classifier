# mnist-classifier
Applies a simple neural network to classify handwritten digits. The mnist dataset needed to run this classifier is found [here](http://yann.lecun.com/exdb/mnist/). The [Lasagne tutorial](http://lasagne.readthedocs.org/en/latest/user/tutorial.html) serve as a major inspiration.

## Features:
- Loads the gzipped csv-files
- Displays 10 randomly selected digits together with corresponding labels before classification
- Runs the classification with batch size 100
- Average traning and test loss is then visualized.
- Displays 10 randomly selected digits together with the predictions from the network.
