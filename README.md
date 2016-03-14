# mnist-classifier
Applies a simple neural network to classify handwritten digits. The mnist dataset needed to run this classifier is automatically downloaded. The [Lasagne tutorial](http://lasagne.readthedocs.org/en/latest/user/tutorial.html) serve as a major inspiration.

As the code stands, the classifier runs with
- 1 hidden layer
- 625 hidden units
- A batch size of 100
- 100 epochs

One run before uploading yielded an accuracy of 97.29% on the test set.

## Features:
- Loads the image files from: http://yann.lecun.com/exdb/mnist/
- Displays 10 randomly selected digits together with corresponding labels before classification
- Runs the classification with batch size 100
- Average traning and test loss is then visualized.
- Displays 10 randomly selected digits together with the predictions from the network.
- Visualizes the weight matrices of 10 random units.
