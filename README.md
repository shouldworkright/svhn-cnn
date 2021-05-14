# Street View House Numbers (SVHN) CNN

This repository houses a series of convolutional neural networks developed for UMass Dartmouth's CIS465 Topics in Computer Vision course. This README contains selections of information from the research paper written as part of this project. The full paper can be accessed under <a href="https://github.com/shouldworkright/svhn-cnn/blob/main/docs/svhn_cnn_research_paper.pdf">docs/svhn_cnn_research_paper.pdf</a>.

## Introduction

The use of convolutional neural networks has become commonplace in the realm of
computer image recognition. There have been several datasets that have been difficult to make
reliable networks to recognize, however, given the presence of background noise and images
taken at various angles. One such dataset is the <a href="http://ufldl.stanford.edu/housenumbers/">Street View House Numbers (SVHN)</a>, which has
many of these issues. This is highly prevalent in many examples of scene text detection, both in
detecting the presence of text and then classifying what it says character by character. In this
study, we focus on recognizing single numbers extracted from images of houses taken from
Google Street View. These images have pieces of other numbers in them as well as various
skewing due to the angle the picture was taken and various background colors. We implement a
CNN and experiment with various parameters and helper functions to examine their effect on
mitigating these noisy background features.

## Dataset

For training our model we first thought to use the <a href="http://yann.lecun.com/exdb/mnist/">Modified National Institute of
Standards and Technology or MNIST database</a> as it contained sixty thousand training images
which made it have lots of variation and would be a challenge for the neural network to learn
from. However, this data contained mostly only handwriting which can be very different from
actual scene text that would be put against the trained model. So that dataset was scrapped and
we moved on to The Street View House Numbers or SVHN dataset. The SVHN is a real-world
image dataset for developing machine learning and object recognition algorithms with minal
requirement for data preprocessing and formatting. This dataset can be seen as similar in flavor
to the MNIST dataset, but it incorporates an order of magnitude more labeled data and comes 2
from a significantly harder, unsolved, real world problem. While the dataset advertises that it
has over six hundred thousand images, this was a bit more than what we need for our model. The
dataset was broken down into three different matlab files, one for training which contained
fifty-eight thousand six hundred and five image, one for testing which contained fourteen
thousand six hundred and fifty-two images, and lastly the additional or extra set which contained
five hundred and thirty-one thousand one hundred and thirty-one images. However, the extra set
was not utilized for this experiment.

## The Newtork Architecture

<p align="center" width="100%">
  <img src="https://github.com/shouldworkright/svhn-cnn/blob/main/docs/assets/fig_9.jpg?raw=true">
</p>

The final iteration of our CNN architecture consists of a series of 4 convolutional layers, each followed by a max pooling layer. The initial convolutional layer includes 32 output filters using ReLU activation. Each subsequent convolutional layer increases the filter size by a factor of 2. The output of the final convolutional-maxpool pair feeds into a flatten layer that reduces the dimensionality of the input. The input of the flatten layer is then fed into a dense layer using ReLU activation that further reduces the dimensionality of the output, followed by a dropout layer with a frequency value of 0.3 to prevent overfitting. The final layer in our architecture is another dense layer that uses softmax activation to produce 10 probabilities, the highest of which representing the most likely digit contained within the image. The ImageDataGenerator parameters were reduced to 8 degrees rotation, 10% height variation, 10% shear intensity, and 95%-105% zoom. The batch size was increased dramatically to 250 over the same 100 epochs. Early stopping and learning rate reduction were also set to monitor validation loss with a patience of 5 epoch in early stopping and 2 epochs in learning rate reduction, which featured the original 50% learning rate reduction. The previous graphs show higher oscillation patterns in the validation loss, which could make validation loss a better candidate for what these functions should monitor.

<p align="center" width="100%">
  <img src="https://github.com/shouldworkright/svhn-cnn/blob/main/docs/assets/fig_10.jpg?raw=true">
</p>

The final model achieved a training accuracy of 90.6%. After validating the model on the test dataset, it was able to achieve a final accuracy of 92.5%. By both decreasing the number of epochs and modifying our network's architecture, the validation accuracy of the model was increased. Again, the effect of learning rate reduction is highly prevalent. After training, the model’s cross entropy loss stood at 0.3. After validating the model with the test dataset, the cross entropy loss stood at 0.27. The results produced here ran analogous to the training accuracy and validation results. It seemed as if a combination of tweaking our CNN’s architecture and slightly decreasing the number of epochs had a net positive impact on our model’s performance. This further decrease in loss could also, at least in part, be attributed to using validation loss in our early stopping and learning rate reduction function as was earlier hypothesized.

<p align="center" width="100%">
  <img src="https://github.com/shouldworkright/svhn-cnn/blob/main/docs/assets/fig_11.jpg?raw=true">
</p>

In comparison to the previous two confusion matrices, no drastic changes have taken place. The overall number of correctly identified digits has increased in comparison to previous results.

## Implementation and Future Work

To create a CNN capable of being implemented into a fully-realized scene text recognition system, the accuracy of said model would have to exceed the accuracy of the final model iteration. Whereas the final model iteration achieved an accuracy of 92.5%, an acceptable CNN would display a validation accuracy in the range of 98-100%.
All CNN iterations were produced using the <a href="https://keras.io/">Keras TensorFlow API</a>. In the future, CNN architectures constructed via alternative Python libraries could be explored. Promising choices include <a href="https://caffe.berkeleyvision.org/">Caffe</a> (a deep learning framework developed by Berkeley Artificial Intelligence Research), <a href="https://mxnet.apache.org/versions/1.8.0/">Apache’s MXNet library</a> (which is compatible with Python, C++, Java, and a host of other programming languages), or <a href="https://pypi.org/project/Theano/">Theano</a> (which may provide superior means of controlling and optimizing low-level matrix operations compared to MIT’s Keras API).
Should such a CNN be successfully constructed, it would be appended to a hardware system that would feed camera frame data into a text detection/localization algorithm. Once the text is successfully localized, each individual digit would be passed to the CNN. The final output filter would produce a digit readable by some other software system. Applications of a fully realized system would include autonomous vehicle address-finding, automated mail delivery, or automated reading of license plates.

## Conclusion

This experiment sought to find a CNN layer structure, hyperparameter, and utility function configuration in order to achieve a model accuracy of over 90% on the SVHN 32x32 dataset. It was discovered that dropout layers between the convolutional and max-pooling layers may have had a negative effect on the model. This is evidenced in the progression of having 2 dropout layers in the first model, 4 in the second, and then just one between the 2 dense layers in the last. A simpler pattern of:

> Convolutional→MaxPooling→Convolutional→MaxPooling→Convolutional→MaxPooling→ Convolutional→MaxPooling→Flatten→Dense→Dropout→Softmax 

seemed to work better than incorporating additional batch normalization layers or having convolutional layers next to one another.
	Unequivocally, learning rate reduction and early stopping both had a clearly beneficial role in training as evidenced by the accuracy and loss curves. It was discovered, however, that observing these graphs in previous versions can assist in providing clues of what metrics these functions should follow. In this case, it was evident that there was more variability in the validation loss curves, which were ultimately chosen as the monitoring method for them. Likewise, the ImageDataGenerator also assisted in refining loss and increasing accuracy, as it was the major differentiating factor between models 1 and 2. The parameters initially presented to this function were likely too high as there was already a high amount of variability in the images. If any number is rotated too far, it is likely that it may no longer be recognized as the same number. In this case, it may even have a detrimental effect in training. The dataset may also have been large enough to account for most of this variation as is, but there was likely some benefit in its addition to the dataset.
  
The AMSGrad variant proposed in the 2018 International Conference on Learning Representations (ICLR 2018) article, <a href="https://arxiv.org/abs/1904.09237">“On the Convergence of Adam and Beyond”</a>, was also tested for its efficacy in these experiments. Though likely not a tremendous influence, its introduction came before a marked improvement between models 1 and 2. Admittedly it’s hard to tell what the exact effect was without isolating its use from other parameter changes.

Model performance seemed greater with a higher number of epochs, especially with the callback functions aiding in the prevention of overtraining. Likewise, larger batch sizes also seemed to have a positive impact on model performance as the first 2 models had batch sizes of 50 and the last a batch size of 250.

There are limitations to the conclusions we can draw from this data, however. Given that models often took upwards of 5+ hours to train, many parameters were changed simultaneously in an attempt to diminish the time needed to train a successful model. This makes it difficult to discern the individual effects of each parameter change. This was attempted to be mitigated by training several different versions in parallel on the same machine, however, this ultimately led to slowed system performance, hardware overheating and eventually the application crashing before it was fully trained. For future iterations and study, it is suggested to break these parameters out into single changes to examine their individual effects. This would be made much more possible by running the models on a high-powered GPU.

### Contributors
Alexander Moulton - https://github.com/moul-10
Daniel Mello - https://github.com/shouldworkright
Ashley Famularo - https://github.com/afamularo99
