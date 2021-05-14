# Street View House Numbers (SVHN) CNN

The use of convolutional neural networks has become commonplace in the realm of
computer image recognition. There have been several datasets that have been difficult to make
reliable networks to recognize, however, given the presence of background noise and images
taken at various angles. One such dataset is the Street View House Numbers (SVHN), which has
many of these issues. This is highly prevalent in many examples of scene text detection, both in
detecting the presence of text and then classifying what it says character by character. In this
study, we focus on recognizing single numbers extracted from images of houses taken from
Google Street View. These images have pieces of other numbers in them as well as various
skewing due to the angle the picture was taken and various background colors. We implement a
CNN and experiment with various parameters and helper functions to examine their effect on
mitigating these noisy background features.

## Dataset

For training our model we first thought to use the Modified National Institute of
Standards and Technology or MNIST database as it contained sixty thousand training images
which made it have lots of variation and would be a challenge for the neural network to learn
from. However, this data contained mostly only handwriting which can be very different from
actual scene text that would be put against the trained model. So that dataset was scrapped and
we moved on to The Street View House Numbers or SVHN dataset. The SVHN is a real-world
image dataset for developing machine learning and object recognition algorithms with minal
requirement for data preprocessing and formatting. This dataset can be seen as similar in flavor
to the MNIST dataset, but it incorporates an order of magnitude more labeled data and comes 2
from a significantly harder, unsolved, real world problem [1]. While the dataset advertises that it
has over six hundred thousand images, this was a bit more than what we need for our model. The
dataset was broken down into three different matlab files, one for training which contained
fifty-eight thousand six hundred and five image, one for testing which contained fourteen
thousand six hundred and fifty-two images, and lastly the additional or extra set which contained
five hundred and thirty-one thousand one hundred and thirty-one images. However, the extra set
was not utilized for this experiment.
