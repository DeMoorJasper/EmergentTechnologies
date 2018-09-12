

# Adapted Example of Scikit-learn website

# Standard scientific Python imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# the dataset is made out of 8x8 arrays of float numbers representing a
# grey-scale image. The digit is known before-hand and present in 'target'
# let's visualize the first 4 digit images.

# this can be done using pandas or standard python,numpy
# benefit of pandas is table-like visualization of contents.
# makes it easier to work with.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

df_images_and_labels = pd.DataFrame(images_and_labels)

plt.show()

for index in np.arange(4):
    plt.subplot(2,4, index + 1)
    plt.axis('off')
    plt.imshow(df_images_and_labels.iloc[index][0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % df_images_and_labels.iloc[index][1])

plt.show()
# To apply the algorithm on the digits we need to put it in
# as a 1-d array of floats. digits.images are 2-d arrays
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# pandas to visualize data transformation
df = pd.DataFrame(data)
df.info()
df.head()

# boot the algorithm
classifier = svm.SVC(gamma=0.001)

# train the algorithm on first half of digit images
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digits on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

#built-in report of scikyt-learn for this algorithm
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


#plot prediction vs image for test set (second half)
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()