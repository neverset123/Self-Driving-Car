## Self-Driving-Car

### CV
#### Calibration
1. asymmetric circle patterns can sometimes yield better results, especially when the camera lens has a high distortion.
2. Take multiple images of the calibration pattern at different angles and distances. The more the variety, the better the calibration, as it will cover more of the camera's field of view.

#### Hough Transformation
if the shape of an aera is known, hough transformation can connect points into boundary lines easily.
1. line in image space is point in hough space
2. point in image space is line in hough space
3. intersected point of lines in hough space is a line that passes through all these points
4. points in image space is lines in hough space that must intersact

#### Gradient
Sobel operator is the heart of canny edge detection. it can calculate in defined direction. with defined threshold of magnitude and direction angle, the target object boundary can be filtered.

#### color
1. with matplotlib.image.imread you will get RGB; while with cv2.imread() you will get BGR
2. There is also HSV color space (hue, saturation, and value), and HLS space (hue, lightness, and saturation). 
3. hue represents color independent of any change in brightness; Lightness and Value represent different ways to measure the relative lightness or darkness of a color. For example, a dark red will have a similar hue but much lower value for lightness than a light red. Saturation also plays a part in this; saturation is a measurement of colorfulness. So, as colors get lighter and closer to white, they have a lower saturation value, whereas colors that are the most intense, like a bright primary color (imagine a bright red, blue, or yellow), have a high saturation value.
4. s channel is mostly stable change for lane detection

### Neural Networks
1. AND OR NOT XOR Operation can be expressed by linear Perceptron (weight and bias)
2. In Keras, lambda layers can be used to create arbitrary functions that operate on each image as it passes through the layer.
3. Recently, pooling layers have fallen out of favor. Dropout is a much better regularizer.
4. regularization can be added to all layers' coresponding weights and bias.

#### AlexNet
parallelization of network

#### VGG
a sequence of (3x3 Conv + 2x2 MaxPooling), it uses 224x224 images as input, good for classification transfer learning.
```
from keras.applications.vgg16 import VGG16
img_path = 'your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
model = VGG16(weights='imagenet', include_top=False)
```

#### GoogLeNet
consist of Inception Module, total number of parameters is small

```
from keras.applications.inception_v3 import InceptionV3

model = InceptionV3(weights='imagenet', include_top=False)
```

#### ResNet
architecture is similar to VGG (repetation of layers)
```
from keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet', include_top=False)
```
#### transfer learning
![](./docs/transfer-learning.png)
1. Case 1: Small Data Set, Similar Data
slice off the end of the neural network
add a new fully connected layer that matches the number of classes in the new data set
randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
train the network to update the weights of the new fully connected layer
2. Case 2: Small Data Set, Different Data
slice off most of the pre-trained layers near the beginning of the network
add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set
randomize the weights of the new fully connected layer; freeze all the weights from the pre-trained network
train the network to update the weights of the new fully connected layer
3. Case 3: Large Data Set, Similar Data
remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
randomly initialize the weights in the new fully connected layer
initialize the rest of the weights using the pre-trained weights
re-train the entire neural network
4. Case 4: Large Data Set, Different Data
remove the last fully connected layer and replace with a layer matching the number of classes in the new data set
retrain the network from scratch with randomly initialized weights
alternatively, you could just use the same strategy as the "large and similar" data case

### Kalman Filter
![](./docs/EKF.PNG)
x: state matrix; F: transition matrix; P: state covariance matrix; Q: process covariance matrix;z: measurement vector;H: projection matrix;R: measurement noise covariance matrix;
The measurement update for lidar will also use the regular Kalman filter equations, since lidar uses linear equations. Only the measurement update for the radar sensor will use the extended Kalman filter equations.
1. Kalman Filter
2. Extended Kalman Filter
use first order taylor expansion (jacobian matrix) to extimate nonlinear transformation (from state space to measurement space in measurment update step).  
in practice for the first step in update, use the h function directly to map predicted locations x from Cartesian to polar coordinates, instead of Jacobian matrix.
3. Unscent Kalman Filter
to tacle nonlinear problem in prediciton step by using sigma points to represent normal distribution. number of sigma points is determined by augmented state vectors
#### MatrixXd and VectorXd
VectorXd: 1D
MatrixXd: multi dimension

#### sensor fusion precedure 
![](./docs/fusion.png)
- first measurement - the filter will receive initial measurements of the bicycle's position relative to the car. These measurements will come from a radar or lidar sensor.
- initialize state and covariance matrices - the filter will initialize the bicycle's position based on the first measurement.
then the car will receive another sensor measurement after a time period Δt.
- predict - the algorithm will predict where the bicycle will be after time Δt. One basic way to predict the bicycle location after Δt is to assume the bicycle's velocity is constant; thus the bicycle will have moved velocity * Δt. In the extended Kalman filter lesson, we will assume the velocity is constant.
- update - the filter compares the "predicted" location with what the sensor measurement says. The predicted location and the measured location are combined to give an updated location. The Kalman filter will put more weight on either the predicted location or the measured location depending on the uncertainty of each value.

## Localization
*  Belief = probability
* sense = product followed by normalization
* move = convolution(addition)
![](./docs/filter.png)
1D markov localization(also called histogram filter), Kalman Filters and Particle Filters are Realizations of Bayes Filter.
localization uses map coordinate, and kalman filter uses vehicle coordinate. 
![](./docs/bayes_rule.png)

### 1D Markov model
assumption
-  future states (the next state) is dependent only upon the current state and not on other preceding states
- all observations are independent


recursive structure
![](./docs/Markov.png)
1. Motion model ->prediction step
$$ p(x_t|x_{t-1}^{(i)}, \mu_t,m) = \sum p(x_t|x_{t-1}^{(i)}, \mu_t, m)bel(x_{t-1}^{(i)}) $$
2. observation model -> update step
$$ p(z_t|x_{t}, z_{1:t-1}, \mu_{1:t},m) = p(z_t|x_{t}, m) $$
--> Markov localization
$$ bel(x_t) = \eta p(z_t|x_{t},m)*\hat{bel}(x_t) $$

#### Implementation steps
- extract sensor observations
    * for each pseudo-position:
        + get the motion model probability
        + determine pseudo ranges
        + get the observation model probability
        + use the motion and observation model probabilities to calculate the posterior probability
- normalize posteriors (see helpers.h for a normalization function)
- update priors (priors --> posteriors)
### particle filters
![](./docs/particle_filter.png)
![](./docs/pseudocode.png)
#### bike motion model
![](./docs/bike_motion_model.png)
#### conversion from vehicle coordinates to map coordiantes??


## path planning
### search
1. A* search
2. dynamic proggramming
try to find some technics !!


## Tools
### interactive widgets
https://github.com/jupyter-widgets/ipywidgets

