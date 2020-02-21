import tensorflow as tf, numpy as np;


np.set_printoptions(linewidth=400);
print(np.get_printoptions());

mnist = tf.keras.datasets.mnist;
(x_train, y_train), (x_test, y_test) = mnist.load_data();

#normalization of dataset
x_train=x_train/255;
x_test=x_test/255;

print("Dataset dimensions:");
print(len(x_train), len(y_train));
print(type(x_train), type(y_train));
print(x_train.shape, y_train.shape);
print(x_train[0]); #visualize normalized example
print("Input should be (as we're not using CNNs):", x_train[0].shape);


#Assumptions:
#not going to use CNN => input is 28*28=784
#output = an integer between 0 and 9
#try relu as activation
#two hidden layers should be enough
#cost function is good as before


#architecture definition:
model = tf.keras.Sequential();
model.add();