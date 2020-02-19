#TODOS:
#OK = 3 inputs
#OK = 1 output
#OK = 3 randomly initialized weights (between 0 and 1)
#OK = activation function is sigmoid
#OK = output = sigmoid ( sum(in*weight) )
#OK = error cost function = - input in neuron * error (actual - label) * sigmoid curve gradient (neuron output * (1-neuron output))
#OK = adjust all weights by -error cost function


import random, math;
import numpy as np;


training_data = np.array([ [0,0,1],  [1,1,1], [1,0,1], [0,1,1] ]);
labeled_data = np.array( [0,1,1,0] );


def sendInput(input, label):
	in1=input[0];
	in2=input[1];
	in3=input[2];

	sum=in1*w1+in2*w2+in3*w3;

	return sigmoid(sum, label);


def sigmoid(x, label):
	output = 1 / (1 + math.exp(-x));
	print("Output is:", output, "; expecting", label, "\n");
	return output;


def calculateLoss(output, label):
	loss = math.pow(label-output,1)/2;
	print("Instance training loss:", loss);
	return loss;

def sigmoidCurveGradient(output):
	return output*(1-output);


#instantiate weights as random between -1 and 1
w1 = random.uniform(-1, 1);
w2 = random.uniform(-1, 1);
w3 = random.uniform(-1, 1);
print("Weights:", w1, w2, w3);


#note to self: don't overtrain!
for i in range(10000):
	print("\nITERATION", i+1, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	
	for x in range(3):
		output = sendInput(training_data[x], labeled_data[x]);
		loss = calculateLoss(output, labeled_data[x]);

		#adjust weights to as per loss result (gradient descent)
		w1 = w1 + (training_data[x][0] * loss * sigmoidCurveGradient(output));
		w2 = w2 + (training_data[x][1] * loss * sigmoidCurveGradient(output));
		w3 = w3 + (training_data[x][2] * loss * sigmoidCurveGradient(output));
		

print("Updated weights:", w1, w2, w3);

print("\n\nTESTING!!!\n\n")
for i in range(4):
	print("Testing with imput", training_data[i]);
	sendInput(training_data[i], labeled_data[i]);

# actual test dataset
print("Testing with imput [1,0,0]");
sendInput([1,0,0], 1);