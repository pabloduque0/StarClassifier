import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class StarClassifier():

    def __init__(self, training_data, training_labels, testing_data, testing_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels

    def train_and_test(self):

        # Parameters
        learning_rate = 0.0001
        training_epochs = 2000
        batch_size = 10
        display_step = 100

        # Network Parameters
        n_input = 100 # 
        n_classes = 2 #

        n_hidden_1 = 2 # 1st layer number of neurons
         
        # tf Graph input
        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Create model
        def multilayer_perceptron(x):
            layer_1 = tf.nn.softmax(tf.matmul(x, weights['h1']))
            output_layer = tf.matmul(layer_1, weights['out'])
            return output_layer

        # Construct model
        logits = multilayer_perceptron(X)

        # Define loss and optimizer
        error = tf.losses.mean_squared_error(predictions=logits, labels=Y)
        #mean_squared_error = np.sqrt(tf.losses.mean_squared_error(logits, Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(error)
        # Initializing the variables
        init = tf.global_variables_initializer()

        ordered_costs = []
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = len(self.training_data)//batch_size
                # Loop over all batches
                i = 0
                while i < len(self.training_data):
                    batch_x = self.training_data[i:i+batch_size]
                    batch_y = self.training_labels[i:i+batch_size]
                    #print("Batch: ", self.training_labels[i:i+batch_size] == self.training_labels[i+batch_size:i+(batch_size*2)])
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, cost = sess.run([train_op, error], feed_dict={X: batch_x, Y: batch_y})
                    ordered_costs.append(cost)
                    # Compute average loss
                    avg_cost += cost / total_batch

                    i += batch_size

                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("Optimization Finished!")

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({X: self.training_data, Y: self.testing_labels}))

        self.plot_costs(ordered_costs, training_epochs, total_batch)

    def plot_costs(self, costs, training_epochs, total_batch):

        plt.plot(np.arange(0, training_epochs*total_batch), costs)
        plt.ylabel('Cost')
        plt.xlabel('Epochs')

