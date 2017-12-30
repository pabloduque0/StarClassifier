import tensorflow as tf

class StarClassifier():

    def __init__(self, training_data, training_labels, testing_data, testing_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels

    def train(self):

        # Parameters
        learning_rate = 0.00001
        training_epochs = 5000
        batch_size = 10
        display_step = 1

        # Network Parameters
        n_input = 100 # 
        n_classes = 2 #

        n_hidden_1 = 51 # 1st layer number of neurons
         
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
            layer_1 = tf.nn.tanh(tf.matmul(x, weights['h1']))
            output_layer = tf.matmul(layer_1, weights['out'])
            return output_layer

        # Construct model
        logits = multilayer_perceptron(X)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = len(self.training_data)//batch_size
                # Loop over all batches
                for i in range(total_batch):
                    batch_x = self.training_data[i:i+batch_size]
                    batch_y = self.training_labels[i:i+batch_size]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                    Y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("Optimization Finished!")

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({X: self.testing_data, Y: self.testing_labels}))