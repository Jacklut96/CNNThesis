import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = '/Users/utente/Desktop/train_filename.tfrecords'
record_iterator = tf.python_io.tf_record_iterator(path=data_path)

with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    # Cast label data into int32
    label = tf.cast(features['train/label'], tf.int32)
    # Reshape image data into the original shape
    #image = tf.reshape(image, [128, 128, 1])
    #print(image)

    # Check the .tfrecord image dimension and plot it
    for string_record in record_iterator:
        # Parse the next example

        example = tf.train.Example()
        example.ParseFromString(string_record)

        img_string = (example.features.feature['train/image']
            .bytes_list
            .value[0])

        # Convert to a numpy array (change dtype to the datatype you stored)
        img_1d = np.fromstring(img_string, dtype=np.float)
        img_1d = np.reshape(img_1d,(128,128))


        i = plt.imshow(img_1d)
        # Print the image shape; does it match your expectations?
        print(img_1d.shape)

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                            min_after_dequeue=10)

    # Initialize all global and local variables
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init_op)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(5):
        img, lbl = sess.run([images, labels])
        img = img.astype(np.uint8)
        for j in range(6):
            plt.subplot(2, 3, j + 1)
            plt.imshow(img[j, ...])
            plt.title('cat' if lbl[j] == 0 else 'dog')
        plt.show()
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()