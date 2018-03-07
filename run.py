import os.path
import tensorflow as tf
import helper
from helper import current_time
import warnings
from distutils.version import LooseVersion
import project_tests as tests

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # scale layer_3_out, layer_4_out
    layer_3_out_scaled = tf.multiply(layer3_out, 0.0001, name='layer3_scaled_out')
    layer_4_out_scaled = tf.multiply(layer4_out, 0.01, name='layer4_scaled_out')

    return image_input, keep_prob, layer_3_out_scaled, layer_4_out_scaled, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    init = tf.truncated_normal_initializer(stddev = 0.01)
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    def conv_1x1(x, num_classes, init=init):
        return tf.layers.conv2d(x, num_classes, 1, padding='same'
                                          ,kernel_initializer=init
                                          ,kernel_regularizer=reg)
    def upsample(x, num_classes, depth, strides, init=init,name=None):
        return tf.layers.conv2d_transpose(x, num_classes, depth ,strides, padding='same'
                                         ,kernel_initializer=init
                                         ,kernel_regularizer=reg
                                         ,name=name)
    
    layer7_1x1 = conv_1x1(vgg_layer7_out, num_classes)
    layer4_1x1 = conv_1x1(vgg_layer4_out, num_classes)
    layer3_1x1 = conv_1x1(vgg_layer3_out, num_classes)
            
    upsample1 = upsample(layer7_1x1, num_classes, 4, 2)  # kernel=4*4, strides=(2,2)?????
    layer1 = tf.layers.batch_normalization(upsample1)
    layer1 = tf.add(layer1, layer4_1x1)
    
    upsample2 = upsample(layer1, num_classes, 4, 2)
    layer2 = tf.layers.batch_normalization(upsample2)
    layer2 = tf.add(layer2, layer3_1x1)
    
    output = upsample(layer2, num_classes, 16, 8, name='fconv_output')

    return output


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # no need reshape logits, labels to 2d shape
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer, labels=correct_label)
                                        ,name = 'cross_entropy_loss')
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # add regularization_loss to loss
    loss = tf.add(cross_entropy_loss,reg_loss, name='total_loss')
#     loss =  cross_entropy_loss
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, name='train_op')
    
    return nn_last_layer, train_op, loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, predictions
             , input_image, correct_label, keep_prob, learning_rate, saver):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    
#     max_loss = 10000
    for epoch_i in range(1,epochs+1):
        epoch_loss = 0
        steps = 1
        for images, labels in get_batches_fn(batch_size):
            _, batch_loss = sess.run([train_op, loss],
                                feed_dict={input_image: images, correct_label: labels, keep_prob:0.75, learning_rate:0.0001})
            epoch_loss += batch_loss
        mean_epoch_loss = epoch_loss*1.0/steps
        # print every Epoch
        print("Epoch {}/{} \t".format(epoch_i, epochs), "Epoch Loss: {:.4f}  Last Batch Loss: {:.4f}".format(mean_epoch_loss, batch_loss))
        steps += 1  

        # every 10 epochs, run on mini_test images, save test images with mask
        if epoch_i%10 == 0:
            mean_iou = helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, predictions, keep_prob, input_image, epoch=epoch_i, mini=True, compute_iou=True)
            print("Epoch{} iou:{} ".format(epoch_i,mean_iou) )

        
        # save last 3 model
        if epoch_i > (epochs-3):
            save_path = saver.save(sess, "./models/models-{}-{}-{}/segmentation_model.ckpt".format(epoch_i,mean_epoch_loss,current_time()))
        # TODO:val_loss is decreasing, save model
        
    # use last model test on whole testset
    final_modeliou = helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, predictions, keep_prob, input_image,epoch=epoch_i, mini=False, compute_iou=True)
    print("Final Model iou: ",final_mean_iou )


def run(num_epochs=50, batch_size=8):
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    logs_dir = './tf_logs'
    
    tests.test_for_kitti_dataset(data_dir)
    epochs = num_epochs
    batch_size = batch_size

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        ##--------------------------------------Construction--------------------------------------##
        print(".....................build network")
        input_image, keep_prob, layer_3_out, layer_4_out, layer_7_out = load_vgg(sess, vgg_path)

        # build fully conv layers; skip layers
        output = layers(layer_3_out, layer_4_out, layer_7_out, num_classes)


        correct_label = tf.placeholder(tf.float32, shape = [None, None, None, num_classes],name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, train_op, loss = optimize(output, correct_label, learning_rate, num_classes) # logits =output, 4d dimension

        #prediction; 
        predictions = tf.cast(tf.argmax(tf.nn.softmax(logits), axis=-1),tf.float32, name='predictions')


        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()

        print("......................training")
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, predictions,
                 input_image, correct_label, keep_prob, learning_rate, saver)


if __name__ == '__main__':
    run()
        