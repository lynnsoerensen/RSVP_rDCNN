from pathlib import Path
import tensorflow as tf

from resources.rcnn_sat.rcnn_sat.bl_net import BLConvLayer


def bl_net(input_tensor, classes, n_timesteps=8, cumulative_readout=False, alphas=[0], beta=0, output_activation='softmax'):
    '''Build the computational graph for the model

    Note that evaluations based on model outputs will reflect instantaneous
    rather than cumulative readouts

    Args:
        input_tensor: Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: int, number of classes to classify images into
        n_timesteps: int, number of model time steps to build
        cumulative_readout: Bool, if True then the outputs correspond to a
            cumulative readout on each time step if True then they
            correspond to a instant readout

    Returns:
        model
    '''

    data_format = tf.keras.backend.image_data_format()
    norm_axis = -1 if data_format == 'channels_last' else -3

    # initialise trainable layers (RCLs and linear readout)
    layers = [
        BLConvLayer(96, 7, 'RCL_0'),
        BLConvLayer(128, 5, 'RCL_1'),
        BLConvLayer(192, 3, 'RCL_2'),
        BLConvLayer(256, 3, 'RCL_3'),
        BLConvLayer(512, 3, 'RCL_4'),
        BLConvLayer(1024, 3, 'RCL_5'),
        BLConvLayer(2048, 1, 'RCL_6'),
    ]
    readout_dense = tf.keras.layers.Dense(
        classes, kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        name='ReadoutDense')

    # initialise list for activations and outputs
    n_layers = len(layers)
    activations = [[None for _ in range(n_layers)]
                   for _ in range(n_timesteps)]

    suppression = [[[None for _ in range(len(alphas))]  # Depending on how many exponentials are used.
                    for _ in range(n_layers)]
                   for _ in range(n_timesteps)]
    suppression_sum = [[None for _ in range(n_layers)]
                       for _ in range(n_timesteps)]

    presoftmax = [None for _ in range(n_timesteps)]
    outputs = [None for _ in range(n_timesteps)]

    # build the model
    for t in range(n_timesteps):
        for n, layer in enumerate(layers):

            # get the bottom-up input
            if n == 0:
                # B conv on the image does not need to be recomputed
                b_input = input_tensor if t == 0 else None
            else:
                # pool b_input for all layers apart from input
                b_input = tf.keras.layers.MaxPool2D(
                    pool_size=(2, 2),
                    name='MaxPool_Layer_{}_Time_{}'.format(n, t)
                )(activations[t][n - 1])

            # get the lateral input
            if t == 0:
                l_input = None
            else:
                l_input = activations[t - 1][n]

            # convolutions
            x_tn = layer(b_input, l_input)
            # batch-normalisation
            x_tn = tf.keras.layers.BatchNormalization(
                norm_axis,
                name='BatchNorm_Layer_{}_Time_{}'.format(n, t))(x_tn)

            if t == 0:
                for a in range(len(alphas)):
                    # Initialize with zeros.
                    suppression[t][n][a] = tf.keras.layers.Lambda(lambda x: x * 0, name='State_Layer_{}_Time_{}_{}'.format(n, t, a))(
                        x_tn)

            else:
                for a in range(len(alphas)):
                    suppression[t][n][a] = tf.keras.layers.Lambda(lambda x: x * alphas[a])(suppression[t - 1][n][a])
                    tmp = tf.keras.layers.Lambda(lambda x: x * (1 - alphas[a]))(activations[t - 1][n])
                    suppression[t][n][a] = tf.keras.layers.Add(name='State_Layer_{}_Time_{}_{}'.format(n, t, a))(
                        [suppression[t][n][a], tmp])

            if len(alphas) > 1:
                # sum up the respective exponentials
                suppression_sum[t][n] = tf.keras.layers.Add(name='State_Layer_{}_Time_{}'.format(n, t))(
                            suppression[t][n])
            else:
                suppression_sum[t][n] = suppression[t][n][0]

            # apply suppression
            supp_tmp = tf.keras.layers.Lambda(lambda x: x * beta)(suppression_sum[t][n])
            x_tn = tf.keras.layers.Subtract(name='Adapt_Layer_{}_Time_{}'.format(n, t))([x_tn, supp_tmp])

            # ReLU
            activations[t][n] = tf.keras.layers.Activation(
                'relu', name='ReLU_Layer_{}_Time_{}'.format(n, t))(x_tn)

        # add the readout layers
        x = tf.keras.layers.GlobalAvgPool2D(
            name='GlobalAvgPool_Time_{}'.format(t)
        )(activations[t][-1])
        presoftmax[t] = readout_dense(x)

        # select cumulative or instant readout
        if cumulative_readout and t > 0:
            x = tf.keras.layers.Add(
                name='CumulativeReadout_Time_{}'.format(t)
            )(presoftmax[:t + 1])
        else:
            x = presoftmax[t]
        outputs[t] = tf.keras.layers.Activation(
            output_activation, name=output_activation + '_Time_{}'.format(t))(x)

    # create Keras model and return
    model = tf.keras.Model(
        inputs=input_tensor,
        outputs=outputs,
        name='bl_net')

    return model


def makeModel_control_adaptation(model_name, n_timesteps=8, trainingSet='ecoset', activation='linear', outputLayer=None, alphas=[0], beta=0):

    weight_dir = Path(__file__).parent.parent.parent / 'resources' / 'rcnn_sat' / 'weights'

    if trainingSet == 'ecoset':
        classes = 565
    elif trainingSet == 'imagenet':
        classes = 1000
    else:
        raise ValueError()


    input_layer = tf.keras.layers.Input((128, 128, 3))


    if trainingSet == 'ecoset':
        classes = 565
    elif trainingSet == 'imagenet':
        classes = 1000

    model = eval(model_name + '_net')(input_layer, n_timesteps=n_timesteps, classes=classes,
                                      output_activation=activation,
                                      alphas=alphas, beta=beta)

    model.load_weights(str(weight_dir /f"{model_name}_{trainingSet}.h5"),by_name=True)

    if outputLayer is not None:

        outputs = [layer.output for layer in model.layers if layer.name.startswith(outputLayer)]
        model = tf.keras.Model(
            inputs=model.input,
            outputs=outputs)

    return model
