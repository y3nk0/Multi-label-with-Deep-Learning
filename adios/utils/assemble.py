"""
Utility functions for constructing MLC models.
"""
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.layers.core import ActivityRegularization

from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.layers import concatenate

from adios.models import MLC

def assemble(name, params):
    if name == 'MLP':
        return assemble_mlp(params)
    elif name == 'ADIOS':
        return assemble_adios(params)
    else:
        raise ValueError("Unknown name of the model: %s." % name)

def assemble_mlp(params):
    """
    Construct an MLP model of the form:
                                X-H-H1-Y
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """

    # X
    X = Input(name='X', shape=(params['X']['dim'],))
    last_layer_name = 'X'
    last_layer = X

    # H
    if 'H' in params:
        kwargs = params['H']['kwargs'] if 'kwargs' in params['H'] else {}
        H_dense = Dense(params['H']['dim'], **kwargs, name='H_dense')(last_layer)
        H_activation = Activation('relu',name='H_activation')(H_dense)
        H_output_name = 'H_activation'
        H_output = H_activation
        if 'batch_norm' in params['H'] and params['H']['batch_norm'] != None:
            H_batch_norm = BatchNormalization(**params['H']['batch_norm'],name='H_batch_norm')(H_output)
            H_output = H_batch_norm
            H_output_name = 'H_batch_norm'
        if 'dropout' in params['H']:
            H_dropout = Dropout(params['H']['dropout'],name='H_dropout')(H_output)
            H_output = H_dropout
            H_output_name = 'H_dropout'
        last_layer_name = H_output_name
        last_layer = H_output

    # H1
    if 'H1' in params:
        kwargs = params['H1']['kwargs'] if 'kwargs' in params['H1'] else {}
        H1_dense = Dense(params['H1']['dim'], **kwargs,name='H1_dense')(last_layer)
        H1_activation = Activation('relu',name='H1_activation')(H1_dense)
        H1_output = H1_activation
        H1_output_name = 'H1_activation'
        if 'batch_norm' in params['H1'] and params['H1']['batch_norm'] != None:
            H1_batch_norm = BatchNormalization(**params['H1']['batch_norm'],name='H1_batch_norm')(H1_output)
            H1_output = H1_batch_norm
            H1_output_name = 'H1_batch_norm'
        if 'dropout' in params['H1']:
            H1_dropout = Dropout(params['H1']['dropout'],name='H1_dropout')(H1_output)
            H1_output = H1_dropout
            H1_output_name = 'H1_dropout'
        last_layer_name = H1_output_name
        last_layer = H1_output

    # Y
    kwargs = params['Y']['kwargs'] if 'kwargs' in params['Y'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y_dense = Dense(params['Y']['dim'], **kwargs,name='Y_dense')(last_layer)
    Y = Activation('sigmoid',name='Y')(Y_dense)
    Y_output = Y
    Y_output_name = 'Y_activation'
    if 'activity_reg' in params['Y']:
       Y_activity_reg = ActivityRegularization(**params['Y']['activity_reg'],name='Y')(Y_output)
       Y_output = Y_activity_reg
       Y_output_name = 'Y_activity_reg'

    # Y = Y_output
    model = MLC(inputs=X, outputs=Y, input_names=params['input_names'], output_names=params['output_names'])

    return model

def assemble_adios(params):
    """
    Construct one of the ADIOS models. The general structure is the following:
                                X-H-(Y0|H0)-H1-Y1,
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """

    # X
    X = Input(name='X', shape=(params['X']['dim'],))
    last_layer_name = 'X'
    last_layer = X

    # H
    if 'H' in params:  # there is a hidden layer between X and Y0
        kwargs = params['H']['kwargs'] if 'kwargs' in params['H'] else {}
        H_Dense = Dense(params['H']['dim'], **kwargs, name='H_dense')(last_layer)
        H_activation = Activation('relu',name='H_activation')(H_Dense)
        H_output_name = 'H_activation'
        H_output = H_activation
        if 'batch_norm' in params['H'] and params['H']['batch_norm'] != None:
            H_batch_norm = BatchNormalization(**params['H']['batch_norm'], name='H_batch_norm')(H_output)
            H_output_name = 'H_batch_norm'
            H_output = H_batch_norm
        if 'dropout' in params['H']:
            H_output = Dropout(params['H']['dropout'], name='H_dropout')(H_output)
            H_output_name = 'H_dropout'
        last_layer_name = 'H_output'
        last_layer = H_output

    # Y0
    kwargs = params['Y0']['kwargs'] if 'kwargs' in params['Y0'] else {}
    if 'W_regularizer' in kwargs:
        kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    Y0_dense = Dense(params['Y0']['dim'], **kwargs, name='Y0_dense')(last_layer)
    Y0_activation = Activation('sigmoid', name='Y0_activation')(Y0_dense)
    Y0_output_name = 'Y0_activation'
    Y0_output = Y0_activation
    if 'activity_reg' in params['Y0']:
        Y0 = ActivityRegularization(**params['Y0']['activity_reg'], name='Y0')(Y0_output)
    #     Y0_output_name = 'Y0_activity_reg'
    #     Y0_output = Y0_activity_reg
    # # model.add(name='Y0')
    # Y0 = Y0_output
    if 'batch_norm' in params['Y0'] and params['Y0']['batch_norm'] != None:
        Y0_batch_norm = BatchNormalization(**params['Y0']['batch_norm'], name='Y0_batch_norm')(Y0_output)
        Y0_output_name = 'Y0_batch_norm'
        Y0_output = Y0_batch_norm

    # H0
    if 'H0' in params:  # we have a composite layer (Y0|H0)
        kwargs = params['H0']['kwargs'] if 'kwargs' in params['H0'] else {}
        H0_dense = Dense(params['H0']['dim'], **kwargs, name='H0_dense')(last_layer)
        H0_activation = Activation('relu', name='H0_activation')(H0_dense)
        H0_output_name = 'H0_activation'
        H0_output = H0_activation
        if 'batch_norm' in params['H0'] and params['H0']['batch_norm'] != None:
            H0_batch_norm = BatchNormalization(**params['H0']['batch_norm'], name='H0_batch_norm')(H0_output)
            H0_output_name = 'H0_batch_norm'
            H0_output = H0_batch_norm
        if 'dropout' in params['H0']:
            H0_dropout = Dropout(params['H0']['dropout'], name='H0_dropout')(H0_output)
            H0_output_name = 'H0_dropout'
            H0_output = H0_dropout
        last_layer_name = ['Y0_output', 'H0_output']
        last_layer = [Y0_output, H0_output]
    else:
        last_layer_name = 'Y0_output'
        last_layer = Y0_output

    # H1
    if 'H1' in params:  # there is a hidden layer between Y0 and Y1
        kwargs = params['H1']['kwargs'] if 'kwargs' in params['H1'] else {}
        if isinstance(last_layer_name, list):
            merged_vector = concatenate(last_layer, axis=-1)
            H1_dense = Dense(params['H1']['dim'], **kwargs, name='H1_dense')(merged_vector)
        else:
            H1_dense = Dense(params['H1']['dim'], **kwargs, name='H1_dense')(last_layer)
        H1_activation = Activation('relu', name='H1_activation')(H1_dense)
        H1_output_name = 'H1_activation'
        H1_output = H1_activation
        if 'batch_norm' in params['H1'] and params['H1']['batch_norm'] != None:
            H1_batch_norm = BatchNormalization(**params['H1']['batch_norm'], name='H1_batch_norm')(H1_output)
            H1_output_name = 'H1_batch_norm'
            H1_output = H1_batch_norm
        if 'dropout' in params['H1']:
            H1_dropout = Dropout(params['H1']['dropout'], name='H1_dropout')(H1_output)
            H1_output_name = 'H1_dropout'
            H1_output = H1_dropout
        last_layer_name = 'H1_output'
        last_layer = H1_output

    # Y1
    kwargs = params['Y1']['kwargs'] if 'kwargs' in params['Y1'] else {}
    if 'W_regularizer' in kwargs:
      kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
    if isinstance(last_layer_name, list):
        # We can then concatenate the two vectors:
        merged_vector = concatenate(last_layer, axis=-1)
        Y1_Dense = Dense(params['Y1']['dim'], **kwargs, name='Y1_dense')(merged_vector)
    else:
        Y1_Dense = Dense(params['Y1']['dim'], **kwargs, name='Y1_dense')(last_layer)

    Y1_activation = Activation('sigmoid', name='Y1_activation')(Y1_Dense)
    Y1_output_name = 'Y1_activation'
    Y1_output = Y1_activation
    if 'activity_reg' in params['Y0']:
        Y1 = ActivityRegularization(**params['Y1']['activity_reg'], name='Y1')(Y1_output)
    #     Y1_output= Y1_activity_reg
    # Y1 = Y1_output

    model = MLC(inputs=X, outputs=[Y0,Y1], input_names=params['input_names'], output_names=params['output_names'])

    return model
