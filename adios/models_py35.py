def assemble_adios_PYTHON3(params):
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
        Y0_activity_reg = ActivityRegularization(**params['Y0']['activity_reg'], name='Y0')(Y0_output)
        Y0_output_name = 'Y0_activity_reg'
        Y0_output = Y0_activity_reg
    # model.add(name='Y0')
    Y0 = Y0_output
    # Y0 = Dense(params['Y0']['dim'])(Y0_output)
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
            H1_dense = Dense(params['H1']['dim'], **kwargs, name='H1_dense')(last_layer)
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
        Y1_activity_reg = ActivityRegularization(**params['Y1']['activity_reg'], name='Y1')(Y1_output)
        Y1_output= Y1_activity_reg
    Y1 = Y1_output
    # Y1 = Dense(num_classes)(Y1_output)

    model = MLC(inputs=X, outputs=[Y0,Y1])

    return model

def assemble_adieu(params):
    """
    Construct our approach based on ADIOS models. The general structure is the following:
                                X-H0-(Y0|H0)-H1-(Y1|H1)-...HN-(YN|HN)
    where all the H-layers are optional and depend on whether they are
    specified in the params dictionary.
    """
    model = MLC()

    # X
    X = Input(name='X', input_shape=(params['X']['dim'],))
    last_layer_name = X

    for num_class in num_classes:

        # Hn
        h_class_num = 'H'+str(num_class)
        if h_class_num in params:  # there is a hidden layer between Y0 and Y1
            kwargs = params[h_class_num]['kwargs'] if 'kwargs' in params[h_class_num] else {}
            if isinstance(last_layer_name, list):
                model.add(Dense(params[h_class_num]['dim'], **kwargs),
                               name=h_class_num+'_dense', inputs=last_layer_name)
            else:
                model.add(Dense(params[h_class_num]['dim'], **kwargs),
                               name=h_class_num+'_dense', input=last_layer_name)
            model.add(Activation('relu'),
                           name=h_class_num+'_activation', input=h_class_num+'_dense')
            HN_output_name = h_class_num+'_activation'
            if 'batch_norm' in params[h_class_num] and params[h_class_num]['batch_norm'] != None:
                model.add(BatchNormalization(**params[h_class_num]['batch_norm']),
                               name=h_class_num+'_batch_norm', input=HN_output_name)
                HN_output_name = h_class_num+'_batch_norm'
            if 'dropout' in params[h_class_num]:
                model.add(Dropout(params[h_class_num]['dropout']),
                               name=h_class_num+'_dropout', input=HN_output_name)
                HN_output_name = h_class_num+'_dropout'
            last_layer_name = HN_output_name

        # Yn
        y_class_num = 'Y'+str(num_class)
        kwargs = params[y_class_num]['kwargs'] if 'kwargs' in params[y_class_num] else {}
        if 'W_regularizer' in kwargs:
          kwargs['W_regularizer'] = l2(kwargs['W_regularizer'])
        if isinstance(last_layer_name, list):
            model.add(Dense(params[y_class_num]['dim'], **kwargs),
                           name=y_class_num+'_dense', inputs=last_layer_name)
        else:
            model.add(Dense(params[y_class_num]['dim'], **kwargs),
                           name=y_class_num+'_dense', input=last_layer_name)
        model.add(Activation('sigmoid'),
                       name=y_class_num+'_activation', input='Y1_dense')
        YN_output_name = y_class_num+'_activation'
        if 'activity_reg' in params['Y'+str(num_class-1)]:
            model.add(ActivityRegularization(**params[y_class_num]['activity_reg']),
                           name=y_class_num+'_activity_reg', input=YN_output_name)
            YN_output_name = y_class_num+'_activity_reg'
        model.add_output(name=y_class_num, input=YN_output_name)

    return model
