import numpy as np
from sklearn.model_selection import train_test_split

def reweight(events,model,batch_size=10000):
    f = model.predict(events, batch_size=batch_size)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights))

def omnifold(theta0,theta_unknown_S,iterations,model,verbose=0):

    weights = np.empty(shape=(iterations, 2, len(theta0))) # new array of given shape, (default type=float), w/o initializing entries
    # shape = (iteration, step, event)

    theta0_G = theta0[:,0] # values in column 0
    theta0_S = theta0[:,1] # values in column 1
    
    labels0 = np.zeros(len(theta0)) # new array of given shape, dtype=float, filled with zeros
    labels_unknown = np.ones(len(theta_unknown_S)) # new array of given shape, (default type), filled with ones
    
    xvals_1 = np.concatenate((theta0_S, theta_unknown_S)) # joins arrays along existing axis
    yvals_1 = np.concatenate((labels0, labels_unknown))   # nb: arrays must have same shape (except in the dimension corresponding to axis(the first, by default)

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels_unknown))

    # initial iterative weights are ones
    weights_pull = np.ones(len(theta0_S)) # same length
    weights_push = np.ones(len(theta0_S))
    
    for i in range(iterations):

        if (verbose>0):
            print("\nITERATION: {}\n".format(i + 1))
            pass
        
        # STEP 1: classify Sim. (which is reweighted by weights_push) to Data
        # weights reweighted Sim. --> Data

        if (verbose>0):
            print("STEP 1\n")
            pass
            
        weights_1 = np.concatenate((weights_push, np.ones(len(theta_unknown_S))))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1) # sklearn: train_test_split splits arrays or matrices into random train and test subsets


        # https://www.tensorflow.org/api_docs/python/tf/keras/Model
        model.compile(loss='binary_crossentropy', # name of objective function, returns weighted loss float tensor
                      optimizer='Adam',           # name of optimizer
                      metrics=['accuracy'])       # metrics to be evaluated by the model during training and testing
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        # Trains the model for a fixed number of epochs (iterations on a dataset)
        model.fit(X_train_1,                                       # input data                                 
                  Y_train_1,                                       # input data
                  sample_weight=w_train_1,                         # array of weights for training samples
                  epochs=20,                                       # number of epochs to train the model (iteration over all data)
                  batch_size=10000,                                # number of samples per gradient update
                  validation_data=(X_test_1, Y_test_1, w_test_1),  # data on which to evaluate the loss and any model metrics at the end of each epoch
                  verbose=verbose)

        weights_pull = weights_push * reweight(theta0_S,model) # updating by reweight product
        weights[i, :1, :] = weights_pull  # assigning weights_pull to array, not exactly sure on [i, :1, :] ?

        # STEP 2: classify Gen. to reweighted Gen. (which is reweighted by weights_pull)
        # weights Gen. --> reweighted Gen.

        if (verbose>0):
            print("\nSTEP 2\n")
            pass

        weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))
        # ones for Gen. (not MC weights), actual weights for (reweighted) Gen.

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2)
        
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        model.fit(X_train_2,
                  Y_train_2,
                  sample_weight=w_train_2,
                  epochs=20,
                  batch_size=2000,
                  validation_data=(X_test_2, Y_test_2, w_test_2),
                  verbose=verbose)
        
        weights_push = reweight(theta0_G,model)
        weights[i, 1:2, :] = weights_push
        pass
        
    return weights
