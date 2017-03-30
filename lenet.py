from yann.network import network
from yann.utils.graph import draw_network
    
# Advaned versions of the CNN
def lenet5 ( dataset= None, verbose = 1 ):             
    """
    This is a version with nesterov momentum and rmsprop instead of the typical sgd. 
    This also has maxout activations for convolutional layers, dropouts on the last
    convolutional layer and the other dropout layers and this also applies batch norm
    to all the layers.  The batch norm is applied by using the ``batch_norm = True`` parameters
    in all layers. This batch norm is applied before activation as is used in the original 
    version of the paper. So we just spice things up and add a bit of steroids to 
    :func:`lenet5`.  This also introduces a visualizer module usage.

    Args: 
        dataset: Supply a dataset.    
        verbose: Similar to the rest of the dataset.    
    """
    optimizer_params =  {        
                "momentum_type"       : 'nesterov',             
                "momentum_params"     : (0.75, 0.95, 30),      
                "optimizer_type"      : 'rmsprop',                
                "id"                  : "main"
                        }

    dataset_params  = {
                            "dataset"   : dataset,
                            "svm"       : False, 
                            "n_classes" : 10,
                            "id"        : 'data'
                    }

    visualizer_params = {
                    "root"       : 'lenet_on_steroids',
                    "frequency"  : 1,
                    "sample_size": 32,
                    "rgb_filters": True,
                    "debug_functions" : False,
                    "debug_layers": False,  # Since we are on steroids this time, print everything.
                    "id"         : 'main'
                        }                      

    net = network(   borrow = True,
                     verbose = verbose )                       
    
    net.add_module ( type = 'optimizer',
                     params = optimizer_params, 
                     verbose = verbose )

    net.add_module ( type = 'datastream', 
                     params = dataset_params,
                     verbose = verbose )

    net.add_module ( type = 'visualizer',
                     params = visualizer_params,
                     verbose = verbose )

    net.add_layer ( type = "input",
                    id = "input",
                    verbose = verbose, 
                    origin = 'data' )
    
    net.add_layer ( type = "conv_pool",
                    origin = "input",
                    id = "conv_pool_1",
                    num_neurons = 64,
                    filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose
                    )

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_1",
                    id = "conv_pool_2",
                    num_neurons = 64,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,
                    #regularize = True,                    
                    verbose = verbose
)
    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_2",
                    id = "conv_pool_3",
                    num_neurons = 128,
                    filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose
                    )
    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_3",
                    id = "conv_pool_4",
                    num_neurons = 128,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose
                    )
    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_4",
                    id = "conv_pool_5",
                    num_neurons = 256,
                    filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose
                    )
    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_5",
                    id = "conv_pool_6",
                    num_neurons = 256,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose
                    )
    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_6",
                    id = "conv_pool_7",
                    num_neurons = 512,
                    filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose
                    )
    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_7",
                    id = "conv_pool_8",
                    num_neurons = 512,
                    filter_size = (3,3),
                    #ool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose
                    )
    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_8",
                    id = "conv_pool_9",
                    num_neurons = 512,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose)

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_9",
                    id = "conv_pool_10",
                    num_neurons = 512,
                    filter_size = (3,3),
                    #ool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose)

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_10",
                    id = "conv_pool_11",
                    num_neurons = 512,
                    filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose)

    net.add_layer ( type = "conv_pool",
                    origin = "conv_pool_11",
                    id = "conv_pool_12",
                    num_neurons = 512,
                    filter_size = (3,3),
                    pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose)

    net.add_layer ( type = "dot_product",
                    origin = "conv_pool_12",
                    id = "hidden_layer1",
                    num_neurons = 4096,
                    #filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose)

    net.add_layer ( type = "dot_product",
                    origin = "hidden_layer1",
                    id = "hidden_layer2",
                    num_neurons = 4096,
                    #filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'relu',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose)

    net.add_layer ( type = "classifier",
                    origin = "hidden_layer2",
                    id = "softmax",
                    num_classes = 102,
                    #filter_size = (3,3),
                    #pool_size = (2,2),
                    activation = 'softmax',
                    #batch_norm = True,           
                    #regularize = True,                             
                    verbose = verbose)

    net.add_layer ( type = "objective",
                    id = "obj",
                    origin = "softmax",
                    objective = "nll",
                    datastream_origin = 'data', 
                    regularization = (0.0001, 0.0001),                
                    verbose = verbose
	)

    learning_rates = (0.05, .0001, 0.001)  
    net.pretty_print()  
    # draw_network(net.graph, filename = 'lenet.png')   

    net.cook()

    net.train( epochs = (20, 20), 
               validate_after_epochs = 1,
               training_accuracy = True,
               learning_rates = learning_rates,               
               show_progress = True,
               early_terminate = True,
               patience = 2,
               verbose = verbose)

    net.test(verbose = verbose)

if __name__ == '__main__':
    import sys
    dataset = None  
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create_dataset':
            from yann.special.datasets import cook_cifar10 
            from yann.special.datasets import cook_mnist
            
            data = cook_cifar10 (verbose = 2)
            # data = cook_mnist()        
            dataset = data.dataset_location()
        else:
            dataset = sys.argv[1]
    else:
        print "provide dataset"
    
    if dataset is None:
        print " creating a new dataset to run through"
        from yann.special.datasets import cook_cifar10 
        from yann.special.datasets import cook_mnist 
        from yann.special.datasets import cook_caltech101 
        #data = cook_cifar10 (verbose = 2)
        # data = cook_mnist()
	data = cook_caltech101(verbose=5)
        dataset = data.dataset_location()

    lenet5 ( dataset, verbose = 2 )
