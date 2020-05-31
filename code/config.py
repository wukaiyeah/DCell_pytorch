# --------------------------------------------------------
# setting parameter
# --------------------------------------------------------
class Config():
    '''
    For passing these parameters
    '''
    def __init__(self):
        # datapath
        self.network = 'Models/Binary_Binary_GO_NN.py' #Model file - must return valid network
        self.save = 'model/' # save directory
        self.load = ''  #load existing net weights
        self.datapath = './' # data folder
        self.train = '../TrainData/GO_0006281_training_data_2016_0' # Training dataset
        self.test = '../TrainData/GO_0006281_testing_data_2016_0' # Testing dataset
        self.topo = '../Topology/GO_0006281_topology' # Ontology graph topology file
        self.model = 'model/DCell_GO:0006281_model.pth'
	# configuration
        self.gindex = '' # Gene index mapping file
        self.devid = 0 # device ID (if using CUDA)
        self.nGPU= 1  # num of gpu devices used
        self.type = 'cuda' # float or cuda
        self.threads = 8 # number of threads
        self.LR = 0.001 # learning rate
        self.LRDecay = 0 # learning rate decay (in # samples)
        self.weightDecay = 0.0 # L2 penalty on the weights
        self.momentum = 0.0 # momentum
        self.batchSize = 8 # batch size
        self.epoch = 200 # number of epochs to train, -1 for unbounded
        self.optimization = 'adam' # optimization method
        self.root = 'GO:0006281' # Root term for DNA repair
