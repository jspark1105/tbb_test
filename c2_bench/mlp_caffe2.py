# Pre-requisites
# 1. Install Anaconda
# wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
# chmod +x anaconda3.sh
# ./anaconda3.sh -b -p ~/local/anaconda3
# 2. Create and activate PyTorch env
# ~/local/anaconda3/bin/conda create -yn pytorch
# source ~/local/anaconda3/bin/activate pytorch
# 3. Install pre-requisite packages
# ~/local/anaconda3/bin/conda install graphviz pydot future automake autoconf ninja numpy pyyaml mkl mkl-include setuptools cmake cffi typing
# 4. Option A (recommended for debugging): Install PyTorch from source
# (related instruction : https://github.com/pytorch/pytorch#from-source)
# git clone --recursive https://github.com/pytorch/pytorch.git
# cd pytorch
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# USE_MKLDNN=0 USE_ROCM=0 with-proxy python setup.py develop
# USE_MKLDNN=0 USE_ROCM=0 with-proxy python setup.py install
# with-proxy conda install protobuf=3.6.1
# 5. Option B: Install PyTorch from package
# ~/local/anaconda3/bin/conda install pytorch-nightly -c pytorch
# 6. Run this script
# python mlp_caffe2.py

### import packages ###
from __future__ import absolute_import, division, print_function, unicode_literals

### parse arguments ###
import argparse
import time

# numpy
import numpy as np

# caffe2
from caffe2.python import caffe2_pb2, core, model_helper, net_drawer, workspace
from numpy import random as ra


parser = argparse.ArgumentParser(description="Train Multi-Level Perceptron MLP")
parser.add_argument("--arch", type=str, default="64-1024-1024-1024-1024-1024-1")
parser.add_argument("--activation-function", type=str, default="relu")
parser.add_argument("--reverse-activation", type=bool, default=False)
parser.add_argument("--loss-function", type=str, default="normF")
parser.add_argument("--mini-batch-size", type=int, default=8192)
parser.add_argument("--data-size", type=int, default=65536)
parser.add_argument("--nepochs", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=0.01)
parser.add_argument("--merge-weight-and-bias", type=bool, default=False)
parser.add_argument("--print-precision", type=int, default=5)
parser.add_argument("--numpy-rand-seed", type=int, default=123)
parser.add_argument("--fwd-graph-file", type=str, default=None)
parser.add_argument("--full-graph-file", type=str, default=None)
parser.add_argument("--random-weight-init", type=bool, default=True)
parser.add_argument("--run-mode", type=int, default=2)
parser.add_argument("--print-model", action="store_true")
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="the number of worker per numa node in the thread pool",
)
parser.add_argument("--num-numa-nodes", type=int, default=1)
args, extra_args = parser.parse_known_args()


### some basic setup ###
np.random.seed(args.numpy_rand_seed)
np.set_printoptions(precision=args.print_precision)
global_options = ["caffe2", "--caffe2_log_level=2", "--caffe2_cpu_numa_enabled=1"] + extra_args
workspace.GlobalInit(global_options)
assert workspace.IsNUMAEnabled()
ln = np.fromstring(args.arch, dtype=int, sep="-")
# test prints
print("mlp arch (" + str(ln.size - 1) + " layers, with input to output dimensions):")
print(ln)

### prepare training data ###
nbatches = int(np.ceil((args.data_size * 1.0) / args.mini_batch_size))
# inputs
m0 = ln[0]
lX = []
# targets
ml = ln[ln.size - 1]
lT = []
for j in range(0, nbatches):
    lX.append([])
    lT.append([])
    mini_batch_per_numa_node = int(
        np.ceil(args.mini_batch_size * 1.0 / args.num_numa_nodes)
    )
    for numa_node_id in range(args.num_numa_nodes):
        n = min(
            mini_batch_per_numa_node,
            args.mini_batch_size - numa_node_id * mini_batch_per_numa_node,
            args.data_size
            - j * args.mini_batch_size
            - numa_node_id * mini_batch_per_numa_node,
        )
        lX[j].append(ra.rand(m0, n).astype(np.float32))
        lT[j].append(ra.rand(ml, n).astype(np.float32))

### define mlp network in caffe2 ###
### create random weights and bias ###
# z = f(y)
# y = Wx+b


class Net:
    def __init__(self, ln, mode):
        # Modes of execution: 0=Forward only; 1=FWD+BWD; 2=FWD+BWD+SGD
        if mode < 0 & mode > 2:
            print(
                "Unknow mode for net"
                + str(mode)
                + "(expected values: \
                0=Forward only; 1=FWD+BWD; 2=FWD+BWD+SGD )"
            )
        else:
            self.addFwdPropOperators(ln)
            _mode = "Forward pass only"
            if mode >= 1:
                self.addBackPropOperators()
                _mode = "Forward + Backward passes"
            if mode == 2:
                self.addTrainingOperators()
                _mode = "Full training (Fwd+Bwd and optimizer)"
            print("Net initialized with execution mode:" + _mode)

    def writeGraphToFile(self, filename):
        graph = net_drawer.GetPydotGraph(self.model.Proto().op, "train", rankdir="LR")
        with open(filename, "wb") as f:
            f.write(graph.create_svg())

    def getLayerParams(self, dimIn, dimOut, layerIndex, numa_node_id, randomInit):
        wstr = "w_{}_{}".format(layerIndex, numa_node_id)
        bstr = "b_{}_{}".format(layerIndex, numa_node_id)
        if args.random_weight_init:
            w = self.model.param_init_net.GivenTensorFill(
                [],
                wstr,
                values=ra.rand(dimOut, dimIn).astype(np.float32),
                shape=[dimOut, dimIn],
                device_option=self.numa_device_options[numa_node_id],
            )
            b = self.model.param_init_net.GivenTensorFill(
                [],
                bstr,
                values=ra.rand(dimOut).astype(np.float32),
                shape=[dimOut],
                device_option=self.numa_device_options[numa_node_id],
            )
        else:
            w = self.model.param_init_net.XavierFill(
                [],
                wstr,
                shape=[dimOut, dimIn],
                device_option=self.numa_device_options[numa_node_id],
            )
            b = self.model.param_init_net.ConstantFill(
                [],
                bstr,
                shape=[dimOut],
                device_option=self.numa_device_options[numa_node_id],
            )
        return [w, b]

    def addFwdPropOperators(self, ln):
        ACTSTR = "act_{}_{}"
        ZSTR = "z_{}_{}"
        # pdb.set_trace()
        self.model = model_helper.ModelHelper(name="mlp")

        self.model.param_init_net.Proto().type = "async_scheduling"
        self.model.param_init_net.Proto().num_workers = args.num_workers

        self.model.net.Proto().type = "async_scheduling"
        self.model.net.Proto().num_workers = args.num_workers

        self.numa_device_options = []
        for numa_node_id in range(args.num_numa_nodes):
            self.numa_device_options.append(caffe2_pb2.DeviceOption())
            self.numa_device_options[numa_node_id].device_type = caffe2_pb2.CPU
            self.numa_device_options[numa_node_id].numa_node_id = numa_node_id

        self.params = []
        for _ in range(args.num_numa_nodes):
            self.params.append([])
        # workspace.FeedBlob(ACTSTR.format(0), X)
        for i in range(1, ln.size):
            for numa_node_id in range(args.num_numa_nodes):
                dimIn = ln[i - 1]
                dimOut = ln[i]
                blobZ = ZSTR.format(i, numa_node_id)
                blobIn = (
                    ACTSTR.format(i - 1, numa_node_id)
                    if i != 1
                    else "X_{}".format(numa_node_id)
                )

                blobOut = (
                    ACTSTR.format(i, numa_node_id)
                    if i != (ln.size - 1)
                    else "Y_pred_{}".format(numa_node_id)
                )
                [w, b] = self.getLayerParams(
                    dimIn, dimOut, i, numa_node_id, args.random_weight_init
                )
                self.model.net.FC(
                    [blobIn, w, b],
                    blobZ,
                    # engine="INTRA_OP_PARALLEL",
                    device_option=self.numa_device_options[numa_node_id],
                )

                self.model.net.Relu(
                    blobZ, blobOut, device_option=self.numa_device_options[numa_node_id]
                )
                self.params[numa_node_id].append(w)
                self.params[numa_node_id].append(b)

    def addBackPropOperators(self):
        self.local_loss = []
        for numa_node_id in range(args.num_numa_nodes):
            dist = self.model.SquaredL2Distance(
                ["Y_{}".format(numa_node_id), "Y_pred_{}".format(numa_node_id)],
                "dist_{}".format(numa_node_id),
                device_option=self.numa_device_options[numa_node_id],
            )
            self.local_loss.append(
                self.model.AveragedLoss(
                    [dist],
                    "loss_{}".format(numa_node_id),
                    device_option=self.numa_device_options[numa_node_id],
                )
            )

        self.loss = self.model.net.Sum(
            ["loss_{}".format(i) for i in range(args.num_numa_nodes)], "loss"
        )

        # print graph to svg file if asked
        if args.fwd_graph_file:
            self.writeGraphToFile(args.fwd_graph_file)

        # X = self.model.StopGradient("X", "X")
        # create gradient ops
        self.gradientMap = self.model.AddGradientOperators(["loss"])
        if args.full_graph_file:
            self.writeGraphToFile(args.full_graph_file)

    def addTrainingOperators(self):
        # ONE is a constant value that is used in the gradient update.
        # We only need to create it once, so it is explicitly placed in param_init_net.
        ONE = self.model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
        workspace.FeedBlob("iter", np.ones(1).astype(np.float32))
        ITER = self.model.net.Iter([], "iter")
        LR = self.model.LearningRate(
            ITER, "LR", base_lr=-1 * args.learning_rate, policy="fixed"
        )
        for numa_node_id in range(args.num_numa_nodes):
            for param in self.params[numa_node_id]:
                param_grad = self.gradientMap[param]
                self.model.WeightedSum(
                    [param, ONE, param_grad, LR],
                    param,
                    device_option=self.numa_device_options[numa_node_id],
                )

    def printParams(self):
        for numa_node_id in range(args.num_numa_nodes):
            for param in mlp.params[numa_node_id]:
                print(workspace.FetchBlob(str(param)))


mode = args.run_mode
# construct the neural network specified above
mlp = Net(ln, mode)
# debug prints
if args.print_model:
    print(mlp.model.net.Proto())

# create net
workspace.RunNetOnce(mlp.model.param_init_net)

flop = 0
for l in range(ln.size - 1):
    flop += ln[l] * ln[l + 1]
flop *= 3 * 2 * args.mini_batch_size

### forward and backward pass ###
if mode >= 1:
    print("loss:")
for k in range(0, args.nepochs):
    dt = 0
    for j in range(0, nbatches):
        print("epoch " + str(k) + ", mini-batch " + str(j) + ":")

        for numa_node_id in range(args.num_numa_nodes):
            workspace.FeedBlob(
                "X_{}".format(numa_node_id), np.transpose(lX[j][numa_node_id])
            )
            workspace.FeedBlob(
                "Y_{}".format(numa_node_id), np.transpose(lT[j][numa_node_id])
            )
        if k == 0 and j == 0:
            workspace.CreateNet(mlp.model.net)

        t = time.time()
        workspace.RunNet(mlp.model.net, 1)
        dt += time.time() - t

        if mode >= 1:
            loss = workspace.FetchBlob("loss")
            print(np.array([loss / args.num_numa_nodes]))

    print("{} sec per epoch {} GF/s".format(dt, nbatches * flop / dt / 1e9))

# mlp.printParams()
