### import packages ###
from __future__ import absolute_import, division, print_function, unicode_literals

### parse arguments ###
import argparse
import pdb

import caffe2

# matplot
import matplotlib.pyplot as plt

# numpy
import numpy as np

# caffe2
from caffe2.python import brew, core, model_helper, net_drawer, workspace
from IPython import display
from matplotlib import pyplot as plt
from numpy import linalg as la, random as ra


parser = argparse.ArgumentParser(description="Train Multi-Level Perceptron MLP")
parser.add_argument("--arch", type=str, default="64-1024-1024-1024-1024-1024-1")
parser.add_argument("--activation-function", type=str, default="relu")
parser.add_argument("--reverse-activation", type=bool, default=False)
parser.add_argument("--loss-function", type=str, default="normF")
parser.add_argument("--mini-batch-size", type=int, default=1)
parser.add_argument("--data-size", type=int, default=1)
parser.add_argument("--nepochs", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=0.01)
parser.add_argument("--merge-weight-and-bias", type=bool, default=False)
parser.add_argument("--print-precision", type=int, default=5)
parser.add_argument("--numpy-rand-seed", type=int, default=123)
parser.add_argument("--fwd-graph-file", type=str, default=None)
parser.add_argument("--full-graph-file", type=str, default=None)
parser.add_argument("--random-weight-init", type=bool, default=True)
parser.add_argument("--run-mode", type=int, default=2)
args = parser.parse_args()

### some basic setup ###
np.random.seed(args.numpy_rand_seed)
np.set_printoptions(precision=args.print_precision)
workspace.GlobalInit(["caffe2", "--caffe2_log_level=2"])
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
    n = min(args.mini_batch_size, args.data_size - (j * args.mini_batch_size))
    lX.append(ra.rand(m0, n).astype(np.float32))
    lT.append(ra.rand(ml, n).astype(np.float32))

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

    def getLayerParams(self, dimIn, dimOut, layerIndex, randomInit):
        wstr = "w_{}".format(layerIndex)
        bstr = "b_{}".format(layerIndex)
        if args.random_weight_init:
            w = self.model.param_init_net.GivenTensorFill(
                [],
                wstr,
                values=ra.rand(dimOut, dimIn).astype(np.float32),
                shape=[dimOut, dimIn],
            )
            b = self.model.param_init_net.GivenTensorFill(
                [], bstr, values=ra.rand(dimOut).astype(np.float32), shape=[dimOut]
            )
        else:
            w = self.model.param_init_net.XavierFill([], wstr, shape=[dimOut, dimIn])
            b = self.model.param_init_net.ConstantFill([], bstr, shape=[dimOut])
        return [w, b]

    def addFwdPropOperators(self, ln):
        ACTSTR = "act_{}"
        ZSTR = "z_{}"
        # pdb.set_trace()
        self.model = model_helper.ModelHelper(name="mlp")
        self.params = []
        # workspace.FeedBlob(ACTSTR.format(0), X)
        for i in range(1, ln.size):
            dimIn = ln[i - 1]
            dimOut = ln[i]
            blobZ = ZSTR.format(i)
            blobIn = ACTSTR.format(i - 1) if i != 1 else "X"

            blobOut = ACTSTR.format(i) if i != (ln.size - 1) else "Y_pred"
            [w, b] = self.getLayerParams(dimIn, dimOut, i, args.random_weight_init)
            self.model.net.FC([blobIn, w, b], blobZ)
            # if not explicitly initializing, this is better, but changes grad-descent code
            # brew.fc(self.model, blobIn, blobZ, dimIn=n, dimOut=m)

            brew.relu(self.model, blobZ, blobOut)
            self.params.append(w)
            self.params.append(b)

    def addBackPropOperators(self):
        dist = self.model.SquaredL2Distance(["Y", "Y_pred"], "dist")
        self.loss = dist.AveragedLoss([], ["loss"])
        # print graph to svg file if asked
        if args.fwd_graph_file != None:
            self.writeGraphToFile(args.fwd_graph_file)

        # X = self.model.StopGradient("X", "X")
        # create gradient ops
        self.gradientMap = self.model.AddGradientOperators(["loss"])
        if args.full_graph_file != None:
            self.writeGraphToFile(args.full_graph_file)

    def addTrainingOperators(self):
        # ONE is a constant value that is used in the gradient update. We only need to create it once, so it is explicitly placed in param_init_net.
        ONE = self.model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
        ITER = brew.iter(self.model, "iter")
        # LR = self.model.LearningRate(ITER, "LR", base_lr=args.learning_rate, policy="step", stepsize=1, gamma=0.999 )
        LR = self.model.LearningRate(
            ITER, "LR", base_lr=-1 * args.learning_rate, policy="fixed"
        )
        # for param in self.model.params:
        for param in self.params:
            # param_grad = self.model.param_to_grad[param]
            param_grad = self.gradientMap[param]
            self.model.WeightedSum([param, ONE, param_grad, LR], param)

    def printParams(self):
        for param in mlp.params:
            print(workspace.FetchBlob(str(param)))


mode = args.run_mode
# construct the neural network specified above
mlp = Net(ln, mode)
# debug prints
# print(mlp)

# create net
workspace.RunNetOnce(mlp.model.param_init_net)
netCreated = False

### forward and backward pass ###
if mode >= 1:
    print("loss:")
for k in range(0, args.nepochs):
    for j in range(0, nbatches):
        print(
            "epoch " + str(k) + ", mini-batch " + str(j) + ":"
        )

        workspace.FeedBlob("X", np.transpose(lX[j]))
        workspace.FeedBlob("Y", np.transpose(lT[j]))
        if not netCreated:
            workspace.CreateNet(mlp.model.net)
            netCreated = True
        workspace.RunNet(mlp.model.name, 1)
        if mode >= 1:
            loss = workspace.FetchBlob("loss")
            print(np.array([loss]))

# mlp.printParams()
