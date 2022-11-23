from models.util import sigmoid, dsigmoid, mse, dmse, relu, drelu
import numpy as np
from scipy.signal import convolve
from scipy.special import softmax
import random


class FullyConnectedLayer:
    def __init__(self, size, num_inputs, activation=sigmoid, activation_derivative=dsigmoid):
        self.activation = activation
        self.d_activation = activation_derivative

        self.w = np.array([[random.uniform(-1, 1) for _ in range(num_inputs)] for __ in range(size)])  # (size x
        # num_inputs)
        self.b = np.array([[random.uniform(-1, 1)] for _ in range(size)])  # (size x 1)
        self.inputs = None  # (num_inputs x 1)
        self.z = None  # (size x 1)
        self.dz = None  # (size x 1)
        self.dw = None  # (size x num_inputs)

    def forward(self, x):
        self.inputs = x.copy().reshape((x.shape[0], 1))
        self.z = np.matmul(self.w, self.inputs) + self.b
        return self.activation(self.z)  # (size x 1)

    def get_da(self):
        return np.matmul(self.w.T, self.dz)

    def backward(self, next_layer=None, der_loss=None):
        if der_loss is not None:
            self.dz = der_loss.reshape((der_loss.shape[0], 1))
        else:
            self.dz = np.multiply(self.d_activation(self.z), next_layer.get_da())
        self.dw = np.matmul(self.dz, self.inputs.T)

    def optimize(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.dz

    def save(self, filename):
        np.save(filename + "_weights", self.w)
        np.save(filename + "_biases", self.b)


class Filter:
    def __init__(self, size, in_channels):
        self.kernels = np.array([[[random.uniform(-1, 1) for _ in range(size)] for __ in range(size)]
                                 for ___ in range(in_channels)])
        self.size = size
        self.in_channels = in_channels

        self.dw = []

    def forward(self, x):
        assert x.shape[0] == self.in_channels

        res = np.zeros((x.shape[1] - self.size + 1, x.shape[2] - self.size + 1))
        for ch_num in range(self.in_channels):
            kernel = self.kernels[ch_num]
            channel = x[ch_num]
            res += convolve(kernel, channel, mode='valid')
        return res

    def backward(self, dz, x):
        assert x.shape[0] == self.kernels.shape[0]
        assert x.shape[1] - self.size + 1 == dz.shape[0]
        assert x.shape[2] - self.size + 1 == dz.shape[1]

        self.dw.clear()
        for ch_num in range(self.kernels.shape[0]):
            self.dw.append(convolve(dz, x[ch_num], mode='valid'))

    def get_dz(self, prev_dz):
        res_arr = []
        for kernel in self.kernels:
            res_arr.append(convolve(np.rot90(np.rot90(kernel)), prev_dz, mode='full'))
        return np.array(res_arr)

    def optimize(self, lr):
        for i in range(self.kernels.shape[0]):
            self.kernels[i] -= self.dw[i] * lr

    def save(self, filename):
        np.save(filename, self.kernels)


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, conv_kernel_size=3,
                 max_pool_size=2, max_pool_stride=2, activation=sigmoid, d_activation=dsigmoid):
        self.filters = [Filter(conv_kernel_size, in_channels) for _ in range(out_channels)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_pool_size = max_pool_size
        self.max_pool_stride = max_pool_stride

        self.activation = activation
        self.d_activation = d_activation

        self.input = None
        self.conv_res = None
        self.z = None
        self.max_pool_index = None

        self.dz = None

    def _max_pool(self, x: np.ndarray):
        h_prev, w_prev = x.shape
        h = int((h_prev - self.max_pool_size) / self.max_pool_stride) + 1
        w = int((w_prev - self.max_pool_size) / self.max_pool_stride) + 1

        downsampled = np.zeros((h, w))
        max_indices = np.zeros((h, w, 2))

        curr_y = out_y = 0
        while curr_y + self.max_pool_size <= h_prev:
            curr_x = out_x = 0
            while curr_x + self.max_pool_size <= w_prev:
                window = x[curr_y:curr_y + self.max_pool_size, curr_x:curr_x + self.max_pool_size]
                downsampled[out_y, out_x] = np.max(window)
                idx = np.unravel_index(np.nanargmax(window), window.shape)
                max_indices[out_y, out_x, 0] = idx[0] + curr_y
                max_indices[out_y, out_x, 1] = idx[1] + curr_x
                curr_x += self.max_pool_stride
                out_x += 1
            curr_y += self.max_pool_stride
            out_y += 1
        return downsampled, max_indices

    def get_dz(self, da: np.ndarray):
        dz = np.multiply(da, self.d_activation(self.z))
        res = np.zeros((dz.shape[0], self.conv_res[0].shape[0], self.conv_res[0].shape[1]))

        assert dz.shape[0] == self.out_channels

        c = 0
        while c < res.shape[0]:
            for i in range(dz[c].shape[0]):
                for j in range(dz[c].shape[1]):
                    h = int(self.max_pool_index[c][i, j, 0])
                    w = int(self.max_pool_index[c][i, j, 1])
                    res[c, h, w] = dz[c, i, j]
            c += 1

        return res

    def forward(self, x):
        self.input = x
        assert x.shape[0] == self.in_channels

        z_arr = []
        conv_res_arr = []
        mp_index_arr = []

        for ch_num in range(self.out_channels):
            conv_res = self.filters[ch_num].forward(x)
            mp_res, mp_index = self._max_pool(conv_res)
            conv_res_arr.append(conv_res)
            mp_index_arr.append(mp_index)
            z_arr.append(mp_res)
        self.conv_res = np.array(conv_res_arr)
        self.z = np.array(z_arr)
        self.max_pool_index = np.array(mp_index_arr)
        return self.activation(np.array(self.z))

    def get_da(self):
        res = np.zeros(self.input.shape)
        for i in range(len(self.filters)):
            res += self.filters[i].get_dz(self.dz[i])
        return res

    def backward(self, next_layer):
        self.dz = self.get_dz(next_layer.get_da())
        assert self.dz.shape[0] == len(self.filters)
        for i in range(len(self.filters)):
            self.filters[i].backward(self.dz[i], self.input)

    def optimize(self, lr):
        for f in self.filters:
            f.optimize(lr)

    def save(self, filename):
        for i in range(len(self.filters)):
            self.filters[i].save(filename + "_filter_" + str(i))


class MiddleLayer:
    def __init__(self, in_channels, num_inputs, activation=sigmoid, d_activation=dsigmoid):
        self.in_channels = in_channels
        self.num_inputs = num_inputs
        self.weights = np.array(
            [[random.uniform(-1, 1) for _ in range(num_inputs)] for __ in range(in_channels)])
        self.biases = np.array([random.uniform(-1, 1) for _ in range(in_channels)])
        self.input = None
        self.activation = activation
        self.d_activation = d_activation

        self.z = None
        self.dz = None
        self.dw = None

    def forward(self, x):
        self.input = x
        assert x.shape[0] == self.in_channels
        assert x.shape[1] * x.shape[2] == self.num_inputs

        res_arr = []

        for i in range(self.in_channels):
            inp = np.ravel(x[i])
            res = np.dot(inp, self.weights[i]) + self.biases[i]
            res_arr.append(res)
        self.z = np.array(res_arr)
        return self.activation(self.z)

    def get_da(self):
        res_arr = []
        for i in range(self.in_channels):
            res = self.dz[i] * self.weights[i]
            res = res.reshape((self.input.shape[1], self.input.shape[2]))
            res_arr.append(res)
        return np.array(res_arr)

    def backward(self, next_layer=None, der_loss=None):
        if der_loss is not None:
            self.dz = der_loss.reshape((der_loss.shape[0], 1))
        else:
            da = next_layer.get_da()
            self.dz = np.multiply(self.d_activation(self.z.reshape((self.z.shape[0], 1))), da)
        res_arr = []
        for i in range(self.in_channels):
            inp = np.ravel(self.input[i])
            res_arr.append(inp * self.dz[i])
        self.dw = np.array(res_arr)

    def optimize(self, lr: float):
        self.weights -= lr * self.dw
        self.biases -= lr * self.dz.ravel()

    def save(self, filename):
        np.save(filename + "_weights", self.weights)
        np.save(filename + "_biases", self.biases)


class CNN:
    def __init__(self):
        self.first_conv_layer = ConvolutionalLayer(1, 5, activation=relu, d_activation=drelu)
        self.second_conv_layer = ConvolutionalLayer(5, 10, activation=relu, d_activation=drelu)
        self.middle_layer = MiddleLayer(10, 25)
        self.pre_out_layer = FullyConnectedLayer(10, 10)
        self.out_layer = FullyConnectedLayer(10, 10, activation=softmax)

    def forward(self, x: np.ndarray):
        res = self.first_conv_layer.forward(x)
        res = self.second_conv_layer.forward(res)
        res = self.middle_layer.forward(res)
        res = self.pre_out_layer.forward(res)
        res = self.out_layer.forward(res)
        return res

    def backward(self, der_loss: np.ndarray):
        self.out_layer.backward(der_loss=der_loss)
        self.pre_out_layer.backward(next_layer=self.out_layer)
        self.middle_layer.backward(next_layer=self.pre_out_layer)
        self.second_conv_layer.backward(self.middle_layer)
        self.first_conv_layer.backward(self.second_conv_layer)

    def optimize(self, lr: float):
        self.first_conv_layer.optimize(lr)
        self.second_conv_layer.optimize(lr)
        self.middle_layer.optimize(lr)
        self.pre_out_layer.optimize(lr)
        self.out_layer.optimize(lr)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float = 0.001, loss=mse, dloss=dmse):
        losses = []
        loss_avg = []
        for epoch in range(epochs):
            ls = 0
            for num_sample in range(x.shape[0]):
                res = self.forward(x[num_sample])
                ls_sample = loss(res, y[num_sample])
                ls += ls_sample
                self.backward(dloss(res, y[num_sample]))
                self.optimize(lr)
                if num_sample % 1000 == 0:
                    print("Epoch ", epoch, ", sample ", num_sample, " loss: ", ls_sample, sep='')
                losses.append(ls_sample)
            loss_avg.append(ls / x.shape[0])
            print("Epoch", epoch, "average loss:", ls)
        return loss_avg, losses

    def save(self, path):
        self.first_conv_layer.save(path + "/conv1")
        self.second_conv_layer.save(path + "/conv2")
        self.middle_layer.save(path + "/flatten")
        self.out_layer.save(path + "/full1")

    def predict(self, x):
        res = []
        for sample in x:
            res.append(self.forward(sample.reshape(1, sample.shape[0], sample.shape[1])).flatten())
        return np.array(res)
