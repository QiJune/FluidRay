import ray
import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import numpy

ray.init()

NUM_WORKERS = 2
BATCH_SIZE = 128


def mnist_network():
    img = fluid.layers.data(name='img', shape=[784])
    hidden = fluid.layers.fc(img, size=100, act='tanh',
                             param_attr='fc.w',
                             bias_attr='fc.b')
    prediction = fluid.layers.fc(hidden, size=10, act='softmax',
                                 param_attr='sftmax.w',
                                 bias_attr='sftmax.b')
    label = fluid.layers.data(name='label',shape=[1],
                              dtype='int64')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)


    avg_loss = fluid.layers.mean(x=loss)
    fluid.backward.append_backward(avg_loss)
    return img, label, avg_loss


@ray.remote
class Worker(object):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.scope = fluid.core.Scope()
        self.program = fluid.Program()
        self.startup = fluid.Program()
        with fluid.program_guard(self.program, self.startup):
            img, label, self.loss = mnist_network()

        self.place = fluid.CPUPlace()
        self.executor = fluid.Executor(self.place)
        self.reader_creator = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=BATCH_SIZE)

        self.reader = self.reader_creator()
        self.feeder = fluid.DataFeeder(feed_list=[img, label], place=self.place)


    def compute_gradient(self, weights):
        for var_name in weights:
            tensor = self.scope.var(var_name).get_tensor()
            tensor.set(weights[var_name][0], self.place)
            tensor.set_lod(weights[var_name][1])

        try:
            data = next(self.reader)
        except:
            self.reader = self.reader_creator()
            data = next(self.reader)

        outs = self.executor.run(self.program,
                     feed=self.feeder.feed(data),
                               scope=self.scope,
                               fetch_list=[ var_name + "@GRAD" for var_name in weights] + [self.loss])
        if self.worker_id == 0:
            print outs[-1]
        return outs[:-1]


@ray.remote
class PServer(object):
    def __init__(self, learning_rate):
        self.scope = fluid.core.Scope()
        self.learning_rate = learning_rate
        self.program = fluid.Program()
        self.startup = fluid.Program()
        with fluid.program_guard(self.program, self.startup):
            mnist_network()

        self.place = fluid.CPUPlace()
        self.executor = fluid.Executor(self.place)
        self.executor.run(self.startup, scope=self.scope)
        self.optimize_program = fluid.Program()

    def apply_gradients(self, *gradients):
        # TODO(qijun) an optimization program is needed
        mean_gradients = numpy.mean(gradients, axis=0)
        weights = self.get_weight()
        for idx, name in enumerate(weights):
            w = weights[name][0]
            w -= self.learning_rate * mean_gradients[idx]
            self.scope.find_var(name).get_tensor().set(w, self.place)

    def get_weight(self):
        weights = dict()
        for p in self.program.global_block().iter_parameters():
            lod_tensor = self.scope.find_var(p.name).get_tensor()
            weights[p.name] = (numpy.copy(numpy.array(lod_tensor)), lod_tensor.lod())
        return weights


if __name__ == '__main__':
    ps = PServer.remote(1e-3 * NUM_WORKERS)
    weights = ps.get_weight.remote()

    works = [Worker.remote(i) for i in range(NUM_WORKERS)]

    while True:
        gradients = [work.compute_gradient.remote(weights) for work in works]
        ps.apply_gradients.remote(*gradients)
        weights = ps.get_weight.remote()




