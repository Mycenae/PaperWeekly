# Solver

The solver orchestrates model optimization by coordinating the network’s forward inference and backward gradients to form parameter updates that attempt to improve the loss. The responsibilities of learning are divided between the Solver for overseeing the optimization and generating parameter updates and the Net for yielding loss and gradients.

Solver负责模型优化，具体是协调网络的前向推理和反向梯度来形成参数更新，以最小化损失。学习的责任由Solver和Net分工，Solver负责监督优化、产生参数更新，Net负责生成损失和梯度。

The Caffe solvers are:

- Stochastic Gradient Descent (type: "SGD"),
- AdaDelta (type: "AdaDelta"),
- Adaptive Gradient (type: "AdaGrad"),
- Adam (type: "Adam"),
- Nesterov’s Accelerated Gradient (type: "Nesterov") and
- RMSprop (type: "RMSProp")

The solver 其功能

1. scaffolds the optimization bookkeeping and creates the training network for learning and test network(s) for evaluation.
- 支持优化的簿记工作，生成训练网络进行学习，以及测试网络进行评估。
2. iteratively optimizes by calling forward / backward and updating parameters
- 通过调用正向/反向过程、更新参数来迭代优化；
3. (periodically) evaluates the test networks
- 周期性的评估测试网络；
4. snapshots the model and solver state throughout the optimization
- 在整个优化过程中对模型和Solver进行快照。

where each iteration 在每次迭代中

1. calls network forward to compute the output and loss
- 调用网络前向过程计算输出和损失和函数；
2. calls network backward to compute the gradients
- 调用网络反向过程来计算梯度；
3. incorporates the gradients into parameter updates according to the solver method
- 根据solver的方法用梯度进行参数更新；
4. updates the solver state according to learning rate, history, and method
- 根据学习率、历史和方法来更新solver状态。

to take the weights all the way from initialization to learned model. 将权重从初始化状态带到学习模型状态。

Like Caffe models, Caffe solvers run in CPU / GPU modes. 和Caffe模型一样，Caffe solver运行在CPU/GPU模式中。

## Methods 方法

The solver methods address the general optimization problem of loss minimization. For dataset $D$, the optimization objective is the average loss over all $|D|$ data instances throughout the dataset

Solver方法处理损失函数最小化的一般优化问题。对于数据集$D$，优化目标是在$|D|$上所有数据样本的平均损失

$$L(W) = \frac{1}{|D|} \sum_i^{|D|} f_W(X^{(i)})+λr(W)$$

where $f_W(X^{(i)})$ is the loss on data instance $X^{(i)}$ and $r(W)$ is a regularization term with weight $λ$. $|D|$ can be very large, so in practice, in each solver iteration we use a stochastic approximation of this objective, drawing a mini-batch of $N<<|D|$ instances:

这里$f_W(X^{(i)})$是在数据样本$X^{(i)}$上的损失，$r(W)$是正则化项，权重为$λ$。$|D|$可能很大，所以在实际情况中，在每个solver迭代中，我们使用目标的随机近似，得到一个mini-batch的样本数量$N<<|D|$：

$$L(W) ≈ \frac{1}{N} \sum_i^N f_W(X^{(i)})+λr(W)$$

The model computes $f_W$ in the forward pass and the gradient $∇f_W$ in the backward pass. 模型在正向过程中计算$f_W$，反向过程中计算梯度$∇f_W$。

The parameter update $ΔW$ is formed by the solver from the error gradient $∇f_W$, the regularization gradient $∇r(W)$, and other particulars to each method. 参数更新$ΔW$由solver根据误差梯度$∇f_W$，正则化梯度$∇r(W)$和其他各种方法的特别项形成。

### SGD

Stochastic gradient descent (type: "SGD") updates the weights $W$ by a linear combination of the negative gradient $∇L(W)$ and the previous weight update $V_t$. The learning rate $α$ is the weight of the negative gradient. The momentum $μ$ is the weight of the previous update.

随机梯度下降法(type: "SGD")用负梯度$∇L(W)$和上一次更新$V_t$的线性组合来更新权值$W$。学习率$α$是负梯度的权值，动量$μ$是上次更新的权值。

Formally, we have the following formulas to compute the update value $V_{t+1}$ and the updated weights $W_{t+1}$ at iteration $t+1$, given the previous weight update $V_t$ and current weights $W_t$:

给定前一次权值更新$V_t$和现在的权重$W_t$，我们用下面的公式来计算第$t+1$次迭代的更新值$V_{t+1}$和更新的权重$W_{t+1}$：

$$V_{t+1}=μV_t−α∇L(W_t)$$
$$W_{t+1}=W_t+V_{t+1}$$

The learning “hyperparameters” (α and μ) might require a bit of tuning for best results. If you’re not sure where to start, take a look at the “Rules of thumb” below, and for further information you might refer to Leon Bottou’s Stochastic Gradient Descent Tricks [1].

学习超参数(α and μ)可能需要一点调整才能得到最佳结果。如果不知道从哪里开始，看一下下面的首要规则，更多信息可以参考Leon Bottou的随机梯度下降法技巧[1]。

[1] L. Bottou. Stochastic Gradient Descent Tricks. Neural Networks: Tricks of the Trade: Springer, 2012.

**Rules of thumb for setting the learning rate α and momentum μ**

设置学习率α和动量μ的经验法则

A good strategy for deep learning with SGD is to initialize the learning rate α to a value around α≈0.01, and dropping it by a constant factor (e.g., 10) throughout training when the loss begins to reach an apparent “plateau”, repeating this several times. Generally, you probably want to use a momentum μ=0.9 or similar value. By smoothing the weight updates across iterations, momentum tends to make deep learning with SGD both stabler and faster.

采用SGD的深度学习的好策略是将学习速率α初始化为0.01左右α≈0.01，在训练过程中，当损失函数达到明显的稳定水平时，将学习速率降低一个常数因子（如1/10），这个过程重复几次。一般来说，你可能会用动量值μ=0.9或类似的值。动量在迭代之间平滑权值更新，会使SGD深度学习更加稳定、快速。

This was the strategy used by Krizhevsky et al. [1] in their famously winning CNN entry to the ILSVRC-2012 competition, and Caffe makes this strategy easy to implement in a SolverParameter, as in our reproduction of [1] at ./examples/imagenet/alexnet_solver.prototxt.

这是Krizhevsky et al. [1]在其ILSVRC2012获胜的CNN模型中采用的策略，Caffe在SolverParameter中使这个策略更加容易实现，我们重新实现了[1]在./examples/imagenet/alexnet_solver.prototxt。

To use a learning rate policy like this, you can put the following lines somewhere in your solver prototxt file: 为使用这一的学习速率策略，你可以将下面的这些行放在你的solver prototxt文件中的某处：

```
base_lr: 0.01     # begin training at a learning rate of 0.01 = 1e-2
lr_policy: "step" # learning rate policy: drop the learning rate in "steps"
                  # by a factor of gamma every stepsize iterations
gamma: 0.1        # drop the learning rate by a factor of 10
                  # (i.e., multiply it by a factor of gamma = 0.1)
stepsize: 100000  # drop the learning rate every 100K iterations
max_iter: 350000  # train for 350K iterations total
momentum: 0.9
```

Under the above settings, we’ll always use momentum μ=0.9. We’ll begin training at a base_lr of α=10e-2 for the first 100,000 iterations, then multiply the learning rate by gamma (γ) and train at α'=αγ=10e-3 for iterations 100K-200K, then at α''=10e−4 for iterations 200K-300K, and finally train until iteration 350K (since we have max_iter: 350000) at α'''=10−5.

在上面的设置中，我们一直使用动量μ=0.9。我们在开始的100,000个迭代中使用的学习速率为α=10e-2，然后将学习速率乘以gamma (γ)，在100K-200K次迭代中使用α'=αγ=10e-3，在200K-300K次迭代中使用α''=10e−4，最后训练直到350K次迭代（因为我们设定最高迭代次数为350K）使用α'''=10−5。

Note that the momentum setting μ effectively multiplies the size of your updates by a factor of 1/(1−μ) after many iterations of training, so if you increase μ, it may be a good idea to decrease α accordingly (and vice versa).

注意动量设定μ在很多次训练迭代后，会将更新乘以一个系数1/(1−μ)，所以如果增加μ，那么应该相应的减小α比较好（反之亦然）。

For example, with μ=0.9, we have an effective update size multiplier of 1/1−0.9=10. If we increased the momentum to μ=0.99, we’ve increased our update size multiplier to 100, so we should drop α (base_lr) by a factor of 10.

比如，当μ=0.9时，我们会得到更新乘数为1/1-0.9=10。所以如果增大动量到μ=0.99，我们就将更新大小乘以了100，那么我们应当相应的将学习速率α (base_lr)降低10倍。

Note also that the above settings are merely guidelines, and they’re definitely not guaranteed to be optimal (or even work at all!) in every situation. If learning diverges (e.g., you start to see very large or NaN or inf loss values or outputs), try dropping the base_lr (e.g., base_lr: 0.001) and re-training, repeating this until you find a base_lr value that works.

注意上述设置只是参考，在各种情况下，绝对不是最优的（甚至不一定能工作）。如果学习发散（例如，开始看到很多非常大的输出或损失值，或NaN，inf），试着降低base_lr（如base_lr: 0.001）然后重新训练，重复这个过程，直到发现能工作的base_lr。

[1] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems, 2012.

### AdaDelta

The AdaDelta (type: "AdaDelta") method (M. Zeiler [1]) is a “robust learning rate method”. It is a gradient-based optimization method (like SGD). The update formulas are

AdaDelta方法(M. Zeiler [1])是一个鲁棒的学习速率方法，是基于梯度的优化方法（和SGD类似），更新公式为

$$(v_t)_i = \frac {RMS((v_{t−1})_i)} {RMS(∇L(W_t))_i} (∇L(W_{t′}))_i$$
$$RMS(∇L(W_t))_i = \sqrt {=E[g^2]+ε}$$
$$E[g^2]_t = δE[g^2]_{t−1}+(1−δ)g^2_t$$
and

$$(W_{t+1})_i = (W_t)_i − α(v_t)_i.$$

[1] M. Zeiler ADADELTA: AN ADAPTIVE LEARNING RATE METHOD. arXiv preprint, 2012.

### AdaGrad

The adaptive gradient (type: "AdaGrad") method (Duchi et al. [1]) is a gradient-based optimization method (like SGD) that attempts to “find needles in haystacks in the form of very predictive but rarely seen features,” in Duchi et al.’s words. Given the update information from all previous iterations $(∇L(W))_{t′}$ for t′∈{1,2,...,t}, the update formulas proposed by [1] are as follows, specified for each component i of the weights W:

$$(W_{t+1})_i=(W_t)_i−α \frac {(∇L(W_t))_i} {\sqrt {\sum^t_{t′=1}(∇L(W_{t′}))^2_i}}$$

Note that in practice, for weights $W∈R^d$, AdaGrad implementations (including the one in Caffe) use only O(d) extra storage for the historical gradient information (rather than the O(dt) storage that would be necessary to store each historical gradient individually).

[1] J. Duchi, E. Hazan, and Y. Singer. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. The Journal of Machine Learning Research, 2011.

### Adam

### NAG (Nesterov’s accelerated gradient)

### RMSprop

## Scaffolding

The solver scaffolding prepares the optimization method and initializes the model to be learned in Solver::Presolve().

Solver支架用Solver::Presolve()准备优化方法并初始化要学习的模型。

```
> caffe train -solver examples/mnist/lenet_solver.prototxt
I0902 13:35:56.474978 16020 caffe.cpp:90] Starting Optimization
I0902 13:35:56.475190 16020 solver.cpp:32] Initializing solver from parameters:
test_iter: 100
test_interval: 500
base_lr: 0.01
display: 100
max_iter: 10000
lr_policy: "inv"
gamma: 0.0001
power: 0.75
momentum: 0.9
weight_decay: 0.0005
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
solver_mode: GPU
net: "examples/mnist/lenet_train_test.prototxt"
```

Net initialization 网络初始化

```
I0902 13:35:56.655681 16020 solver.cpp:72] Creating training net from net file: examples/mnist/lenet_train_test.prototxt
[...]
I0902 13:35:56.656740 16020 net.cpp:56] Memory required for data: 0
I0902 13:35:56.656791 16020 net.cpp:67] Creating Layer mnist
I0902 13:35:56.656811 16020 net.cpp:356] mnist -> data
I0902 13:35:56.656846 16020 net.cpp:356] mnist -> label
I0902 13:35:56.656874 16020 net.cpp:96] Setting up mnist
I0902 13:35:56.694052 16020 data_layer.cpp:135] Opening lmdb examples/mnist/mnist_train_lmdb
I0902 13:35:56.701062 16020 data_layer.cpp:195] output data size: 64,1,28,28
I0902 13:35:56.701146 16020 data_layer.cpp:236] Initializing prefetch
I0902 13:35:56.701196 16020 data_layer.cpp:238] Prefetch initialized.
I0902 13:35:56.701212 16020 net.cpp:103] Top shape: 64 1 28 28 (50176)
I0902 13:35:56.701230 16020 net.cpp:103] Top shape: 64 1 1 1 (64)
[...]
I0902 13:35:56.703737 16020 net.cpp:67] Creating Layer ip1
I0902 13:35:56.703753 16020 net.cpp:394] ip1 <- pool2
I0902 13:35:56.703778 16020 net.cpp:356] ip1 -> ip1
I0902 13:35:56.703797 16020 net.cpp:96] Setting up ip1
I0902 13:35:56.728127 16020 net.cpp:103] Top shape: 64 500 1 1 (32000)
I0902 13:35:56.728142 16020 net.cpp:113] Memory required for data: 5039360
I0902 13:35:56.728175 16020 net.cpp:67] Creating Layer relu1
I0902 13:35:56.728194 16020 net.cpp:394] relu1 <- ip1
I0902 13:35:56.728219 16020 net.cpp:345] relu1 -> ip1 (in-place)
I0902 13:35:56.728240 16020 net.cpp:96] Setting up relu1
I0902 13:35:56.728256 16020 net.cpp:103] Top shape: 64 500 1 1 (32000)
I0902 13:35:56.728270 16020 net.cpp:113] Memory required for data: 5167360
I0902 13:35:56.728287 16020 net.cpp:67] Creating Layer ip2
I0902 13:35:56.728304 16020 net.cpp:394] ip2 <- ip1
I0902 13:35:56.728333 16020 net.cpp:356] ip2 -> ip2
I0902 13:35:56.728356 16020 net.cpp:96] Setting up ip2
I0902 13:35:56.728690 16020 net.cpp:103] Top shape: 64 10 1 1 (640)
I0902 13:35:56.728705 16020 net.cpp:113] Memory required for data: 5169920
I0902 13:35:56.728734 16020 net.cpp:67] Creating Layer loss
I0902 13:35:56.728747 16020 net.cpp:394] loss <- ip2
I0902 13:35:56.728767 16020 net.cpp:394] loss <- label
I0902 13:35:56.728786 16020 net.cpp:356] loss -> loss
I0902 13:35:56.728811 16020 net.cpp:96] Setting up loss
I0902 13:35:56.728837 16020 net.cpp:103] Top shape: 1 1 1 1 (1)
I0902 13:35:56.728849 16020 net.cpp:109]     with loss weight 1
I0902 13:35:56.728878 16020 net.cpp:113] Memory required for data: 5169924
```

Loss 损失

```
I0902 13:35:56.728893 16020 net.cpp:170] loss needs backward computation.
I0902 13:35:56.728909 16020 net.cpp:170] ip2 needs backward computation.
I0902 13:35:56.728924 16020 net.cpp:170] relu1 needs backward computation.
I0902 13:35:56.728938 16020 net.cpp:170] ip1 needs backward computation.
I0902 13:35:56.728953 16020 net.cpp:170] pool2 needs backward computation.
I0902 13:35:56.728970 16020 net.cpp:170] conv2 needs backward computation.
I0902 13:35:56.728984 16020 net.cpp:170] pool1 needs backward computation.
I0902 13:35:56.728998 16020 net.cpp:170] conv1 needs backward computation.
I0902 13:35:56.729014 16020 net.cpp:172] mnist does not need backward computation.
I0902 13:35:56.729027 16020 net.cpp:208] This network produces output loss
I0902 13:35:56.729053 16020 net.cpp:467] Collecting Learning Rate and Weight Decay.
I0902 13:35:56.729071 16020 net.cpp:219] Network initialization done.
I0902 13:35:56.729085 16020 net.cpp:220] Memory required for data: 5169924
I0902 13:35:56.729277 16020 solver.cpp:156] Creating test net (#0) specified by net file: examples/mnist/lenet_train_test.prototxt
```

Completion 结束

```
I0902 13:35:56.806970 16020 solver.cpp:46] Solver scaffolding done.
I0902 13:35:56.806984 16020 solver.cpp:165] Solving LeNet
```

## Updating Parameters 参数更新

The actual weight update is made by the solver then applied to the net parameters in Solver::ComputeUpdateValue(). The ComputeUpdateValue method incorporates any weight decay r(W) into the weight gradients (which currently just contain the error gradients) to get the final gradient with respect to each network weight. Then these gradients are scaled by the learning rate α and the update to subtract is stored in each parameter Blob’s diff field. Finally, the Blob::Update method is called on each parameter blob, which performs the final update (subtracting the Blob’s diff from its data).

实际的权值更新由solver完成然后应用于网络参数中，对应方法为Solver::ComputeUpdateValue()。这个方法将权值衰减r(W)整合入权值梯度中（现在只包含误差梯度）得到最终对每个网络权值的梯度。然后这些梯度乘以学习速率α，减法的更新存储在每个参数blob的diff域中。最后，在每个参数blob上调用Blob::Update方法，执行最终的更新（从blob的data中减去diff）

## Snapshotting and Resuming 创建快照和恢复

The solver snapshots the weights and its own state during training in Solver::Snapshot() and Solver::SnapshotSolverState(). The weight snapshots export the learned model while the solver snapshots allow training to be resumed from a given point. Training is resumed by Solver::Restore() and Solver::RestoreSolverState().

solver用Solver::Snapshot()和Solver::SnapshotSolverState()对训练过程中的权值和自身状态创建快照。权值快照导出学习的模型，而solver快照使训练可以从一个给定点恢复。训练恢复用Solver::Restore()和Solver::RestoreSolverState()。

Weights are saved without extension while solver states are saved with .solverstate extension. Both files will have an _iter_N suffix for the snapshot iteration number.

保存的权值文件没有扩展名，而solver状态的保存文件扩展名为.solverstate。两种文件都有_iter_N后缀，即快照时迭代的次数。

Snapshotting is configured by: 配置快照方法如下：

```
# The snapshot interval in iterations.
snapshot: 5000
# File path prefix for snapshotting model weights and solver state.
# Note: this is relative to the invocation of the `caffe` utility, not the
# solver definition file.
snapshot_prefix: "/path/to/model"
# Snapshot the diff along with the weights. This can help debugging training
# but takes more storage.
snapshot_diff: false
# A final snapshot is saved at the end of training unless
# this flag is set to false. The default is true.
snapshot_after_train: true
```

in the solver definition prototxt. 在solver定义prototxt文件中。