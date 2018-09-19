# AutoGraph: Easy control flow for graphs

AutoGraph helps you write complicated graph code using normal Python. Behind the scenes, AutoGraph automatically transforms your code into the equivalent TensorFlow graph code. AutoGraph already supports much of the Python language, and that coverage continues to grow. For a list of supported Python language features, see the Autograph capabilities and limitations.

AutoGraph使用正常的Python帮助你编写复杂图的代码。在后台，AutoGraph自动将代码转换成等效的TensorFlow图代码。AutoGraph已经支持大多数Python语言，并不断增长。对于支持的Python语言特征，详见Autograph capabilities and limitations文档。

## Setup 安装

To use AutoGraph, install the latest version of TensorFlow: 安装最新版TensorFlow以使用AutoGraph：

```
! pip install -q -U tf-nightly

keras 2.2.2 has requirement keras-applications==1.0.4, but you'll have keras-applications 1.0.5 which is incompatible.
keras 2.2.2 has requirement keras-preprocessing==1.0.2, but you'll have keras-preprocessing 1.0.3 which is incompatible.
```

Import TensorFlow, AutoGraph, and any supporting modules: 导入所有支持模块

```
from __future__ import division, print_function, absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow.contrib import autograph


import numpy as np
import matplotlib.pyplot as plt
```

We'll enable eager execution for demonstration purposes, but AutoGraph works in both eager and graph execution environments: `tf.enable_eager_execution()` 我们激活eager execution进行展示，但AutoGraph在eager execution和graph execution环境中都可以运行。

> Note: AutoGraph converted code is designed to run during graph execution. When eager exectuon is enabled, use explicit graphs (as this example shows) or `tf.contrib.eager.defun`.

> 注意：AutoGraph转换的代码是为graph execution准备的。当激活eager execution时，显式的使用graph（本例就是这样的）或`tf.contrib.eager.defun`.

## Automatically convert Python control flow 自动转换Python控制流

AutoGraph will convert much of the Python language into the equivalent TensorFlow graph building code. AutoGraph会将大多数Python语言转换为对应的TensorFlow图代码。

> Note: In real applications batching is essential for performance. The best code to convert to AutoGraph is code where the control flow is decided at the batch level. If making decisions at the individual example level, you must index and batch the examples to maintain performance while applying the control flow logic.

> 注意：在真实应用中，分批次对于性能来说非常重要。最适合AutoGraph转换的代码就是控制流在批次层次决定的代码。如果在个体样本水平做决定，必须使用索引并对样本分批次，在应用控制流逻辑的同时来维持性能。

AutoGraph converts a function like: AutoGraph像下面一样转换函数：

```
def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0.0
  return x
```

To a function that uses graph building: 对于一个使用图的函数：`print(autograph.to_code(square_if_positive))`

```
from __future__ import print_function
import tensorflow as tf

def tf__square_if_positive(x):
  try:
    with tf.name_scope('square_if_positive'):

      def if_true():
        with tf.name_scope('if_true'):
          x_1, = x,
          x_1 = x_1 * x_1
          return x_1,

      def if_false():
        with tf.name_scope('if_false'):
          x_2, = x,
          x_2 = 0.0
          return x_2,
      x = ag__.utils.run_cond(tf.greater(x, 0), if_true, if_false)
      return x
  except:
    ag__.rewrite_graph_construction_error(ag_source_map__)



tf__square_if_positive.autograph_info__ = {}
```

Code written for eager execution can run in a tf.Graph with the same results, but with the benfits of graph execution: 为eager execution写的代码可以在tf.Graph中运行，得到相同的结果，但也有graph execution的好处：

```
print('Eager results: %2.2f, %2.2f' % (square_if_positive(tf.constant(9.0)), 
                                       square_if_positive(tf.constant(-9.0))))

Eager results: 81.00, 0.00
```

Generate a graph-version and call it: 生成一个graph版的，然后调用：

```
tf_square_if_positive = autograph.to_graph(square_if_positive)

with tf.Graph().as_default():  
  # The result works like a regular op: takes tensors in, returns tensors.
  # You can inspect the graph using tf.get_default_graph().as_graph_def()
  g_out1 = tf_square_if_positive(tf.constant( 9.0))
  g_out2 = tf_square_if_positive(tf.constant(-9.0))
  with tf.Session() as sess:
    print('Graph results: %2.2f, %2.2f\n' % (sess.run(g_out1), sess.run(g_out2)))


Graph results: 81.00, 0.00
```

AutoGraph supports common Python statements like while, for, if, break, and return, with support for nesting. Compare this function with the complicated graph verson displayed in the following code blocks:

AutoGraph支持普通的Python语句如while, for, if, break和return，支持嵌套。将这个函数与如下的复杂的graph版比较：

```
# Continue in a loop
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s

print('Eager result: %d' % sum_even(tf.constant([10,12,15,20])))

tf_sum_even = autograph.to_graph(sum_even)

with tf.Graph().as_default(), tf.Session() as sess:
    print('Graph result: %d\n\n' % sess.run(tf_sum_even(tf.constant([10,12,15,20]))))
```

`Eager result: 42`
`Graph result: 42`

`print(autograph.to_code(sum_even))`

```
from __future__ import print_function
import tensorflow as tf

def tf__sum_even(items):
  try:
    with tf.name_scope('sum_even'):
      s = 0

      def extra_test(s_2):
        with tf.name_scope('extra_test'):
          return True

      def loop_body(loop_vars, s_2):
        with tf.name_scope('loop_body'):
          c = loop_vars
          continue_ = tf.constant(False)

          def if_true():
            with tf.name_scope('if_true'):
              continue__1, = continue_,
              continue__1 = tf.constant(True)
              return continue__1,

          def if_false():
            with tf.name_scope('if_false'):
              return continue_,
          continue_ = ag__.utils.run_cond(tf.greater(c % 2, 0), if_true,
              if_false)

          def if_true_1():
            with tf.name_scope('if_true_1'):
              s_1, = s_2,
              s_1 += c
              return s_1,

          def if_false_1():
            with tf.name_scope('if_false_1'):
              return s_2,
          s_2 = ag__.utils.run_cond(tf.logical_not(continue_), if_true_1,
              if_false_1)
          return s_2,
      s = ag__.for_stmt(items, extra_test, loop_body, (s,))
      return s
  except:
    ag__.rewrite_graph_construction_error(ag_source_map__)



tf__sum_even.autograph_info__ = {}
```

## Decorator

If you don't need easy access to the original Python function, use the `convert` decorator: 如果不需要轻易访问原始Python函数，使用`convert` decorator：

```
@autograph.convert()
def fizzbuzz(i, n):
  while i < n:
    msg = ''
    if i % 3 == 0:
      msg += 'Fizz'
    if i % 5 == 0:
      msg += 'Buzz'
    if msg == '':
      msg = tf.as_string(i)
    print(msg)
    i += 1
  return i

with tf.Graph().as_default():
  final_i = fizzbuzz(tf.constant(10), tf.constant(16))
  # The result works like a regular op: takes tensors in, returns tensors.
  # You can inspect the graph using tf.get_default_graph().as_graph_def()
  with tf.Session() as sess:
    sess.run(final_i)
```

`Tensor("fizzbuzz/while/loop_body/cond_2/Merge:0", shape=(), dtype=string)`

## Examples

Let's demonstrate some useful Python language features. 我们展示几个有用的Python语言特征。

### Assert

AutoGraph automatically converts the Python assert statement into the equivalent tf.Assert code: AutoGraph自动转换Python assert语句为相应的tf.Assert代码：

```
@autograph.convert()
def inverse(x):
  assert x != 0.0, 'Do not pass zero!'
  return 1.0 / x

with tf.Graph().as_default(), tf.Session() as sess:
  try:
    print(sess.run(inverse(tf.constant(0.0))))
  except tf.errors.InvalidArgumentError as e:
    print('Got error message:\n    %s' % e.message)
```

```
Got error message:
    assertion failed: [Do not pass zero!]
     [[node inverse/Assert/Assert (defined at /tmp/tmp5y2lbzay.py:7)  = Assert[T=[DT_STRING], summarize=3, _device="/job:localhost/replica:0/task:0/device:CPU:0"](inverse/NotEqual, inverse/Assert/Assert/data_0)]]
```

### Print

Use the Python print function in-graph: 在graph中使用Python print函数：

```
@autograph.convert()
def count(n):
  i=0
  while i < n:
    print(i)
    i += 1
  return n
    
with tf.Graph().as_default(), tf.Session() as sess:
    sess.run(count(tf.constant(5)))
```

`Tensor("count/while/Identity:0", shape=(), dtype=int32)`

### Lists

Append to lists in loops (tensor list ops are automatically created): 用循环在列表中追加元素（张量列表操作自动创建）

```
@autograph.convert()
def arange(n):
  z = []
  # We ask you to tell us the element dtype of the list
  autograph.set_element_type(z, tf.int32)
  
  for i in tf.range(n):
    z.append(i)
  # when you're done with the list, stack it
  # (this is just like np.stack)
  return autograph.stack(z) 


with tf.Graph().as_default(), tf.Session() as sess:
    sess.run(arange(tf.constant(10)))
```

### Nested control flow

```
@autograph.convert()
def nearest_odd_square(x):
  if x > 0:
    x = x * x
    if x % 2 == 0:
      x = x + 1
  return x

with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(nearest_odd_square(tf.constant(4))))
    print(sess.run(nearest_odd_square(tf.constant(5))))
    print(sess.run(nearest_odd_square(tf.constant(6))))
```

`17 25 37`

### While loop

```
@autograph.convert()
def square_until_stop(x, y):
  while x < y:
    x = x * x
  return x
    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(square_until_stop(tf.constant(4), tf.constant(100))))
```

`256`

### For loop

```
@autograph.convert()
def squares(nums):

  result = []
  autograph.set_element_type(result, tf.int64)

  for num in nums: 
    result.append(num * num)
    
  return autograph.stack(result)
    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(squares(tf.constant(np.arange(10)))))
```

`[ 0  1  4  9 16 25 36 49 64 81]`

### Break

```
@autograph.convert()
def argwhere_cumsum(x, threshold):
  current_sum = 0.0
  idx = 0
  for i in tf.range(len(x)):
    idx = i
    if current_sum >= threshold:
      break
    current_sum += x[i]
  return idx

N = 10
with tf.Graph().as_default():  
  with tf.Session() as sess:
    idx = argwhere_cumsum(tf.ones(N), tf.constant(float(N/2)))
    print(sess.run(idx))
```

`5`

## Interoperation with tf.Keras

Now that you've seen the basics, let's build some model components with autograph. 现在我们看过了基础知识，我们用AutoGraph建立一些模型组件。

It's relatively simple to integrate autograph with tf.keras. 将AutoGraph与tf.keras集成到一起相对是比较简单的

### Stateless functions 无状态函数

For stateless functions, like collatz shown below, the easiest way to include them in a keras model is to wrap them up as a layer using tf.keras.layers.Lambda. 对于无状态函数，比如下面的collatz，在keras模型中包含它们的最简单方法是把它们包装成一个层，使用tf.keras.layers.Lambda。

```
import numpy as np

@autograph.convert()
def collatz(x):
  x = tf.reshape(x,())
  assert x > 0
  n = tf.convert_to_tensor((0,)) 
  while not tf.equal(x, 1):
    n += 1
    if tf.equal(x%2, 0):
      x = x // 2
    else:
      x = 3 * x + 1
      
  return n

with tf.Graph().as_default():
  model = tf.keras.Sequential([
    tf.keras.layers.Lambda(collatz, input_shape=(1,), output_shape=())
  ])
  
result = model.predict(np.array([6171]))
result
```

`array([261], dtype=int32)`

### Custom Layers and Models

The easiest way to use AutoGraph with Keras layers and models is to @autograph.convert() the call method. See the TensorFlow Keras guide for details on how to build on these classes. 将AutoGraph和Keras层一起使用的最简单方法是@autograph.convert()，详见TensorFlow Keras guide。

Here is a simple example of the stocastic network depth technique: 下面是一个简单例子：

```
# `K` is used to check if we're in train or test mode.
K = tf.keras.backend

class StocasticNetworkDepth(tf.keras.Sequential):
  def __init__(self, pfirst=1.0, plast=0.5, *args,**kwargs):
    self.pfirst = pfirst
    self.plast = plast
    super().__init__(*args,**kwargs)
        
  def build(self,input_shape):
    super().build(input_shape.as_list())
    self.depth = len(self.layers)
    self.plims = np.linspace(self.pfirst, self.plast, self.depth + 1)[:-1]
    
  @autograph.convert()
  def call(self, inputs):
    training = tf.cast(K.learning_phase(), dtype=bool)  
    if not training: 
      count = self.depth
      return super(StocasticNetworkDepth, self).call(inputs), count
    
    p = tf.random_uniform((self.depth,))
    
    keeps = (p <= self.plims)
    x = inputs
    
    count = tf.reduce_sum(tf.cast(keeps, tf.int32))
    for i in range(self.depth):
      if keeps[i]:
        x = self.layers[i](x)
      
    # return both the final-layer output and the number of layers executed.
    return x, count
```

Let's try it on mnist-shaped data: 在mnist形状的数据上测试一下：

`train_batch = np.random.randn(64, 28, 28, 1).astype(np.float32)`

Build a simple stack of conv layers, in the stocastic depth model: 在随机深度模型中构建简单的卷积层叠加：

```
with tf.Graph().as_default() as g:
  model = StocasticNetworkDepth(
        pfirst=1.0, plast=0.5)

  for n in range(20):
    model.add(
          layers.Conv2D(filters=16, activation=tf.nn.relu,
                        kernel_size=(3, 3), padding='same'))

  model.build(tf.TensorShape((None, None, None, 1)))
  
  init = tf.global_variables_initializer()
```

Now test it to ensure it behaves as expected in train and test modes:

```
# Use an explicit session here so we can set the train/test switch, and
# inspect the layer count returned by `call`
with tf.Session(graph=g) as sess:
  init.run()
 
  for phase, name in enumerate(['test','train']):
    K.set_learning_phase(phase)
    result, count = model(tf.convert_to_tensor(train_batch, dtype=tf.float32))

    result1, count1 = sess.run((result, count))
    result2, count2 = sess.run((result, count))

    delta = (result1 - result2)
    print(name, "sum abs delta: ", abs(delta).mean())
    print("    layers 1st call: ", count1)
    print("    layers 2nd call: ", count2)
    print()
```

```
test sum abs delta:  0.0
    layers 1st call:  20
    layers 2nd call:  20

train sum abs delta:  0.00029145874
    layers 1st call:  19
    layers 2nd call:  16
```

## Advanced example: An in-graph training loop

The previous section showed that AutoGraph can be used inside Keras layers and models. Keras models can also be used in AutoGraph code.前一节展示了AutoGraph可以在Keras层和模型中间使用。Keras模型也可以用在AutoGraph代码中。

Since writing control flow in AutoGraph is easy, running a training loop in a TensorFlow graph should also be easy. 由于在AutoGraph中写控制流很容易，在TensorFlow图中运行一个训练循环应当也很容易。

This example shows how to train a simple Keras model on MNIST with the entire training process—loading batches, calculating gradients, updating parameters, calculating validation accuracy, and repeating until convergence—is performed in-graph.

这个例子展示了怎样在MNIST上训练简单的keras模型，整个训练过程都是在in-graph执行的，包括载入批次，计算梯度，更新参数，计算验证准确率，重复直到收敛。

### Download data

```
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
```

### Define the model

```
def mlp_model(input_shape):
  model = tf.keras.Sequential((
      tf.keras.layers.Dense(100, activation='relu', input_shape=input_shape),
      tf.keras.layers.Dense(100, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')))
  model.build()
  return model


def predict(m, x, y):
  y_p = m(tf.reshape(x, (-1, 28 * 28)))
  losses = tf.keras.losses.categorical_crossentropy(y, y_p)
  l = tf.reduce_mean(losses)
  accuracies = tf.keras.metrics.categorical_accuracy(y, y_p)
  accuracy = tf.reduce_mean(accuracies)
  return l, accuracy


def fit(m, x, y, opt):
  l, accuracy = predict(m, x, y)
  # Autograph automatically adds the necessary `tf.control_dependencies` here.
  # (Without them nothing depends on `opt.minimize`, so it doesn't run.)
  # This makes it much more like eager-code.
  opt.minimize(l)
  return l, accuracy


def setup_mnist_data(is_training, batch_size):
  if is_training:
    ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds = ds.shuffle(batch_size * 10)
  else:
    ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

  ds = ds.repeat()
  ds = ds.batch(batch_size)
  return ds


def get_next_batch(ds):
  itr = ds.make_one_shot_iterator()
  image, label = itr.get_next()
  x = tf.to_float(image) / 255.0
  y = tf.one_hot(tf.squeeze(label), 10)
  return x, y 
```

### Define the training loop

```
# Use `recursive = True` to recursively convert functions called by this one.
@autograph.convert(recursive=True)
def train(train_ds, test_ds, hp):
  m = mlp_model((28 * 28,))
  opt = tf.train.AdamOptimizer(hp.learning_rate)
  
  # We'd like to save our losses to a list. In order for AutoGraph
  # to convert these lists into their graph equivalent,
  # we need to specify the element type of the lists.
  train_losses = []
  autograph.set_element_type(train_losses, tf.float32)
  test_losses = []
  autograph.set_element_type(test_losses, tf.float32)
  train_accuracies = []
  autograph.set_element_type(train_accuracies, tf.float32)
  test_accuracies = []
  autograph.set_element_type(test_accuracies, tf.float32)
  
  # This entire training loop will be run in-graph.
  i = tf.constant(0)
  while i < hp.max_steps:
    train_x, train_y = get_next_batch(train_ds)
    test_x, test_y = get_next_batch(test_ds)

    step_train_loss, step_train_accuracy = fit(m, train_x, train_y, opt)
    step_test_loss, step_test_accuracy = predict(m, test_x, test_y)
    if i % (hp.max_steps // 10) == 0:
      print('Step', i, 'train loss:', step_train_loss, 'test loss:',
            step_test_loss, 'train accuracy:', step_train_accuracy,
            'test accuracy:', step_test_accuracy)
    train_losses.append(step_train_loss)
    test_losses.append(step_test_loss)
    train_accuracies.append(step_train_accuracy)
    test_accuracies.append(step_test_accuracy)
    i += 1
  
  # We've recorded our loss values and accuracies 
  # to a list in a graph with AutoGraph's help.
  # In order to return the values as a Tensor, 
  # we need to stack them before returning them.
  return (autograph.stack(train_losses), autograph.stack(test_losses),  
          autograph.stack(train_accuracies), autograph.stack(test_accuracies))
```

Now build the graph and run the training loop:

```
with tf.Graph().as_default() as g:
  hp = tf.contrib.training.HParams(
      learning_rate=0.005,
      max_steps=500,
  )
  train_ds = setup_mnist_data(True, 50)
  test_ds = setup_mnist_data(False, 1000)
  (train_losses, test_losses, train_accuracies,
   test_accuracies) = train(train_ds, test_ds, hp)

  init = tf.global_variables_initializer()
  
with tf.Session(graph=g) as sess:
  sess.run(init)
  (train_losses, test_losses, train_accuracies,
   test_accuracies) = sess.run([train_losses, test_losses, train_accuracies,
                                test_accuracies])
  
plt.title('MNIST train/test losses')
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.xlabel('Training step')
plt.ylabel('Loss')
plt.show()
plt.title('MNIST train/test accuracies')
plt.plot(train_accuracies, label='train accuracy')
plt.plot(test_accuracies, label='test accuracy')
plt.legend(loc='lower right')
plt.xlabel('Training step')
plt.ylabel('Accuracy')
plt.show()
```

`Step Tensor("train/while/Identity:0", shape=(), dtype=int32) train loss: Tensor("train/while/loop_body/fit/Identity_2:0", shape=(), dtype=float32) test loss: Tensor("train/while/loop_body/predict/Mean:0", shape=(), dtype=float32) train accuracy: Tensor("train/while/loop_body/fit/predict/Mean_1:0", shape=(), dtype=float32) test accuracy: Tensor("train/while/loop_body/predict/Mean_1:0", shape=(), dtype=float32)`

![Image](https://www.tensorflow.org/guide/autograph_files/output_56_1.png)
![Image](https://www.tensorflow.org/guide/autograph_files/output_56_2.png)