# Tensors 张量

TensorFlow, as the name indicates, is a framework to define and run computations involving tensors. A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.

张量是定义和运行张量计算的框架。张量是矢量和矩阵的泛化形式，可以拥有更高的维度。TensorFlow内部将张量表示为n维基础数据类型的阵列。

When writing a TensorFlow program, the main object you manipulate and pass around is the `tf.Tensor`. A `tf.Tensor` object represents a partially defined computation that will eventually produce a value. TensorFlow programs work by first building a graph of `tf.Tensor` objects, detailing how each tensor is computed based on the other available tensors and then by running parts of this graph to achieve the desired results.

当写TensorFlow程序时，主要的操作对象就是`tf.Tensor`。一个`tf.Tensor`对象代表部分定义的计算，最终会产生一个值。TensorFlow程序首先构建一个计算图，包含一些`tf.Tensor`对象，计算图中详细叙述了每个张量怎样根据其他张量来进行计算，然后通过（部分）运行这个计算图，可以得到想要的结果。

A `tf.Tensor` has the following properties: `tf.Tensor`有如下属性：

- a data type (float32, int32, or string, for example) 数据类型
- a shape 形状

Each element in the Tensor has the same data type, and the data type is always known. The shape (that is, the number of dimensions it has and the size of each dimension) might be only partially known. Most operations produce tensors of fully-known shapes if the shapes of their inputs are also fully known, but in some cases it's only possible to find the shape of a tensor at graph execution time.

张量中的每个元素数据类型都一样，而且数据类型都是已知的。形状，也就是维数，和每个维度的大小，可能只会确定一部分。对多数操作来说，如果输入张量的形状全部已知，那么生成的张量，其形状也都是确定全部知道的，但也有一些情况只能在计算图运行的时候才能确定张量的形状。

Some types of tensors are special, and these will be covered in other units of the TensorFlow guide. The main ones are: 一些张量的类型比较特殊，将会在TensorFlow指南的其他部分讲述。主要的类型有：

- `tf.Variable` 变量
- `tf.constant` 常量
- `tf.placeholder` 占位符
- `tf.SparseTensor` 稀疏张量

With the exception of `tf.Variable`, the value of a tensor is immutable, which means that in the context of a single execution tensors only have a single value. However, evaluating the same tensor twice can return different values; for example that tensor can be the result of reading data from disk, or generating a random number.

除了`tf.Variable`，其他张量的值都是不可改变的，意思是，在每次执行的上下文环境中，张量都只有一个单独的值。如果对张量求值两次，可能返回不同的值，比如张量可以是从磁盘上读的数据的结果，或者是随机产生的数值。

## Rank

The rank of a `tf.Tensor` object is its number of dimensions. Synonyms for rank include order or degree or n-dimension. Note that rank in TensorFlow is not the same as matrix rank in mathematics. As the following table shows, each rank in TensorFlow corresponds to a different mathematical entity:

一个`tf.Tensor`对象的rank是其维数。rank的同义词包括order, 或degree, 或n-dimension。注意TensorFlow中的rank和数学矩阵中的rank不一样。如下表所示，TensorFlow每个rank对应一种不同的数学实体：

Rank | Math entity
--- | ---
0	| Scalar (magnitude only)
1	| Vector (magnitude and direction)
2	| Matrix (table of numbers)
3	| 3-Tensor (cube of numbers)
n	| n-Tensor (you get the idea)

### Rank 0

The following snippet demonstrates creating a few rank 0 variables: 下面的代码片段展示了创建几个rank 0的变量：

```
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
```

> Note: A string is treated as a single item in TensorFlow, not as a sequence of characters. It is possible to have scalar strings, vectors of strings, etc.

### Rank 1

To create a rank 1 `tf.Tensor` object, you can pass a list of items as the initial value. For example: 要创建一个rank 1的`tf.Tensor`对象，可以将几个值作为其初始值，比如：

```
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)
```

### Higher ranks

A rank 2 `tf.Tensor` object consists of at least one row and at least one column: rank 2的`tf.Tensor`对象包括至少一行一列的元素：

```
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)
```

Higher-rank Tensors, similarly, consist of an n-dimensional array. For example, during image processing, many tensors of rank 4 are used, with dimensions corresponding to example-in-batch, image width, image height, and color channel.

类似的，高阶张量包括n维阵列。比如，在图像处理中，很多张量是rank 4的，其维数分别对应，批次样本数，图像高度，图像宽度，色彩通道数。

```
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color
```

### Getting a `tf.Tensor` object's rank

To determine the rank of a `tf.Tensor` object, call the `tf.rank` method. For example, the following method programmatically determines the rank of the `tf.Tensor` defined in the previous section: 为确定`tf.Tensor`对象的rank，调用`tf.rank`方法。

```
r = tf.rank(my_image)
# After the graph runs, r will hold the value 4.
```

### Referring to `tf.Tensor` slices 引用`tf.Tensor`切片

Since a `tf.Tensor` is an n-dimensional array of cells, to access a single cell in a `tf.Tensor` you need to specify n indices.

由于`tf.Tensor`对象时一个n维阵列，要引用`tf.Tensor`中一个元素，需要指定n个索引。

For a rank 0 tensor (a scalar), no indices are necessary, since it is already a single number. For a rank 1 tensor (a vector), passing a single index allows you to access a number: 对于rank 0张量（标量），不需要指定索引，因为本身就是一个数值。对于rank 1的张量（矢量），需要指定一个索引来引用其中某个数字：

`my_scalar = my_vector[2]`

Note that the index passed inside the [] can itself be a scalar `tf.Tensor`, if you want to dynamically choose an element from the vector. 注意在方括号[]中传递的索引本身也可以是一个`tf.Tensor`对象，如果需要从矢量中动态选择不同的元素，可以试试这样。

For tensors of rank 2 or higher, the situation is more interesting. For a `tf.Tensor` of rank 2, passing two numbers returns a scalar, as expected: 对于rank 2或更高的张量，情况更加有意思。对于rank 2的`tf.Tensor`，传递2个数作为索引返回一个张量：

`my_scalar = my_matrix[1, 2]`

Passing a single number, however, returns a subvector of a matrix, as follows: 如果只传递一个数作为索引，那么会返回矩阵的一个子向量，如下所示：

`my_row_vector = my_matrix[2]`
`my_column_vector = my_matrix[:, 3]`

The : notation is python slicing syntax for "leave this dimension alone". This is useful in higher-rank Tensors, as it allows you to access its subvectors, submatrices, and even other subtensors. 注意冒号:是Python切片的语法，表示不要管这一维，这在高阶张量中非常有用，因为可以访问子向量、子矩阵甚至是子张量。

## Shape

The shape of a tensor is the number of elements in each dimension. TensorFlow automatically infers shapes during graph construction. These inferred shapes might have known or unknown rank. If the rank is known, the sizes of each dimension might be known or unknown.

张量的形状是在每个维度上元素的数量。TensorFlow在计算图构建的时候自动推断形状。这些推断出的形状其rank可能已知或未知。如果rank已知，每个维度上的元素数量也可能是已知或未知。

The TensorFlow documentation uses three notational conventions to describe tensor dimensionality: rank, shape, and dimension number. The following table shows how these relate to one another:

TensorFlow文档用三种标记习惯来描述张量：rank, shape和dimension number。下表所示的三个量之间的关系。

Rank | Shape | Dimension number | Example
-- | --- | --- | ---
0	| []	| 0-D	| A 0-D tensor. A scalar.
1	| [D0]	| 1-D	| A 1-D tensor with shape [5].
2	| [D0, D1]	| 2-D	| A 2-D tensor with shape [3, 4].
3	| [D0, D1, D2]	| 3-D	| A 3-D tensor with shape [1, 4, 3].
n	| [D0, D1, ... Dn-1]	| n-D	| A tensor with shape [D0, D1, ... Dn-1].

Shapes can be represented via Python lists / tuples of ints, or with the `tf.TensorShape`. 形状可以用Python列表或元组来表示，或用`tf.TensorShape`表示。

### Getting a tf.Tensor object's shape

There are two ways of accessing the shape of a `tf.Tensor`. While building the graph, it is often useful to ask what is already known about a tensor's shape. This can be done by reading the shape property of a `tf.Tensor` object. This method returns a TensorShape object, which is a convenient way of representing partially-specified shapes (since, when building the graph, not all shapes will be fully known).

两种方法访问`tf.Tensor`的形状。在构建计算图时，经常会访问张量的已知形状。可以读取`tf.Tensor`对象的形状属性。这个方法返回一个TensorShape对象，这个对象可以很方便的表示部分确定的张量形状（当构建计算图时，形状信息不一定会全知道）。

It is also possible to get a `tf.Tensor` that will represent the fully-defined shape of another `tf.Tensor` at runtime. This is done by calling the `tf.shape` operation. This way, you can build a graph that manipulates the shapes of tensors by building other tensors that depend on the dynamic shape of the input `tf.Tensor`.

在运行时可能得到`tf.Tensor`对象，其表示的是另一个全已知的`tf.Tensor`形状，这是通过调用`tf.shape`操作得到的。通过这种方法，可以构建一个计算图，根据输入对象`tf.Tensor`的形状来构建其他张量。

For example, here is how to make a vector of zeros with the same size as the number of columns in a given matrix: 比如，下面是怎样通过给定矩阵构建一个张量，张量的维数是矩阵的列数：

`zeros = tf.zeros(my_matrix.shape[1])`

### Changing the shape of a `tf.Tensor`

The number of elements of a tensor is the product of the sizes of all its shapes. The number of elements of a scalar is always 1. Since there are often many different shapes that have the same number of elements, it's often convenient to be able to change the shape of a `tf.Tensor`, keeping its elements fixed. This can be done with `tf.reshape`.

张量中元素的数量是其所有维度的大小的成绩。标量的元素数量永远是1。有很多不同形状的张量，其元素数量是一样的，可以通过改变其形状在不同张量间转换，这是通过调用函数`tf.reshape`来实现的。

The following examples demonstrate how to reshape tensors: 下面的例子是如何改变张量的形状：

```
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!
```

## Data types

In addition to dimensionality, Tensors have a data type. Refer to the `tf.DType` page for a complete list of the data types. 除了维度，张量还有数据类型。参考`tf.DType`页面查看所有的数据类型。

It is not possible to have a `tf.Tensor` with more than one data type. It is possible, however, to serialize arbitrary data structures as strings and store those in `tf.Tensors`. 一个`tf.Tensor`对象只能有一种数据类型，但是可以将任意类型的`tf.Tensor`序列化为字符串进行保存。

It is possible to cast `tf.Tensors` from one datatype to another using `tf.cast`: 可以用`tf.cast`进行数据类型转换：

```
# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
```

To inspect a `tf.Tensor`'s data type use the `Tensor.dtype` property. 用`Tensor.dtype`来查看张量的数据类型属性。

When creating a `tf.Tensor` from a python object you may optionally specify the datatype. If you don't, TensorFlow chooses a datatype that can represent your data. TensorFlow converts Python integers to `tf.int32` and python floating point numbers to `tf.float32`. Otherwise TensorFlow uses the same rules numpy uses when converting to arrays.

当从Python对象生成一个`tf.Tensor`对象时，可以部分指定数据类型。如果不指定，TensorFlow会选择一个数据类型代表你的数据。TensorFlow将Python整数转换成`tf.int32`，Python浮点数转换为`tf.float32`。其他转换阵列类型的情况下，TensorFlow使用与Numpy相同的原则。

## Evaluating Tensors

Once the computation graph has been built, you can run the computation that produces a particular `tf.Tensor` and fetch the value assigned to it. This is often useful for debugging as well as being required for much of TensorFlow to work.

建立好计算图之后，可以运行计算，取得一些特定的`tf.Tensor`的值。这是TensorFlow的运行方式，也在调试时很方便。

The simplest way to evaluate a Tensor is using the `Tensor.eval` method. For example: 最简单的求值方式是使用`Tensor.eval`方法，比如：

```
constant = tf.constant([1, 2, 3])
tensor = constant * constant
print(tensor.eval())
```

The eval method only works when a default `tf.Session` is active (see Graphs and Sessions for more information). 当有默认`tf.Session`活跃时，eval方法才是可用的。

`Tensor.eval` returns a numpy array with the same contents as the tensor. `Tensor.eval`方法返回一个numpy阵列，其值与张量相同。

Sometimes it is not possible to evaluate a `tf.Tensor` with no context because its value might depend on dynamic information that is not available. For example, tensors that depend on placeholders can't be evaluated without providing a value for the placeholder.

有时候没有上下文无法求`tf.Tensor`的值，因为其值可能与一些动态信息有关，而这些动态信息可能不可用。比如，依靠placeholder的张量，如果不给placeholder赋值的话，就不能求值。

```
p = tf.placeholder(tf.float32)
t = p + 1.0
t.eval()  # This will fail, since the placeholder did not get a value.
t.eval(feed_dict={p:2.0})  # This will succeed because we're feeding a value to the placeholder.
```

Note that it is possible to feed any `tf.Tensor`, not just placeholders. 注意可以给任何`tf.Tensor`赋值，不一定需要是placeholder。

Other model constructs might make evaluating a `tf.Tensor` complicated. TensorFlow can't directly evaluate `tf.Tensor`s defined inside functions or inside control flow constructs. If a `tf.Tensor` depends on a value from a queue, evaluating the `tf.Tensor` will only work once something has been enqueued; otherwise, evaluating it will hang. When working with queues, remember to call `tf.train.start_queue_runners` before evaluating any `tf.Tensor`s.

## Printing Tensors

For debugging purposes you might want to print the value of a `tf.Tensor`. While `tfdbg` provides advanced debugging support, TensorFlow also has an operation to directly print the value of a `tf.Tensor`.

为了调试，可能需要打印`tf.Tensor`的值。`tfdbg`提供了高级调试支持，TensorFlow有一个操作可以直接打印张量值。

Note that you rarely want to use the following pattern when printing a `tf.Tensor`: 注意尽量不要使用下面的方式打印张量：

```
t = <<some tensorflow operation>>
print(t)  # This will print the symbolic tensor when the graph is being built.
          # This tensor does not have a value in this context.
```

This code prints the `tf.Tensor` object (which represents deferred computation) and not its value. Instead, TensorFlow provides the `tf.Print` operation, which returns its first tensor argument unchanged while printing the set of `tf.Tensor`s it is passed as the second argument.

这种代码打印的是`tf.Tensor`对象，代表的是延期计算对象，不是其真值。TensorFlow有一个`tf.Print`操作，可以返回其第一个张量参数不变，而打印`tf.Tensor`对象作为第二个参数。

To correctly use `tf.Print` its return value must be used. See the example below 为正确的使用`tf.Print`函数，其返回值必须使用。

```
t = <<some tensorflow operation>>
tf.Print(t, [t])  # This does nothing
t = tf.Print(t, [t])  # Here we are using the value returned by tf.Print
result = t + 1  # Now when result is evaluated the value of `t` will be printed.
```

When you evaluate `result` you will evaluate everything `result` depends upon. Since `result` depends upon t, and evaluating t has the side effect of printing its input (the old value of t), t gets printed.

当求`result`的值时，必须求所有`result`相关的值。由于`result`依赖t，求t的值又会引发打印输入t的操作，所以t被打印出来了。