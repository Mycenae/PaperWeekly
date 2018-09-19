# Save and Restore 保存和恢复

The `tf.train.Saver` class provides methods to save and restore models. The `tf.saved_model.simple_save` function is an easy way to build a saved model suitable for serving. Estimators automatically save and restore variables in the model_dir.

`tf.train.Saver`类提供了保存和恢复模型的方法。`tf.saved_model.simple_save`函数是构建适用于服务的保存模型的简单方法。Estimators在model_dir中自动保存恢复变量。

## Save and restore variables 保存和恢复变量

TensorFlow Variables are the best way to represent shared, persistent state manipulated by your program. The `tf.train.Saver` constructor adds save and restore ops to the graph for all, or a specified list, of the variables in the graph. The Saver object provides methods to run these ops, specifying paths for the checkpoint files to write to or read from.

TensorFlow变量是程序表示共享、持续状态的最好方法。`tf.train.Saver` constructor 将图中的所有变量或指定的变量的保存和恢复操作加入到图中。Saver对象提供了运行这些操作的方法，指定checkpoint文件的写入和读取路径。

Saver restores all variables already defined in your model. If you're loading a model without knowing how to build its graph (for example, if you're writing a generic program to load models), then read the Overview of saving and restoring models section later in this document.

Saver恢复模型中已定义的所有变量。如果你装载一个模型却不知道怎样构建其图，比如写一个一般的装载模型的程序，那么阅读后面的Overview of saving and restoring models一节。

TensorFlow saves variables in binary checkpoint files that map variable names to tensor values. TensorFlow在二进制checkpoint文件中保存变量，其中有变量名到张量值的映射。

> Caution: TensorFlow model files are code. Be careful with untrusted code. See Using TensorFlow Securely for details.

> 注意：TensorFlow模型是代码，对于不受信任的代码要小心，详见Using TensorFlow Securely文档。

### Save variables 保存变量

Create a Saver with `tf.train.Saver()` to manage all variables in the model. For example, the following snippet demonstrates how to call the `tf.train.Saver.save` method to save variables to checkpoint files:

用`tf.train.Saver()`创建一个Saver来管理模型中的所有变量。比如，下面的代码片段展示了怎样调用`tf.train.Saver.save`方法来将变量保存到checkpoint文件中：

```
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
```

### Restore variables 恢复变量

The `tf.train.Saver` object not only saves variables to checkpoint files, it also restores variables. Note that when you restore variables you do not have to initialize them beforehand. For example, the following snippet demonstrates how to call the `tf.train.Saver.restore` method to restore variables from the checkpoint files:

`tf.train.Saver`对象不仅可以把变量保存到checkpoint文件中，还可以恢复变量。注意当恢复变量时，不需要先初始化变量。比如，下面的代码片段展示了怎样调用`tf.train.Saver.restore`方法来从checkpoint文件中恢复变量：

```
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

> Note: There is not a physical file called /tmp/model.ckpt. It is the prefix of filenames created for the checkpoint. Users only interact with the prefix instead of physical checkpoint files.

> 注意：没有一个实体文件名为 /tmp/model.ckpt，这是checkpoint文件的文件名前缀。用户只和checkpoint文件的前缀交互，不和实体的checkpoint文件交互。

### Choose variables to save and restore 选择保存和恢复的变量

If you do not pass any arguments to `tf.train.Saver()`, the saver handles all variables in the graph. Each variable is saved under the name that was passed when the variable was created.

如果不给`tf.train.Saver()`传递任何参数，那么saver会处理图中的所有变量。每个变量都用创建变量时的名字保存起来。

It is sometimes useful to explicitly specify names for variables in the checkpoint files. For example, you may have trained a model with a variable named "weights" whose value you want to restore into a variable named "params".

有时候需要显式的指定checkpoint文件中变量的名字。比如，你可能训练了一个模型，有个变量名称为"weights"，其值想要恢复到变量"params"中。

It is also sometimes useful to only save or restore a subset of the variables used by a model. For example, you may have trained a neural net with five layers, and you now want to train a new model with six layers that reuses the existing weights of the five trained layers. You can use the saver to restore the weights of just the first five layers.

有时候想要保存或恢复模型中的一部分变量。比如，训练了一个5层的神经网络，你现在想要训练一个6层的新模型，而想复用已经训练的5层的权重。你可以用saver来恢复前5层的权重。

You can easily specify the names and variables to save or load by passing to the `tf.train.Saver()` constructor either of the following:

可以很容易的指定要保存或恢复的变量和名称，只要向`tf.train.Saver()` constructor传递下面任一参数：

- A list of variables (which will be stored under their own names). 变量列表
- A Python dictionary in which keys are the names to use and the values are the variables to manage. 一个Python字典，其中key为使用的名称，value为要管理的变量。

Continuing from the save/restore examples shown earlier: 继续前面的保存/恢复例子：

```
tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")

  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

Notes: 注意：

- You can create as many Saver objects as you want if you need to save and restore different subsets of the model variables. The same variable can be listed in multiple saver objects; its value is only changed when the `Saver.restore()` method is run.

- 如果需要保存和恢复模型不同子集的变量，可以创建任意多个Saver对象。相同的变量可以出现在多个saver对象中，其值只在`Saver.restore()`方法运行时改变。

- If you only restore a subset of the model variables at the start of a session, you have to run an initialize op for the other variables. See `tf.variables_initializer` for more information.

- 如果在session开始处只恢复模型的一部分变量，那么需要对其他变量运行一个初始化操作。详见`tf.variables_initializer`文档。

- To inspect the variables in a checkpoint, you can use the inspect_checkpoint library, particularly the print_tensors_in_checkpoint_file function.

- 为检查checkpoint中的变量，可以使用inspect_checkpoint库，尤其是print_tensors_in_checkpoint_file函数。

- By default, Saver uses the value of the `tf.Variable.name` property for each variable. However, when you create a Saver object, you may optionally choose names for the variables in the checkpoint files.

- 默认情况下，Saver使用每个变量的`tf.Variable.name`属性。但是当创建Saver对象的时候，可以选择性的为checkpoint文件中的变量命名。

### Inspect variables in a checkpoint 检查checkpoint文件中的变量

We can quickly inspect variables in a checkpoint with the inspect_checkpoint library. Continuing from the save/restore examples shown earlier: 我们可以用inspect_checkpoint库来快速检查checkpoint中的变量。继续上面的保存/恢复的例子：

```
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]
```

## Save and restore models 保存和恢复模型

Use SavedModel to save and load your model—variables, the graph, and the graph's metadata. This is a language-neutral, recoverable, hermetic serialization format that enables higher-level systems and tools to produce, consume, and transform TensorFlow models. TensorFlow provides several ways to interact with `SavedModel`, including the `tf.saved_model` APIs, `tf.estimator.Estimator`, and a command-line interface.

使用SavedModel来保存和恢复模型变量、图和图的元数据。这是一个与语言无关的，可恢复的，密封的序列化格式，使高层系统和工具都可以创建，消耗以及变换TensorFlow模型。TensorFlow提供了几种方式来与`SavedModel`互动，包括`tf.saved_model` API，`tf.estimator.Estimator`，和一个命令行接口。

## Build and load a SavedModel 创建并载入一个SavedModel

### Simple save 简单保存

The easiest way to create a SavedModel is to use the `tf.saved_model.simple_save` function: 创建SavedModel最简单的方法是调用`tf.saved_model.simple_save`函数：

```
simple_save(session,
            export_dir,
            inputs={"x": x, "y": y},
            outputs={"z": z})
```

This configures the SavedModel so it can be loaded by TensorFlow serving and supports the Predict API. To access the classify, regress, or multi-inference APIs, use the manual SavedModel builder APIs or an `tf.estimator.Estimator`.

这对SavedModel进行了配置，这样就可以被TensorFlow serving载入，支持Predict API。要访问分类、回归或多推理 API，使用SavedModel builder API手册或`tf.estimator.Estimator`。

### Manually build a SavedModel 手工建造一个SavedModel

If your use case isn't covered by `tf.saved_model.simple_save`, use the manual builder APIs to create a SavedModel.

如果你的使用案例不是`tf.saved_model.simple_save`的情况，使用手工builder API来创建一个SavedModel。

The `tf.saved_model.builder.SavedModelBuilder` class provides functionality to save multiple MetaGraphDefs. A MetaGraph is a dataflow graph, plus its associated variables, assets, and signatures. A MetaGraphDef is the protocol buffer representation of a MetaGraph. A signature is the set of inputs to and outputs from a graph.

`tf.saved_model.builder.SavedModelBuilder`类具有保存多个MetaGraphDefs的功能。MetaGraph是一个数据流图，加上与之相关的变量、assets和signatures。MetaGraphDef是MetaGraph的protocol buffer表示。一个signature是一个图的输入输出集合。

If assets need to be saved and written or copied to disk, they can be provided when the first MetaGraphDef is added. If multiple MetaGraphDefs are associated with an asset of the same name, only the first version is retained.

如果asset需要保存并写入磁盘，那么在加入第一个MetaGraphDef时就可以提供。如果多个MetaGraphDefs与同一名称的asset相关联，只有第一个会得到保留。

Each MetaGraphDef added to the SavedModel must be annotated with user-specified tags. The tags provide a means to identify the specific MetaGraphDef to load and restore, along with the shared set of variables and assets. These tags typically annotate a MetaGraphDef with its functionality (for example, serving or training), and optionally with hardware-specific aspects (for example, GPU).

加入SavedModel的每个MetaGraphDef都必须有用户指定的标签注释。标签是一种识别特定的MetaGraphDef来载入和恢复的方法，以及共享的变量和asset集。这些标签一般对MetaGraphDef的功能进行注释，比如是serving或训练，也可能有硬件相关的方面，如GPU。

For example, the following code suggests a typical way to use SavedModelBuilder to build a SavedModel: 比如，下面的代码是用SavedModelBuilder来构建SavedModel的典型方法：

```
export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets,
                                       strip_default_attrs=True)
...
# Add a second MetaGraphDef for inference.
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([tag_constants.SERVING], strip_default_attrs=True)
...
builder.save()
```

Forward compatibility via strip_default_attrs=True 前向兼容

Following the guidance below gives you forward compatibility only if the set of Ops has not changed.

下面的指南给出了前向兼容性，如果操作集合没有改变，就可以使用下面的指南。

The SavedModelBuilder class allows users to control whether default-valued attributes must be stripped from the NodeDefs while adding a meta graph to the SavedModel bundle. Both `SavedModelBuilder.add_meta_graph_and_variables` and `SavedModelBuilder.add_meta_graph` methods accept a Boolean flag strip_default_attrs that controls this behavior.

SavedModelBuilder类使用户可以控制，当meta graph加入SavedModel包时，默认值的属性是否要从NodeDefs上剥离。`SavedModelBuilder.add_meta_graph_and_variables`和`SavedModelBuilder.add_meta_graph`方法都接收Boolean标志strip_default_attrs来控制这个行为。

If strip_default_attrs is False, the exported `tf.MetaGraphDef` will have the default valued attributes in all its `tf.NodeDef` instances. This can break forward compatibility with a sequence of events such as the following:

如果strip_default_attrs为False，那么导出的`tf.MetaGraphDef`会将默认值的属性存储在所有`tf.NodeDef`实例中。这会打乱前向兼容性，其事件如下所示：

- An existing Op (Foo) is updated to include a new attribute (T) with a default (bool) at version 101.
- 已经存在的操作Foo在版本更新（版本101）时包含了一个新的属性T，默认值bool。
- A model producer such as a "trainer binary" picks up this change (version 101) to the OpDef and re-exports an existing model that uses Op Foo.
- 一个创建模型者，如trainer binary，发现了版本101对OpDef的变化，重新将其导出给了已存在的使用Op Foo的模型。
- A model consumer (such as Tensorflow Serving) running an older binary (version 100) doesn't have attribute T for Op Foo, but tries to import this model. The model consumer doesn't recognize attribute T in a NodeDef that uses Op Foo and therefore fails to load the model.
- 模型消耗者，比如TensorFlow Serving，运行老版本的（版本100）没有操作Op Foo的新属性T，但试图导入这个模型。模型消耗者没有识别出NodeDef中的属性T，载入模型失败。
- By setting strip_default_attrs to True, the model producers can strip away any default valued attributes in the NodeDefs. This helps ensure that newly added attributes with defaults don't cause older model consumers to fail loading models regenerated with newer training binaries.
- 设置strip_default_attrs为True，模型创建者可以剥离任何NodeDef中的默认值属性。这确保了新加入的属性不会导致模型消耗者载入模型失败。

See compatibility guidance for more information. 详见兼容性指南。

### Loading a SavedModel in Python 在Python中载入SavedModel

The Python version of the SavedModel loader provides load and restore capability for a SavedModel. The load operation requires the following information: Python版的SavedModel载入器提供了保存和恢复SavedModel的能力。载入操作需要下面的信息：

- The session in which to restore the graph definition and variables. 恢复图定义和变量的session
- The tags used to identify the MetaGraphDef to load. 要载入的识别MetaGraphDef的标签
- The location (directory) of the SavedModel. SavedModel的位置（路径）

Upon a load, the subset of variables, assets, and signatures supplied as part of the specific MetaGraphDef will be restored into the supplied session. 在一次载入中，变量、assets和signatures子集将恢复到提供的session中：

```
export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...
```

### Load a SavedModel in C++ 在C++中载入SavedModel

The C++ version of the SavedModel loader provides an API to load a SavedModel from a path, while allowing SessionOptions and RunOptions. You have to specify the tags associated with the graph to be loaded. The loaded version of SavedModel is referred to as SavedModelBundle and contains the MetaGraphDef and the session within which it is loaded.

C++版的SavedModel载入器提供了API从一个路径载入SavedModel，还可以传入SessionOptions和RunOptions参数。你必须指定图相关的tags进行载入。载入版的SavedModel是SavedModelBundle类型的，包含了MetaGraphDef和载入的session。

```
const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &bundle);
```

### Load and serve a SavedModel in TensorFlow serving 在TensorFlow Serving中载入和serve一个SavedModel

You can easily load and serve a SavedModel with the TensorFlow Serving Model Server binary. See instructions on how to install the server, or build it if you wish.你可以用TensorFlow Serving Model Server binary轻松的载入并serve一个SavedModel。查看服务器安装指导。

Once you have the Model Server, run it with: 安装好了Model Server之后，运行下面命令：

`tensorflow_model_server --port=port-numbers --model_name=your-model-name --model_base_path=your_model_base_path`

Set the port and model_name flags to values of your choosing. The model_base_path flag expects to be to a base directory, with each version of your model residing in a numerically named subdirectory. If you only have a single version of your model, simply place it in a subdirectory like so: * Place the model in /tmp/model/0001 * Set model_base_path to /tmp/model

设定端口和模型名称标志位。模型基准路径标志位应当是一个基准路径，其中模型的每个版本都在数字命名的子文件夹中。如果你的模型只有一个版本，只需要将其放在这样的子文件夹下：* Place the model in /tmp/model/0001 * Set model_base_path to /tmp/model

Store different versions of your model in numerically named subdirectories of a common base directory. For example, suppose the base directory is /tmp/model. If you have only one version of your model, store it in /tmp/model/0001. If you have two versions of your model, store the second version in /tmp/model/0002, and so on. Set the --model-base_path flag to the base directory (/tmp/model, in this example). TensorFlow Model Server will serve the model in the highest numbered subdirectory of that base directory.

在一个共同基准文件夹中，将不同版本的模型存储在数字命名的子文件夹中。比如，假设基准文件夹是/tmp/model。如果只有一个版本模型，就放在/tmp/model/0001文件夹中。如果有两个模型版本，将第二个放在文件夹/tmp/model/0002中，等等。将--model-base_path标志位设为基准文件夹，比如，/tmp/model。TensorFlow Model Server将运行数字最大的那个版本的模型。

### Standard constants 标准常数

SavedModel offers the flexibility to build and load TensorFlow graphs for a variety of use-cases. For the most common use-cases, SavedModel's APIs provide a set of constants in Python and C++ that are easy to reuse and share across tools consistently.

SavedModel可以在很多情况下构建和载入TensorFlow计算图。在最常用的情况中，SavedModel的API提供了一个常数集，在Python和C++中易于重用和分享。

Standard MetaGraphDef tags

You may use sets of tags to uniquely identify a MetaGraphDef saved in a SavedModel. A subset of commonly used tags is specified in: Python or C++ link

你可以使用在SavedModel中唯一指定的MetaGraphDef中的tag集。一些常用的tag为：Python or C++ link

Standard SignatureDef constants

A SignatureDef is a protocol buffer that defines the signature of a computation supported by a graph. Commonly used input keys, output keys, and method names are defined in: Python or C++ link

SignatureDef是定义了一个图支持的signature计算的protocol buffer。常用的input keys, output keys和方法名定义在：Python or C++ link

## Using SavedModel with Estimators 在Estimator中使用SavedModel

After training an Estimator model, you may want to create a service from that model that takes requests and returns a result. You can run such a service locally on your machine or deploy it in the cloud.

训练了一个Estimator模型后，你可能想从模型创建一个服务，这个服务可以接收请求，并返回一个结果。你可以在本地运行这个服务，也可以将其部署在云端。

To prepare a trained Estimator for serving, you must export it in the standard SavedModel format. This section explains how to:

为准备一个训练过的Estimator进行Serving，你必须将其导出成为标准的SavedModel格式。这一节讲解怎样进行：

- Specify the output nodes and the corresponding APIs that can be served (Classify, Regress, or Predict).
- 指定可以served的输出节点以及相应的API（Classify, Regress, Predict）
- Export your model to the SavedModel format. 将模型导出到SavedModel格式
- Serve the model from a local server and request predictions. 在本地serve这个模型并要求预测。

### Prepare serving inputs 准备serving输入

During training, an `input_fn()` ingests data and prepares it for use by the model. At serving time, similarly, a `serving_input_receiver_fn()` accepts inference requests and prepares them for the model. This function has the following purposes:

在训练过程中，函数`input_fn()`吸收数据以备模型使用。在serving时候，类似的，函数`serving_input_receiver_fn()`接收推理请求，准备好后交给模型。这个函数的目的有以下两个：

- To add placeholders to the graph that the serving system will feed with inference requests.
- 向图中增加placeholder，serving系统将用推理请求进行feed。
- To add any additional ops needed to convert data from the input format into the feature Tensors expected by the model.
- 增加另外的ops，将输入数据格式转换成特征张量，输入模型。

The function returns a `tf.estimator.export.ServingInputReceiver` object, which packages the placeholders and the resulting feature Tensors together. 函数返回`tf.estimator.export.ServingInputReceiver`对象，这个对象将placeholder和得到的特征张量打包在一起。

A typical pattern is that inference requests arrive in the form of serialized `tf.Examples`, so the `serving_input_receiver_fn()` creates a single string placeholder to receive them. The `serving_input_receiver_fn()` is then also responsible for parsing the `tf.Examples` by adding a `tf.parse_example` op to the graph.

典型的模式是，推理请求的输入数据为序列化的`tf.Examples`，所以`serving_input_receiver_fn()`函数创建一个字符串的placeholder来接收。那么函数`serving_input_receiver_fn()`要负责解析`tf.Examples`，于是向图中加入了一个`tf.parse_example`操作

When writing such a `serving_input_receiver_fn()`, you must pass a parsing specification to `tf.parse_example` to tell the parser what feature names to expect and how to map them to Tensors. A parsing specification takes the form of a dict from feature names to `tf.FixedLenFeature`, `tf.VarLenFeature`, and `tf.SparseFeature`. Note this parsing specification should not include any label or weight columns, since those will not be available at serving time—in contrast to a parsing specification used in the `input_fn()` at training time.

当写`serving_input_receiver_fn()`这样的函数时，必须向`tf.parse_example`传递一个解析规则，告诉解析器想要什么特征名称，怎样将其映射到张量。解析规则是字典的形式，从特征名称到`tf.FixedLenFeature`, `tf.VarLenFeature`, 和 `tf.SparseFeature`。注意这个解析规则不能包括任何标签或权值列，因为这在serving时候是不可用的，形成对比的是，训练时`input_fn()`函数用的解析规则。

In combination, then: 合起来，就是：

```
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
```

The `tf.estimator.export.build_parsing_serving_input_receiver_fn` utility function provides that input receiver for the common case.

`tf.estimator.export.build_parsing_serving_input_receiver_fn`工具函数提供了输入接收器的一般形式。

> Note: when training a model to be served using the Predict API with a local server, the parsing step is not needed because the model will receive raw feature data.

> 注意：当训练模型时，如果这个模型要在本地服务器上用Predict API进行服务，解析步骤是不需要的，因为模型会接收原始特征数据。

Even if you require no parsing or other input processing—that is, if the serving system will feed feature Tensors directly—you must still provide a `serving_input_receiver_fn()` that creates placeholders for the feature Tensors and passes them through. The `tf.estimator.export.build_raw_serving_input_receiver_fn` utility provides for this.

即使你不需要解析，或有其他的输入处理过程，也就是说，serving系统会直接接收到特征张量，你还是需要提供一个函数`serving_input_receiver_fn()`，函数要创建特征张量的placeholders并传递给它们。工具函数`tf.estimator.export.build_raw_serving_input_receiver_fn`提供了这种功能。

If these utilities do not meet your needs, you are free to write your own `serving_input_receiver_fn()`. One case where this may be needed is if your training `input_fn()` incorporates some preprocessing logic that must be recapitulated at serving time. To reduce the risk of training-serving skew, we recommend encapsulating such processing in a function which is then called from both `input_fn()` and `serving_input_receiver_fn()`.

如果这些工具仍然不能满足需要，可以写自己的`serving_input_receiver_fn()`。一种需要这样的情况是，如果你的训练`input_fn`包括了一些预处理逻辑，而且在serving时还需要进行概括。为减少训练-serving时的情况不同的风险，我们推荐将这种处理封装在一个函数中，然后在`input_fn()` and `serving_input_receiver_fn()`都进行调用。

Note that the `serving_input_receiver_fn()` also determines the input portion of the signature. That is, when writing a `serving_input_receiver_fn()`, you must tell the parser what signatures to expect and how to map them to your model's expected inputs. By contrast, the output portion of the signature is determined by the model.

注意`serving_input_receiver_fn()`函数还决定了signature的输入部分。也就是说，当写一个`serving_input_receiver_fn()`函数时，你必须告诉解析器要期待什么样的signature，以及怎样将其映射到模型期待的输入中。输出的signature部分是由模型决定的。

### Specify the outputs of a custom model 指定定制模型的输出

When writing a custom `model_fn`, you must populate the export_outputs element of the `tf.estimator.EstimatorSpec` return value. This is a dict of {name: output} describing the output signatures to be exported and used during serving.

当写一个定制的`model_fn`函数时，你必须填充`tf.estimator.EstimatorSpec`返回的export_outputs元素。这是一个{name: output}的字典，描述的是要导出的输出signature，在serving的时候使用。

In the usual case of making a single prediction, this dict contains one element, and the name is immaterial. In a multi-headed model, each head is represented by an entry in this dict. In this case the name is a string of your choice that can be used to request a specific head at serving time.

在普通情况时，要进行单个预测，这个字典包括一个元素，其名称是不重要的。在多头模型中，每个head都由字典入口代表。在这种情况下，其名称是一个字符串，是在serving时用作请求特定的head的选择。

Each output value must be an `ExportOutput` object such as `tf.estimator.export.ClassificationOutput`, `tf.estimator.export.RegressionOutput`, or `tf.estimator.export.PredictOutput`.

每个输出值必须是一个`ExportOutput`对象，如`tf.estimator.export.ClassificationOutput`, `tf.estimator.export.RegressionOutput`, 或 `tf.estimator.export.PredictOutput`。

These output types map straightforwardly to the TensorFlow Serving APIs, and so determine which request types will be honored.

这些输出的类型直接映射到TensorFlow Serving API，也就确定了哪个请求类型将要用到。

> Note: In the multi-headed case, a SignatureDef will be generated for each element of the export_outputs dict returned from the `model_fn`, named using the same keys. These SignatureDefs differ only in their outputs, as provided by the corresponding ExportOutput entry. The inputs are always those provided by the `serving_input_receiver_fn`. An inference request may specify the head by name. One head must be named using signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY indicating which SignatureDef will be served when an inference request does not specify one.

>注意：在多头情况下，每个元素都会从`model_fn`产生一个SignatureDef，内容是export_outputs字典，名称使用同样的key。这些SignatureDef只在输出处不一样，这是由相应的ExportOutput入口提供的。输入永远由`serving_input_receiver_fn`提供。一个推理请求可能指定head的名称。当推理请求没有指定的时候，一个head必须使用signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY命名，以指出哪个SignatureDef将会被serve。

### Perform the export 执行导出

To export your trained Estimator, call `tf.estimator.Estimator.export_savedmodel` with the export base path and the `serving_input_receiver_fn`. 要导出训练好的Estimator，调用`tf.estimator.Estimator.export_savedmodel`，准备好导出基础路径和`serving_input_receiver_fn`。

`estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn, strip_default_attrs=True)`

This method builds a new graph by first calling the `serving_input_receiver_fn()` to obtain feature Tensors, and then calling this Estimator's `model_fn()` to generate the model graph based on those features. It starts a fresh Session, and, by default, restores the most recent checkpoint into it. (A different checkpoint may be passed, if needed.) Finally it creates a time-stamped export directory below the given export_dir_base (i.e., export_dir_base/< timestamp>), and writes a SavedModel into it containing a single MetaGraphDef saved from this Session.

这个方法构建了一个新图，首先调用`serving_input_receiver_fn()`得到特征张量，然后调用这个Estimator的`model_fn()`生成基于这些特征的模型图。它开始了一个新的Session，默认把最近的checkpoint恢复进去。如果需要的话，另一个不同的checkpoint可能被传递。最后它在给定的export_dir_base中创建了带有时间戳的导出目录，即export_dir_base/< timestamp>，然后将SavedModel写入，包括了这个Session里的单个MetaGraphDef。

> Note: It is your responsibility to garbage-collect old exports. Otherwise, successive exports will accumulate under export_dir_base.

> 注意：你必须收集旧的导出的垃圾。不然，后续的导出将在export_dir_base中堆积。

### Serve the exported model locally 本地serve导出模型

For local deployment, you can serve your model using TensorFlow Serving, an open-source project that loads a SavedModel and exposes it as a gRPC service.

对于本地部署，你可以使用TensorFlow Serving serve你的模型，这是个开源项目，载入一个SavedModel，将其展现为一个gRPC服务。

First, install TensorFlow Serving. 首先，安装TensorFlow Serving。

Then build and run the local model server, substituting export_dir_base with the path to the SavedModel you exported above: 然后build并运行本地模型服务器，用导出的SavedModel的路径代替export_dir_base：

```
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=$export_dir_base
```

Now you have a server listening for inference requests via gRPC on port 9000! 现在有了一个服务器通过gRPC服务在端口9000监听推理请求。

### Request predictions from a local server 从本地服务器上请求预测

The server responds to gRPC requests according to the PredictionService gRPC API service definition. (The nested protocol buffers are defined in various neighboring files).

根据PredictionService gRPC API服务定义，服务器响应gRPC请求。嵌套的protocol buffers定义在多个相邻的文件中。

From the API service definition, the gRPC framework generates client libraries in various languages providing remote access to the API. In a project using the Bazel build tool, these libraries are built automatically and provided via dependencies like these (using Python for example):

从API服务定义中，gRPC框架生成不同语言的客户端库，为API提供远程访问。在使用Bazel build工具的工程中，这些库自动生成，通过下面这些依赖性提供（使用Python作为例子）：

```
  deps = [
    "//tensorflow_serving/apis:classification_proto_py_pb2",
    "//tensorflow_serving/apis:regression_proto_py_pb2",
    "//tensorflow_serving/apis:predict_proto_py_pb2",
    "//tensorflow_serving/apis:prediction_service_proto_py_pb2"
  ]
```

Python client code can then import the libraries thus: Python客户端代码可以导入这些库：

```
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
```

> Note: `prediction_service_pb2` defines the service as a whole and so is always required. However a typical client will need only one of classification_pb2, regression_pb2, and predict_pb2, depending on the type of requests being made.

> 注意：`prediction_service_pb2`定义了服务整体，一直也都是这样要求的。但是，根据请求的类型，一个典型的客户端只需要classification_pb2, regression_pb2, and predict_pb2中的一个。

Sending a gRPC request is then accomplished by assembling a protocol buffer containing the request data and passing it to the service stub. Note how the request protocol buffer is created empty and then populated via the generated protocol buffer API.

组合一个包括请求数据的protocol buffer，将其传递到服务stub上，就完成了发送一个gRPC请求。注意这些请求的protocol buffer是怎样创建为空，然后又通过产生的protocol buffer API填充的。

```
from grpc.beta import implementations

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))

result = stub.Classify(request, 10.0)  # 10 secs timeout
```

The returned result in this example is a `ClassificationResponse` protocol buffer. 这个例子中返回的结果是一个`ClassificationResponse` protocol buffer。

This is a skeletal example; please see the Tensorflow Serving documentation and examples for more details.这是一个骨架性的例子；详见Tensorflow Serving文档和例子。

> Note: `ClassificationRequest` and `RegressionRequest` contain a `tensorflow.serving.Input` protocol buffer, which in turn contains a list of `tensorflow.Example` protocol buffers. `PredictRequest`, by contrast, contains a mapping from feature names to values encoded via `TensorProto`. Correspondingly: When using the Classify and Regress APIs, TensorFlow Serving feeds serialized `tf.Examples` to the graph, so your `serving_input_receiver_fn()` should include a `tf.parse_example()` Op. When using the generic Predict API, however, TensorFlow Serving feeds raw feature data to the graph, so a pass through `serving_input_receiver_fn()` should be used.

> 注意：ClassificationRequest和RegressionRequest包含一个`tensorflow.serving.Input` protocol buffer, 其中又包含了一个`tensorflow.Example` protocol buffers列表。而`PredictRequest`包含了从特征名称到`TensorProto`编码的值的映射。相应的，当使用Classify和Regress API时，TensorFlow Serving将`tf.Examples` feed给图，所以`serving_input_receiver_fn()`应当包括一个`tf.parse_example()`操作。但当使用一般性的Predict API时，TensorFlow Serving将原始特征数据feed给图，所以需要经过`serving_input_receiver_fn()`传递一下。

## CLI to inspect and execute SavedModel 通过命令行来检查并执行SavedModel

You can use the SavedModel Command Line Interface (CLI) to inspect and execute a SavedModel. For example, you can use the CLI to inspect the model's SignatureDefs. The CLI enables you to quickly confirm that the input Tensor dtype and shape match the model. Moreover, if you want to test your model, you can use the CLI to do a sanity check by passing in sample inputs in various formats (for example, Python expressions) and then fetching the output.

可以使用SavedModel命令行CLI来检查并执行一个SavedModel。比如，你可以使用CLI来检查模型的SignatureDefs。命令行可以很快的确认输入张量的数据类型和形状是否符合模型。如果你想测试模型，可以使用命令行来进行一个sanity check，将样本输入以多种格式（如，Python语句）传入，取回输出。

### Install the SavedModel CLI 安装（太简单，不翻译了）

Broadly speaking, you can install TensorFlow in either of the following two ways: 可以使用下面两种方法安装TensorFlow：

- By installing a pre-built TensorFlow binary.
- By building TensorFlow from source code.

If you installed TensorFlow through a pre-built TensorFlow binary, then the SavedModel CLI is already installed on your system at pathname bin\saved_model_cli. 

If you built TensorFlow from source code, you must run the following additional command to build saved_model_cli:

`$ bazel build tensorflow/python/tools:saved_model_cli`

### Overview of commands 命令概览

The SavedModel CLI supports the following two commands on a MetaGraphDef in a SavedModel: 支持两个命令

- show, which shows a computation on a MetaGraphDef in a SavedModel.
- run, which runs a computation on a MetaGraphDef.

### show command

A SavedModel contains one or more MetaGraphDefs, identified by their tag-sets. To serve a model, you might wonder what kind of SignatureDefs are in each model, and what are their inputs and outputs. The show command let you examine the contents of the SavedModel in hierarchical order. Here's the syntax:

一个SavedModel包含一个或多个MetaGraphDefs，由其tag-sets识别。要serve一个模型，你可能会想，在每个模型中都是什么类型的SignatureDefs，它们的输入和输出都是什么。show命令可以让你层次式的检查SavedModel中的内容。这里是语法：

```
usage: saved_model_cli show [-h] --dir DIR [--all]
[--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
```

For example, the following command shows all available MetaGraphDef tag-sets in the SavedModel: 例如，下面的命令显示出SavedModel中所有可用的MetaGraphDef tag-sets：

```
$ saved_model_cli show --dir /tmp/saved_model_dir
The given SavedModel contains the following tag-sets:
serve
serve, gpu
```

The following command shows all available SignatureDef keys in a MetaGraphDef: 下面的命令显示出MetaGraphDef中所有可用的SignatureDef keys：

```
$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
following keys:
SignatureDef key: "classify_x2_to_y3"
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"
```

If a MetaGraphDef has multiple tags in the tag-set, you must specify all tags, each tag separated by a comma. For example: 如果一个MetaGraphDef有在tag-set里有多个tags，你必须指定所有tags，每个tag由逗号分开。比如：

`$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu`

To show all inputs and outputs TensorInfo for a specific SignatureDef, pass in the SignatureDef key to signature_def option. This is very useful when you want to know the tensor key value, dtype and shape of the input tensors for executing the computation graph later. For example:

为显示一个指定的SignatureDef的所有输入和输出TensorInfo，将SignatureDef key传入signature_def选项。当你想知道张量键值、数据类型和输入张量的形状的时候很有用，比如

```
$ saved_model_cli show --dir \
/tmp/saved_model_dir --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['x'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: x:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['y'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: y:0
Method name is: tensorflow/serving/predict
```

To show all available information in the SavedModel, use the --all option. For example:使用选项来显示出SavedModel中的所有可用信息，比如：

```
$ saved_model_cli show --dir /tmp/saved_model_dir --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/classify

...

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/predict
```

### run command

Invoke the run command to run a graph computation, passing inputs and then displaying (and optionally saving) the outputs. Here's the syntax: 调用run命令来运行一个图的运算，传入输入然后显示输出。下面是语法：

```
usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
                           SIGNATURE_DEF_KEY [--inputs INPUTS]
                           [--input_exprs INPUT_EXPRS]
                           [--input_examples INPUT_EXAMPLES] [--outdir OUTDIR]
                           [--overwrite] [--tf_debug]
```

The run command provides the following three ways to pass inputs to the model: run命令有以下三种方式向模型传递输入：

- --inputs option enables you to pass numpy ndarray in files.
- --input_exprs option enables you to pass Python expressions.
- --input_examples option enables you to pass tf.train.Example.

#### --inputs

To pass input data in files, specify the --inputs option, which takes the following general format: 要用文件传递输入数据，指定选项--inputs，一般格式如下：

`--inputs <INPUTS>`

where INPUTS is either of the following formats: 这里INPUTS是如下格式之一

- < input_key>=< filename>
- < input_key>=< filename>[< variable_name>]

You may pass multiple INPUTS. If you do pass multiple inputs, use a semicolon to separate each of the INPUTS. 可以传入多个INPUTS。如果你确实传入了多个输入，用分号分隔每个INPUTS。

saved_model_cli uses numpy.load to load the filename. The filename may be in any of the following formats: .npy or .npz or pickle format

saved_model_cli使用numpy.load来载入文件名。文件名可以用以下三种格式：.npy .npz or pickle

A .npy file always contains a numpy ndarray. Therefore, when loading from a .npy file, the content will be directly assigned to the specified input tensor. If you specify a variable_name with that .npy file, the variable_name will be ignored and a warning will be issued.

一个.npy文件永远包含一个numpy ndarray。所以，当从一个npy文件中载入，其内容将会直接赋值给指定的输入张量。如果你用这个npy文件指定了一个variable_name，这个variable_name将会被忽略，产生一个警告。

When loading from a .npz (zip) file, you may optionally specify a variable_name to identify the variable within the zip file to load for the input tensor key. If you don't specify a variable_name, the SavedModel CLI will check that only one file is included in the zip file and load it for the specified input tensor key.

当从一个npz文件从载入，你可以选择性的指定一个variable_name来识别zip文件中的变量，为输入张量key载入。如果不指定variable_name，SavedModel CLI会检查只有一个文件在zip文件中，然后将其载入给指定的输入张量key。

When loading from a pickle file, if no variable_name is specified in the square brackets, whatever that is inside the pickle file will be passed to the specified input tensor key. Otherwise, the SavedModel CLI will assume a dictionary is stored in the pickle file and the value corresponding to the variable_name will be used.

当从一个pickle文件载入时，如果没有variable_name指定，不管pickle文件中有什么，都会被传到指定的输入张量key中。不然，SavedModel CLI将会假设pickle文件中存储了一个字典，与variable_name对应的值会被使用。

#### --input_exprs

To pass inputs through Python expressions, specify the --input_exprs option. This can be useful for when you don't have data files lying around, but still want to sanity check the model with some simple inputs that match the dtype and shape of the model's SignatureDefs. For example:

要通过Python表达式传入输入，指定选项--input_exprs。当没有现成的数据文件，但仍然想对模型进行sanity check时，使用一些简单的符合模型SignatureDefs的数据类型和形状的输入。比如

`<input_key>=[[1],[2],[3]]`

In addition to Python expressions, you may also pass numpy functions. For example: 除了Python表达式，还可以传入numpy函数，比如：

`<input_key>=np.ones((32,32,3))`

(Note that the numpy module is already available to you as np.)注意numpy模块已经是可用的。

#### --input_examples

To pass `tf.train.Example` as inputs, specify the --input_examples option. For each input key, it takes a list of dictionary, where each dictionary is an instance of `tf.train.Example`. The dictionary keys are the features and the values are the value lists for each feature. For example:

要传入`tf.train.Example`对象作为输入，指定选项--input_examples。对于每个输入key，都需要一个字典列表，这里每个字典都是`tf.train.Example`的一个实例。字典key是特征，value是每个特征的value列表。比如：

`<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`

#### Save output 保存输出

By default, the SavedModel CLI writes output to stdout. If a directory is passed to --outdir option, the outputs will be saved as npy files named after output tensor keys under the given directory.

默认情况下，SavedModel CLI输出到标准输出。如果--outdir选项传入一个字典，这个输出将会保存为npy文件，命名为输出张量key，在给定的目录下。

Use --overwrite to overwrite existing output files.

#### TensorFlow debugger (tfdbg) integration

If --tf_debug option is set, the SavedModel CLI will use the TensorFlow Debugger (tfdbg) to watch the intermediate Tensors and runtime graphs or subgraphs while running the SavedModel.

#### Full examples of run

Given: 给定

1. Your model simply adds x1 and x2 to get output y.
2. All tensors in the model have shape (-1, 1).
3. You have two npy files:
- /tmp/my_data1.npy, which contains a numpy ndarray [[1], [2], [3]].
- /tmp/my_data2.npy, which contains another numpy ndarray [[0.5], [0.5], [0.5]].

To run these two npy files through the model to get output y, issue the following command:

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npy;x2=/tmp/my_data2.npy \
--outdir /tmp/out
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

Let's change the preceding example slightly. This time, instead of two .npy files, you now have an .npz file and a pickle file. Furthermore, you want to overwrite any existing output file. Here's the command:我们简单的改变一下前面这个例子。这次，不是2个npy文件，而是一个npz文件和一个pickle文件。而且还需要覆盖现有的输出文件，命令如下：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y \
--inputs x1=/tmp/my_data1.npz[x];x2=/tmp/my_data2.pkl --outdir /tmp/out \
--overwrite
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

You may specify python expression instead of an input file. For example, the following command replaces input x2 with a Python expression: 可以指定python表达式，而不是输入文件。比如，下面的命令将输入x2替换为一个Python表达式：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npz[x] \
--input_exprs 'x2=np.ones((3,1))'
Result for output key y:
[[ 2]
 [ 3]
 [ 4]]
```

To run the model with the TensorFlow Debugger on, issue the following command: 要在开启TensorFlow Debugger的状态下运行模型，运行以下命令：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def serving_default --inputs x=/tmp/data.npz[x] --tf_debug
```

## Structure of a SavedModel directory SavedModel目录结构

When you save a model in SavedModel format, TensorFlow creates a SavedModel directory consisting of the following subdirectories and files: 当你用SavedModel格式保存了一个模型，TensorFlow创建了一个SavedModel目录，包括下面的子目录和文件：

```
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb|saved_model.pbtxt
```

where:这里

- assets is a subfolder containing auxiliary (external) files, such as vocabularies. Assets are copied to the SavedModel location and can be read when loading a specific MetaGraphDef.
- assets是一个子目录，包括辅助（外部）文件，比如词汇表。Assets拷入SavedModel目录位置，当载入MetaGraphDef的时候可以被读取。
- assets.extra is a subfolder where higher-level libraries and users can add their own assets that co-exist with the model, but are not loaded by the graph. This subfolder is not managed by the SavedModel libraries.
- assets.extra是一个子目录，高层库和用户可以将他们自己与模型共存的assets加入，但没有被图载入。这个子目录不被SavedModel库管理。
- variables is a subfolder that includes output from `tf.train.Saver`.
- variable是一个子目录，包括`tf.train.Saver`的输出。
- saved_model.pb or saved_model.pbtxt is the SavedModel protocol buffer. It includes the graph definitions as MetaGraphDef protocol buffers.
- saved_model.pb或saved_model.pbtxt是SavedModel protocol buffer。包括图用MetaGraphDef protocol buffers的定义。

A single SavedModel can represent multiple graphs. In this case, all the graphs in the SavedModel share a single set of checkpoints (variables) and assets. For example, the following diagram shows one SavedModel containing three MetaGraphDefs, all three of which share the same set of checkpoints and assets:

单个SavedModel可以代表多个图。在这个情况中，所有SavedModel中的图分享一个checkpoints集合和assets集合。比如，下图展示了一个SavedModel包括3个MetaGraphDefs，所有3个都分享同样的checkpoints and assets集合。

![Image](https://www.tensorflow.org/images/SavedModel.svg)

Each graph is associated with a specific set of tags, which enables identification during a load or restore operation.

每个图都与特定的tag集合相关联，这在载入或恢复操作的时候用来识别。