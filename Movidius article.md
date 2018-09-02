# A Brief Guide to Intel Movidius Neural Compute Stick with Raspberry Pi 3
# Intel Movidius神经网络计算棒与Raspberry Pi 3使用指南

from [Deep Learning Turkey](https://medium.com/deep-learning-turkey/a-brief-guide-to-intel-movidius-neural-compute-stick-with-raspberry-pi-3-f60bf7683d40) in 2018-03-04

注：其实Movidius官网上的文档更详细

Today, low-power consumption is indispensable for autonomous/unmanned vehicles and IoT (Internet of Things) devices. In order to develop deep learning inference application at the edge, we can use Intel’s both energy efficient and low cost Movidius USB stick!

今天，低能耗对于自动/无人驾驶和物联网设备是不可或缺的。为在edge端开发深度学习推理应用，我们可以使用Intel的低能耗Movidius USB计算棒，同时它也很便宜。

Movidius Neural Compute Stick (NCS) is produced by Intel and it can be run without any need of Intenet. This software development kit enables rapid prototyping, validation, and deployment of deep neural networks. Profiling, tuning, and compiling a DNN on a development computer with the tools are provided in the Intel Movidius Neural Compute SDK. The Movidius NCS’ compute capability comes from Myriad 2 VPU (Vision Processing Unit).

Movidius神经网络计算棒(NCS)产于Intel，运行时无需Internet接入。这种软件开发套件可以快速开发原型、验证并部署深度神经网络。Intel Movidius神经网络计算SDK使开发平台能为深度神经网络提供成型、精调并编译的功能。Movidius NCS的计算能力来自Myriad 2 VPU，即视觉处理单元。

Running Deep Learning models efficiently on low capacity graph processors is very painful. Movidius allows us to optimize the operation of large models such as GoogLeNet (thanks to Yann LeCun) with multi-use support. It is an easy-to-use kit that allows you to design and implement applications such as classification and object recognition as physical products.

在低能耗图像处理器上想有效运行深度学习模型是非常痛苦的。Movidius使我们能优化像GoogLeNet这样的规模较大模型。套件容易使用，这样就可以设计实现分类或目标识别这样的应用成为实体产品。

**We can simply think of Movidius NCS as a GPU running on USB (Graphics Processing Unit)**. However, training of the model is not performed on this unit, the trained model works optimally on the unit and is intended to be used in physical environments for testing purposes.

**我们可以将Movidius NCS简单的想象成USB供电的GPU**。但模型的训练不是在这个部件上面进行的，训练好的模型最适合在这上面运行，可以在物理环境中用作测试。

- We can use with Ubuntu 16.04 or Raspberry Pi 3 Raspbian Stretch.
- It supports two DNN frameworks (TensorFlow and Caffe).
- Movidius Myriad 2 VPU works with Caffe based convolutional neural networks.
- We can run complex deep learning models like SqueezeNet, GoogLeNet and AlexNet on your computer with low processing capability.

- 我们可以用Ubuntu 16.04或Raspberry Pi 3 Raspbian Stretch进行开发。
- 支持两种深度神经网络框架，即Tensorflow Lite和Caffe。
- Movidius Myriad 2 VPU运行基于Caffe的卷积神经网络上。
- 可以在低能耗电脑上运行复杂的深度学习模型如SqueezeNet, GoogLeNet以及AlexNet。

![Image](https://movidius.github.io/ncsdk/images/ncs_workflow.jpg)

Movidius Neural Compute Stick Workflow 工作流程

## APPLICATION

It is so simple to run a image classification demo. Now we can use [NC App Zoo](https://github.com/movidius/ncappzoo) repo for classifying an image. We need to take graph file to activate application of Movidius NCS. It has compiled GoogLeNet model for ready to run. This application needs some files.

运行一个图像分类的demo非常简单。现在我们用NC App Zoo repo对一幅图像进行分类，我们需要输入图文件（指DNN流程图文件）来激活Movidius NCS应用，已经编译好了GoogLeNet模型可以运行，这个应用需要一些文件。

`make` command is used for creating the files that Movidius needs as graph file. Graph file is a demo of image-classifier.

`make`命令可以用来生成Movidius需要的图文件。这个图文件是图像分类器的demo。

### Follow the steps below for a quick applicaiton: 按下面的步骤来快速得到应用

- For using property of NCSDK API *add (import)* mvnc library 为使用NCSDK API我们*add (import)* mvnc library

```
import mvnc.mvncapi as mvnc
```

- You can access to Movidius NCS using API like any other USB Devices. Also you can use parallel Movidius devices at ones if you need more capacity to compute your model. For now, one kit is enough for this application. Select and open process:

- 就像访问其他USB设备一样，可以用API访问Movidius NCS。如果需要更多模型计算能力，还可以并行使用Movidius设备。目前，一套设备对这种应用已经足够了。选择并打开process

```
# Look for enumerated Intel Movidius NCS device(s); quit program if none found.
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()
# Get a handle to the first enumerated device and open it
 device = mvnc.Device(devices[0])
 device.OpenDevice()
```

- Pretrained GoogLeNet model for using compiled graph file:

- 使用编译过的graph文件中的预训练GoogLeNet模型

```
# Read the graph file into a buffer
 with open(GRAPH_PATH, mode='rb') as f:
   blob = f.read()
# Load the graph buffer into the NCS
graph = device.AllocateGraph(blob)
```

- Need to do some pre-processing before loading the image in Movidius NCS: 在将图像装载进Movidius NCS之前，需要做一些预处理

```
# Read & resize image [Image size is defined during training]
img = print_img = skimage.io.imread( IMAGES_PATH )
img = skimage.transform.resize( img, IMAGE_DIM, preserve_range=True)
# Convert RGB to BGR [skimage reads image in RGB, but Caffe uses BGR]
img = img[:, :, ::-1]
# Mean subtraction & scaling [A common technique used to center the data]
img = img.astype( numpy.float32 )
img = ( img — IMAGE_MEAN ) * IMAGE_STDDEV
```

- Use `LoadTensor()` for loading the image to Movidius NCS: 用`LoadTensor()`将图像装载进Movidius NCS

```
# Load the image as a half-precision floating point array 
graph.LoadTensor( img.astype( numpy.float16 ), 'user object' )
```

- Give the input image to pretrained model then getting output using by `GetResult()`: 将图像送入预训练模型，然后用`GetResult()`得到输出

```
# Get the results from NCS
 output, userobj = graph.GetResult()
```

- Printing prediction of model output and corresponding labels. Also displaying the input image at the same time.

- 将模型预测结果和对应的标签打印出来，同时显示输入图像

```
# Print the results
print('\n — — — — predictions — — — — ')
labels = numpy.loadtxt(LABELS_FILE_PATH, str, delimiter = '\t')
order = output.argsort()[::-1][:6]
for i in range( 0, 5 ):
print ('prediction ' + str(i) + ' is' + labels[order[i]])
# Display the image on which inference was performed
skimage.io.imshow(IMAGES_PATH)
skimage.io.show()
```

![Image](https://cdn-images-1.medium.com/max/800/1*kZ-ubckG1VuzEv-Vg6m8Og.png)

Test Result of Image Classification Problem (Fail)

![Image](https://cdn-images-1.medium.com/max/800/1*8Lb1WAG109ih8iqUC1e9BQ.png)

Test Result of Image Classification Problem (Successful)

- At the last step, *clear* and *shutdown* to Movidius NCS device for use it again: 最后一步，清除并关闭Movidius NCS以备后用

```
graph.DeallocateGraph()
device.CloseDevice()
```

References:
1. https://developer.movidius.com/
2. https://www.movidius.com/news/movidius-and-dji-bring-vision-based-autonomy-to-dji-phantom-4
3. https://github.com/movidius/ncappzoo
4. https://software.intel.com/en-us/articles/build-an-image-classifier-in-5-steps-on-the-intel-movidius-neural-compute-stick
5. https://towardsdatascience.com/getting-started-with-intel-movidius-d8ba13e7d3ae