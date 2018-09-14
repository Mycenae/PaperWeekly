# Keras: 基于Python的深度学习库

尽管还有很多论文没看，但Keras中的名词已经基本都熟悉了，本文试图对Keras的各种概念、函数、参数进行简单的分类、总结，以梳理哪里还需要进一步熟悉加强学习。

## Keras程序流程：
1. 构建模型：输入数据的尺寸，经过各层的定义         model =sequential() / model.add(...)
2. 外围配置：损失函数，优化器，模型性能度量         model.compile(optimizer, loss, metrics)
3. 网络训练：指定训练数据，batch size，训练epoch数 model.fit(data, labels, epoch, batch_size) or model.train_on_batch(...)
4. 模型评估或数据预测：利用测试集评估模型性能 model.evaluate(data, batch_size) 或利用训练好的模型对得到的数据进行预测 model.predict(...)

## 基本概念和注意事项
1. 符号计算：搭水管-送水理论
2. 张量及轴
3. 数据格式：tensorflow [sample_num, height, width, channels], channels_last \
            caffe/theano [sample_sum, channels, height, width], channels_first
4. functional model: sequential/graph -> sequential/functional model
5. batch, epoch etc.
6. HDF5: Hierachical Data Format v5, python lib h5py, 保存模型时用到hdf5文件类型，保存内容包括(i)模型结构(ii)权重(iii)训练配置（损失函数，优化器等）(iv)优化器的状态，以便从上次训练中断的地方开始
7. 注意tensorflow和theano中参数顺序的不同，如卷积核，BN层参数；注意shuffle和validation split的顺序；

## Keras.Model
### common methods and attributes:
1. model.layers, model.inputs, model.outputs, model.summary(), model.get_config(), model.get_weights(), model.set_weights(), model.to_json, model.to_yaml, model.save_weights(), model.load_weights()
2. 创建子类，定制模型，实现call()方法

### Sequential Model
**compile, fit, evaluate, predict**, \
train_on_batch, test_on_batch, predict_on_batch, fit_generator, evaluate_generator, predict_generator, get_layer

### functional model api
**compile, fit, evaluate, predict**, \
train_on_batch, test_on_batch, predict_on_batch, fit_generator, evaluate_generator, predict_generator, get_layer

## Layers

### common methods and attributes:
1. layer.get_weights(), layer.set_weights(), layer.get_config()
2. layer.input, layer.output, layer.input_shape, layer.output_shape
3. layer.get_input_at(), layer.get_output_at(), layer.get_input_shape_at(), layer.get_output_shape_at()

### core network layers
**Dense(), Activation(), Dropout(), Flatten(), Input(),** \
Reshape(), Permute(), RepeatVector(), Lambda(), ActivityRegularization(), Masking(), SpatialDropout1D(), SpatialDropout2D(), SpatialDropout3D()

### convolutional layers
**Conv1D(), Conv2D(), SeparableConv1D(), SeparableConv2D()**, \
Conv2DTranspose(), Conv3D(), Cropping1D(), Cropping2D(), Cropping3D(), UpSampling1D(), UpSampling2D(), UpSampling3D(), ZeroPadding1D(), ZeroPadding2D(), ZeroPadding3D()

### pooling layers
**MaxPoolingxD(), AveragePoolingxD()**, GlobalMaxPoolingxD(), GlobalAveragePoolingxD(), x=1,2,3

### locally-connected layers
LocallyConnected1D(), LocallyConnected2D()

### Recurrent layers
RNN, GRU, LSTM, etc

### Embeddings

### Merge layers
**Add(), Substract(), Multiply(), Average(), Maximum(), Concatenate(), Dot()**

### Advanced Activations
**Softmax(), ReLU()**, LeakyReLU(), PReLU(), ELU(), ThresholdedReLU()

### Normalization
**BatchNormalization()**

### Noise
GaussianNoise(), GaussianDropout(), AlphaDropout()

### Wrappers
TimeDistributed(), Bidirectional()

### writing your own layer
build(input_shape), call(x), compute_output_shape(input_shape)

## Data preprocessing

### series preprocessing

### text preprocessing

### image preprocessing
**keras.preprocessing.image.ImageDataGenerator()**\
apply_transfrom(), fit(), flow(), flow_from_directory(), get_random_transform(), random_transform(), standardize()

## misc

### Losses
**mean_squared_error, categorical_crossentropy**, etc

### Metrics
binary_accuracy, categorical_accuracy

### Optimizers
SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, TFOptimizer

### Activations
softmax, relu, sigmoid, tanh, linear, etc.

### Initializers
**Initializer(): Zeros(), Ones(), Constant(), RandomNormal(), RandomUniform(), TruncatedNormal(), VarianceScaling()**, Orthognal(), Identity(), lecun_uniform(), glorot_normal(), glorot_uniform(), he_normal(), lecun_normal(), he_uniform()

### Regularizers
kernel_regularizer, bias_regularizer, activity_regularizer; l1, l2, l1_l2

### constraints
kernel_constraint, bias_constraint; max_norm, non_neg, unit_norm, min_max_norm

### callbacks
Callback(), BaseLogger, History(), ModelCheckpoint(), EarlyStopping(), LearningRateScheduler(), Tensorboard(), CSVLogger(), ReduceLROnPlateau(), LambdaCallback(), TerminateOnNaN(), ProgbarLogger(), RemoteMonitor()

### Sample Datasets
MNIST, Fashion-MNIST, CIFAR10, CIFAR100, IMDB, Reuters, Boston_housing

### Pretrained model
Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet, DenseNet, NASNet

### utilities
to_categorical(), normalize(), get_file(), print_summary(), plot_model(), multi_gpu_model(), HDF5Matrix()