# Conditional Generative Adversarial Nets

Mehdi Mirza University of Montreal

## Abstract

Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models. In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator. We show that this model can generate MNIST digits conditioned on class labels. We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.

GANs是最近提出的训练生成式模型的新方法。在本文中，我们提出条件版的GANs，其构建方法是，将我们期望的数据y作为条件施加到生成器和判别器之上。我们证明了，这个模型可以生成，以类别标签为条件的MNIST数字。我们还描述了，这个模型怎样用于学习一个多模的模型，并给出了一个图像加标签的初级应用示例，证明了这种方法可以生成描述性的标签，而这个标签不是训练标签的一部分。

## 1 Introduction 引言

Generative adversarial nets were recently introduced as an alternative framework for training generative models in order to sidestep the difficulty of approximating many intractable probabilistic computations.

GANs是最近提出的框架，是一种训练生成式模型的方法，可以绕过估计很多难以处理的概率计算的困难。

Adversarial nets have the advantages that Markov chains are never needed, only backpropagation is used to obtain gradients, no inference is required during learning, and a wide variety of factors and interactions can easily be incorporated into the model.

对抗网络有一个优势，就是再也不需要Markov链了，只需要反向传播来得到梯度，在学习的时候不需要推理过程，模型可以很容易的集成很多因素和互动。

Furthermore, as demonstrated in [8], it can produce state of the art log-likelihood estimates and realistic samples. 而且，如[8]中所给出的，可以得到目前最好的log概率估计结果和很真实的样本。

In an unconditioned generative model, there is no control on modes of the data being generated. However, by conditioning the model on additional information it is possible to direct the data generation process. Such conditioning could be based on class labels, on some part of data for inpainting like [5], or even on data from different modality.

在无条件的生成式模型中，对于生成的数据的模式没有任何控制。但是，通过采用额外的信息给模型添加条件，则很可能引导数据生成过程。这种条件可能是基于类别标签的，可能是基于部分数据的，如[5]一样进行修补，甚至是从不同模式的数据的。

In this work we show how can we construct the conditional adversarial net. And for empirical results we demonstrate two set of experiment. One on MNIST digit data set conditioned on class labels and one on MIR Flickr 25,000 dataset [10] for multi-modal learning.

在本文中，我们给出了如何构建条件对抗网络的方法。通过经验性的结果，我们给出两组试验。一个是在MNIST数字数据集上，施加的条件是类别标签；一个是在MIR Flickr 25000数据集的，进行多模态学习。

## 2 Related Work 相关工作

### 2.1 Multi-modal Learning For Image Labelling

Despite the many recent successes of supervised neural networks (and convolutional networks in particular) [13, 17], it remains challenging to scale such models to accommodate an extremely large number of predicted output categories. A second issue is that much of the work to date has focused on learning one-to-one mappings from input to output. However, many interesting problems are more naturally thought of as a probabilistic one-to-many mapping. For instance in the case of image labeling there may be many different tags that could appropriately applied to a given image, and different (human) annotators may use different (but typically synonymous or related) terms to describe the same image.

尽管最近监督神经网络（尤其是卷积神经网络）有很多成功，但将这些模型扩展，以适应大量predicted output categories，还是非常有挑战的。第二个问题是，迄今为止的工作，关注的是从输入到输出的一对一的映射学习。但是，很多有趣的问题，很可能是一个概率上一对多的映射。比如，在图像加标签的情形中，对于给定的图像，可能会有很多不同的标签都是合适的，不同的人类标注者会使用不同的术语（但通常是同义词，或相关的词），来描述同样的图像。

One way to help address the first issue is to leverage additional information from other modalities: for instance, by using natural language corpora to learn a vector representation for labels in which geometric relations are semantically meaningful. When making predictions in such spaces, we benefit from the fact that when prediction errors we are still often ‘close’ to the truth (e.g. predicting ’table’ instead of ’chair’), and also from the fact that we can naturally make predictive generalizations to labels that were not seen during training time. Works such as [3] have shown that even a simple linear mapping from image feature-space to word-representation-space can yield improved classification performance.

帮助解决第一个问题的一种方法是，利用其他模态的额外信息：比如，使用自然语言语料库学习到标签的一个向量表示，而标签中有语义上有意义的几何关系。当在这种空间中进行预测时，我们会从中受益，因为在预测时，其预测误差通常会很接近事实真相（如，预测“桌子”，而不是“椅子”），而且我们可以很自然的预测性的泛化标签，这些标签在训练中并没有看到。[3]这样的工作表明，即使是从图像特征空间到词语表示空间的简单线性映射，也可以得到改进的分类性能。

One way to address the second problem is to use a conditional probabilistic generative model, the input is taken to be the conditioning variable and the one-to-many mapping is instantiated as a conditional predictive distribution.

解决第二个问题的一个方法是，使用条件式概率生成模式，输入是条件变量，一对多的映射实例化为一种条件预测分布。

[16] take a similar approach to this problem, and train a multi-modal Deep Boltzmann Machine on the MIR Flickr 25,000 dataset as we do in this work. [16]对这个问题采用了一种类似的方法，在MIR Flickr 25000数据集上训练了一个多模DBM，我们在本文中也是这样做的。

Additionally, in [12] the authors show how to train a supervised multi-modal neural language model, and they are able to generate descriptive sentence for images. 另外，在[12]中，作者展示了怎样训练一个有监督的多模态自然语言模型，可以对图像生成描述性的语句。

## 3 Conditional Adversarial Nets

### 3.1 Generative Adversarial Nets

Generative adversarial nets were recently introduced as a novel way to train a generative model. They consists of two ‘adversarial’ models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. Both G and D could be a non-linear mapping function, such as a multi-layer perceptron.

GANs是最近提出的用于训练生成式模型的新方法。它们包含两个对抗的模型：一个生成式模型G，捕获数据分布，一个判别式模型D，估计样本是从训练数据，而不是从G中生成的概率。G和D都可以是非线性映射函数，比如多层感知机(MLP)。

To learn a generator distribution pg over data data x, the generator builds a mapping function from a prior noise distribution pz(z) to data space as G(z;θg). And the discriminator, D(x;θd), outputs a single scalar representing the probability that x came form training data rather than pg .

为在数据x上学习得到生成器分布pg，生成器从先验噪声分布pz(z)到数据空间构建一个映射函数，记为G(z;θg)。判别器D(x;θd)，输出一个标量，表示x是从训练数据中来的，而不是从pg中得到的概率。

G and D are both trained simultaneously: we adjust parameters for G to minimize log(1 − D(G(z)) and adjust parameters for D to minimize logD(X), as if they are following the two-player min-max game with value function V(G, D):

G和D同时进行训练：我们调整G的参数，以最小化log(1 − D(G(z))，调整D的参数，以最小化logD(X)，即，它们使用下面的价值函数V(G,D)来进行双玩家min-max游戏：

$$min_G max_D V(D,G) = E_{x∼pdata(x)} [logD(x)] + E_{z∼pz(z)} [log(1−D(G(z)))]$$(1)

### 3.2 Conditional Adversarial Nets

Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information y. y could be any kind of auxiliary information, such as class labels or data from other modalities. We can perform the conditioning by feeding y into the both the discriminator and generator as additional input layer.

GANs可以拓展为条件式模型，生成器和判别器都以一些额外信息y为条件。y可以是任何辅助信息，比如类别标签，或其他模态中的数据。我们将y作为生成器和判别器的额外输入层，从而为其施加条件。

In the generator the prior input noise pz(z), and y are combined in joint hidden representation, and the adversarial training framework allows for considerable flexibility in how this hidden representation is composed(For now we simply have the conditioning input and prior noise as inputs to a single hidden layer of a MLP, but one could imagine using higher order interactions allowing for complex generation mechanisms that would be extremely difficult to work with in a traditional generative framework).

在生成器中，先验输入噪声pz(z)和y结合到一起，形成联合隐藏表示；对抗训练框架允许几种形式的隐藏表示联合方式（现在我们只是简单的将条件输入和先验噪声作为MLP的单个隐藏层输入，但也可以是更高阶的互动，得到更复杂的生成机制，在传统的生成式框架中将很难起作用）。

In the discriminator x and y are presented as inputs and to a discriminative function (embodied again by a MLP in this case). 在判别器中，x和y是判别器函数的输入（在这种情况中，仍然是MLP）。

The objective function of a two-player minimax game would be as Eq 2 两玩家minmax游戏的目标函数如式2表示：

$$min_G max_D V(D,G) = E_{x∼pdata(x)} [logD(x|y)] + E_{z∼pz(z)} [log(1−D(G(z|y)))]$$(2)

Fig 1 illustrates the structure of a simple conditional adversarial net. 图1给出了一个简单的条件对抗网络的结构。

## 4 Experimental Results

### 4.1 Unimodal

We trained a conditional adversarial net on MNIST images conditioned on their class labels, encoded as one-hot vectors. 我们在MNIST图像中，以其类别标签为条件（编码为独热向量），训练了一个条件对抗网络。

In the generator net, a noise prior z with dimensionality 100 was drawn from a uniform distribution within the unit hypercube. Both z and y are mapped to hidden layers with Rectified Linear Unit (ReLu) activation [4, 11], with layer sizes 200 and 1000 respectively, before both being mapped to second, combined hidden ReLu layer of dimensionality 1200. We then have a final sigmoid unit layer as our output for generating the 784-dimensional MNIST samples.

在生成器网络中，100维的先验噪声z符合单位超体内的均匀分布。z和y都使用ReLU激活映射到隐藏层，层的大小分别为200和1000，在映射到第二个隐藏层之前，两个ReLU隐藏层结合成1200维。我们然后有一个最终的sigmoid单元层作为我们的输出，生成784维的MNIST样本。

The discriminator maps x to a maxout [6] layer with 240 units and 5 pieces, and y to a maxout layer with 50 units and 5 pieces. Both of the hidden layers mapped to a joint maxout layer with 240 units and 4 pieces before being fed to the sigmoid layer. (The precise architecture of the discriminator is not critical as long as it has sufficient power; we have found that maxout units are typically well suited to the task.)

判别器将x映射到一个maxout层，有240个单元和5片，将y映射到一个maxout层，有50个单元和5片。在送入sigmoid层之前，两个隐藏层映射到一个联合maxout层，有240个单元和4片。（判别器的精确架构并不重要，只要其有足够的能量；我们发现maxout单元通常很适合这类任务。）

The model was trained using stochastic gradient decent with mini-batches of size 100 and initial learning rate of 0.1 which was exponentially decreased down to .000001 with decay factor of 1.00004. Also momentum was used with initial value of .5 which was increased up to 0.7. Dropout [9] with probability of 0.5 was applied to both the generator and discriminator. And best estimate of log-likelihood on the validation set was used as stopping point.

模型的训练使用随机梯度下降，mini-batches大小为100，初始学习速率为0.1，以指数级下降到.000001，衰减系数为1.00004。也使用了动量，初始值为.5，然后增加到.7。生成器和判别器都使用了概率为0.5的dropout。在验证集上log似然的最佳估计，就是停止点。

Table 1 shows Gaussian Parzen window log-likelihood estimate for the MNIST dataset test data. 1000 samples were drawn from each 10 class and a Gaussian Parzen window was fitted to these samples. We then estimate the log-likelihood of the test set using the Parzen window distribution. (See [8] for more details of how this estimate is constructed.)

表1给出了，MNIST数据集测试数据的Gaussian Parzen窗的log似然估计。每10个类别选出1000个样本，使用Gaussian Parzen窗来拟合这些样本。我们然后使用Parzen窗分布来估计测试集的log似然。（[8]中有这个估计是怎样构建的细节）。

The conditional adversarial net results that we present are comparable with some other network based, but are outperformed by several other approaches – including non-conditional adversarial nets. We present these results more as a proof-of-concept than as demonstration of efficacy, and believe that with further exploration of hyper-parameter space and architecture that the conditional model should match or exceed the non-conditional results.

我们得到的条件对抗网络的结果，与基于一些其他网络的类似，但不如几种其他方法的，包括非条件对抗网络的。我们给出这些结果，更多的是概念验证，而不是功效的验证；我们相信，进一步探索超参数空间和结构，条件模型会与非条件模型结果类似，甚至超过。

Fig 2 shows some of the generated samples. Each row is conditioned on one label and each column is a different generated sample. 图2给出了一些生成的样本。每一行是以一个标签为条件的，每一列都是一个不同的生成的样本。

### 4.2 Multimodal

Photo sites such as Flickr are a rich source of labeled data in the form of images and their associated user-generated metadata (UGM) — in particular user-tags. 照片网站，如Flickr，是标注数据的丰富源泉，其形式是图像和其关联的用户生成的元数据(UGM)，特别是用户标签。

User-generated metadata differ from more ‘canonical’ image labelling schems in that they are typically more descriptive, and are semantically much closer to how humans describe images with natural language rather than just identifying the objects present in an image. Another aspect of UGM is that synoymy is prevalent and different users may use different vocabulary to describe the same concepts — consequently, having an efficient way to normalize these labels becomes important. Conceptual word embeddings [14] can be very useful here since related concepts end up being represented by similar vectors.

用户生成的元数据与经典的图像标注方案不同，因为这些一般更具有描述性，语义上更接近于人类怎样用自然语言描述这些图像，而不仅仅是识别图像中存在哪些物体。UGM的另一方面是，其中同义词非常多，不同的用户会使用不同的词汇来描述相同的概念，结果是，有一种高效的标准化这些标签的方法非常重要。概念词汇潜入[14]这里会非常有用，因为相关的概念最终会由类似的向量所表示。

In this section we demonstrate automated tagging of images, with multi-label predictions, using conditional adversarial nets to generate a (possibly multi-modal) distribution of tag-vectors conditional on image features.

在本节中，我们展示的是图像的自动标签，并进行多标签预测，使用的是对抗网络在图像特征的条件下生成（可能是多模的）标签向量分布。

For image features we pretrain a convolutional model similar to the one from [13] on the full ImageNet dataset with 21,000 labels [15]. We use the output of the last fully connected layer with 4096 units as image representations.

对于图像特征，我们在21000个标签的完整ImageNet数据集上，预训练一个卷积模型，与[13]中的类似。我们使用最后一个全连接层（4096个单元）的输出作为图像表示。

For the word representation we first gather a corpus of text from concatenation of user-tags, titles and descriptions from YFCC100M 2 dataset metadata. After pre-processing and cleaning of the text we trained a skip-gram model [14] with word vector size of 200. And we omitted any word appearing less than 200 times from the vocabulary, thereby ending up with a dictionary of size 247465.

对于词汇表示，我们首先收集YFCC100M 2数据集的元数据中的用户标记、标题和描述拼接形成的语料库。在对文本进行预处理和清洗后，我们训练一个skip-gram模型，词向量大小为200。我们忽略了词汇表中任何出现次数少于200的词汇，最后形成的词典大小为247465。

We keep the convolutional model and the language model fixed during training of the adversarial net. And leave the experiments when we even backpropagate through these models as future work.

我们在训练对抗网络的时候，保持卷积模型和语言模型固定。

For our experiments we use MIR Flickr 25,000 dataset [10], and extract the image and tags features using the convolutional model and language model we described above. Images without any tag were omitted from our experiments and annotations were treated as extra tags. The first 150,000 examples were used as training set. Images with multiple tags were repeated inside the training set once for each associated tag.

对我们的试验，我们使用MIR Flickr 25000数据集，使用上面所述的卷积模型和语言模型来提取图像特征和标签特征。没有标签的图像在我们的试验中直接忽略掉，标注是被当作额外的标签。前150000个样本用作训练集。具有多标签的图像，对每个相关的标签，在训练集中就重复一次。

For evaluation, we generate 100 samples for each image and find top 20 closest words using cosine similarity of vector representation of the words in the vocabulary to each sample. Then we select the top 10 most common words among all 100 samples. Table 4.2 shows some samples of the user assigned tags and annotations along with the generated tags.

对于评估，我们对每个图像生成100个样本，找到top 20最接近的词语，使用的是词汇向量表示的cosine相似度。然后我们在所有100个样本中选择top 10最常用的词汇。表4.2给出了，用户指定的标签和标注，以及生成的标签的一些例子。

The best working model’s generator receives Gaussian noise of size 100 as noise prior and maps it to 500 dimension ReLu layer. And maps 4096 dimension image feature vector to 2000 dimension ReLu hidden layer. Both of these layers are mapped to a joint representation of 200 dimension linear layer which would output the generated word vectors.

最好的模型的生成器，以100维的高斯噪声为输入，将其映射为500维的ReLU层。将4096维的图像特征向量映射为2000维的ReLU隐含层。这两层都映射到200维线性层的联合表示上，然后输出生成的词向量。

The discriminator is consisted of 500 and 1200 dimension ReLu hidden layers for word vectors and image features respectively and maxout layer with 1000 units and 3 pieces as the join layer which is finally fed to the one single sigmoid unit.

判别器是由500维和1200维的ReLU隐含层组成，这分别是词向量和图像特征，联合层是1000个单元、3片的maxout层，最后送入一个sigmoid单元。

The model was trained using stochastic gradient decent with mini-batches of size 100 and initial learning rate of 0.1 which was exponentially decreased down to .000001 with decay factor of 1.00004. Also momentum was used with initial value of .5 which was increased up to 0.7. Dropout with probability of 0.5 was applied to both the generator and discriminator.

模型用SGD进行训练，mini-batches大小为100，初始学习速率为0.1，以指数降低到.000001，衰减因子为1.00004。还使用了动量，初始值为.5，然后增加到0.7。生成器和判别器都使用了概率为0.5的dropout。

The hyper-parameters and architectural choices were obtained by cross-validation and a mix of random grid search and manual selection (albeit over a somewhat limited search space.)

超参数和架构选择是通过交叉验证、随机网格搜索和手工选择的混合方法得到的。

## 5 Future Work

The results shown in this paper are extremely preliminary, but they demonstrate the potential of conditional adversarial nets and show promise for interesting and useful applications.

本文中给出的结果是非常初步的，但证明了条件对抗网络的潜能，可能得到有趣有用的应用。

In future explorations between now and the workshop we expect to present more sophisticated models, as well as a more detailed and thorough analysis of their performance and characteristics.

未来的探索，我们希望给出更复杂的模型，以及其性能和特性的详细彻底的分析。

Also, in the current experiments we only use each tag individually. But by using multiple tags at the same time (effectively posing generative problem as one of ‘set generation’) we hope to achieve better results.

还有，在现在的试验里，我们只独立的使用了每个标签。但通过同时使用多标签（将生成式问题作为“集合生成”），我们希望得到更好的结果。

Another obvious direction left for future work is to construct a joint training scheme to learn the language model. Works such as [12] has shown that we can learn a language model for suited for the specific task.

另一个明显的方向是，构建语言模型的联合训练方案。[12]这样的工作已经表明，我们可以对特定的任务学习一个语言模型。