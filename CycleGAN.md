# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

Jun-Yan Zhu et al. Berkeley AI Research (BAIR)

## 0 Abstract

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F : Y → X and introduce a cycle consistency loss to enforce F(G(X)) ≈ X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

图像到图像的翻译一类视觉和图形学问题，其目标是学习输入图像到输出图像的映射，使用的是对齐的图像对组成的训练集。但是，对于很多任务来说，成对的训练数据是不可用的。我们提出一种方法，学习在没有成对图像的情况下，将图像从源领域X翻译到目标领域Y。我们的目标是，学习到一个映射G : X → Y，使得用对抗损失难以区分，G(X)的图像分布和Y的分布。因为这个映射是高度不受约束的，我们将其与一个逆映射F : Y → X组成映射对，提出了一个循环一致性损失，以确保F(G(X)) ≈ X，反之亦然。在几个任务上给出定性的结果，这些任务都没有可用的成对训练数据，包含集合风格迁移，目标变容，季节迁移，照片增强，等。与几种之前的方法的定量比较，证明了我们的方法确实更优秀。

## 1. Introduction

What did Claude Monet see as he placed his easel by the bank of the Seine near Argenteuil on a lovely spring day in 1873 (Figure 1, top-left)? A color photograph, had it been invented, may have documented a crisp blue sky and a glassy river reflecting it. Monet conveyed his impression of this same scene through wispy brush strokes and a bright palette.

What if Monet had happened upon the little harbor in Cassis on a cool summer evening (Figure 1, bottom-left)? A brief stroll through a gallery of Monet paintings makes it possible to imagine how he would have rendered the scene: perhaps in pastel shades, with abrupt dabs of paint, and a somewhat flattened dynamic range.

We can imagine all this despite never having seen a side by side example of a Monet painting next to a photo of the scene he painted. Instead, we have knowledge of the set of Monet paintings and of the set of landscape photographs. We can reason about the stylistic differences between these two sets, and thereby imagine what a scene might look like if we were to “translate” it from one set into the other.

In this paper, we present a method that can learn to do the same: capturing special characteristics of one image collection and figuring out how these characteristics could be translated into the other image collection, all in the absence of any paired training examples.

本文中，我们提出一种方法，可以学习来做这样的事：捕捉到一个图像集的特殊性质，将这些特性迁移到另一个图像集合中，整个过程没有任何成对的训练样本。

This problem can be more broadly described as image-to-image translation [22], converting an image from one representation of a given scene, x, to another, y, e.g., grayscale to color, image to semantic labels, edge-map to photograph. Years of research in computer vision, image processing, computational photography, and graphics have produced powerful translation systems in the supervised setting, where example image pairs {$x_i,y_i$}$^N_{i=1}$ are available (Figure 2, left), e.g., [11, 19, 22, 23, 28, 33, 45, 56, 58, 62]. However, obtaining paired training data can be difficult and expensive. For example, only a couple of datasets exist for tasks like semantic segmentation (e.g., [4]), and they are relatively small. Obtaining input-output pairs for graphics tasks like artistic stylization can be even more difficult since the desired output is highly complex, typically requiring artistic authoring. For many tasks, like object transfiguration (e.g., zebra↔horse, Figure 1 top-middle), the desired output is not even well-defined.

这个问题是一类更广的图像到图像翻译问题，将一幅图像从一个场景的一种表示x，转化成另一种y，即，灰度到彩色，图像到语义标签，边缘图到照片。在计算机视觉、图像处理、计算摄影学和图形学中多年的研究，得到了监督设置下很强大的翻译系统，其中样本图像对{$x_i,y_i$}$^N_{i=1}$是可用的。但是，要得到成对的训练数据，这是困难昂贵的。比如，像语义分割这种任务，只有几个数据集可用，而且相对较小。得到图形学任务的输入-输出对，如艺术风格化，可能是更困难的，因为期望的输出是高度复杂的，一般需要进行艺术创作。对于很多任务，如目标变容，（如，斑马与马的互相转换），期望的输出甚至都没法定义。

We therefore seek an algorithm that can learn to translate between domains without paired input-output examples (Figure 2, right). We assume there is some underlying relationship between the domains – for example, that they are two different renderings of the same underlying scene – and seek to learn that relationship. Although we lack supervision in the form of paired examples, we can exploit supervision at the level of sets: we are given one set of images in domain X and a different set in domain Y. We may train a mapping G : X → Y such that the output $\hat y$ = G(x), x ∈ X, is indistinguishable from images y ∈ Y by an adversary trained to classify $\hat y$ apart from y. In theory, this objective can induce an output distribution over $\hat y$ that matches the empirical distribution $p_{data}(y)$ (in general, this requires G to be stochastic) [16]. The optimal G thereby translates the domain X to a domain $\hat Y$ distributed identically to Y. However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – there are infinitely many mappings G that will induce the same distribution over $\hat y$. Moreover, in practice, we have found it difficult to optimize the adversarial objective in isolation: standard procedures often lead to the well-known problem of mode collapse, where all input images map to the same output image and the optimization fails to make progress [15].

因此，我们寻找一种算法，在没有成对的输入-输出样本下（图2，右），学习在不同领域中进行图像翻译。我们假设，在这些不同领域中有一些潜在的关系，如，它们是同一潜在场景的不同呈现，然后我们寻求去学习这种关系。虽然我们缺少成对样本形式的监督，我们可以在集合的层次上寻找监督关系：我们给定了领域X中的图像集合，和领域Y中的另一个图像集合。我们可以训练一个映射G: X → Y，使输出$\hat y$ = G(x), x ∈ X，在对抗训练得到的判别器下，无法与图像y ∈ Y区别开来。理论上，这个目标函数会带来$\hat y$的输出分布，与经验分布$p_{data}(y)$要匹配，（一般来说，这需要G是随机的）。因此，最佳的G从领域X翻译到领域$\hat Y$，而领域$\hat Y$与Y的分布完全一致。但是，这样的一种翻译不能确保，单个的输入x和输出y是有意义成对的，在G会带来相同的$\hat y$分布的条件下，有无数多的映射G存在。而且，实践中，我们发现很难单独最优化对抗目标：标准的过程通常会导致著名的模式失败问题，其中所有输入图像映射到相同的输出图像，导致优化过程无法有进展。

These issues call for adding more structure to our objective. Therefore, we exploit the property that translation should be “cycle consistent”, in the sense that if we translate, e.g., a sentence from English to French, and then translate it back from French to English, we should arrive back at the original sentence [3]. Mathematically, if we have a translator G : X → Y and another translator F : Y → X, then G and F should be inverses of each other, and both mappings should be bijections. We apply this structural assumption by training both the mapping G and F simultaneously, and adding a cycle consistency loss [64] that encourages F(G(x)) ≈ x and G(F(y)) ≈ y. Combining this loss with adversarial losses on domains X and Y yields our full objective for unpaired image-to-image translation.

这些问题要求，在目标函数上增加更多结构。因此，我们研究了翻译的一个性质，即，翻译应当具有循环一致性，如如果我们从一个语句从英语翻译到法语，然后从法语再翻译回英语，我们应当得到相同的原始语句。数学上来说，如果我们有一个翻译器G : X → Y，和另一个翻译器F : Y → X，那么G和F应当是互逆的，这两个映射都应当是双向单射的。我们同时训练映射G和F，应用这个架构假设，并增加一个循环一致性损失，鼓励F(G(x)) ≈ x ，以及G(F(y)) ≈ y。在领域X和Y上，将这个损失与对抗损失结合到一起，得到我们的完整目标函数，进行不成对的图像到图像的翻译。

We apply our method to a wide range of applications, including collection style transfer, object transfiguration, season transfer and photo enhancement. We also compare against previous approaches that rely either on hand-defined factorizations of style and content, or on shared embedding functions, and show that our method outperforms these baselines. We provide both PyTorch and Torch implementations. Check out more results at our [website](https://junyanz.github.io/CycleGAN/).

我们将此方法应用很多应用中，包括集合风格迁移，目标变容，季节迁移和照片增强。我们还与之前的方法进行比较，即那些依赖于手动定义的风格与内容的分解，或依赖于共享的嵌入函数的，表明我们的方法超过了这些基准。我们给出PyTorch和Torch实现，详见我们的网站。

## 2. Related work

**Generative Adversarial Networks (GANs)** [16, 63] have achieved impressive results in image generation [6, 39], image editing [66], and representation learning [39, 43, 37]. Recent methods adopt the same idea for conditional image generation applications, such as text2image [41], image inpainting [38], and future prediction [36], as well as to other domains like videos [54] and 3D data [57]. The key to GANs’ success is the idea of an adversarial loss that forces the generated images to be, in principle, indistinguishable from real photos. This loss is particularly powerful for image generation tasks, as this is exactly the objective that much of computer graphics aims to optimize. We adopt an adversarial loss to learn the mapping such that the translated images cannot be distinguished from images in the target domain.

GANs在图像生成，图像编辑和表示学习中取得了令人印象深刻的结果。最近的方法采用了相同的思想，进行条件图像生成应用，如text2image，图像修补和未来预测，以及其他领域，如视频和3D数据。GANs成功的关键是对抗损失的思想，使生成的图像与真实的图像无法分辨开来。这种损失对于图像生成任务尤其强大，因为这正是大多数计算机图形学任务想要优化的目标函数。我们采用了对抗损失来学习这个映射，这样翻译的图像无法与目标领域的图像区分开来。

**Image-to-Image Translation**. The idea of image-to-image translation goes back at least to Hertzmann et al.’s Image Analogies [19], who employ a non-parametric texture model [10] on a single input-output training image pair. More recent approaches use a dataset of input-output examples to learn a parametric translation function using CNNs (e.g., [33]). Our approach builds on the “pix2pix” framework of Isola et al. [22], which uses a conditional generative adversarial network [16] to learn a mapping from input to output images. Similar ideas have been applied to various tasks such as generating photographs from sketches [44] or from attribute and semantic layouts [25]. However, unlike the above prior work, we learn the mapping without paired training examples.

**图像到图像的翻译**。其思想至少要回到Hertzmann等人的图像类比的文章[19]，其采用了一个非参数的纹理模型[10]，在单个输入-输出图像对上进行训练。最近的方法使用一个输入-输出样本的数据集，来使用CNNs学习一个参数化的翻译函数。我们的方法在Isola等的pix2pix框架之上构建，使用了cGAN网络来学习从输入到输出图像的映射。类似的思想也应用到来很多任务中，如从草图中生成图像，或从属性和语义分布中生成图像。但是，与上述的工作不同，我们不用成对的训练样本来学习这个映射。

**Unpaired Image-to-Image Translation**. Several other methods also tackle the unpaired setting, where the goal is to relate two data domains: X and Y. Rosales et al. [42] propose a Bayesian framework that includes a prior based on a patch-based Markov random field computed from a source image and a likelihood term obtained from multiple style images. More recently, CoGAN [32] and cross-modal scene networks [1] use a weight-sharing strategy to learn a common representation across domains. Concurrent to our method, Liu et al. [31] extends the above framework with a combination of variational autoencoders [27] and generative adversarial networks [16]. Another line of concurrent work [46, 49, 2] encourages the input and output to share specific “content” features even though they may differ in “style“. These methods also use adversarial networks, with additional terms to enforce the output to be close to the input in a predefined metric space, such as class label space [2], image pixel space [46], and image feature space [49].

**不成对的图像到图像翻译**。有几种启发方法也在处理不成对的设置下的问题，其目标是将两个数据领域X和Y关联起来。Rosales等提出一个贝叶斯框架。最近，CoGAN和跨模态场景网络使用来一种权重共享的策略，在跨领域中学习一个通用表示。与我们的工作同时，Liu等拓展来上述框架，将VAE与GAN结合了起来。同时的另一条工作线，鼓励输入和输出共享特定的内容特征，但其风格可能是不同的。这些方法也使用了GAN，但加入了另外的项，使输出与输入在预定义的度量空间中接近，比如类别标签空间，图像像素空间，和图像特征空间。

Unlike the above approaches, our formulation does not rely on any task-specific, predefined similarity function between the input and output, nor do we assume that the input and output have to lie in the same low-dimensional embedding space. This makes our method a general-purpose solution for many vision and graphics tasks. We directly compare against several prior and contemporary approaches in Section 5.1.

与上面的方法不同，我们的方法不依赖任何与任务相关的，预定义的输入输出间的相似度函数，我们也没有假设，输入和输出要在相同的低维嵌入空间中。这使我们的方法是很多视觉和图形学任务的通用解。我们在5.1节中直接与几种之前和现在的工作进行比较。

**Cycle Consistency**. The idea of using transitivity as a way to regularize structured data has a long history. In visual tracking, enforcing simple forward-backward consistency has been a standard trick for decades [24, 48]. In the language domain, verifying and improving translations via “back translation and reconciliation” is a technique used by human translators [3] (including, humorously, by Mark Twain [51]), as well as by machines [17]. More recently, higher-order cycle consistency has been used in structure from motion [61], 3D shape matching [21], co-segmentation [55], dense semantic alignment [65, 64], and depth estimation [14]. Of these, Zhou et al. [64] and Godard et al. [14] are most similar to our work, as they use a cycle consistency loss as a way of using transitivity to supervise CNN training. In this work, we are introducing a similar loss to push G and F to be consistent with each other. Concurrent with our work, in these same proceedings, Yi et al. [59] independently use a similar objective for unpaired image-to-image translation, inspired by dual learning in machine translation [17].

**循环一致性**。使用传递性作为结构化数据的规范方式，这种思想有很长的历史。在视觉跟踪中，增加简单的前向-后向一致性，几十年中一直是标准技巧。在语言领域，通过“反向翻译和一致性“验证和改进翻译，是人类翻译者和机器使用的一种技术。最近，高阶循环一致性已经用于了从运动恢复结构，3D形状匹配，协同分割，密集语义对齐，和深度估计。在这之中，Zhou等和Godard等是与我们的工作最接近的，因为他们使用循环一致性损失，作为一种使用传递性来监督CNN训练的方式。在本文中，我们引入了类似的损失函数，使G和F互相一致。与我们的工作同时，Yi等独立的使用了类似的目标函数，进行不成对的图像到图像翻译，这是受到了机器翻译中对偶学习的启发。

**Neural Style Transfer** [13, 23, 52, 12] is another way to perform image-to-image translation, which synthesizes a novel image by combining the content of one image with the style of another image (typically a painting) based on matching the Gram matrix statistics of pre-trained deep features. Our primary focus, on the other hand, is learning the mapping between two image collections, rather than between two specific images, by trying to capture correspondences between higher-level appearance structures. Therefore, our method can be applied to other tasks, such as painting→ photo, object transfiguration, etc. where single sample transfer methods do not perform well. We compare these two methods in Section 5.2.

**神经风格迁移**是另一种进行图像到图像翻译的方法，通过将一幅图像的内容与另一幅图像的风格（一般是一幅绘画）结合起来，合成一幅新的图像，其方法是匹配预训练深度特征统计的Gram矩阵。另一方面，我们的主要焦点，是在两个图像集合之间学习映射，而不是在两幅特定的图像之间学习，我们学习的方法是试图捕获高层外表结构之间的对应性。因此，我们的方法可以用于其他任务，如绘画到照片的转换，目标变容，等等，在这些任务中，单样本迁移的方法效果不会很好。我们在5.2节中比较这两种方法。

## 3. Formulation

Our goal is to learn mapping functions between two domains X and Y given training samples $\{x_i\}^N_{i=1}$ where $x_i$ ∈ X and $\{y_j\}^M_{j=1}$ where $y_j$ ∈ Y(We often omit the subscript i and j for simplicity). We denote the data distribution as x ∼ $p_{data}(x)$ and y ∼ $p_{data}(y)$. As illustrated in Figure 3 (a), our model includes two mappings G : X → Y and F : Y → X. In addition, we introduce two adversarial discriminators DX and DY , where DX aims to distinguish between images {x} and translated images {F(y)}; in the same way, DY aims to discriminate between {y} and {G(x)}. Our objective contains two types of terms: adversarial losses [16] for matching the distribution of generated images to the data distribution in the target domain; and cycle consistency losses to prevent the learned mappings G and F from contradicting each other.

我们的目标是，在给定训练样本$\{x_i\}^N_{i=1}$和$\{y_j\}^M_{j=1}$的情况下，其中$x_i$ ∈ X，$y_j$ ∈ Y（简化起见，我们回忽略掉上标i和j），学习两个领域X和Y间的映射函数。我们将数据分布表示为，x ∼ $p_{data}(x)$，y ∼ $p_{data}(y)$。如图3a所示，我们的模型包含两个映射G : X → Y和F : Y → X。另外，我们引入了两个对抗判别器DX和DY，其中DX的目标是区分{x}和翻译过来的图像{F(y)}；以同样的方式，DY的目标是区分{y}和{G(x)}。我们的目标函数包含两类项：对抗损失，用于匹配生成图像的分布，和目标领域的数据分布；循环一致性损失，来防止学习的映射G和F互相矛盾。

Figure 3: (a) Our model contains two mapping functions G : X → Y and F : Y → X, and associated adversarial discriminatorsDY andDX.DY encouragesGtotranslateXintooutputsindistinguishablefromdomainY,andviceversa for DX and F. To further regularize the mappings, we introduce two cycle consistency losses that capture the intuition that if we translate from one domain to the other and back again we should arrive at where we started: (b) forward cycle-consistency loss: x → G(x) → F (G(x)) ≈ x, and (c) backward cycle-consistency loss: y → F (y) → G(F (y)) ≈ y

### 3.1. Adversarial Loss

We apply adversarial losses [16] to both mapping functions. For the mapping function G : X → Y and its discriminator DY , we express the objective as:

我们将对抗损失应用到两个映射函数中。对于映射函数G : X → Y，及其判别器DY，其目标函数为：

$$L_{GAN}(G, DY , X, Y) = E_{y∼pdata(y)} [log DY (y)] + + E_{x∼pdata(x)} [log(1 − DY (G(x))]$$(1)

where G tries to generate images G(x) that look similar to images from domain Y, while DY aims to distinguish between translated samples G(x) and real samples y. G aims to minimize this objective against an adversary D that tries to maximize it, i.e., $min_G max_{DY} L_{GAN}(G, DY , X, Y)$. We introduce a similar adversarial loss for the mapping function F : Y → X and its discriminator DX as well: i.e.,$min_F max_{DX} L_{GAN}(F,DX,Y,X)$.

其中G要生成的图像G(x)，看起来与领域Y中的图像很类似，而DY的目标是区分翻译的样本G(x)和真实样本y。G的目标是最小化目标函数，而D要最大化目标函数，即，$min_G max_{DY} L_{GAN}(G, DY , X, Y)$。我们对映射F : Y → X，以及其判别器DX，提出来类似的对抗损失，即，$min_F max_{DX} L_{GAN}(F,DX,Y,X)$。

### 3.2. Cycle Consistency Loss

Adversarial training can, in theory, learn mappings G and F that produce outputs identically distributed as target domains Y and X respectively (strictly speaking, this requires G and F to be stochastic functions) [15]. However, with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution. Thus, adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. To further reduce the space of possible mapping functions, we argue that the learned mapping functions should be cycle-consistent: as shown in Figure 3 (b), for each image x from domain X, the image translation cycle should be able to bring x back to the original image, i.e., x → G(x) → F (G(x)) ≈ x. We call this forward cycle consistency. Similarly, as illustrated in Figure 3 (c), for each image y from domain Y , G and F should also satisfy backward cycle consistency: y → F (y) → G(F(y)) ≈ y. We incentivize this behavior using a cycle consistency loss:

理论上，对抗训练学习到的映射G和F，产生的输出其分布与目标领域Y和X分别一致（严格来说，这需要G和F是随机函数）。但是，在足够大的容量下，一个网络可以将同样的输入图像集，映射到目标领域的图像的任意随机排列，而任意学习到的映射得到的输出分布，都可以匹配目标领域。因此，对抗损失本身并不能保证，学习到的函数可以将输入xi映射到期望的输出yi上。为进一步降低可能的映射函数的空间，我们认为，学习到的映射函数应当是循环一致的：如图3b所示，对于领域X中的每个图像x，图像翻译的循环应当将x带回到原始图像中，即，x → G(x) → F (G(x)) ≈ x。我们称这个为前向循环一致性。类似的，如图3c所示，对于领域Y中的每幅图像y，G和F应当也满足反向循环一致性：y → F (y) → G(F(y)) ≈ y。我们使用一个循环一致性损失来鼓励这种行为：

$$L_{cyc} (G, F) = E_{x∼pdata(x)} [||F(G(x)) − x||_1] + + E_{y∼pdata(y)} [||G(F(y)) − y||_1]$$(2)

In preliminary experiments, we also tried replacing the L1 norm in this loss with an adversarial loss between F(G(x)) and x, and between G(F(y)) and y, but did not observe improved performance.

在初步的试验中，我们还试着将损失中的L1范数替换为F(G(x))与x之间的对抗损失，以及G(F(y))与y之间的对抗损失，但没有观察到性能有什么改进。

The behavior induced by the cycle consistency loss can be observed in Figure 4: the reconstructed images F(G(x)) end up matching closely to the input images x.

循环一致性损失带来的结果可以如图4中观察到：重建的图像F(G(x))与输入图像x之间有紧密的匹配。

Figure 4: The input images x, output images G(x) and the reconstructed images F(G(x)) from various experiments. From top to bottom: photo↔Cezanne, horses↔zebras, winter→summer Yosemite, aerial photos↔Google maps.

### 3.3. Full Objective

Our full objective is: 我们完整的目标函数为：

$$L(G, F, DX , DY) = L_{GAN}(G, DY , X, Y) + L_{GAN}(F,DX,Y,X) + λL_{cyc} (G, F)$$(3)

where λ controls the relative importance of the two objectives. We aim to solve: 其中λ控制的是两个目标函数的相对重要性。我们的目标是求解：

$$G∗,F∗ = argmin_{G,F} max_{DX,DY} L(G,F,DX,DY)$$(4)

Notice that our model can be viewed as training two “autoencoders” [20]: we learn one autoencoder F ◦ G : X → X jointly with another G ◦ F : Y → Y. However, these autoencoders each have special internal structures: they map an image to itself via an intermediate representation that is a translation of the image into another domain. Such a setup can also be seen as a special case of “adversarial autoencoders” [34], which use an adversarial loss to train the bottleneck layer of an autoencoder to match an arbitrary target distribution. In our case, the target distribution for the X → X autoencoder is that of the domain Y.

注意我们的模型可以视为训练两个自动编码机：我们学习自动编码机F ◦ G : X → X的同时也学习另外一个G ◦ F : Y → Y。但是，这些自动编码机每个都其特殊的内部结构：它们将一幅图像映射其本身，通过的是一个中间表示，而这个中间表示是图像到另一个领域的翻译。这样一种设置，也可以看作是“对抗自动编码机”的一种特殊情况。对抗自动编码机，使用对抗损失来训练自动编码机的瓶颈层，使其匹配任意的目标分布。在我们的情况中，X → X的自动编码机的目标分布是领域Y的分布。

In Section 5.1.4, we compare our method against ablations of the full objective, including the adversarial loss $L_{GAN}$ alone and the cycle consistency loss $L_{cyc}$ alone, and empirically show that both objectives play critical roles in arriving at high-quality results. We also evaluate our method with only cycle loss in one direction and show that a single cycle is not sufficient to regularize the training for this under-constrained problem.

在5.1.4节中，我们将此方法与完整目标函数的分离进行了对比，包括单独的对抗损失$L_{GAN}$和单独的循环一致性损失$L_{cyc}$，通过经验表明，这两个目标函数都起到了非常关键的作用，两个损失一起才能达到高质量的结果。我们还评估了一个方向的循环损失，表明，单个循环是不足以对训练这个约束不足的问题进行规范化的。

## 4. Implementation

**Network Architecture**. We adopt the architecture for our generative networks from Johnson et al. [23] who have shown impressive results for neural style transfer and super-resolution. This network contains two stride-2 convolutions, several residual blocks [18], and two fractionally-strided convolutions with stride 1/2. We use 6 blocks for 128 × 128 images and 9 blocks for 256 × 256 and higher-resolution training images. Similar to Johnson et al. [23], we use instance normalization [53]. For the discriminator networks we use 70 × 70 PatchGANs [22, 30, 29], which aim to classify whether 70 × 70 overlapping image patches are real or fake. Such a patch-level discriminator architecture has fewer parameters than a full-image discriminator and can work on arbitrarily-sized images in a fully convolutional fashion [22].

**网络架构**。我们采用Johnson等[23]的网络作为我们生成器网络的架构，其架构在神经风格迁移和超分辨率上得到了令人印象深刻的结果。这个网络包含了2个步长为2的卷积，几个残差模块，2个分数步长的卷积，步长为1/2。我们对128×128的训练图像使用6个模块，对256×256和更高分辨率的训练图像使用9个模块。与Johnson等[23]类似，我们使用了实例归一化。对于判别器网络，我们使用70×70的PatchGANs，其目标是对70×70的图像块进行分类，是真实的还是合成的。这样的图像块级的判别器架构，其参数比完整图像的判别器参数更少，可以在任意大小的图像上以全卷积的方式应用。

**Training details**. We apply two techniques from recent works to stabilize our model training procedure. First, for $L_{GAN}$ (Equation 1), we replace the negative log likelihood objective by a least-squares loss [35]. This loss is more stable during training and generates higher quality results. In particular, for a GAN loss $L_{GAN}(G, D, X, Y)$, we train the G to minimize $E_{x∼pdata(x)}[(D(G(x)) − 1)^2]$ and train the D to minimize $E_{y∼pdata(y)}[(D(y) − 1)^2] + E_{x∼pdata(x)}[D(G(x))^2]$.

**训练细节**。我们应用最近文献中提出的两种技术，来稳定模型训练过程。首先，对于$L_{GAN}$（式1），我们将log似然目标函数替换为最小二乘损失。这个损失在训练时更加稳定，可以生成更高质量的结果。特别的，对于GAN损失$L_{GAN}(G, D, X, Y)$，我们训练G以最小化$E_{x∼pdata(x)}[(D(G(x)) − 1)^2]$，训练D以最小化$E_{y∼pdata(y)}[(D(y) − 1)^2] + E_{x∼pdata(x)}[D(G(x))^2]$。

Second, to reduce model oscillation [15], we follow Shrivastava et al.’s strategy [46] and update the discriminators using a history of generated images rather than the ones produced by the latest generators. We keep an image buffer that stores the 50 previously created images.

第二，为降低模型震荡，我们按照Shrivastava等的策略，使用生成的图像历史来更新判别器，而不是使用生成器生成的最新图像来更新。我们维持一个图像缓存，保存之前生成的50幅图像。

For all the experiments, we set λ = 10 in Equation 3. We use the Adam solver [26] with a batch size of 1. All networks were trained from scratch with a learning rate of 0.0002. We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero over the next 100 epochs. Please see the appendix (Section 7) for more details about the datasets, architectures, and training procedures.

对于所有的试验，我们设式3中的λ = 10。我们使用Adam solver，batch size为1。所有网络都从头训练，学习速率为0.0002。我们对前100轮数据的训练，使用相同的学习速率，在后面100轮训练中线性的逐渐降低到0。关于数据集、架构和训练过程详见附录。

## 5. Results 结果

We first compare our approach against recent methods for unpaired image-to-image translation on paired datasets where ground truth input-output pairs are available for evaluation. We then study the importance of both the adversarial loss and the cycle consistency loss and compare our full method against several variants. Finally, we demonstrate the generality of our algorithm on a wide range of applications where paired data does not exist. For brevity, we refer to our method as CycleGAN. The PyTorch and Torch code, models, and full results can be found at our [website](https://junyanz.github.io/CycleGAN/).

我们首先与最近的不成对图像到图像翻译方法进行比较，我们在成对的数据集上进行比较，其中真值的输入-输出对可以用于评估。我们然后研究对抗损失和循环一致性损失的重要性，将完整的方法与几种变体进行比较。最后，我们在很多应用上证明轮我们算法的泛化性，其中成对的数据是不存在的。为简化，我们称本文的方法为CycleGAN。代码已经开源。

### 5.1. Evaluation

Using the same evaluation datasets and metrics as “pix2pix” [22], we compare our method against several baselines both qualitatively and quantitatively. The tasks include semantic labels↔photo on the Cityscapes dataset [4], and map↔aerial photo on data scraped from Google Maps. We also perform ablation study on the full loss function.

我们与pix2pix使用相同的评估数据集和度量标准，我们与几种基准方法进行轮定性和定量的比较。任务包括，在Cityscapes数据集上的语义标签到图像的相互转换，在从谷歌地图上爬取下来的数据上，地图与空中图像的相互转换。我们还对完整的损失函数进行了几种分离比较试验。

#### 5.1.1 Evaluation Metrics

**AMT perceptual studies**. On the map↔aerial photo task, we run “real vs fake” perceptual studies on Amazon Mechanical Turk (AMT) to assess the realism of our outputs. We follow the same perceptual study protocol from Isola et al. [22], except we only gather data from 25 participants per algorithm we tested. Participants were shown a sequence of pairs of images, one a real photo or map and one fake (generated by our algorithm or a baseline), and asked to click on the image they thought was real. The first 10 trials of each session were practice and feedback was given as to whether the participant’s response was correct or incorrect. The remaining 40 trials were used to assess the rate at which each algorithm fooled participants. Each session only tested a single algorithm, and participants were only allowed to complete a single session. The numbers we report here are not directly comparable to those in [22] as our ground truth images were processed slightly differently and the participant pool we tested may be differently distributed from those tested in [22] (due to running the experiment at a different date and time). Therefore, our numbers should only be used to compare our current method against the baselines (which were run under identical conditions), rather than against [22].

**AMT感官研究**。在地图到空中图像的相互转换任务中，我们用AMT进行“真实vs合成“的感官研究，以评估我们输出的真实性。我们采用的感官研究准则与Isola等[22]的一样，但我们测试过程中，每个算法只采集25位参与者的数据。系统给予参与者一系列图像对，一个是真实图像/地图，一个是合成的（由我们的算法或基准算法生成），然后要求点击他们认为是真实的图像。每个session的前10次测验是练习，会给予反馈，即参与者的响应是正确的还是不正确的。剩下的40次测验用来评估每个算法可以骗过多少参与者。每个session只测试一个算法，参与者只能完整一个session。这里我们得到的数量，与[22]中的数量不能直接进行比较，因为我们的真值图像的处理略有不同，我们测试的参与者与[22]中测试的分布会有不同（因为试验在不同的日期与时间进行）。因此，我们的数量只能用于比较目前的方法与基准（这是在相同的条件下运行的），而不能与[22]进行比较。

**FCN score**. Although perceptual studies may be the gold standard for assessing graphical realism, we also seek an automatic quantitative measure that does not require human experiments. For this, we adopt the “FCN score” from [22], and use it to evaluate the Cityscapes labels→photo task. The FCN metric evaluates how interpretable the generated photos are according to an off-the-shelf semantic segmentation algorithm (the fully-convolutional network, FCN, from [33]). The FCN predicts a label map for a generated photo. This label map can then be compared against the input ground truth labels using standard semantic segmentation metrics described below. The intuition is that if we generate a photo from a label map of “car on the road”, then we have succeeded if the FCN applied to the generated photo detects “car on the road”.

**FCN分数**。虽然感官研究是评估图形真实性的黄金标准，我们还使用一种量化的度量，不需要进行人类试验。为此，我们采用[22]中的FCN分数，用其来评估在Cityscapes上的语义标签到图像的生成任务。FCN度量标准评估的是生成的图像的可解释性，根据的是现成的语义分割算法FCN。FCN从生成的图像中预测一个标签图，标签图与输入的真值标签图进行比较，使用的是标准的语义分割度量标准，如下所述。原理的直觉是，如果我们从”car on the road“的标签图生成了一幅图像，如果FCN对生成的图像进行分割，检测到的也是”car on the road“，那么我们就成功了。

**Semantic segmentation metrics**. To evaluate the performance of photo→labels, we use the standard metrics from the Cityscapes benchmark [4], including per-pixel accuracy, per-class accuracy, and mean class Intersection-Over-Union (Class IOU) [4].

**语义分割标准**。为评估图像生成标签的应用的性能，我们使用Cityscapes基准测试中的标准度量，包括逐像素的准确率，每类别的准确率，和平均类别IoU。

#### 5.1.2 Baselines

**CoGAN**[32]. This method learns one GAN generator for domain X and one for domain Y , with tied weights on the first few layers for shared latent representations. Translation from X to Y can be achieved by finding a latent representation that generates image X and then rendering this latent representation into style Y.

这种方法学习的是一个GAN生成器，从领域X到另一个领域Y，前几层的权重对共享的隐式表示固定。从X到Y的翻译，可以通过找到一种隐式表示，生成图像X，然后将这种隐式表示呈现为Y的风格。

**SimGAN**[46]. Like our method, Shrivastava et al.[46] uses an adversarial loss to train a translation from X to Y. The regularization term $||x − G(x)||_1$ is used to penalize making large changes at pixel level.

和我们的方法类似，Shrivastava等使用对抗损失来训练X到Y的翻译。正则化项$||x − G(x)||_1$用于惩罚在像素级的大的变化。

**Feature loss + GAN**. We also test a variant of SimGAN [46] where the L1 loss is computed over deep image features using a pretrained network (VGG-16 relu4_2 [47]), rather than over RGB pixel values. Computing distances in deep feature space, like this, is also sometimes referred to as using a “perceptual loss” [8, 23].

我们还测试了SimGAN的一种变体，其中在深度图像特征中使用预训练网络（VGG-16 relu4_2）计算L1损失，而不是在RGB像素值上计算。像这样在深度特征空间上计算距离，有时候也被称为使用一个“感官损失”。

**BiGAN/ALI**[9, 7]. Unconditional GANs [16] learn a generator G : Z → X, that maps a random noise z to an image x. The BiGAN [9] and ALI [7] propose to also learn the inverse mapping function F : X → Z. Though they were originally designed for mapping a latent vector z to an image x, we implemented the same objective for mapping a source image x to a target image y.

无条件的GANs学习的生成器G : Z → X，将随机噪声z映射到图像x。BiGAN [9]和ALI [7]提出学习一个逆映射函数F : X → Z。虽然最初设计是用于将隐式向量z映射到图像x，我们对将源图像x到目标图像y的映射，也实现了相同的目标函数。

**pix2pix**[22]. We also compare against pix2pix [22], which is trained on paired data, to see how close we can get to this “upper bound” without using any paired data.

我们还与pix2pix进行了比较，这是在成对数据上进行训练的，看看我们不使用成对数据，与这个上限的距离有多远。

For a fair comparison, we implement all the baselines using the same architecture and details as our method, except for CoGAN [32]. CoGAN builds on generators that produce images from a shared latent representation, which is incompatible with our image-to-image network. We use the public implementation of CoGAN instead.

为公平比较，我们使用相同的架构和细节来实现所有的基准，除了CoGAN，我们使用CoGAN的公开实现。

#### 5.1.3 Comparison against baselines

As can be seen in Figure 5 and Figure 6, we were unable to achieve compelling results with any of the baselines. Our method, on the other hand, can produce translations that are often of similar quality to the fully supervised pix2pix.

如图5和图6所示，这些基准无法得到较好的结果。我们的方法则可以生成与全监督的pix2pix类似的翻译结果。

Table 1 reports performance regarding the AMT perceptual realism task. Here, we see that our method can fool participants on around a quarter of trials, in both the maps→aerial photos direction and the aerial photos→maps direction at 256 × 256 resolution3. All the baselines almost never fooled participants.

表1提出了AMT感官真实性任务的比较。这里，我们看到，我们的方法可以骗过大约1/4的参与者。但所有基准几乎无法骗到参与者。

Table 2 assesses the performance of the labels→photo task on the Cityscapes and Table 3 evaluates the opposite mapping (photos→labels). In both cases, our method again outperforms the baselines.

表2评估了在Cityscapes上标签到图像任务上的性能，表3评估了相反的映射（图像到语义标签）。在两种情况下，我们的方法都再一次超过了基准的方法。

#### 5.1.4 Analysis of the loss function

In Table 4 and Table 5, we compare against ablations of our full loss. Removing the GAN loss substantially degrades results, as does removing the cycle-consistency loss. We therefore conclude that both terms are critical to our results. We also evaluate our method with the cycle loss in only one direction: GAN + forward cycle loss $E_{x∼pdata(x)} [||F(G(x))−x||_1 ]$, or GAN + backward cycle loss $E_{y∼pdata(y)}[||G(F(y))−y||_1]$ (Equation 2) and find that it often incurs training instability and causes mode collapse, especially for the direction of the mapping that was removed. Figure 7 shows several qualitative examples.

在表4和表5中，我们对完整的损失进行了分离试验。去掉GAN损失，会使结果严重降质，去掉循环一致性损失也会这样。因此我们得出结论，这两项对我们的结果都很关键。我们还评估了，本文方法在只有单个方向的循环损失时的结果：GAN+前向循环损失$E_{x∼pdata(x)} [||F(G(x))−x||_1 ]$，或GAN+反向循环损失$E_{y∼pdata(y)}[||G(F(y))−y||_1]$，发现这会带来训练的不稳定，导致模式崩溃，尤其是对于移除掉的那个方向。图7给出了几个定性的例子。

#### 5.1.5 Image reconstruction quality

In Figure 4, we show a few random samples of the reconstructed images F(G(x)). We observed that the reconstructed images were often close to the original inputs x, at both training and testing time, even in cases where one domain represents significantly more diverse information, such as map↔aerial photos.

在图4中，我们给出了重建图像F(G(x))的几个随机样本。我们观察到，重建的图像与原始输入x通常很接近，在训练时和测试时都是这样，甚至在一个领域表示明显多样的信息时候也是这样，如地图到空中图像的相互转换。

#### 5.1.6 Additional results on paired datasets

Figure 8 shows some example results on other paired datasets used in “pix2pix” [22], such as architectural labels↔photos from the CMP Facade Database [40], and edges↔shoes from the UT Zappos50K dataset [60]. The image quality of our results is close to those produced by the fully supervised pix2pix while our method learns the mapping without paired supervision.

图8给出了在成对数据集上的一些结果例子，这个数据集是用在pix2pix中的。我们结果的图像质量与全监督方法pix2pix的结果类似，而我们的方法没有用成对的监督进行学习。

### 5.2. Applications

We demonstrate our method on several applications where paired training data does not exist. Please refer to the appendix (Section 7) for more details about the datasets. We observe that translations on training data are often more appealing than those on test data, and full results of all applications on both training and test data can be viewed on our project website.

我们给出了几种应用中我们方法的结果，这些应用中是没有成对的训练数据的。数据集详见附件。我们观察到，在训练集上的翻译结果，比对测试数据的更有吸引力，在训练数据和测试数据上所有应用的完整结果见项目网站。

**Collection style transfer (Figure 10 and Figure 11)**. We train the model on landscape photographs downloaded from Flickr and WikiArt. Unlike recent work on “neural style transfer” [13], our method learns to mimic the style of an entire collection of artworks, rather than transferring the style of a single selected piece of art. Therefore, we can learn to generate photos in the style of, e.g., Van Gogh, rather than just in the style of Starry Night. The size of the dataset for each artist/style was 526, 1073, 400, and 563 for Cezanne, Monet, Van Gogh, and Ukiyo-e.

我们从Flickr和WikiArt上下载了风景图，然后训练我们的模型。与最近的“神经风格迁移”不一样的是，我们的方法学习的是模仿整个艺术品集合的风格，而不是单幅艺术品的风格的迁移。因此，我们可以生成梵高风格的图像，而不止是starry night风格的图像。每个艺术家/风格的数据集大小分别为526, 1073, 400, and 563 for Cezanne, Monet, Van Gogh, and Ukiyo-e。

**Object transfiguration (Figure 13)** The model is trained to translate one object class from ImageNet [5] to another (each class contains around 1000 training images). Turmukhambetov et al. [50] propose a subspace model to translate one object into another object of the same category, while our method focuses on object transfiguration between two visually similar categories.

模型训练用于将ImageNet的一个目标类别翻译到另一个类别（每个类别包含大约1000幅训练图像）。Turmukhambetov等提出一个子空间模型，将一个目标翻译到同样类别的另一个目标，而我们的方法关注的则是两个视觉上类似的类别的目标的变容。

**Season transfer (Figure 13)** The model is trained on 854 winter photos and 1273 summer photos of Yosemite downloaded from Flickr. 从Flickr上下载了Yosemite的854幅冬天图像和1273幅夏天图像，然后进行了训练。

**Photo generation from paintings (Figure 12)** For painting→photo, we find that it is helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output. In particular, we adopt the technique of Taigman et al. [49] and regularize the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator: i.e., $L_{identity}(G,F) = E_{y∼pdata(y)}[||G(y) − y||_1] + E_{x∼pdata(x)} [||F (x) − x||_1 ]$.

对于绘画到图像的转换，我们发现，引入额外的损失，鼓励保持输入到输出的色彩组成，是很有帮助的。特别是，我们采用了Taigman等的技术，当目标领域的真实样本是生成器的输入时，对生成器进行正则化为一个恒等映射，即：$L_{identity}(G,F) = E_{y∼pdata(y)}[||G(y) − y||_1] + E_{x∼pdata(x)} [||F (x) − x||_1 ]$.

Without $L_{identity}$, the generator G and F are free to change the tint of input images when there is no need to. For example, when learning the mapping between Monet’s paintings and Flickr photographs, the generator often maps paintings of daytime to photographs taken during sunset, because such a mapping may be equally valid under the adversarial loss and cycle consistency loss. The effect of this identity mapping loss are shown in Figure 9.

没有$L_{identity}$，生成器G和F很随意的改变输入图像的色调，而这根本没有任何必要。比如，当学习莫奈的绘画与Flickr的图像的映射时，生成器经常将日间的绘画映射到落日时的照片，因为这样一个映射在对抗损失和循环一致性损失下都是可行的。恒等映射损失的效果如图9所示。

In Figure 12, we show additional results translating Monet’s paintings to photographs. This figure and Figure 9 show results on paintings that were included in the training set, whereas for all other experiments in the paper, we only evaluate and show test set results. Because the training set does not include paired data, coming up with a plausible translation for a training set painting is a nontrivial task. Indeed, since Monet is no longer able to create new paintings, generalization to unseen, “test set”, paintings is not a pressing problem.

在图12中，我们给出了将莫奈的绘画翻译为照片的更多结果。图12和图9给出的结果，其绘画是在训练集上的，而本文中的其他所有试验，我们只给出测试集的结果。因为训练集不包含成对的数据，对训练集的绘画给出一个可行的翻译，这并不是一个容易的任务。确实，因为莫奈无法再创作新的绘画了，泛化到未看到过的测试集绘画，这不是一个很大的问题。

**Photo enhancement (Figure 14)** We show that our method can be used to generate photos with shallower depth of field. We train the model on flower photos downloaded from Flickr. The source domain consists of flower photos taken by smartphones, which usually have deep DoF due to a small aperture. The target contains photos captured by DSLRs with a larger aperture. Our model successfully generates photos with shallower depth of field from the photos taken by smartphones.

我们展示了本文方法用于生成更浅深度野的图像。我们从Flickr上下载花的图像进行模型训练。源领域是手机拍摄的花的照片，通常DoF很深。目标领域是DSLRs拍摄的照片。我们的模型，从手机拍摄的图像中，成功的生成了更浅DoF的图像。

**Comparison with Gatys et al. [13]** In Figure 15, we compare our results with neural style transfer [13] on photo stylization. For each row, we first use two representative artworks as the style images for [13]. Our method, on the other hand, can produce photos in the style of entire collection. To compare against neural style transfer of an entire collection, we compute the average Gram Matrix across the target domain and use this matrix to transfer the “average style” with Gatys et al [13].

在图15中，我们与神经风格迁移的结果进行了比较。对于每一行，我们首先使用两幅代表作作为[13]的风格图像。我们的方法则可以生成整个集合的作品风格的图像。为与整个集合的神经风格迁移进行比较，我们计算了整个目标领域上的平均Gram Matrix，使用这个矩阵来迁移其平均风格。

Figure 16 demonstrates similar comparisons for other translation tasks. We observe that Gatys et al. [13] requires finding target style images that closely match the desired output, but still often fails to produce photorealistic results, while our method succeeds to generate natural-looking results, similar to the target domain.

图16给出来其他翻译任务的类似比较。我们观察到，[13]需要目标风格图像与期望的输出要很接近，但有时候仍然不能得到很真实的结果，但我们的方法总是能成功的得到看起来很自然的结果，与目标领域类似。

## 6. Limitations and Discussion

Although our method can achieve compelling results in many cases, the results are far from uniformly positive. Figure 17 shows several typical failure cases. On translation tasks that involve color and texture changes, like many of those reported above, the method often succeeds. We have also explored tasks that require geometric changes, with little success. For example, on the task of dog→cat transfiguration, the learned translation degenerates into making minimal changes to the input (Figure 17). This failure might be caused by our generator architectures which are tailored for good performance on the appearance changes. Handling more varied and extreme transformations, especially geometric changes, is an important problem for future work.

虽然我们的方法在很多情况下都可以得到很好的结果，但结果还不是在所有情况下都很好。图17给出了一些失败的情况。在涉及到色彩变换和纹理变换的翻译任务中，与上面给出的一样，本文的方法通常会成功。我们还研究了需要几何变换的任务，但没有成功。如，在狗到猫变容的任务中，学习到的翻译只能对输入作出很小的改变。这种失败可能是因为，生成器架构在外表变化上表现非常好。处理变化更多的，更极端的变换，尤其是几何变换，是未来工作的重要问题。

Some failure cases are caused by the distribution characteristics of the training datasets. For example, our method has got confused in the horse → zebra example (Figure 17, right), because our model was trained on the wild horse and zebra synsets of ImageNet, which does not contain images of a person riding a horse or zebra.

一些失败情况是由训练数据集的分布性质导致的。如，在马到斑马的变化中，我们的方法会失败，因为模型的训练，是在ImageNet中的野马和斑马的子集中，这不包含一个人骑着一匹马，或斑马的图像。

We also observe a lingering gap between the results achievable with paired training data and those achieved by our unpaired method. In some cases, this gap may be very hard – or even impossible – to close: for example, our method sometimes permutes the labels for tree and building in the output of the photos→labels task. To resolve this ambiguity may require some form of weak semantic supervision. Integrating weak or semi-supervised data may lead to substantially more powerful translators, still at a fraction of the annotation cost of the fully-supervised systems.

在使用成对训练数据时得到的结果，与使用不成对方法得到的结果，还是有一些差距的。在一些情况下，这个差距可能非常难弥合的，甚至是不可能弥合的：比如，本文的方法有时候，在照片到标签的任务中，在输出中会改变标签的顺序，如树木和建筑物。为解决这个歧义，可能需要一些弱的语义监督。采用弱监督或半监督，会得到强大的多的翻译器，而且标注代价是全监督系统的很小一部分。

Nonetheless, in many cases completely unpaired data is plentifully available and should be made use of. This paper pushes the boundaries of what is possible in this “unsupervised” setting.

尽管如此，在很多情况中，完全不成对的图像是大量可用的，也应该使用这些数据。本文以这种无监督的设置推进了研究前沿。

## 7. Appendix

### 7.1. Training details

We train our networks from scratch, with a learning rate of 0.0002. In practice, we divide the objective by 2 while optimizing D, which slows down the rate at which D learns, relative to the rate of G. We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero over the next 100 epochs. Weights are initialized from a Gaussian distribution N (0, 0.02).

**Cityscapes label↔Photo** 2975 training images from the Cityscapes training set [4] with image size 128 × 128. We used the Cityscapes val set for testing.

### 7.2. Network architectures

We provide both PyTorch and Torch implementations.

**Generator architectures** We adopt our architectures from Johnson et al. [23]. We use 6 residual blocks for 128 × 128 training images, and 9 residual blocks for 256 × 256 or higher-resolution training images. Below, we follow the naming convention used in the Johnson et al.’s Github repository.

Let c7s1-k denote a 7 × 7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1. dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection padding was used to reduce artifacts. Rk denotes a residual block that contains two 3 × 3 con- volutional layers with the same number of filters on both layer. uk denotes a 3 × 3 fractional-strided-Convolution- InstanceNorm-ReLU layer with k filters and stride 1.

The network with 6 residual blocks consists of: c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

The network with 9 residual blocks consists of: c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

**Discriminator architectures** For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, we apply a convo- lution to produce a 1-dimensional output. We do not use InstanceNorm for the first C64 layer. We use leaky ReLUs with a slope of 0.2. The discriminator architecture is: C64-C128-C256-C512