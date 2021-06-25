# BRIEF: Binary Robust Independent Elementary Features

Michael Calonder, Vincent Lepetit, Christoph Strecha, and Pascal Fua

CVLab, EPFL, Lausanne, Switzerland

## 0. Abstract

We propose to use binary strings as an efficient feature point descriptor, which we call BRIEF. We show that it is highly discriminative even when using relatively few bits and can be computed using simple intensity difference tests. Furthermore, the descriptor similarity can be evaluated using the Hamming distance, which is very efficient to compute, instead of the L2 norm as is usually done.

我们提出使用二值串作为一种高效的特征点描述子，我们称之为BRIEF。我们证明了，这是具有高度区分性的，即使使用的少数几个bits，而且可以用简单的灰度差测试进行计算。而且，描述子相似性可以使用Hamming距离进行评估，计算起来非常高效，通常是使用L2范数来进行计算的。

As a result, BRIEF is very fast both to build and to match. We compare it against SURF and U-SURF on standard benchmarks and show that it yields a similar or better recognition performance, while running in a fraction of the time required by either.

结果是，BRIEF构建和匹配起来非常快速。我们将其与SURF和U-SURF在标准基准测试中进行了比较，表明其得到了类似的或更好的识别性能，而运行时间只是其一小部分。

## 1. Introduction

Feature point descriptors are now at the core of many Computer Vision technologies, such as object recognition, 3D reconstruction, image retrieval, and camera localization. Since applications of these technologies have to handle ever more data or to run on mobile devices with limited computational resources, there is a growing need for local descriptors that are fast to compute, fast to match, and memory efficient.

特征点描述子现在是很多计算机视觉技术的核心，比如目标识别，3D重建，图像检索，和相机定位。由于这些技术的应用需要处理非常多的数据，或在计算资源有限的移动设备上运行，所以非常需要可以快速计算，快速匹配，占用内存很少的局部描述子。

One way to speed up matching and reduce memory consumption is to work with short descriptors. They can be obtained by applying dimensionality reduction, such as PCA [1] or LDA [2], to an original descriptor such as SIFT [3] or SURF [4]. For example, it was shown in [5–7] that floating point values of the descriptor vector could be quantized using very few bits per value without loss of recognition performance. An even more drastic dimensionality reduction can be achieved by using hash functions that reduce SIFT descriptors to binary strings, as done in [8]. These strings represent binary descriptors whose similarity can be measured by the Hamming distance.

一种加速匹配和降低内存消耗的方法，是使用短描述子。这可以使用降维来得到，比如对SIFT或SURF这样的原始描述子使用PCA或LDA。比如，[5-7]中展示了，浮点值的描述子向量可以量化到很少的bits，而不损失识别性能。一种更加极端的降维方式，可以使用hash函数得到，将SIFT描述子降到二值串，如[8]。这些串表示二值描述子，其相似性可以用Hamming距离度量。

While effective, these approaches to dimensionality reduction require first computing the full descriptor before further processing can take place. In this paper, we show that this whole computation can be shortcut by directly computing binary strings from image patches. The individual bits are obtained by comparing the intensities of pairs of points along the same lines as in [9] but without requiring a training phase. We refer to the resulting descriptor as BRIEF.

这些方法虽然很有效，但是仍然需要首先计算完整的描述子，然后才能进一步处理。本文中，我们展示了这整个计算可以简化，直接从图像块中计算二值串。通过比较同样直线上的点对亮度，就可以得到这些单独的bits，就像[9]一样，但是不需要训练阶段。我们称得到的描述子为BRIEF。

Our experiments show that only 256 bits, or even 128 bits, often suffice to obtain very good matching results. BRIEF is therefore very efficient both to compute and to store in memory. Furthermore, comparing strings can be done by computing the Hamming distance, which can be done extremely fast on modern CPUs that often provide a specific instruction to perform a XOR or bit count operation, as is the case in the latest SSE [10] instruction set.

我们的试验表明，只用256 bits，甚至是128 bits，就足以得到很好的匹配结果。BRIEF因此在计算和在内存中存储都非常高效。而且，比较串可以通过计算Hamming距离得到，在现代CPUs中这可以极快的计算得到。

This means that BRIEF easily outperforms other fast descriptors such as SURF and U-SURF in terms of speed, as will be shown in the Results section. Furthermore, it also outperforms them in terms of recognition rate in many cases, as we will demonstrate using benchmark datasets.

这意味着，BRIEF可以很轻松的在速度上超过其他的快速描述子，如SURF和U-SURF，这在结果这一节中可以看到。而且，在很多情况中，识别率也超过了这些模型，我们会使用基准测试数据集证明。

## 2. Related Work

The SIFT descriptor [3] is highly discriminant but, being a 128-vector, is relatively slow to compute and match. This can be a drawback for real-time applications such as SLAM that keep track of many points as well as for algorithms that require storing very large numbers of descriptors, for example for large-scale 3D reconstruction.

SIFT描述子是具有高度区分性的，但是，作为一个128维的向量，在计算和匹配上都相对较慢。对实时应用来说，这可以是一个缺陷，这可以是一个缺陷，比如SLAM，跟踪很多点，以及需要存储大量描述子的算法，比如大规模3D重建。

There are many approaches to solving this problem by developing faster to compute and match descriptors, while preserving the discriminative power of SIFT. The SURF descriptor [4] represents one of the best known ones. Like SIFT, it relies on local gradient histograms but uses integral images to speed up the computation. Different parameter settings are possible but, since using only 64 dimensions already yields good recognition performances, that version has become very popular and a de facto standard. This is why we compare ourselves to it in the Results section.

有很多求解这个问题的方法，提出计算和匹配很快的算子，同时保持SIFT的区分能力。SURF描述子是这其中最有名的一种。与SIFT相似，SURF依赖于局部梯度直方图，但使用积分图像来加速计算。可以有不同的参数设置，但由于只使用了64维就已经得到了很好的识别性能，所以这个版本就已经很流行了，成为了事实上的标准。这也是我们为什么在结果小节中与其进行比较。

SURF addresses the issue of speed but, since the descriptor is a 64-vector of floating points values, representing it still requires 256 bytes. This becomes significant when millions of descriptors must be stored. There are three main classes of approaches to reducing this number.

SURF处理了速度的问题，但是，由于描述子是64维的浮点数，表示仍然需要256 bytes。这在需要存储数百万描述子时就成为了明显的问题。降低这个数，有三种主要的方法。

The first involves dimensionality reduction techniques such as Principal Component Analysis (PCA) or Linear Discriminant Embedding (LDE). PCA is very easy to perform and can reduce descriptor size at no loss in recognition performance [1]. By contrast, LDE requires labeled training data, in the form of descriptors that should be matched together, which is more difficult to obtain. It can improve performance [2] but can also overfit and degrade performance.

第一种是降维方法，比如PCA或LDE。PCA很容易进行，而且可以在不损失识别性能的时候，降低描述子大小[1]。对比起来，LDE需要标注的训练数据，需要描述子可以被匹配到一起的数据，这很难得到。LDE可以改进性能，但是也会过拟合，导致性能损失。

A second way to shorten a descriptor is to quantize its floating-point coordinates into integers coded on fewer bits. In [5], it is shown that the SIFT descriptor can be quantized using only 4 bits per coordinate. Quantization is used for the same purpose in [6, 7]. It is a simple operation that results not only in memory gain but also in faster matching as computing the distance between short vectors can then be done very efficiently on modern CPUs. In [6], it is shown that for some parameter settings of the DAISY descriptor, PCA and quantization can be combined to reduce its size to 60 bits. However, in this approach the Hamming distance cannot be used for matching because the bits are, in contrast to BRIEF, arranged in blocks of four and hence cannot be processed independently.

第二种缩短描述子的方法是，将浮点数坐标进行量化，成为在少量bits上的整数编码。在[5]表明，SIFT可以量化成每个坐标只使用4 bits。量化在[6,7]中也进行了相同的使用。这是一个简单的运算，不仅存储更加高效，而且匹配更加快速，因为计算短向量之间的距离在现代CPUs中可以很高效的计算。在[6]中展示了，对DAISY描述子的一些参数设置，PCA和量化可以一起使用，将大小降低到60 bits。但是，在这种方法中，Hamming距离不能用于匹配，因为这些bits是以4的模块来安排的，不能独立处理，而BRIEF则形成了对比。

A third and more radical way to shorten a descriptor is to binarize it. For example, [8] drew its inspiration from Locality Sensitive Hashing (LSH) [11] to turn floating-point vectors into binary strings. This is done by thresholding the vectors after multiplication with an appropriate matrix. Similarity between descriptors is then measured by the Hamming distance between the corresponding binary strings. This is very fast because the Hamming distance can be computed very efficiently with a bitwise XOR operation followed by a bit count. The same algorithm was applied to the GIST descriptor to obtain a binary description of an entire image [12]. Another way to binarize the GIST descriptor is to use nonlinear Neighborhood Component Analysis [12, 13], which seems more powerful but probably slower at run-time.

第三种更加激进的缩短描述子方法，是将其二值化。比如，[8]从LSH[11]得到灵感，将浮点数向量转化成二值串。首先与一个合适的矩阵进行相乘，然后对向量使用阈值，就可以做到。描述子之间的相似度，是通过对应的二值串的Hamming距离来进行度量的。这非常快速，因为Hamming距离可以进行非常高效的计算，即用逐个bit的XOR运算和bit计数进行。同样的算法应用到了GIST描述子，得到了整幅图像的二值描述[12]。另一种二值化GIST描述子的方法是，使用非线性NCA[12,13]，似乎非常强力，但在运行时可能更慢。

While all three classes of shortening techniques provide satisfactory results, relying on them remains inefficient in the sense that first computing a long descriptor then shortening it involves a substantial amount of time-consuming computation. By contrast, the approach we advocate in this paper directly builds short descriptors by comparing the intensities of pairs of points without ever creating a long one. Such intensity comparisons were used in [9] for classification purposes and were shown to be very powerful in spite of their extreme simplicity. Nevertheless, the present approach is very different from [9] and [14] because it does not involve any form of online or offline training.

所有三种缩短描述子的技术都给出了令人满意的结果，但靠这些仍然效率不高，因为需要首先计算一个长的描述子，然后对其进行缩短，这需要很长的时间。对比起来，我们在本文中推出的方法，直接构建了短描述子，比较了点对之间的灰度，不需要创建一个长的描述子。这种亮度比较用在[9]中作为分类方法，虽然极其简单，但已经证明了非常强力。尽管如此，提出的方法与[9,14]是非常不一样的，因为不需要任何形式的在线或离线训练。

## 3. Method

Our approach is inspired by earlier work [9, 15] that showed that image patches could be effectively classified on the basis of a relatively small number of pairwise intensity comparisons. The results of these tests were used to train either randomized classification trees [15] or a Naive Bayesian classifier [9] to recognize patches seen from different viewpoints. Here, we do away with both the classifier and the trees, and simply create a bit vector out of the test responses, which we compute after having smoothed the image patch.

我们的方法是受到[9,15]的早期工作启发的，他们证明了，以较小数量的成对亮度对比为基础，图像块就可以被有效的分类。这些测试的结果，用于训练随机分类树[15]，或朴素贝叶斯分类器[9]，识别不同视角看到的图像块。这里，我们既采用了分类器，也采用了树，简单的从这些测试响应中创建了一个bit向量，在平滑了图像块之后进行这样的计算。

More specifically, we define test τ on patch p of size S × S as 更具体的，我们在大小为S × S的图像块p上定义测试τ

$$τ(p; x, y) := \left\{ \begin{matrix} 1 & if p(x)<p(y) \\ 0 & otherwise \end{matrix} \right.$$(1)

where p(x) is the pixel intensity in a smoothed version of p at $x = (u, v)^⊤$. Choosing a set of nd (x, y)-location pairs uniquely defines a set of binary tests. We take our BRIEF descriptor to be the nd-dimensional bitstring

其中p(x)是p经过平滑后在$x = (u, v)^⊤$处的灰度。选择nd个(x,y)位置对的集合，就定义了二值测试的集合。我们的BRIEF描述子就是nd维的bit串

$$f_{n_d} (p) := \sum_{1≤i≤n_d} 2^{i-1} τ(p; x_i, y_i)$$(2)

In this paper we consider nd = 128, 256, and 512 and will show in the Results section that these yield good compromises between speed, storage efficiency, and recognition rate. In the remainder of the paper, we will refer to BRIEF descriptors as BRIEF-k, where k = nd/8 represents the number of bytes required to store the descriptor.

本文中，我们考虑nd = 128, 256, 512，在结果小节中展示了，在速度，存储效率和识别率之间得到了很好的折中。在本文剩下的篇幅中，我们称BRIEF描述子为BRIEF-k，其中k=nd/8，表示存储描述子所需的bytes数量。

When creating such descriptors, the only choices that have to be made are those of the kernels used to smooth the patches before intensity differencing and the spatial arrangement of the (x, y)-pairs. We discuss these in the remainder of this section.

当创建这种描述子时，需要做的选择，只有两个，一个是在亮度对比前平滑图像块所需的滤波核，和(x,y)对的空间安排。我们在本节剩下的篇幅中讨论。

To this end, we use the Wall dataset that we will describe in more detail in section 4. It contains five image pairs, with the first image being the same in all pairs and the second image shot from a monotonically growing baseline, which makes matching increasingly more difficult. To compare the pertinence of the various potential choices, we use as a quality measure the recognition rate in image pairs that will be precisely defined at the beginning of section 4. In short, for both images of a pair and for a given number of corresponding keypoints between them, it quantifies how often the correct match can be established using BRIEF for description and the Hamming distance as the metric for matching. This rate can be computed reliably because the scene is planar and the homography between images is known. It can therefore be used to check whether points truly correspond to each other or not.

为此，我们使用Wall数据集，在第4部分进行详述。数据集包含5个图像对，在所有图像对中，第一幅图像都是一样的，第二幅图像是从逐渐单调增长的基准上拍摄的，这使得匹配越来越难。为比较各种可能的选择的相关性，我们使用图像对的识别率作为质量度量，在第4部分的开始进行精确定义。简短来说，对于一对图像中的两幅，对它们之间给定数量的对应关键点，使用BRIEF进行描述，使用Hamming距离作为匹配的度量，识别率量化了确立正确的匹配的比率。这个比率可以很可靠的计算，因为场景是平面的，图像之间的对应关系是已知的。因此可以用于检查这些点是否真正的对应彼此。

### 3.1. Smoothing Kernels

By construction, the tests of Eq. 1 take only the information at single pixels into account and are therefore very noise-sensitive. By pre-smoothing the patch, this sensitivity can be reduced, thus increasing the stability and repeatability of the descriptors. It is for the same reason that images need to be smoothed before they can be meaningfully differentiated when looking for edges. This analogy applies because our intensity difference tests can be thought of as evaluating the sign of the derivatives within a patch.

在构建的时候，式1的测试只考虑了单个像素上的信息，因此对噪声非常敏感。通过对图像块进行预平滑，这种敏感性可以降低，因此增加了描述子的稳定性和重复性。基于同样的原因，图像在有意义的比较寻找边缘之前，需要被平滑。这种类比是可以应用的，因为我们的灰度差异测试可以被认为是，评估在一个图像块中的导数符号。

Fig. 1 illustrates the effects of increasing amounts of Gaussian smoothing on the recognition rates for variances of Gaussian kernel ranging from 0 to 3. The more difficult the matching, the more important smoothing becomes to achieving good performance. Furthermore, the recognition rates remain relatively constant in the 1 to 3 range and, in practice, we use a value of 2. For the corresponding discrete kernel window we found a size of 9×9 pixels be necessary and sufficient.

图1描述了增加高斯滤波的程度对识别率的影响，高斯滤波核的方差从0增加到了3。匹配越难，平滑变得就越重要，可以获得更好的性能。而且，在1-3的范围内，识别率基本保持一致，所以在实践中，我们使用的值为2。对于对应的离散核窗口，我们发现大小9x9是充分而且必须的。

### 3.2. Spatial Arrangement of the Binary Tests

Generating a length nd bit vector leaves many options for selecting the nd test locations (xi, yi) of Eq. 1 in a patch of size S × S. We experimented with the five sampling geometries depicted by Fig. 2. Assuming the origin of the patch coordinate system to be located at the patch center, they can be described as follows.

生成长度为nd bit向量，在大小为SxS的图像块中，选择式1中nd个测试位置(xi, yi)，有很多选项。我们采用5种采样几何进行试验，如图2所示。假设图像块坐标系统的原点在图像块中心，可以描述如下。

I) (X, Y) ∼ i.i.d. Uniform(−S/2, S/2): The (xi, yi) locations are evenly distributed over the patch and tests can lie close to the patch border. 位置在图像块中均匀分布，可以在图像块边缘的附近。

II) (X, Y) ∼ i.i.d. Gaussian(0, S^2/25): The tests are sampled from an isotropic Gaussian distribution. Experimentally we found S/2 = 5/2σ ⇔ σ^2 = S^2/25 to give best results in terms of recognition rate. 测试是从各向同性的高斯分布中采样得到的。通过试验发现，S/2 = 5/2σ ⇔ σ^2 = S^2/25会得到了最好的识别率结果。

III) X ∼ i.i.d. Gaussian(0, S^2/25) , Y ∼ i.i.d. Gaussian(xi, S^2/100) : The sampling involves two steps. The first location xi is sampled from a Gaussian centered around the origin while the second location is sampled from another Gaussian centered on xi. This forces the tests to be more local. Test locations outside the patch are clamped to the edge of the patch. Again, experimentally we found S/4 = 5/2σ ⇔ σ^2 = S^2/100 for the second Gaussian performing best. 采样包括两个步骤。第一个位置xi是从以原点为中心的高斯函数中采样的，而第二个位置是从另一个以xi为中心的高斯函数中采样得到的。这使得测试更加局部。在图像块之外的测试位置被缩到图像块的边缘处。我们再次通过试验发现，第一个高斯分布的参数为S/4 = 5/2σ ⇔ σ^2 = S^2/100会得到最好的结果。

IV) The (xi, yi) are randomly sampled from discrete locations of a coarse polar grid introducing a spatial quantization. 从粗糙的极坐标网格中随机采样出离散的位置点。

V) ∀i : xi = (0, 0)^⊤ and yi is takes all possible values on a coarse polar grid containing nd points. xi固定为原点，yi是从粗糙的极坐标网格点上采样得到。

For each of these test geometries we compute the recognition rate and show the result in Fig. 3. Clearly, the symmetrical and regular G V strategy loses out against all random designs G I to G IV, with G II enjoying a small advantage over the other three in most cases. For this reason, in all further experiments presented in this paper, it is the one we will use.

对每种测试几何，我们都计算了识别率，在图3中给出了结果。很明显，对称和规则的GV策略识别率很低，比不上GI到GIV的随机设计，而GII在多数情况中都略有优势。基于此原因，在本文所有进一步的试验中，我们都使用GII的设计。

### 3.3. Distance Distributions

In this section, we take a closer look at the distribution of Hamming distances between our descriptors. To this end we extract about 4000 matching points from the five image pairs of the Wall sequence. For each image pair, Fig. 4 shows the normalized histograms, or distributions, of Hamming distances between corresponding points (in blue) and non-corresponding points (in red). The maximum possible Hamming distance being 32 · 8 = 256 bits, unsurprisingly, the distribution of distances for non-matching points is roughly Gaussian and centered around 128. As could also be expected, the blue curves are centered around a smaller value that increases with the baseline of the image pairs and, therefore, with the difficulty of the matching task.

本节中，我们仔细观察一下我们的描述子之间的Hamming距离的分布。为此，我们从Wall序列中的5个图像对中提取了大约4000个匹配点。对每个图像对，图4给出了对应点（蓝色）和非对应点（红色）的Hamming距离的归一化的直方图，或分布。最大可能的Hamming距离为32 · 8 = 256 bits，非匹配点的距离分布是大约高斯的，以128为中心，这并不令人惊讶。还可以期待的是，蓝色曲线以更小的值为中心，并对着图像对的基准而增加，也就是匹配任务的难度。

Since establishing a match can be understood as classifying pairs of points as being a match or not, a classifier that relies on these Hamming distances will work best when their distributions are most separated. As we will see in section 4, this is of course what happens with recognition rates being higher in the first pairs of the Wall sequence than in the subsequent ones.

由于确立一个匹配可以理解成，将点对分类成匹配或不匹配，依赖于Hamming距离的分类器，当其分布最能分开的情况下，会效果最好。我们在第4部分会看到，在Wall序列的第一对中，识别率更高，在随后的对中更低一些，这是理所当然的。

## 4. Results

In this section, we compare our method against several competing approaches. Chief among them is the latest OpenCV implementation of the SURF descriptor [4], which has become a de facto standard for fast-to-compute descriptors. We use the standard SURF64 version, which returns a 64-dimensional floating point vector and requires 256 bytes of storage. Because BRIEF, unlike SURF, does not correct for orientation, we also compare against U-SURF [4], where the U stands for upright and means that orientation also is ignored [4].

本节中，我们与其他几种方法进行了比较。主要是OpenCV最新的SURF描述子实现[4]，这已经是快速计算描述子的一个事实上的标准了。我们使用标准的SURF64版本，会返回一个64维的浮点向量，需要256 bytes存储。因为BRIEF与SURF不同，并没有对方向进行修正，我们还与U-SURF进行了比较，其中U表示竖直，意思是方向可以忽略。

To this end, we use the six publicly available test image sequences depicted by Fig. 5. They are designed to test robustness to 为此，我们使用了6个公开可用的测试图像序列，如图5所示。它们设计是用于测试对下面变化的稳健性

- viewpoint changes (Wall, Graffiti, Fountain), 视角变化
- compression artifacts (Jpeg), 压缩带来的失真
- illumination changes (Light), 光照变化
- and image blur (Trees). 图像模糊

For each one, we consider 5 image pairs by matching the first of 6 images to the other five they contain. Note that, the Wall and Graffiti scenes being planar, the images are related by homographies that are used to compute the ground truth. The viewpoints for Jpeg, Light, and Trees being almost similar, the images are also taken to be related by a homography, which is very close to being the identity. By contrast, the Fountain scene is fully three-dimensional and the ground truth is computed from laser scan data. Note also that the 5 pairs in Wall and Fountain are sorted in order of increasing baseline so that pair 1|6 is much harder to match than pair 1|2, which negatively affects the performance of all the descriptors considered here.

对每个图像序列，我们都考虑5个图像对，将6幅图像的第一个与其他5幅图像进行匹配。注意，Wall和Graffiti的场景是平面的，图像是通过单应性关联起来的，用于计算真值。对Jpeg，Light和Trees的视角是类似的，图像也通过单应性进行关联。对比起来，Fountain场景是全3D的，真值是通过激光扫描数据计算得到的。注意，在Wall和Fountain中的5个对是以增加基准的方式排序的，所以1|6对比1|2对要更难匹配，这对这里考虑的所有描述子的性能都有负面的影响。

For evaluation purposes, we rely on two straightforward metrics, elapsed CPU time and recognition rate. The former simply is averaged measured wall clock time over many repeated runs. Given an image pair, the latter is computed as follows: 为了评估的目的，我们计算了两种直接的度量，即计算CPU耗时和识别率。前者就是简单的在很多次重复的运行时间进行平均。给定一个图像对，后者的计算如下：

- Pick N interest points from the first image, infer the N corresponding points in the other from the ground truth data, and compute the 2N associated descriptors using the method under consideration. 从第一幅图像中选择N个兴趣点，从真值数据中在另外一幅图像中推理得到N个对应点，使用考虑的方法计算2N个相关的描述子。

- For each point in the first set, find the nearest neighbor in the second one and call it a match. 对第一个集合中的每个点，在第二个集合中找到最接近的邻域，称之为一个匹配。

- Count the number of correct matches n_c and take the recognition rate to be r = n_c/N. 计算正确匹配的数量nc，识别率为r=n_c/N。

Although this may artificially increase the recognition rates, only the absolute recognition rate numbers are to be taken with caution. Since we apply the same procedure to all descriptors, and not only ours, the relative rankings we obtain are still valid and speak in BRIEF’s favor. To confirm this, we detected SURF-points in both images of each test pair and computed their (SURF- or BRIEF-) descriptors, matched these descriptors to their nearest neighbor, and applied a standard left-right consistency check. Even though this setup now involves outliers, BRIEF continued to outperform SURF in the same proportion as before.

虽然这可以人为的增加识别率，只有绝对识别率数量是要小心对待的。由于我们将同样的过程应用到所有的描述子中，并不只是我们的，我们得到的相对排名仍然是有效的，是支持BRIEF的。为确认，我们在每个测试对的两幅图像中检测到SURF点，并计算其SURF或BRIEF描述子，将这些描述子与其最接近的邻居进行匹配，并应用了一个标准的左-右一致性检查。即使这个设置现在会有一些离群点，BRIEF在同样的比例中，会一直超过SURF的表现。

In the remainder of this section we will use these metrics to show that the computational requirements of BRIEF are much lower than those of all other methods while achieving better recognition rates than SURF on all sequences except Graffiti. This is explained both by the fact that this dataset requires strong rotation invariance, which BRIEF does not provide, and by the very specific nature of the Graffiti images. They contain large monochrome areas on which our intensity difference tests are often uninformative. In other words, this data set clearly favors descriptors based on gradient histograms, as has already been noted [7]. When comparing recognition rates against those of U-SURF, BRIEF still does better on Wall, Fountain, Trees, and similarly on Light and Jpeg.

在本节剩余篇幅中，我们会使用这些度量来展示BRIEF的计算需求比所有其他方法都要低很多，同时在除了Graffiti的所有序列中都比SURF获得了更好的识别率。因为这个数据集需要很强的旋转不变性，而BRIEF没有提供，而且Graffiti图像有很特有的本质，包含很大的单色区域，我们的灰度差异测试通常不包含任何信息。换句话说，这个数据集很明显支持基于梯度直方图的描述子，这一点[7]已经指出来了。当比较与U-SURF的识别率，BRIEF仍然在Wall，Fountain，Trees要好，在Light和Jpeg上性能类似。

In other words, on data sets such as those that involve only modest amounts of in-plane rotation, there is a cost not only in terms of speed but also of recognition rate to achieving orientation invariance, as already pointed out in [4]. This explains in part why both BRIEF and U-SURF outperform SURF. Therefore, when not actually required, orientation correction should be avoided. This is an important observation because there are more and more cases, such as when using a mobile phone equipped with an orientation sensor, when orientation invariance stops being a requirement. This is also true in many urban contexts where photos tend to be taken at more or less canonical orientations and for mobile robotics where the robot’s attitude is known.

换句话说，在那些平面内旋转不太剧烈的数据集上，要获得方向不变性，其代价只是计算速度上的，还有识别率，[4]中已经指出了这一点。这部分解释了BRIEF和U-SURF为什么超过了SURF。因此，当并不真正需要时，方向修正应当避免。这是一个很重要的观察，因为有越来越多的情况，比如使用带有方向传感器的手机，这时方向不变性就不是一种必须。这在很多城市环境中也是成立的，图像的拍摄一般都是经典方向，对于移动机器人也是成立的，因为机器人的朝向是已知的。

**Recognition Rate as a Function of Descriptor Size**. Since many practical problems involve matching a few hundred feature points, we first use N = 512 to compare the recognition rate of BRIEF using either 128, 256, or 512 tests, which we denote as BRIEF-16, BRIEF-32, and BRIEF-64. The trailing number stands for the number of bytes required to store the descriptor. Recall that since both SURF and U-SURF return 64 floating point numbers, they require 256 bytes of storage and are therefore at least four times bigger.

识别率作为描述子大小的函数。由于很多实际问题涉及到匹配几百个特征点，我们首先使用N=512来比较使用128，256或512个测试的BRIEF的识别率，我们表示为BRIEF-16，BRIEF-32，BRIEF-64。后面的数字代表存储描述子所需的bytes的数量。回忆一下，由于SURF和U-SURF返回的都是64位浮点数，他们需要256 bytes的存储，因此至少要大了4倍。

As shown in Fig. 6, BRIEF-64 outperforms SURF and U-SURF in all sequences except Graffiti while using a descriptor that is four times smaller. Unsurprisingly, BRIEF-32 does not do quite as well but still compares well against SURF and U-SURF. BRIEF-16 is too short and shows the limits of the approach.

如图6所示，BRIEF-64在除了Graffiti的所有序列中都超过了SURF和U-SURF，而使用的描述子小了4倍。BRIEF-32的效果没有那么好，但与SURF和U-SURF也相当，这不令人惊讶。BRIEF-16太短了，这是一个局限。

To show that this behavior is not an artifact for the number N of feature points used for testing purposes, we present similar recognition rates for values of N ranging from 512 to 4096 in Fig. 7. As could be expected, the recognition rates drop as N increases for all the descriptors but the rankings remain unchanged.

为证明这个行为对于为测试目的所用的特征点数量N并不是一个人为现象，我们给出了对于值N在512到4096的范围内，类似的识别率，如图7所示。就像预期的一样，当N增加时，识别率在下降，对于所有的描述子都是一样的，但排名保持不变。

In Fig. 8, we use the wall data set to plot recognition rates as a function of the number of tests. We clearly see a saturation effect beyond 200 tests for the easy cases and an improvement up to 512 for the others. This tallies with the results of Fig. 6 showing that BRIEF-32 (256 tests) yields near optimal results for the short baseline pairs and that BRIEF-64 (512 tests) is more appropriate for the others.

在图8中，我们使用Wall数据集，画出了识别率作为测试点数量的函数。我们很明显看到，对于简单的情况，在超过200个测试后，就出现了饱和现象，对于其他情况，在超过512时才出现饱和现象。这与图6的结果是一致的，即BRIEF-32（256个测试）对于短基准得到了接近最佳的结果，而BRIEF-64（512测试）对于其他的结果更加合适。

**Influence of Feature Detector**. To perform the experiments described above, we used SURF keypoints so that we could run both SURF, U-SURF, and BRIEF on the same points. This choice was motivated by the fact that SURF requires an orientation and a scale and U-SURF a scale, which the SURF detector provides.

特征检测器的影响。为进行上面所述的试验，我们使用SURF关键点，这样我们可以在相同的点上运行SURF，U-SURF和BRIEF。这个选择是受到下面的事实推动，即SURF需要一个方向和一个尺度，而U-SURF需要一个尺度，而SURF检测器会提供。

However, in practice, using the SURF detector in conjunction with BRIEF would negate part of the considerable speed advantage that BRIEF enjoys over SURF. It would make much more sense to use a fast detector such as [16]. To test the validity of this approach, we therefore recomputed our recognition rates on the Wall sequence using CenSurE keypoints instead of SURF keypoints. As can be seen in Fig. 9-left, BRIEF works even slightly better for CenSurE points than for SURF points.

但是，在实际中，使用SURF检测器与BRIEF一起会使BRIEF相对于SURF的速度优势没那么明显。使用一个快速检测器，如[16]，会有意义的多。为测试这种方法的有效性，我们因此重新计算了在Wall序列上使用CenSureE关键点，而不是SURF关键点的识别率。如图9左所示，BRIEF在使用CenSurE点时候，效果比SURF点效果甚至要好一些。

**Orientation Sensitivity**. BRIEF is not designed to be rotationally invariant. Nevertheless, as shown by our results on the 5 test data sets, it tolerates small amounts of rotation. To quantify this tolerance, we take the first image of the Wall sequence with N = 512 points and match these against points in a rotated version of itself, where the rotation angle ranges from 0 to 180 degrees.

方向敏感性。BRIEF并不是旋转不变的。尽管如此，我们在5个测试数据集上的结果表明，对于小幅度的旋转，是会有相当的容忍度的。为量化这种容忍度，我们在Wall序列第一幅图像中取N=512个点，将其与旋转版的图像进行匹配，旋转角度从0度到180度。

Fig. 9-right depicts the recognition rate of BRIEF-32, SURF, and U-SURF. Since the latter does not correct for orientation either, its behavior is very similar or even a bit worse than that of BRIEF: Up to 10 to 15 degrees, there is little degradation followed by a precipitous drop. SURF, which attempts to compensate for orientation changes, does better for large rotations but worse for small ones, highlighting once again that orientation-invariance comes at a cost.

图9右给出了BRIEF-32，SURF和U-SURF的识别率。由于后者并没有修正方向，其行为非常类似，甚至比BRIEF要差一些：在10到15度范围内，几乎没有多少性能下降，然后就有很陡峭的下降。SURF本身补偿了方向变化，在大的旋转上表现很好，但对于小的旋转则略差，再次凸显了方向不变性是有一定代价的。

To complete the experiment, we plot a fourth curve labeled as O-BRIEF-32, where the “O” stands for orientation correction. In other words, we run BRIEF-32 on an image rotated using the orientation estimated by SURF. O-BRIEF-32 is not meant to represent a practical approach but to demonstrate that the response to in-plane rotations is more a function of the quality of the orientation estimator rather than of the descriptor itself, as evidenced by the fact that O-BRIEF-32 and SURF are almost perfectly superposed.

为完成试验，我们画出了第四条曲线，我们称之为O-BRIEF-32，其中O代表方向修正。换句话说，我们用SURF估计方向，对图像进行旋转，然后运行BRIEF-32。O-BRIEF-32并不表示一个实际的方法，但可以证明，对平面内旋转的响应更多的是方向估计质量的功能，而不是描述子本身的，在图中O-BRIEF-32和SURF几乎是完美重合的，这就是证据。

**Estimating Speed**. In a practical setting where either speed matters or computational resources are limited, not only should a descriptor exhibit the highest possible recognition rates but also be computationally as cheap as possible. Matching a number of points between two images typically involves three steps:

估计速度。在一个实际的设置中，速度非常重要，或计算资源受限，一个描述子不仅应当给出最高的识别率，还要计算速度尽量的快。在两幅图像中匹配一定数量的点，一般有三个步骤：

1) Detecting the feature points. 检测到特征点。

2) Computing the description vectors. 计算描述向量。

3) Matching, which means finding the nearest neighbor in descriptor space. 匹配，意思是在描述子空间找到最接近的邻居。

For affine-invariant methods such as SURF, the first step can involve a costly scale-space search for local maxima. In the case of BRIEF, any fast detector such as CenSurE [16] or FAST [17] can be used. BRIEF is therefore at an advantage there. 对于仿射不变的方法，如SURF，第一步会在尺度空间搜索中寻找局部极值，计算量很大。在BRIEF的情况中，任何快速检测器，比如CenSurE或FAST都可以进行使用。BRIEF因此在这里有优势。

The following table gives timing results for the second and third steps for 512 keypoints, measured on a 2.66 GHz/Linux x86-64 machine, in milliseconds: 下表给出了512个关键点的第二和第三步骤的计时结果，单位为毫秒：

| | BRIEF-16 | BRIEF-32 | BRIEF-64 | SURF-64
--- | --- | --- | --- | ---
Descriptor computation | 8.18 | 8.87 | 9.57 | 335
Matching (exact NN) | 2.19 | 4.35 | 8.16 | 28.3

As far as building the descriptors is concerned, we observe a 35- to 41-fold speed-up over SURF where the time for performing and storing the tests remains virtually constant. U-SURF being about 1/3 faster than SURF [4], the equivalent number should be an 23- to 27-fold speed increase. Because BRIEF spends by far the most CPU time with smoothing, approximate smoothing techniques based on integral images may yield extra speed. For matching, we observe a 4- to 13-fold speed-up over SURF. The matching time scales quadratically with the number of bits used in BRIEF but the absolute values remain extremely low within the useful range. Furthermore, in theory at least, these computation times could be driven almost to zero using the POPCNT instruction from SSE4.2 [10]. Because only the latest Intel Core i7 CPUs support this instruction, we were unable to exploit it and used a straight-forward SSE2/SSE4.1 implementation instead.

在构建描述子上，我们观察到速度比SURF提升了35到41倍，而进行测试和保存测试的时间基本上是常数。U-SURF比SURF快了1/3，等价的是速度提升了23到27倍。因为BRIEF的大多数CPU时间用在平滑上，近似平滑的技术可能会有额外的提速，如基于积分图像的技术。对于匹配，我们观察到比SURF有4倍到13倍的加速。匹配时间随着BRIEF中bits数量有平方式的提高，但其绝对值仍然非常低。而且，至少在理论上，这些计算时间使用SSE4.2中的POPCNT指令，可以基本上降低到0。因为只有最新的Intel Core i7 CPUs才支持这种指令，我们不能进行利用，只使用了SSE2/SSE4.1的实现来替代。

## 5. Conclusion

We have introduced the BRIEF descriptor that relies on a relatively small number of intensity difference tests to represent an image patch as a binary string. Not only is construction and matching for this descriptor much faster than for other state-of-the-art ones, it also tends to yield higher recognition rates, as long as invariance to large in-plane rotations is not a requirement.

我们提出了BRIEF描述子，依靠少量灰度差值测试来将一个图像块表示为一个二值串。描述子的构建和匹配都比目前最好的快了很多，而且识别率也会更高一些，不需要对很大的平面内旋转具有不变性。

It is an important result from a practical point of view because it means that real-time matching performance can be achieved even on devices with very limited computational power. It is also important from a more theoretical viewpoint because it confirms the validity of the recent trend [18, 12] that involves moving from the Euclidean to the Hamming distance for matching purposes.

从实际角度来说这很重要，因为这意味着在计算能力很有限的设备上也可以达到实时匹配的性能。从更加理论的视角来说也很重要，因为这确认了最近匹配的趋势，从欧式距离转移到了Hamming距离。

In future work, we will incorporate orientation and scale invariance into BRIEF so that it can compete with SURF and SIFT in a wider set of situations. Using fast orientation estimators, there is no theoretical reason why this could not be done without any significant speed penalty.

在未来的工作中，我们会将方向和尺度不变性集成到BRIEF中，这样可以在更多情况下与SURF和SIFT进行竞争。使用快速方向估计器，理论上应当不会对速度有明显的损失。