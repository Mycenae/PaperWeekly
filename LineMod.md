# Gradient Response Maps for Real-Time Detection of Textureless Objects

Stefan Hinterstoisser et. al. 

## 0. Abstract

We present a method for real-time 3D object instance detection that does not require a time-consuming training stage, and can handle untextured objects. At its core, our approach is a novel image representation for template matching designed to be robust to small image transformations. This robustness is based on spread image gradient orientations and allows us to test only a small subset of all possible pixel locations when parsing the image, and to represent a 3D object with a limited set of templates. In addition, we demonstrate that if a dense depth sensor is available we can extend our approach for an even better performance also taking 3D surface normal orientations into account. We show how to take advantage of the architecture of modern computers to build an efficient but very discriminant representation of the input images that can be used to consider thousands of templates in real time. We demonstrate in many experiments on real data that our method is much faster and more robust with respect to background clutter than current state-of-the-art methods.

我们提出了一种方法进行3D目标实例实时检测，这种方法不需要非常耗时的训练阶段，可以处理没有纹理的目标。我们方法的核心是，一种新的图像表示进行模板匹配，其设计对小的图像变换是稳健的。这种稳健性是基于大量的图像梯度方向，使我们在解析图像时可以在少量像素上就进行测试，并用少量模板就可以表示一个3D目标。另外，我们证明了，如果一个密集深度传感器是可用的，我们可以将方法拓展，考虑3D表面法线方向，得到更好的性能。我们展示了，怎样利用现代计算机的架构，来构建输入图像的，一种高效但非常有区分能力的表示，可以实时考虑数千个模板。我们在很多实验中证明了，在真实数据中，我们的方法比目前最好的方法，在有背景杂乱的情况下，更加快速，更加稳健。

**Index Terms** - Computer vision, real-time detection and object recognition, tracking, multimodality template matching.

## 1. Introduction

Real-time object instance detection and learning are two important and challenging tasks in computer vision. Among the application fields that drive development in this area, robotics especially has a strong need for computationally efficient approaches as autonomous systems continuously have to adapt to a changing and unknown environment and to learn and recognize new objects.

实时目标检测和学习，是计算机视觉中两个重要并有挑战的任务。在推动这个领域前进的应用中，机器人学尤其需要在计算上很高效的方法，因为这种自动系统一直需要适应一种变化的、未知的环境，需要学习识别新的目标。

For such time-critical applications, real-time template matching is an attractive solution because new objects can be easily learned and matched online, in contrast to statistical-learning techniques that require many training samples and are often too computationally intensive for real-time performance [1], [2], [3], [4], [5]. The reason for this inefficiency is that those learning approaches aim at detecting unseen objects from certain object classes instead of detecting a priori, known object instances from multiple viewpoints. Classical template matching tries to achieve the latter in classical template matching where generalization is not performed on the object class but on the viewpoint sampling. While this is considered as an easier task, it does not make the problem trivial, as the data still exhibit significant changes in viewpoint, in illumination, and in occlusion between the training and the runtime sequence.

对于这样时间上很关键的应用，实时模板匹配是一种很好的方法，因为新的目标可以很容易的在线学习并匹配，而比较起来，统计学习方法需要很多训练样本，而且计算量很大，很难达到实时的性能。这种效率不高的原因是，这些学习方法的目标是从特定目标类别中检测出没看到的目标，而不是多角度的检测已知的目标。经典的模板匹配则尝试实现后者的目标，在目标类别上并没有进行泛化，而只是对采样的视角进行了泛化。这是一个比较简单的任务，但这并不是没有意义的，因为在不同视角、光照和遮挡情况下，训练时和运行时的序列数据仍然有很多变化。

When the object is textured enough for keypoints to be found and recognized on the basis of their appearance, this difficulty has been successfully addressed by defining patch descriptors that can be computed quickly and used to characterize the object [6]. However, this kind of approach will fail on textureless objects such as those of Fig. 1, whose appearance is often dominated by their projected contours.

当目标有足够的纹理，那么可以基于其外观找到并检测出关键点，通过定义图像块的描述子，这对于目标是可以很快的计算并表达目标的特征，这个问题可以得到了很好的处理[6]。但是，对于无纹理的目标，这种方法失败了，如图1所示，其外观主要由其投影的轮廓所决定。

To overcome this problem, we propose a novel approach based on real-time template recognition for rigid 3D object instances, where the templates can be both built and matched very quickly. We will show that this makes it very easy and virtually instantaneous to learn new incoming objects by simply adding new templates to the database while maintaining reliable real-time recognition.

为解决这个问题，我们提出一种基于实时模板识别的刚性3D目标实体检测的新方法，其中模板的构建和匹配都可以很快的进行。我们展示了，这使得学习新来的目标非常简单，几乎是立刻就完成，只要简单的给数据库加入新的模板，同时可以保持可靠的实时识别。

However, we also wish to keep the efficiency and robustness of statistical methods, as they learn how to reject unpromising image locations very quickly and tend to be very robust because they can generalize well from the training set. We therefore propose a new image representation that holds local image statistics and is fast to compute. It is designed to be invariant to small translations and deformations of the templates, which has been shown to be a key factor to generalization to different viewpoints of the same object [6]. In addition, it allows us to quickly parse the image by skipping many locations without loss of reliability.

但是，我们还希望保持统计方法的效率和稳健性，因为它们学习拒绝没有希望的图像位置时非常快，而且非常稳健，因为从训练集的泛化非常好。我们因此提出了一种新的图像表示，包含图像的局部统计量，计算起来很快。其设计对于模板的小的平移和形变是具有不变性的，这对于泛化到同样目标在不同视角下的成像，非常关键。另外，在不失可靠性的情况下通过跳过很多位置，这允许我们很快的解析图像。

Our approach is related to recent and efficient template matching methods [7], [8] which consider only images and their gradients to detect objects. As such, they work even when the object is not textured enough to use feature point techniques, and learn new objects virtually instantaneously. In addition, they can directly provide a coarse estimation of the object pose, which is especially important for robots which have to interact with their environment. However, similarly to previous template matching approaches [9], [10], [11], [12], they suffer severe degradation of performance or even failure in the presence of strong background clutter such as the one displayed in Fig. 1.

我们的方法与最近的高效模板匹配方法相关[7, 8]，只用图像及其梯度来检测目标。当目标没有足够多的纹理来使用特征点技术时候，也可以使用，而且学习新目标几乎是立刻的。另外，它们可以直接给出目标姿态的粗糙估计，这对于机器人来说非常重要，因为其需要与环境互动。但是，与之前的模板匹配方法类似，在背景存在很多杂乱时，会有严重的性能下降甚至失败，比如图1中所示的情况。

We therefore propose a new approach that addresses this issue while being much faster for larger templates. Instead of making the templates invariant to small deformations and translations by considering dominant orientations only as in [7], we build a representation of the input images which has similar invariance properties but consider all gradient orientations in local image neighborhoods. Together with a novel similarity measure, this prevents problems due to too strong gradients in the background, as illustrated by Fig. 1.

我们因此提出了一种新方法来处理这个问题，对于更大的模板也非常快。我们没有像[7]那样考虑主要的方向来使模板对于小的变形和平移具有不变性，而是构建了输入图像的表示，有着类似的不变性，但在局部图像邻域中考虑所有的梯度方向。与新的相似性度量一起，这避免了由于背景中有很强的梯度导致的问题，如图1所示。

To avoid slowing down detection when using this finer method, we have to carefully consider how modern CPUs work. A naive implementation would result in many "memory cache misses," which slow down the computations, and we thus show how to structure our image representation in memory to prevent these and to additionally exploit heavy SSE parallelization. We consider this as an important contribution: Because of the nature of the hardware improvements, it is no longer guaranteed that legacy code will run faster on the new versions of CPUs [13]. This is particularly true for computer vision, where algorithms are often computationally expensive. It is now required to take the CPU architecture into account, which is not an easy task.

在使用这种更精细的方法时，为避免使检测变慢，我们必须仔细的考虑现代CPU是怎样工作的。自然的实现会导致很多“内存缓存击不中”，这使得计算变慢，因此我们展示怎样在内存中使我们的图像表示结构化，以避免这种结果，另外采用很重的SSE并行化。我们认为这是一个很重要的贡献：因为硬件改进的本质，所以不能保证老的代码会在新版本的CPUs上运行更快。对计算机视觉来说，这尤其正确，其中算法通常计算量很大。这需要将CPU架构考虑进来，这并不容易。

For the case where a dense depth sensor is available, we describe an extension of our method where additional depth data are used to further increase the robustness by simultaneously leveraging the information of the 2D image gradients and 3D surface normals. We propose a method that robustly computes 3D surface normals from dense depth maps in real-time, making sure to preserve depth discontinuities on occluding contours and to smooth out discretization noise of the sensor. The 3D normals are then used together with the image gradients and in a similar way.

如果深度传感器可用，我们的方法可以进行拓展，利用深度数据来进一步增加稳健性，同时利用2D梯度信息和3D表面法线。我们提出一种方法，从密集深度图中实时稳健的计算3D表面法线，确保在遮挡的轮廓处保持深度的不连续性，平滑掉传感器的离散化噪声。然后3D法线的利用方式，与图像梯度是类似的。

In the remainder of the paper, we first discuss related work before we explain our approach. We then discuss the theoretical complexity of our approach. We finally present experiments and quantitative evaluations for challenging scenes. 本文的剩下部分，我们首先讨论相关的工作，然后解释了我们的方法，我们然后讨论了我们的方法的理论复杂度，最后，给出了试验和在有挑战的场景中进行了定量评估。

## 2. Related Work

Template matching has played an important role in tracking-by-detection applications for many years. This is due to its simplicity and its capability of handling different types of objects. It neither needs a large training set nor a time-consuming training stage, and can handle low-textured or textureless objects, which are, for example, difficult to detect with feature points-based methods [6], [14]. Unfortunately, this increased robustness often comes at the cost of an increased computational load that makes naive template matching inappropriate for real-time applications. So far, several works have attempted to reduce this complexity.

模板匹配在通过检测进行跟踪的应用中，在很多年中都起了很重要的作用。这是因为其简单，和能处理不同类型的目标的能力。这既不需要一个大型训练集，也没有很耗时的训练阶段，可以处理纹理很少，或没有纹理的目标，这用基于特征点的方法是很难检测的。不幸的是，这样好的稳健性，通常代价是计算复杂度增加，使模板检测不适用于实时应用。迄今为止，有几个工作试图降低这种复杂性。

An early approach to Template Matching [12] and its extension [11] include the use of the Chamfer distance between the template and the input image contours as a
dissimilarity measure. For instance, Gavrila and Philomin [11] introduced a coarse-to-fine approach in shape and parameter space using Chamfer Matching [9] on the Distance Transform (DT) of a binary edge image. The Chamfer Matching minimizes a generalized distance between two sets of edge points. Although fast when using the Distance Transform, the disadvantage of the Chamfer Transform is its sensitivity to outliers, which often result from occlusions.

模板匹配的早起方法[12]及其扩展[11]，将模板和输入图像轮廓之间的Chamfer距离作为不相似程度的度量。比如，Gavrila and Philomin [11]提出了一种由粗糙到精细的方法，在形状和参数空间使用Chamfer匹配，是二值边缘图的距离变换，来进行匹配。Chamfer匹配最小化两个边缘点集之间的广义距离。使用距离变换时虽然快，Chamfer变换的一个劣势是，对离群点很敏感，而遮挡常常导致离群点。

Another common measure on binary edge images is the Hausdorff distance [15]. It measures the maximum of all distances from each edge point in the image to its nearest neighbor in the template. However, it is sensitive to occlusions and clutter. Huttenlocher et al. [10] tried to avoid that shortcoming by introducing a generalized Hausdorff distance which only computes the maximum of the kth largest distances between the image and the model edges and the lth largest distances between the model and the image edges. This makes the method robust against a certain percentage of occlusions and clutter. Unfortunately, a prior estimate of the background clutter in the image is required but not always available. Additionally, computing the Hausdorff distance is computationally expensive and prevents its real-time application when many templates are used.

二值边缘的另一个常见度量是Hausdorff距离。其度量的是图像中每个边缘点到模板中所有邻域的最近点的所有距离的最大值。但是，其对遮挡和杂乱背景也很敏感。Huttenlocher等[10]提出了一种广义Hausdorff距离，计算图像和模板边缘的第k大距离，和模板和图像边缘的第l大距离，来避免这个缺陷。这使这个方法对一定程度的遮挡和杂乱是稳健的。不幸的是，这需要先对背景的杂乱有先验估计，但这并不总是可用的。另外，计算Hausdorff距离的计算量很大，当使用很多模板时，不能进行实时应用。

Both Chamfer Matching and the Hausdorff distance can easily be modified to take the orientation of edge points into account. This drastically reduces the number of false positives as shown in [12], but unfortunately also increases the computational load.

Chamfer匹配和Hausdorff距离可以很容易的修改，将边缘点的方向纳入考虑。这极大的降低了假阳性的数量，如[12]所示，但不幸的是，也增加了计算量。

The methof of [16] is also based on the Distance Transform; however, it is invariant to scale changes and robust enough against planar perspective distortions to do real-time matching. Unfortunately, it is restricted to objects with closed contours, which are not always available.

[16]也是基于距离变换的；但是，对尺度变化是不变的，对平面的视角扭曲也是足够稳健的，可以进行实时匹配。不幸的是，只能对有封闭轮廓的目标进行使用，这并不总是可用的。

All these methods use binary edge images obtained with a contour extraction algorithm, using the Canny detector [17], for example, and they are very sensitive to illumination changes, noise, and blur. For instance, if the image contrast is lowered, the number of extracted edge pixels progressively decreases, which has the same effect as increasing the amount of occlusion.

所有这些方法都使用二值边缘图，可以使用轮廓提取算法得到，或使用Canny检测器得到，这对光照变化、噪声和模糊非常敏感。比如，如果图像对比度降低，提取出的边缘像素的数量逐渐减少，这与增加遮挡量有一样的效果。

The method proposed in [18] tries to overcome these limitations by considering the image gradients in contrast to the image contours. It relies on the dot product as a similarity measure between the template gradients and those in the image. Unfortunately, this measure rapidly declines with the distance to the object location or when the object appearance is even slightly distorted. As a result, the similarity measure must be evaluated densely and with many templates to handle appearance variations, making the method computationally costly. Using image pyramids provides some speed improvements; however, fine but important structures tend to be lost if one does not carefully sample the scale space.

[18]中提出的方法，通过考虑图像梯度与图像轮廓的对比，来克服这个限制。其依赖于模板梯度与图像中的梯度的点乘作为相似度度量。不幸的是，这个度量随着到目标位置的距离增加而迅速降低，或在目标外观略微形变时迅速降低。结果是，相似度度量必须密集评估，要有很多模板来处理外观的变化，使这个方法计算代价很大。使用图像金字塔会得到一些速度改进，但是，如果不仔细选择尺度空间，精细但重要的结构可能丢失。

Contrary to the above-mentioned methods, there are also approaches addressing the general visual recognition problem: They are based on statistical learning and aim at detecting object categories rather than a priori, known object instances. While they are better at category generalization, they are usually much slower during learning and runtime, which makes them unsuitable for online applications.

与上面所述的方法相反，还有处理通用识别识别问题的方法：它们是基于统计学习的，其目标是检测目标类别，而不是先验的，已知的目标实例。它们更擅长于类别泛化，在训练和运行时通常很慢，这使其不适于在线应用。

For example, Amit et al. [19] proposed a coarse to fine approach that spreads gradient orientations in local neighborhoods. The amount of spreading is learned for each object part in an initial stage. While this approach - used for license plate reading - achieves high recognition rates, it is not real-time capable.

比如，Amit等[19]提出了一种由粗糙到精细的方法，在局部邻域中延展梯度方向。延展的程度在初始阶段就对每个目标部分进行学习。这种方法是用于车牌读取的，获得了很高的识别率，但是并不是实时的。

Histogram of Gradients (HoG) [1] is another related and very popular method. It statistically describes the distribution of intensity gradients in localized portions of the image. The approach is computed on a dense grid with uniform intervals and uses overlapping local histogram normalization for better performance. It has proven to give reliable results but tends to be slow due to the computational complexity.

HoG是另一种相关的，非常流行的方法。其描述了在图像的一个局部块中，灰度梯度分布的统计量。这种方法是在均匀间隔的密集网格上计算的，使用了重叠的局部直方图归一化得到更好的性能。其已证明可以得到可靠的结果，但由于计算复杂度较高，所以较慢。

Ferrari et al. [4] provided a learning-based method that recognizes objects via a Hough-style voting scheme with a nonrigid shape matcher on object boundaries of a binary edge image. The approach applies statistical methods to learn the model from few images that are only constrained within a bounding box around the object. While giving very good classification results, the approach is neither appropriate for object tracking in real time due to its expensive computation nor is it precise enough to return the accurate pose of the object. Additionally, it is sensitive to the results of the binary edge detector, an issue that we discussed before.

Ferrari等[4]给出了一种基于学习的方法，通过一种类似Hough方式的投票方案来识别物体，在二值边缘图的目标边缘上进行非刚性形状匹配。这个方法使用统计方法来从少量图像上学习模型，这个图像是包围目标的边界框所约束的图像。这个方法可以得到非常好的分类结果，但由于计算量大，不能够实时的用于目标跟踪，也不够准确以得到目标的准确姿态。另外，其对于二值边缘检测器的结果也很敏感，这个问题我们前面讨论过。

Kalal et al. [20] very recently developed an online learning-based approach. They showed how a classifier can be trained online in real time, with a training set generated automatically. However, as we will see in the experiments, this approach is only suitable for smooth background transitions and not appropriate to detect known objects over unknown backgrounds.

Kalal等[20]最近提出了一种在线的基于学习的方法。他们展示了，分类器可以在线实时训练，训练集自动生成。但是，我们在试验中可以看到，这种方法只适用于光滑的背景变化，对于在未知背景中已知目标的检测是不合适的。

Opposite to the above-mentioned learning-based methods, there are also approaches that are specifically trained on different viewpoints. As with our template-based approach, they can detect objects under different poses, but typically require a large amount of training data and a long offline training phase. For example, in [5], [21], [22], one or several classifiers are trained to detect faces or cars under various views.

与上述基于学习的方法不同，还有在不同视角专门训练的方法。与我们的基于模板的方法相似，他们可以检测不同姿态的物体，但需要大量训练数据，和很长的离线训练阶段。比如，在[5,21,22]中，训练了一个或几个分类器来检测不同视角的人脸或车。

More recent approaches for 3D object detection are related to object class recognition. Stark et al. [23] rely on 3D CAD models and generate a training set by rendering them from different viewpoints. Liebelt and Schmid [24] combine a geometric shape and pose prior with natural images. Su et al. [25] use a dense, multiview representation of the viewing sphere combined with a part-based probabilistic representation. While these approaches are able to generalize to the object class, they are not real-time capable and require expensive training.

最近的3D目标检测的方法与目标类别识别相关。Stark等[23]依赖于3D CAD模型，在不同视角上对其渲染来生成训练集。Liebelt和Schmid[24]将自然图像与几何形状和姿态的先验结合到一起。Su等[25]使用一个密集，多视角表示与基于部位的概率表示的结合。这些方法能泛化到目标类别，但是不能实时进行，需要很昂贵的训练。

From the related works which also take into account depth data there are mainly approaches related to pedestrian detection [26], [27], [28], [29]. They use three kinds of cues: image intensity, depth, and motion (optical flow). The most recent approach of Enzweiler et al. [26] builds part-based models of pedestrians in order to handle occlusions caused by other objects and not only self-occlusions modeled in other approaches [27], [29]. Besides pedestrian detection, there has been an approach to object classification, pose estimation, and reconstruction introduced by Sun et al. [30]. The training data set is composed of depth and image intensities, while the object classes are detected using the modified Hough transform. While quite effective in real applications, these approaches still require exhaustive training using large training data sets. This is usually prohibited in robotic applications, where the robot has to explore an unknown environment and learn new objects online.

相关的工作将深度数据也纳入考虑，主要有行人识别[26,27,28,29]。它们使用三种线索：图像灰度，深度和运动（光流）。最近的Enzweiler等[26]构建了基于部位的行人模型，以处理被其他目标的遮挡，而并不局限于其他方法建模的自我遮挡[27,29]。除了行人识别，还有目标分类、姿态估计和重建的方法[30]。训练集是由深度数据和图像组成的，目标类别是使用修改的Hough变换检测的。在实时应用中非常有效，但这些方法仍然需要大量训练，需要有大型训练数据集。这在机器人应用中是不可行的，其中机器人需要探索未知的环境，在线学习新的目标。

As mentioned in the introduction, we recently proposed a method to detect textureless 3D object instances from different viewpoints based on templates [7]. Each object is represented as a set of templates, relying on local dominant gradient orientations to build a representation of the input images and the templates. Extracting the dominant orientations is useful to tolerate small translations and deformations. It is fast to perform and, most of the time, discriminant enough to avoid generating too many false positive detections.

如在引言中提到的，我们最近提出了一种基于模板的从多个视角检测无纹理3D目标实例的方法[7]。每个目标是用模板集合表示的，依靠局部主要梯度方向来构建输入图像和模板的表示。提取的主要方向，对于容忍小的平移和形变是有用的。其执行起来很快，多数时间足够有区分能力，防止生成太多假阳性检测。

However, we noticed that this approach degrades significantly when the gradient orientations are disturbed by stronger gradients of different orientations coming from background clutter in the input images. In practice, this often happens in the neighborhood of the silhouette of an object, which is unfortunate as the silhouette is a very important cue especially for textureless objects. The method we propose in this paper does not suffer from this problem while running at the same speed. Additionally, we show how to extend our approach to handle 3D surface normals at the same time if a dense depth sensor like the Kinect is available. As we will see, this increases the robustness significantly.

但是，我们注意到，这个方法在输入图像背景中的杂乱有更强的不同方向的梯度时，其梯度方向会受到干扰，性能也会有很大的下降。实践中，这通常会在目标的轮廓的边缘发生，这非常不幸，因为轮廓是非常重要的线索，尤其是对于没有纹理的目标。我们在本文提出的方法，则没有这个问题，而且可以以相同的速度运行。另外，我们展示了，将我们的方法进行拓展，在有Kinect这样的密集深度传感器时，以同时处理3D表面法线。我们将会看到，这显著提升了稳健性。

## 3. Proposed Approach

In this section, we describe our template representation and show how a new representation of the input image can be built and used to parse the image to quickly find objects. We will start by deriving our similarity measure, emphasizing the contribution of each aspect of it. We also show how we implement our approach to efficiently use modern processor architectures. Additionally, we demonstrate how to integrate depth data to increase robustness if a dense depth sensor is available.

本节中，我们描述了我们的模板表示，展示了输入图像的新的表示怎样进行构建，并用于快速解析图像以找到目标。我们首先推导相似性度量，强调每部分的贡献。我们还展示了怎样实现我们的方法，以高效的利用现代处理器架构。另外，我们展示了，在密集深度相机可用时，怎样将深度数据整合到一起，以提升稳健性。

### 3.1 Similarity Measure

Our unoptimized similarity measure can be seen as the measure defined by Steger in [18] modified to be robust to small translations and deformations. Steger suggests using

我们未优化的相似性度量可以视为，[18]中Steger定义的度量经过修改，对小的平移和形变稳健。Steger建议使用

$$E_{Steger}(I,T,c) = \sum_{r∈P} |cos(ori(O,r) - ori(I, c+r))|$$(1)

where ori(O, r) is the gradient orientation in radians at location r in a reference image O of an object to detect. Similarly, ori(I, c+r) is the gradient orientation at c shifted by r in the input image I. We use a list, denoted by P, to define the locations r to be considered in O. This way we can deal with arbitrarily shaped objects efficiently. A template T is therefore defined as a pair T = (O, P).

其中ori(O,r)是在参考图像O中位置r处一个要检测的目标的梯度方向，单位为rad。类似的，ori(I,c+r)是输入图像I中在c经过r偏移的梯度方向。我们使用一个列表P，来定义O中要考虑的位置r。这样我们可以高效的处理任意形状的目标。模板T因此可以定义为T=(O,P)。

Each template T is created by extracting a small set of its most discriminant gradient orientations from the corresponding reference image, as shown in Fig. 2, and by storing their locations. To extract the most discriminative gradients we consider the strength of their norms. In this selection process, we also take the location of the gradients into account to avoid an accumulation of gradient orientations in one local area of the object while the rest of the object is not sufficiently described. If a dense depth sensor is available, we can extend our approach with 3D surface normals, as shown on the right side of Fig. 2.

每个模板T的创建，是从对应的参考图像中提取一小部分最有区分性的梯度方向，如图2所示，并存储其位置。为提取最具有区分性的梯度，我们考虑其范数的强度。在选择的过程中，我们还讲梯度的位置纳入进来，以避免梯度方向在目标的一个局部区域的聚积，而目标的其他位置没有得到充分的描述。如果密集深度传感器可用，我们可以用3D表面法线来拓展我们的方法，如图2右所示。

Considering only the gradient orientations and not their norms makes the measure robust to contrast changes, and taking the absolute value of the cosine allows it to correctly handle object occluding boundaries: It will not be affected if the object is over a dark background or a bright background.

只考虑梯度方向，不考虑其范数，使这个度量对于对比度变化更加稳健，取cos值得绝对值，使其可以正确的处理目标遮挡的边缘：如果目标在暗的背景中，还是在亮的背景中，都不会影响。

The similarity measure of (1) is very robust to background clutter, but not to small shifts and deformations. A common solution is to first quantize the orientations and to use local histograms like in SIFT [6] or HoG [1]. However, this can be unstable when strong gradients appear in the background. In DOT [7], we kept the dominant orientations of a region. This was faster than building histograms, but suffers from the same instability. Another option is to apply Gaussian convolution to the orientations like in DAISY [31], but this would be too slow for our purpose.

相似性度量(1)对背景的杂乱非常稳健，但对小的偏移和形变则不是。一个常见的解决方法是，首先量化方向，然后使用局部的直方图，如SIFT或HOG。但是，当背景中出现很强的梯度时，这就不稳定了。在DOT[7]中，我们保留一个区域中的主要方向。这比构建直方图要快，但有同样的不稳定的问题。另一个选项是，对方向进行高斯卷积，如DAISY[31]，但这对于我们的方法拉说，太慢了。

We therefore propose a more efficient solution. We introduce a similarity measure that, for each gradient orientation on the object, searches in a neighborhood of the associated gradient location for the most similar orientation in the input image. This can be formalized as

我们因此提出了一个更高效的解决方案。我们提出了一个新的相似性度量，对目标上的每个梯度方向，在相关的梯度位置的邻域，在输入图像中搜索最相似的方向。这可以表述成

$$E(I,T,c) = \sum_{r∈P} (max_{t∈R(c+r)} |cos(ori(O,r) - ori(I, t))|)$$(2)

where R(c+r) = [c+r-T/2, c+r+T/2] x [c+r-T/2, c+r+T/2] defines  the neighborhood of size T centered on location c + r in the input image. Thus, for each gradient we align the local neighborhood exactly to the associated gradient location, whereas in DOT, the gradient orientation is adjusted only to some regular grid. We show below how to compute this measure efficiently.

其中，R(c+r) = [c+r-T/2, c+r+T/2] x [c+r-T/2, c+r+T/2]，定义了以输入图像的位置c+r为中心的邻域大小T。因此，对每个梯度，我们将局部邻域与相关的梯度位置进行严格对齐，而在DOT中，梯度方向只调整到一些规则网格上。我们在下面展示，怎么搞笑的计算这个度量。

### 3.2 Computing the Gradient Orientations

Before we continue with our approach, we shortly discuss why we use gradient orientations and how we extract them easily. 在我们开始我们的方法前，我们简短的讨论，怎样使用梯度方向，怎样很容易的提取它们。

We chose to consider image gradients because they proved to be more discriminant than other forms of representations [6], [18] and are robust to illumination change and noise. Additionally, image gradients are often the only reliable image cue when it comes to textureless objects. Considering only the orientation of the gradients and not their norms makes the measure robust to contrast changes, and taking the absolute value of cosine between them allows it to correctly handle object occluding boundaries: It will not be affected if the object is over a dark background or a bright background.

我们选择考虑图像梯度，因为这比其他形式的表示更加有区分性，对光照变化和噪声更加稳健。另外，在无纹理目标的情况下，图像梯度可能是唯一可靠的图像线索了。只考虑梯度的方向，而不考虑其范数，使这个度量对对比度变化更加稳健，取其cos值的差的绝对值，使其可以更准确的处理目标遮挡的边缘：目标是在一个暗色的背景上，还是在一个亮色的背景上，这都不会受到影响。

To increase robustness, we compute the orientation of the gradients on each color channel of our input image separately and for each image location use the gradient orientation of the channel whose magnitude is largest, as done in [1], for example. Given an RGB color image I, we compute the gradient orientation map I_G(x) at location x with

为提高稳健性，我们在输入图像的每个色彩通道中分别计算梯度的方向，对每个图像位置，使用梯度幅度最大的通道的梯度方向，[1]中也是这样做的。给定一个RGB彩色图像I，我们在位置x上计算梯度方向图I_G(x)

$$I_G(x) = ori(\hat C(x))$$(3)

where

$$\hat C(x) = argmax_{C∈{R,G,B}} |∂C/∂x|$$(4)

and R, G, B are the RGB channels of the corresponding color image.

In order to quantize the gradient orientation map we omit the gradient direction, consider only the gradient orientation, and divide the orientation space into n_0 equal spacings, as shown in Fig. 3. To make the quantization robust to noise, we assign to each location the gradient whose quantized orientation occurs most often in a 3 x 3 neighborhood. We also keep only the gradients whose norms are larger than a small threshold. The whole unoptimized process takes about 31 ms on the CPU for a VGA image.

为量化梯度方向图，我们忽略了其梯度direction，只考虑了梯度的方向，将方向空间分成n_0个相等的间隔，如图3所示。为使这个量化对噪声稳健，我们给每个位置指定的梯度是，其量化的方向在3x3的邻域中发生最多的方向。我们还只保留了范数大于一个小的阈值的梯度。这整个未优化的过程，在CPU上对一个VGA的图像来说，耗时31ms。

### 3.3 Spreading the Orientations

In order to avoid evaluating the max operator in (2) every time a new template must be evaluated against an image location, we first introduce a new binary representation — denoted by J — of the gradients around each image location. We will then use this representation together with lookup tables to efficiently precompute these maximal values.

为在新模板与一个图像位置进行评估时，为避免每次都要计算(2)中的最大化算子，我们首先在每个图像位置附近引入梯度新的二值表示，表示为J。然后我们使用这个表示与查找表一起，来高效的预计算这些最大值。

The computation of J is depicted in Fig. 4. We first quantize orientations into a small number of $n_o$ values as done in previous approaches [6], [1], [7]. This allows us to "spread" the gradient orientations ori(I, t) of the input image I around their locations to obtain a new representation of the original image.

J的计算如图4所示。我们首先将方向量化成$n_o$个值，和[1,6,7]一样。这使我们可以将输入图像的梯度方向ori(I, t)在其位置上进行扩散，以得到原始图像的新表示。

For efficiency, we encode the possible combinations of orientations spread to a given image location m using a binary string: Each individual bit of this string corresponds to one quantized orientation, and is set to 1 if this orientation is present in the neighborhood of m. The strings for all the image locations form the image J on the right part of Fig. 4. These strings will be used as indices to access lookup tables for fast precomputation of the similarity measure, as it is described in the next section.

为效率，我们将方向扩散到给定图像m的可能的方向组合用一个二值字符串来进行编码：这个字符串的每个bit都代表了一个量化的方向，如果在m的邻域中存在这个方向，就设为1。所有图像位置的字符串，形成了图4右的图像J。这个字符串会用于访问查找表的索引，进行相似度的预计算，在下一节中进一步阐述。

J can be computed very efficiently: We first compute a map for each quantized orientation, whose values are set to 1 if the corresponding pixel location in the input image has this orientation and 0 if it does not. J is then obtained by shifting these maps over the range of [-T/2, +T/2] x [-T/2, +T/2] and merging all shifted versions with an OR operation.

J可以进行高效的计算：我们首先对每个量化的方向计算一个图，如果输入图像的对应像素处有这个方向，就设为1，否则设为0。将这个图在[-T/2, +T/2] x [-T/2, +T/2]的范围内平移，用OR运算合并所有的平移版，然后就得到了J。

### 3.4 Precomputing Response Maps

As shown in Fig. 5, J is used together with lookup tables to precompute the value of the max operation in (2) for each location and each possible orientation ori(O, r) in the template. We store the results into 2D maps $S_i$. Then, to evaluate the similarity function, we will just have to sum values read from these $S_i$s.

如图5所示，J与查找表一起，对目标中的每个可能的方向ori(O, r)和每个位置，计算(2)中的max运算的值。我们将结果存储成2D图$S_i$。然后，为评估相似度函数，我们只需要将从这些$S_i$中读出这些值，然后求和，得到这些相似度函数的值。

We use a lookup table $τ_i$ for each of the $n_o$ quantized orientations, computed offline as 我们对$n_o$个量化的方向的每一个，都使用一个查找表$τ_i$，离线计算如下

$$τ_i[L] = max_{l∈L} |cos(i-l)|$$(5)

where 其中

- i is the index of the quantized orientations; to keep the notations simple, we also use i to represent the corresponding angle in radians; i是量化方向的索引；为保持表示简单，我们用i也表示对应的角度；

- L is a list of orientations appearing in a local neighborhood of a gradient with orientation i as described in Section 3.3. In practice, we use the integer value corresponding to the binary representation of L as an index to the element in the lookup table. L是一个方向列表，在梯度方向i的邻域出现，如3.3节所示。实践中，我们使用对应L的二值表示的整数值，作为查找表中元素的索引。

For each orientation i, we can now compute the value at each location c of the response map $S_i$ as 对每个方向i，我们现在在响应图$S_i$的每个位置c上计算值

$$S_i(c) = τ_i[J(c)]$$(6)

Finally, the similarity measure of (2) can be evaluated as 最后，式(2)的相似性度量可以计算为

$$E(I,T,c) = \sum_{r∈P} S_{ori(O,r)} (c+r)$$(7)

Since the maps $S_i$ are shared between the templates, matching several templates against the input image can be done very fast once the maps are computed. 由于图$S_i$在不同模板间是共享的，一旦图计算结束后，将几个模板与输入图像之间的匹配，可以很快的计算得到。

### 3.5 Linearizing the Memory for Parallelization

Thanks to (7), we can match a template against the whole input image by only adding the values in the response maps $S_i$. However, one of the advantages of spreading the orientations as was done in Section 3.3 is that it is sufficient to do the evaluation only every Tth pixel without reducing the recognition performance. If we want to exploit this property efficiently, we have to take into account the architecture of modern computers.

有了(7)式，我们只需要对响应图$S_i$中的值相加，就可以将模板与整个图像进行匹配。但是，像3.3节所说的将方向进行扩展的一个优势，是每T个像素进行计算就足够了，不会降低识别性能。如果我们想要高效的利用这个性质，我们还要考虑现代计算机的架构。

Modern processors do not read only one data value at a time from the main memory but several ones simultaneously, called a cache line. Accessing the memory at random places results in a cache miss and slows down the computations. On the other hand, accessing several values from the same cache line is very cheap. As a consequence, storing data in the same order as they are read speeds up the computations significantly. In addition, this allows parallelization: For instance, if 8-bit values are used as it is the case for our $S_i$ maps, SSE instructions can perform operations on 16 values in parallel. On multicore processors or on the GPU, even more operations can be performed simultaneously. For example, the NVIDIA Quadro GTX 590 can perform 1,024 operations in parallel.

现代处理器从内存中不是一次读一个数据值，而是一次读多个，称为一个缓存线。随机位置访问内存会得到缓存丢失，使计算速度变慢。另一方面，从相同的缓存线中访问几个值是非常容易的结果是，在数据被读取时，以相同的顺序存储数据，会显著加速计算。另外，这还允许并行化：比如，如果使用8-bit值，我们的$S_i$图就是这个情况，SSE指令可以并行的对16个值进行运算。在多核处理器或在GPU上，可以并行的进行更多计算。比如，NVIDIA Quadro GTX 590可以并行进行1024个计算。

Therefore, as shown in Fig. 6, we store the precomputed response maps $S_i$ into memory in a cache-friendly way: We restructure each response map so that the values of one row that are T pixels apart on the x-axis are now stored next to each other in memory. We continue with the row which is T pixels apart on the y-axis once we finished with the current one.

因此，我们将预计算的响应图$S_i$以缓存友好的方式存储到内存中，如图6所示：我们重新组织每个响应图的结构，这样一行的值在内存中是存储在一起的，即在x轴上T个像素以内的。一旦当前的这个结束，我们对y轴上的T像素以外的继续进行存储。

Finally, as described in Fig. 7, computing the similarity measure for a given template at each sampled image location can be done by adding the linearized memories with an appropriate offset computed from the locations r in the templates.

最后，如图7所示，对一个给定的模板，在每个采样的图像位置，计算相似度度量，可以通过将线性化的内存，与从模板中位置r上计算得到的合适的偏置相加，完成计算。

### 3.6 Extension to Dense Depth Sensors

In addition to color images, recent commodity hardware like the Kinect allows to capture dense depth maps in real time. If these depth maps are aligned to the color images, we can make use of them to further increase the robustness of our approach as we have recently shown in [32].

除了彩色图像，最近的硬件如Kinect，可以实时捕获密集深度图。如果这些深度图与彩色图像对齐，我们可以利用深度图进一步改进我们方法的稳健性，我们最近在[32]中进行了展示。

Similarly to the image cue, we decided to use quantized surface normals computed from a dense depth map in our template representation, as shown in Fig. 8. They allow us to represent both close and far objects while fine structures are preserved.

与图像线索类似，我们决定在我们的模板表示中，利用从密集深度图中计算得到的量化的表面法线，如图8所示。这使我们既可以表示近的目标，也可以表示远的目标，同时精细的结构还得到了保持。

In the following, we propose a method for the fast and robust estimation of surface normals in a dense range image. Around each pixel location x, we consider the first order Taylor expansion of the depth function D(x)

下面，我们提出一种方法，从密集深度图像中快速稳健的估计表面法线。在每个像素位置x，我们考虑深度函数D(x)的一阶Taylor展开

$$D(x+dx) - D(x) = dx^T∇D + h.o.t$$(8)

Within a patch defined around x, each pixel offset dx yields an equation that constrains the value of ∇D, allowing us to estimate an optimal gradient $∇\hat D$ in a least-square sense. This depth gradient corresponds to a 3D plane going through three points X, $X_1$, and $X_2$:

在x附近定义的一个图像块内，每个像素偏移dx产生了一个等式，约束了∇D的值，使我们可以在最小二乘的意义上估计最优梯度$∇\hat D$。这个深度梯度对应着一个经过三个点 X, $X_1$和$X_2$的3D平面

$$X = \vec v(x)D(x)$$(9)
$$X_1 = \vec v(x+[1,0]^T) (D(x)+[1,0]∇\hat D)$$(10)
$$X_2 = \vec v(x+[0,1]^T) (D(x)+[0,1]∇\hat D)$$(11)

where $\vec v(x)$ is the vector along the line of sight that goes through pixel x and is computed from the internal parameters of the depth sensor. The normal to the surface at the 3D point that projects on x can be estimated as the normalized cross-product of $X_1 - X$ and $X_2 - X$.

其中$\vec v(x)$是穿过像素x沿着视线的向量，是从深度传感器的内部参数计算得到的。在投影到x的3D点上的表面法线，可以估计为$X_1 - X$和$X_2 - X$的归一化点积。

However, this would not be robust around occluding contours, where the first order approximation of (8) no longer holds. Inspired by bilateral filtering, we ignore the contributions of pixels whose depth difference with the central pixel is above a threshold. In practice, this approach effectively smooths out quantization noise on the surface, while still providing meaningful surface normal estimates around strong depth discontinuities. Our similarity measure is then defined as the dot product of the normalized surface normals, instead of the cosine difference for the image gradients in (2). We otherwise apply the same technique we apply to the image gradients. The combined similarity measure is simply the sum of the measure for the image gradients and the one for the surface normals.

但是，这在遮挡的轮廓附近是不会稳健的，因为(8)式的一阶近似不再成立。受双边滤波启发，我们忽略了一些像素的贡献，即与中心像素的深度差大于一定阈值的。实践中，这个方法有效的平滑掉了表面上的量化噪声，而在深度强不连续处仍然会给出有意义的表面法线估计。我们的相似性度量，这时就定义为归一化表面法线的点积，而不是(2)中的梯度的cos差值。然后我们应用那些与图像梯度相同的技术。结合的相似性度量，就是图像梯度的度量，和表面法线度量的和。

To make use of our framework we have to quantize the 3D surface normals into n_0 bins. This is done by measuring the angles between the computed normals and a set of n_0 precomputed vectors. These vectors are arranged in a circular cone shape originating from the peak of the cone pointing toward the camera. To make the quantization robust to noise, we assign to each location the quantized value that occurs most often in a 3 x 3 neighborhood. The whole process is very efficient and needs only 14 ms on the CPU and less than 1 ms on the GPU.

为利用我们的框架，我们需要量化3D表面法线成n_0个bins。这通过在计算的法线与n_0个预计算的向量集之间度量得到。这些向量以圆形的椎体排列，从椎体的尖峰处指向相机。为使量化对噪声稳健，我们对每个位置指定了在3x3邻域中最经常出现的值。整个过程非常高效，在CPU上只需要14ms计算，在GPU上1ms。

### 3.7 Computation Time Study

In this section, we compare the numbers of operations required by the original method from [18] and the method we propose.

### 3.8 Experimental Validation

We compared our approach, which we call LINE (for LINEearizing the memory), to DOT [7], HOG [1], TLD [20], and the Steger method [18]. For these experiments we used three different variations of LINE: LINE-2D that uses the image gradients only, LINE-3D that uses the surface normals only, and LINE-MOD, for multimodal, which uses both.

### 3.9 Robustness

We used six sequences made of over 2,000 real images each. Each sequence presents illumination and large viewpoint changes over a heavily cluttered background. Ground truth is obtained with a calibration pattern attached to each scene that enables us to know the actual location of the object. The templates were learned over a homogeneous background.

We consider the object to be correctly detected if the location given back is within a fixed radius of the ground truth position.

As we can see in the left columns of Figs. 11 and 12, LINE-2D mostly outperforms all other image-based approaches. The only exception is the method of Steger, which gives similar results. This is because our approach and the one of Steger use similar score functions. However, the advantage of our method in terms of computation times is very clear from Fig. 10.

The reason for the weak detection results of TLD is that while this method works well under smooth background transition, it is not suitable to detect known objects over unknown backgrounds.

If a dense depth sensor is available we can further increase the robustness without becoming slower at runtime. This is depicted in the left columns of Figs. 11 and 12, where LINE-MOD always outperforms all the other approaches and shows only a few false positives. We believe that this is due to the complementarity of the object features that compensate for the weaknesses of each other (see Fig. 9). The depth cue alone often performs not very well.

The superiority of LINE-MOD becomes more obvious in Table 1: If we set the threshold for each approach to allow for 97 percent true positive rate and only evaluate the hypothesis with the largest response, we obtain for LINE-MOD a high detection rate with a very small false positive rate. This is in contrast to LINE-2D, where the true positive rate is often over 90 percent, but the false positive rate is not negligible. The true positive rate is computed as the ratio of correct detections and the number of images; similarly, the false positive rate is the ratio of the number of incorrect detections and the number of images.

One reason for this high robustness is the good separability of the multimodal approach as shown in the middle of Figs. 11 and 12: In contrast to LINE-2D, where we have a significant overlap between true and false positives, LINE-MOD separates at a specific threshold—about 80 in our implementation—almost all true positives well from almost all false positives. This has several advantages. First, we will detect almost all instances of the object by setting the threshold to this specific value. Second, we also know that almost every returned template with a similarity score above this specific value is a true positive. Third, the threshold is always around the same value, which supports the conclusion that it might also work well for other objects.

### 3.10 Speed

Learning new templates only requires extracting and storing the image features (and, if used, the depth features), which is almost instantaneous. Therefore, we concentrate on runtime performance.

The runtimes given in Fig. 10 show that the general LINE approach is real time and can parse a VGA image with over 3,000 templates with about 10 fps on the CPU. The small difference of computation times between LINE-MOD and LINE-2D and LINE-3D comes from the slightly slower preprocessing step of LINE-MOD, which includes the two preprocessing steps of LINE-2D and LINE-3D.

DOT is initially faster than LINE but becomes slower as the number of templates increases. This is because the runtime of LINE is independent of the template size, whereas the runtime of DOT is not. Therefore, to handle larger objects DOT has to use larger templates, which makes the approach slower once the number of templates increases.

Our implementation of Steger et al. is approximately 100 times slower than our LINE-MOD method. Note that we use four pyramid levels for more efficiency, which is one of the reasons for the different speed improvement given in Section 3.7, where we assumed no image pyramid.

TLD uses a tree classifier similar to [33], which is the reason why the timings stay relatively equal with respect to the number of templates. Since this paper is concerned with detection, for this experiment we consider only the detection component of TLD and not the tracking component.

### 3.11 Occlusion

We also tested the robustness of LINE-2D and LINE-MOD with respect to occlusion. We added synthetic noise and illumination changes to the images, incrementally occluded the six different objects of Section 3.9 and measured the corresponding response values. As expected, the similarity measures used by LINE-2D and LINE-MOD behave linearly in the percentage of occlusion, as reported in Fig. 10. This is a desirable property since it allows detection of partly occluded templates by setting the detection threshold with respect to the tolerated percentage of occlusion.

We also experimented with real scenes where we first learned our six objects in front of a homogeneous background and then added heavy 2D and 3D background clutter. For recognition we incrementally occluded the objects. We define our object as correctly recognized if the template with the highest response is found within a fixed radius of the ground truth object location. The average recognition result is displayed in Fig. 10: With 20 percent occlusion for LINE-2D and with over 30 percent occlusion for LINE-MOD we are still able to recognize objects.

### 3.12 Number of Templates

We discuss here the average number of templates needed to detect an arbitrary object from a large number of viewpoints. In our implementation, approximately 2,000 templates are needed to detect an object with 360 degree tilt rotation, 90 degree inclination rotation and in-plane rotations of $\pm$ 80 degree—tilt and inclination cover the half-sphere of Fig. With the number of templates given here, the detection works for scale changes in the range of [1.0, 2.0].

### 3.13 Examples

Figs. 14, 15, and 16 show the output of our methods on textureless objects in different heavy cluttered inside and outside scenes. The objects are detected under partial occlusion, drastic pose, and illumination changes. In Figs. 14 and 15, we only use gradient features, whereas in Fig. 16, we also use 3D normal features. Note that we could not apply LINE-MOD outside since the Primesense device was not able to produce a depth map under strong sunlight.

### 3.14 Failure Cases

Fig. 13 shows the limitations of our method. It tends to produce false positives and false negatives in case of motion blur. False positives and false negatives can also be produced when some templates are not discriminative enough.

## 4. Conclusion

We presented a new method that is able to detect 3D textureless objects in real time under heavily background clutter, illumination changes, and noise. We also showed that if a dense depth sensor is available, 3D surface normals can be robustly and efficiently computed and used together with 2D gradients to further increase the recognition performance. We demonstrated how to take advantage of the architecture of modern computers to build a fast but very discriminant representation of the input images that can be used to consider thousands of arbitrarily sized and arbitrarily shaped templates in real time. Additionally, we have shown that our approach outperforms state-of-the-art methods with respect to the combination of recognition rate and speed, especially in heavily cluttered environments.

我们提出一种新方法，可以在背景非常杂乱、光照变化和噪声的情况下，实时检测3D无纹理目标。我们还展示了，如果密集深度相机可用，3D表面法线可以稳健高效的计算，并与2D梯度一起使用，进一步改进识别性能。我们展示了，怎样利用现代计算机架构，来构建输入图像快速而又很有区分性的表示，可以用于实时考虑数千个任意形状、任意大小的模板。另外，我们还证明了。我们的方法比目前最好的方法性能要更好，包括识别速度和识别率，尤其是在高度杂乱的环境中。