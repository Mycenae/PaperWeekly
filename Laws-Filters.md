# Rapid texture identification

Kenneth I. Laws, Massachusetts

## 0. Abstract

A method is presented for classifying each pixel of a textured image, and thus for segmenting the scene. The "texture energy" approach requires only a few convolutions with small (typically 5x5) integer coefficient masks, followed by a moving-window absolute average operation. Normalization by the local mean and standard deviation eliminates the need for histogram equalization. Rotation-invariance can also be achieved by using averages of the texture energy features. The convolution masks are separable, and can be implemented with 1-dimensional (vertical and horizontal) or multipass 3x3 convolutions. Special techniques permit rapid processing on general-purpose digital computers.

提出了一种纹理图像中像素分类的方法，因此可以对场景进行分割。这个纹理能量方法只需要与很小的（一般是5x5）整数系数掩模进行几个卷积，然后进行滑窗的绝对平均运算。局部均值和标准差的归一化，消除了直方图均化的需要。通过使用纹理能量特征的平均，还可以得到旋转不变性。卷积掩模是可分的，可以进行1d实现或多次3x3卷积实现。可以采用特殊的技术，在通用目标计算机上进行快速处理。

## 1. Introduction

Image textures arise from many physical sources. Cellular textures are composed of repeated similar elements such as leaves on a tree or bricks in a wall. Other texture types include flow patterns, fiber masses, and stress cracks. A complete analysis of any texture would require modeling of the underlying physical structure. Texture analysis is fundamental to some applications, such as metal surface inspection and geologic fault identification, for which appropriate theories of texture generation are required.

图像纹理因为很多物理原因出现。网状的纹理是由类似的元素重复构成的，比如树上的树叶，或墙上的砖。其他纹理类型包括流体的类型，纤维物质和压力裂缝。任何纹理的完整分析需要对潜在的物理结构进行建模。纹理分析对一些应用是非常基础的，比如金属表面检查，和地质学缺陷检查，其中需要纹理生成的合适理论。

In other applications, such as radiographic diagnosis, texture recognition is more important than knowledge of the physical generating mechanism. General-purpose image analysis systems, such as the human visual system, use texture as an aid in segmentation and interpretation of scenes. Built-in models of physical texture generators are not required for texture discrimination.

在其他应用中，比如放射成像诊断，纹理识别比其物理生成机制更加重要。通用目标的图像分析系统，比如人眼视觉系统，使用纹理来帮助分割和场景解释。纹理区分，不需要物理纹理生成的内建模型。

This paper describes a set of "texture energy" transforms that provide texture measures for each pixel of a monochrome image. The transforms require only simple convolution and moving-average techniques. They can be made invariant to changes in luminance, contrast, and rotation without histogram equalization or other preprocessing operations.

本文描述了一个纹理能量变换的集合，对灰度图像的每个像素都计算得到一个纹理度量。这个变换只需要简单的卷积和滑动平均技术。不需要直方图均化或其他预处理运算，就可以得到对光照，对比度和旋转的不变性。

Texture energy is the amount of variation within a filtered image window. A particular texture energy measure thus depends on the spatial filter, on the window size, and on the method of measuring average variation within the window. A particular set of texture energy measures that have been found useful in distinguishing natural textures is presented, along with an example of texture classification using these local texture energy measures.

纹理能量是在一个滤波图像窗口中变化的量。一种特定的纹理能量依赖于空间滤波器，依赖于窗口大小，依赖于在窗口中度量平均变化的方法。提出了一种特定的纹理能量度量集合，对于区分自然图像的纹理是很有用的，还有使用这些局部纹理能量度量来进行纹理分类的例子。

## 2. Texture energy classification

Figure 1 diagrams the sequence of images used in measuring and identifying textures; Figure 2 diagrams the corresponding sequence of image transform operations. The original image is first filtered with a set of small convolution masks to generate the set of filtered images, F. Each filtered image, F, is then processed with a nonlinear "local texture energy" filter to generate the corresponding texture energy image, E. This filter is a moving-window average of the absolute filtered image values. Such moving-window operations are very fast even on general-purpose digital computers.

图1给出了用于度量和识别纹理的图像序列；图2给出了图像变换运算的对应顺序。原始图像首先用小型卷积掩模集合进行滤波，生成滤波后的图像，F。每个滤波后的图像F，然后用非线性局部纹理能量滤波器进行处理，以生成对应的纹理能量图像E。这个滤波器是绝对滤波图像值的滑动窗口平均。这样的滑窗运算在通用计算机上是非常快的。

The next step is the linear combination of texture energy planes into a smaller set of principal component planes, C. This is an optional data compression step. Other optional combination operations are normalization (for illumination invariance) and averaging (for rotational invariance).

下一步是将纹理能量平面线性组合成，主要部分平面的小集合C。这是可选的数据压缩步骤。其他可选的组合运算有，归一化（得到光照不变性）和平均（得到旋转不变性）。

The final output is a segmented image or classification map, M. Either texture energy planes or principal component planes may be used as input to the pixel classifier. Classification is simple and fast if the texture classes are known a priori. Clustering or segmentation algorithms must be used if texture classes are unknown.

最后的输出是分割的图像或分类图，M。用纹理能量平面或主要部分平面作为像素分类器的输入。如果纹理类别是已知的，那么分类是简单迅速的。如果纹理类别未知，那么就需要用聚类或分割算法。

## 3. Implemention

This section presents a specific implementation that is complete except for classification coefficients (which must be estimated for each new application). The 5x5 filter mask size and 15x15 texture energy window size are specific for the illustrated example, but also are probably close to optimal for textures near the resolution limit of a standard 512x512 display.

本节给出了一种具体的完整实现，除了分类系数以外（这需要对每种新应用进行估计）。在这个例子中，具体是5x5的滤波器掩模大小和15x15纹理能量窗口大小，但对于接近显示极限的标准512x512分辨率而言，这种纹理可能也是接近最优的。

Figure 2 diagrams the sequence of operators needed to generate a classification map using this technique. All processing can be performed in the spatial domain (i.e., without Fourier transforms) using shift register or tapped delay line techniques. This paper describes a specific moving-window implementation for general-purpose digital computers. Timing data for this implementation may be found in Ref. 1; comparative classification results are also given there and are summarized in Ref. 2.

图2给出了使用这种技术生成分类图所需的算子序列。所有的处理都可以在空域进行处理（即，不需要进行Fourier变换），使用shift register或tapped delay line技术。本文描述了一种具体的通用计算机上的滑窗实现。这种实现的定时数据可以在[1]中找到；其中也有比较分类的结果，在[2]中也进行了总结。

Let the original image be I(r,c), where r and c are the row and column indices. The subscript ranges are unimportant since boundary effects are ignored. Only five rows (or columns) need to be available at one time in the implementation described here. Figure 3(a) represents a buffer or storage area holding this data; it is shown centered on row r of the original image. During processing the image rows enter at the bottom, move up through the buffer, and "drop off" the top. For efficient implementation, pointers to rows should be changed rather than moving data rows.

令原始图像为I(r,c)，其中r和c是行和列的索引。下标的范围是不重要的，因为忽略了边缘效果。这里描述的实现，一次只需要有5行（或列）可用就可以。图3(a)表示了保存这些数据的一个buffer或存储区域；其中心是原图形的第r行。在处理图像时，图像的行从底部进入，逐步往上移动穿过buffer，并丢弃掉上面的部分。为高效的进行实现，指向行的指针应当变化，而不要变化移动的数据行。

The original image must be convolved with a set of coefficient masks to form a set of filtered images (Ref. 3). Figure 3(b) represents the output row, F(r,c), produced by convolving the image buffer of Fig. 3(a) with a 5x5 filter mask.

原始图像需要与一系列系数掩模集进行卷积，以形成滤波器图像组[3]。图3(b)表示将图3(a)的图像buffer与5x5滤波器掩模进行卷积得到的输出的行，F(r,c)。

A very fast filter implementation is possible using the coefficient vectors of Fig. 4. The justification of these coefficients is given in Ref. 1. One of the vectors is used to form a weighted sum of the buffer rows, and another is convolved with the combined row. The result is the filtered row $F_{vh}(r,c)$, where vh (e.g., L5E5) represents the particular combination of vertical and horizontal coefficient vectors. Identical filtered output can be obtained by convolving with a 5x5 mask or by twice convolving with certain 3x3 filter masks (Refs. 1,2). This can be the fastest technique if special convolution hardware is available.

使用图4的系数矢量，可以进行快速的滤波器实现。这些系数的选择在[1]中给出。一个矢量是用于对buffer行进行加权求和，另一个是与结合的行进行卷积。得到的结果是滤波后的行$F_{vh}(r,c)$，其中vh（如，L5E5）表示垂直和水平系数矢量的特定组合。与5x5的掩模卷积，和与特定的3x3滤波器掩模卷积两次，可以得到相同的滤波结果[1,2]。如果有特殊的卷积硬件可用，这可以是最快的技术。

Next, each filtered image must be converted to a texture energy image. Local texture energy is here measured by the sum of absolute values in a window, or local region, of the filtered image. It is similar to a local standard deviation if the filtered image is zero-mean. (All of the suggested vh filters are zero-mean except L5L5, which will be used only for luminance and contrast normalization.)

下一步，每种滤波的图像必须转化成纹理能量图。局部纹理能量这里要通过，滤波后图像的局部区域或窗口中的绝对值之和，进行度量。如果滤波后图像的均值为0，这与局部标准差是类似的。（除了L5L5，所有的vh滤波器都是0均值的，这可以用于光照和对比度归一化。）

We shall assume a 15x15 pixel moving window for the texture energy transform, which requires buffering of 15 rows of each filtered image. The texture energy measure for a filtered image window is

我们用15x15的滑窗来计算纹理能量变换，这需要对每个滤波后的图像的15行进行缓存。对一个滤波后的图像窗口，其纹理能量度量为

$$E_{vh}(r,c) = \sum_{j=c-7}^{c+7} \sum_{i=r-7}^{r+7} |E_{vh} (i,j)|$$(1)

An efficient method of computing this measure for each image pixel is available. To compute a texture energy image, the sum of absolute values in each column is first computed:

对每个像素，计算这种度量有高效的方法。为计算纹理能量图像，首先计算每列的绝对值之和：

$$S_{vh} (r,c) = \sum_{i=r-7}^{r+7} |F_{vh} (i,c)|$$(2)

The texture energy values, or window sums, are then computed from the row of column sums: 对纹理能量值，或窗口之和，是从列和的行进行计算得到的：

$$E_{vh}(r,c) = \sum_{i=c-7}^{c+7} |S_{vh} (r,i)|$$(3)

As each subsequent image row is read, the column sums can be updated by subtracting out the absolute values on the top row of each filtered image buffer, updating the buffer, and adding in the absolute values of the new bottom row. Similarly, each horizontal sum can be computed from the previous one by subtracting one column sum and adding another:

由于读取了每个后续的图像行，其列和的更新，可以通过减去每个滤波后图像缓存中顶行的绝对值，更新缓存，并加上新加入的底行的绝对值。类似的，每个水平的和的计算，可以通过之前的和减去一个列和，再加上另一个来进行：

$$E_{vh}(r,c) = E_{vh}(r,c-1) - S_{vh}(r,c-8) + S_{vh} (r,c+7)$$(4)

The texture energy transform requires essentially constant time regardless of the window size. This would permit very large (hence very reliable) texture energy windows to be used when image texture regions are known to be large, although the number of buffered filtered image rows would also be large.

纹理能量变换的计算需要的时间是恒定的，与窗口大小无关。当图像纹理区域已知很大时，虽然缓存的滤波图像行的数量会非常大，这可以允许使用很大的纹理能量窗口（因此非常可靠）。

An even simpler implementation is possible using a "fading memory" for each filtered image. Only the column sum of absolute filtered values, $S_{vh}(r,c)$, is stored, thus eliminating the need for a 15 row buffer for each filtered image. Each time the image buffer is shifted one line, the new filtered column sum row is computed as:

对每个滤波后的图像使用fading memory技术，可以得到更简单的实现。只存储滤波后值绝对值的列和$S_{vh}(r,c)$，因此不需要对每个滤波后的图像存储15行缓存。每次图像缓存偏移1行，新的滤波后的列和可以计算为：

$$S_{vh} (r,c) = 14/15 S_{vh} (r,c) + |F_{vh} (r,c)|$$(5)

Thus one-fifteenth of the total is replaced by the new filtered row. This method of reducing storage requirements has not been tested and is not further considered here.

因此综合的1/15替换为新滤波的行。这种降低存储要求的方法还没有进行测试，因此这里没有进一步考虑。

An optional next step is combination of the texture energy measures to form a smaller number of more useful texture measures. Three operations may be used: normalization, rotational averaging, and extraction of principal components. Each is optional, and should be applied only if warranted by the application.

下一步是可选的，将纹理能量度量结合到一起，形成更少更有用的纹理度量。可以使用三种运算：归一化，旋转平均，和主要成分的提取。每个都是可选的，而且只再应用批准后才应当应用。

Normalization is used to adjust for luminance and contrast of the original image so that textures differing only in luminance or contrast will not be distinguished. Normalization is achieved by subtracting local average luminance from each texture energy row and /or dividing by a local average contrast. These local measures may be the moving-average mean and standard deviation of the L5L5 filtered image, computed in much the same way as the texture energy measures.

归一化用于对原图的光照和对比度进行调整，这样只有光照和对比度不同的纹理不会得到区分。归一化的得到，是对每个纹理能量行减去局部平均光照，和/或除以一个局部平均对比度。这些局部措施可能是L5L5滤波图像的滑动平均均值和标准差，与纹理能量度量计算的方法基本类似。

Rotational averaging is used when rotated versions of a texture field are to be considered identical (although a boundary between such fields may still be differentiated). Rotational invariance is achieved by averaging (or simply adding) matched pairs of texture energy measures. The L5E5 and E5L5 filters, for instance, measure vertical and horizontal edge content; their sum measures total edge content. This has been found more effective than filtering with rotation-invariant transforms (e.g., Sobel gradient magnitude) and then computing the texture energy (Ref. 1).

当认为纹理场的旋转版本是一样的时候，就需要用到旋转平均（这些场之间的边缘仍然需要被区分）。旋转不变性的获得，是通过平均匹配上的纹理能量度量对（或简单的相加）。比如，L5E5和E5L5滤波器度量的是垂直和水平的边缘内容；其和度量的是总边缘内容。这比用旋转不变的变换进行滤波（如，Sobel梯度幅度），然后计算纹理能量，更加有效。

Extraction of principal components may be used to reduce the number of features passed to the classifier. The normalized and/or rotation-invariant texture energy rows are linearly combined to form a smaller number of data rows. The weighting coefficients for these sums can be derived from principal components analysis (also known as eigen analysis or Karhunen-Loeve rotation).

主要成分的提取，可用于降低传入都分类器的特征数量。归一化的和/或旋转不变的纹理能量行，线性结合到一起，以形成更少的数据行。这些和的加权系数可以从PCA中推导得到（也称为eigen分析或Karhunen-Loeve旋转）。

The final operation in Fig. 2 is classification. The classifier computes linear discriminant functions from the measured texture energy values for each pixel (Ref. 4). (Quadratic, or cross-product, terms might increase the classification accuracy of these functions.) The texture class for which the discriminant is greatest is determined to be the source class for that pixel neighborhood. Note that assignment to known classes (e.g., water, forest, urban, cultivated land) can be done pixel by pixel, whereas segmentation by clustering requires that each of the texture energy images be available in its entirety.

图2中的最后运算是分类。分类器从对每个像素度量的纹理能量值中计算得出线性区分函数[4]。（二次的，或点积项，可能会增加这些函数的分类准确率。）对这个像素的邻域，判别式最大的纹理类别，确定为源类别。注意，指定为已知的类别（如，水，森林，都市，耕种的土地），可以逐个像素进行，而通过聚类的分割，则要求每个纹理能量图像都完全可用。

## 4. Example

Figure 5 shows the results of one texture classification experiment (Ref. 1). The top half of the composite texture image consists of eight 128x128 pixel blocks of grass, raffia, sand, wool, pigskin, leather, water, and wood. The lower-left quadrant consists of 32*32 blocks of these textures, and the lower-right quadrant of 16x16 blocks. Histogram equalization of each texture and each quadrant have removed all first-order differences in the texture field gray level distributions. This equalization was important for research in texture analysis, but has little effect on the texture measures described here.

图5给出了一个纹理分类试验的结果[1]。复合纹理图像的上半部分是8个128x128的像素块，分别是grass, raffia, sand, wool, pigskin, leather, water和wood。下左的1/4是这些纹理的32*32块，下右的1/4是16x16的块。每种纹理和每个1/4块的直方图均化，已经去除了纹理场灰度分布的所有一阶差异。这种均化对纹理分析中的研究很重要，但对这里描述的纹理度量没有什么效果。

The classification map, Fig. 5(b), shows the assigned source labels as eight different gray levels. Fifteen filter combinations were used. (As few as nine 5x5 filters would have given similar results, as would the nine 3x3 center-weighted filter masks.) Normalized texture energy was computed over 15x15 pixel moving windows using Eqs. 2-4. Rotational averaging and principal component extraction were not used. The classifier had no knowledge of the mosaic structure of the composite image. The same texture fields, but different samples, were used for training and for testing the classifier.

图5(b)的分类图将指定的源标签展示为8种不同的灰度级。使用了15种滤波器组合。（9种5x5的滤波器会给出类似的结果，9种3x3的中心加权的滤波器掩模也会得到类似的结果。）在15x15像素的滑动窗口上使用式2-4计算了归一化的纹理能量。旋转平均和PCA提取并没有使用。分类器并不知道复合图像的马赛克结构。同样的纹理场，但是不同的样本，被用于训练和测试分类器。

The center-weighted coefficient vectors in Fig. 4 were chosen after extensive experimentation. The W5 vector was not found useful but is listed for completeness. Either the E5 or S5 vector could also be omitted with little degradation in classifier performance. Other sets of shorter and longer vectors have been tested, but none gave as low a classification error rate for the experimental test set.

在很多试验后，我们选择了图4中的中心加权的系数向量。W5向量并没有什么用，但为了完整仍然列了出来。E5或S5矢量可以忽略，对分类器性能也不会造成很大的影响。其他更短和更长的向量进行了测试，但对于试验中的测试集，都给出了更高的分类错误率。

The classification map shows excellent results: 87% classification accuracy for the interiors of the 128x128 blocks. This contrasts with only 72% accuracy for classification done with co-occurrence texture measures (Ref. 5). Hundreds of other filter and moving- window statistic combinations tested (Ref. 1) on this data have performed less well than the texture energy measures described here.

分类图给出了非常好的结果：对内部的128x128模块，给出了87%分类准确率。用共现纹理度量[5]只有72%的分类准确率。其他滤波和滑窗统计量的组合在这个数据上测试了几百次[1]，但都比这里描述的纹理能量度量效果要差一些。

Errors tend to occur in clumps, indicating non-representative regions of the texture fields. Grass, for instance, had only 78% recognition accuracy over the entire 512x512 training image because sharpness of focus varied greatly from the top of the training image to the bottom. Classification errors in the other texture samples may represent irregularities in the available texture fields.

误差通常会成堆产生，说明纹理场有表示不足的区域。比如grass，在整个512x512的图像中只有78%的识别准确率，因为训练图像中从上到下，角点的锐利度差别很大。其他纹理样本中的分类错误率，在可用的纹理场中会表现出不规则性。

Greater accuracy in large area classification can be achieved with larger texture energy windows. A 31x31 window, for instance, yields overall accuracy greater than 97% for the full 512x512 training images (Ref. 1). Smaller windows permit greater resolution with less accuracy. Figure 4(b) shows that the 15x15 window correctly identifies the centers of nearly all 32x32 texture blocks, and provides some discrimination of the 16x16 texture blocks. A human observer could probably do little better at this resolution, but would be able to detect the moasic block structure.

在大面积分类中，采用更大的纹理能量窗口，可以得到更好的准确率。比如31x31的窗口，在完整的512x512训练图像中，可以得到总体上超过97%的准确率[1]。更小的窗口可以进行更小的分辨率，准确率也更低。图4(b)展示了15x15的窗口，正确的识别出了几乎所有32x32纹理块的中心，对16x16纹理块也给出了一些区分性。人类观察者在这种分辨率上可能不会做到更好的结果，但会检测到这种马赛克模块结构。

## 5. Conclusion

Excellent classification results were obtained for the very difficult problem of Fig. 5(a). Lower accuracy may be expected for real-world problems unless color or multispectral data is available. Addition of cross-product terms to the discriminant functions might raise classification accuracy, as would a second-level segmentation applying adjacency constraints and higher-level knowledge. The rapid texture measurement and classification techniques presented here seem promising as stepping-stones to better image analysis systems.

对图5(a)的很困难的问题，给出了非常好的分类结果。对真实世界的问题，估计效果会略差一些，除非有彩色数据或多光谱数据。为判别函数加入点积项，可能会提高分类准确率。这里给出的快速纹理度量和分类技术，可能是更好的图像分析系统的一步。