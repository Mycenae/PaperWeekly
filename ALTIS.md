# ALTIS: A fast and automatic lung and trachea CT-image segmentation method

**Purpose:** The automated segmentation of each lung and trachea in CT scans is commonly taken as a solved problem. Indeed, existing approaches may easily fail in the presence of some abnormalities caused by a disease, trauma, or previous surgery. For robustness, we present ALTIS (implementation is available at http://lids.ic.unicamp.br/downloads) — a fast automatic lung and trachea CT-image segmentation method that relies on image features and relative shape- and intensity-based characteristics less affected by most appearance variations of abnormal lungs and trachea.

**目的**：每个肺和气管在CT中的自动分割一般认为是一个已经解决的问题。确实，现有的方法在患病、创伤或之前的手术导致的非正常情况的情况下，很容易失败。为了稳健性，我们提出了ALTIS，一种快速自动肺和气管CT图像分割算法，算法依靠的是图像特征和基于相对形状和灰度值的特征，受到不正常肺和气管的外形变化较小。

**Methods:** ALTIS consists of a sequence of image foresting transforms (IFTs) organized in three main steps: (a) lung-and-trachea extraction, (b) seed estimation inside background, trachea, left lung, and right lung, and (c) their delineation such that each object is defined by an optimum-path forest rooted at its internal seeds. We compare ALTIS with two methods based on shape models (SOSM-S and MALF), and one algorithm based on seeded region growing (PTK).

**方法**：ALTIS包括一系列图像foresting变换(IFTs)，组织成了三个主要的步骤：(a)肺和气管的提取，(b)在背景、气管，左肺和右肺中的种子估计，(c)其勾画的确定，使每个目标由一个最优路径forest定义，其起源在其内部种子上。我们将ALTIS与两种基于形状模型的方法(SOSM-S和MALF）、和一种基于种子区域生长(PTK)的方法进行了比较。

**Results:** The experiments involve the highest number of scans found in literature — 1255 scans, from multiple public data sets containing many anomalous cases, being only 50 normal scans used for training and 1205 scans used for testing the methods. Quantitative experiments are based on two metrics, DICE and ASSD. Furthermore, we also demonstrate the robustness of ALTIS in seed estimation. Considering the test set, the proposed method achieves an average DICE of 0.987 for both lungs and 0.898 for the trachea, whereas an average ASSD of 0.938 for the right lung, 0.856 for the left lung, and 1.316 for the trachea. These results indicate that ALTIS is statistically more accurate and considerably faster than the compared methods, being able to complete segmentation in a few seconds on modern PCs.

**结果**：试验包含了文献中发现的最高数量的扫描，1255幅，从多个公开数据集中得到的，包含了很多反常案例，50个正常的扫描用于训练，1205个扫描用作测试。定量试验是基于两种度量，DICE和ASSD。而且，我们还证明了ALTIS在种子估计上的稳健性。考虑测试集，提出的方法得到的两个肺的平均DICE为0.987，气管为0.898，而右肺的平均ASSD为0.938，左肺为0.856，气管为1.316。这些结果说明，与对比方法比较，ALTIS在统计上更准确，速度更快，在现代PC上几秒钟内就得到了完全的分割。

**Conclusion:** ALTIS is the most effective and efficient choice among the compared methods to segment left lung, right lung, and trachea in anomalous CT scans for subsequent detection, segmentation, and quantitative analysis of abnormal structures in the lung parenchyma and pleural space.

**结论**：ALTIS在肺部和气管在异常CT扫描中的几种被比较方法中，是最有效和高效的，可以进行后续的在肺薄壁组织和胸膜空间异常结构检测，分割和定量分析。

## 1. Introduction

Respiratory diseases (RD) are disorders that can affect the lungs and airways. According to the World Health Organization and the Forum of International Respiratory Societies, RD are among the top ten leading causes of death in the world. They also estimate that RD will become the third leading cause of worldwide death in the next 30 yr.

呼吸系统疾病RD是可以影响肺部和气道的疾病。根据WHO和国际呼吸协会论坛，RD是世界上10大致死疾病之一。他们还估计，在未来30年中，RD会成为世界范围内第三大致死原因之一。

Computerized tomography (CT) is the standard procedure for analysis and diagnosis of RD by providing images of the thorax with high spatial resolution and reasonable contrast. In order to automate the CT image analysis of the thorax, the segmentation of each lung and trachea can be used as a first step for the detection of lung nodules, diagnosis of respiratory diseases, analysis of intrathoracic airway trees, density evaluation of the lung parenchyma, measurement of trachea stenosis, and quantification of emphysema. However, its success may be compromised due to variations in intensity, shape, and size of the lungs, patient movement during scanning, congenital anomalies, thoracic deformities from trauma or previous surgery, and the presence of anomalies caused by a disease.

CT是分析和诊断RD的标准过程，提供了胸部高分辨率、合理对比度的图像。为自动分析胸部CT图像，每个肺和气管的自动分割，是后续检测肺结节，诊断呼吸系统疾病，分析胸内气道树，肺薄壁的密度评估，气管狭窄程度的度量，肺气肿的定量评估等过程的第一步。但是，由于亮度、形状、肺部的大小的变化，在扫描时患者的运动，先天性异常，由于创伤或之前的手术受到的胸部形变，或者由于患病导致的存在异常。

The automated segmentation of each lung and trachea in CT scans is often taken as a solved problem. However, several methods do not separate all three objects, other methods rely on image properties and relative shape- and intensity-based characteristics that may change considerably in abnormal cases, and many approaches are validated on a relatively low number of images. The segmentation of anomalous images, as we address here, is certainly an open problem and this can be easily verified in recent works based on deep learning. The aforementioned approaches nonetheless, still present a few limitations and need further investigation. That is, they do not separate the lungs and trachea, usually process the images in a slice-by-slice fashion, despite three-dimensional (3D) networks are available, and have not been proven to generalize to new data sets without fine tuning on annotated training sets from the same data set, differently from what we will show in this paper.

每个肺部和气管在CT中的自动分割，通常认为是一个已经解决的问题。但是，几种方法并没有将这三者分开，其他方法依赖于图像性质，和基于相对形状和灰度的性质，这在异常的情况下会变化很大，很多方法的验证，是在较少的图像上的。异常图像的分割，是我们在这里要处理的情况，当然是一个开放的问题，这在最近的基于深度学习的工作中可以很容易得到验证。之前提到的方法，仍然提出了一些限制，需要更进一步的研究。即，他们并没有区分开肺和气管，处理图像的方式是逐个slice的形式进行的，尽管3D网络是可用的，但并未证明可以不经过精调的情况下，很好的泛化到新的数据集上，这与我们在本文中看到的是不一样的。

Classical methods for lung segmentation start by thresholding the original voxel intensities to extract lungs and trachea as the largest connected component a strategy sensitive to abnormal intensity variations which may also exclude structures brighter than the air inside the lungs (e.g., nodules, vessels, tumors). Some methods also require user interaction to select seeds inside the objects. Morphological operations and/or shape models can amend the problem, but they are not failproof (Fig. 1). Indeed, most recent methods adopt seeded region growing, shape models, or deep learning. Some seed-based methods assume that exists an axial slice above the lungs, which is suitable to estimate seeds inside the trachea. Unfortunately, this is not always the case (Fig. 2). Shape models can solve some of those limitations, but they usually fail when lungs are deformed by disease or abnormal shape of the thoracic cage (Fig. 3). Moreover, they often require high computational time for deformable registration. In [14], for example, the authors report 55s per scan for a first segmentation based on seeded region growing, followed by 4s to detect errors, and, whenever necessary, 120 min to correct errors using shape models (unacceptable for clinical use).

经典的肺分割方法，开始先用原始体素灰度用阈值进行处理，将肺部和气管提取为一个最大的连通部分，这个策略对非正常的灰度变化是敏感的，还会排除掉肺内比空气更亮的结构（如，结节，血管，肿瘤）。一些方法还需要用户交互来选择目标内部的种子。形态学运算和/或形状模型可以修复这个问题，但并不是预防故障的（图1）。确实，最近的方法采用了种子区域生长，形状模型，或深度学习。一些基于种子的方法假设，存在一个肺上的轴向slice，适合于估计气管中的种子。不幸的是，这种假设并不一定成立（图2）。形状模型可以解决一部分这些限制，但当肺被疾病或非正常胸廓形状所形变时（图3），这就通常会失败。比如，在[14]中，作者采用基于种子区域生长，对CT进行首次分割，每scan要耗费55s，然后还要用4s进行错误检测，在需要的时候，用120min来修正使用形状模型的错误（对临床使用是不可接受的）。

Deep neural networks have achieved great results in many fields, including medical imaging. However, we could not find a network that could be optimized from a small training set, as we do for this study, and used to segment the trachea, left lung, and right lung. Although it is possible to train a deep neural network to separate the lungs and trachea, the results will be hardly accurate without exploiting some connectivity-based image operator, such as done in methods that combine neural networks with graph cuts for semantic segmentation. In [21], for instance, the authors present a progressive and multipath holistically nested neural network (P-HNN) to segment pathological lungs in CT images. Yet, it does not separate the trachea, left lung, and right lung. Another example is [16], where the authors present a 3D autoencoder network (Seg3DNet) to segment the lungs of canines and porcine in CT images. Seg3DNet was trained with 1756 human scans and fine tuned using thoracic four-dimensional (4D) CT images of 13 animals (canine and porcine). The requirement for fine tuning reinforces our statement that deep neural networks usually require annotated training images for each new data set. Since the network uses 3D kernels, the 4D images are treated as series of 3D scans. Additionally, the network does not consider the trachea as an object in the segmentation process and does not separate the left and right lungs.

DNN在很多领域都取得了很好的结果，包括医学图像。但是，我们没有发现一个网络可以在一个小型训练集上优化的很好，这样就无法在这个研究中很好的分割气管、左肺和右肺。虽然可能训练一个神经网络来区分肺和气管，但不利用基于连接性的图像算子的话，结果很难准确，比如有的工作将神经网络与图割结合到了一起进行语义分割。比如在[21]中，作者提出了一个渐进的和多路径全嵌套神经网络来在CT图像中分割病理的肺。但是，并没有分离出气管、左肺、右肺。另一个例子是[16]，其中作者提出了一个3D自动编码器网络(Seg3DNet)来分割狗和猪的肺部。Seg3DNet用1756个人类扫描进行训练，并用13个动物的胸部4D CT图像进行精调。精调的要求强化了我们的说法，即DNN通常对每个新的数据集都需要标注好的训练图像。由于网络使用的是3D核，4D图像被认为是3D扫描的序列。另外，网络在分割的过程中，并没有将气管作为一个目标，而且也没有区分左肺和右肺。

In this paper, we present a fast method for lungs and trachea segmentation and validate it on a high number of test images from distinct data sets (to the best of our knowledge, the highest number in literature). For that, we use the image foresting transform (IFT) framework to design a fast sequence of image processing operations based on relative shape- and intensity-based features, and image properties that are robust to account for most appearance variations of abnormal lungs, separating the trachea, left lung, and right lung. Three main relative features are considered in this paper: the lungs and trachea are usually darker than the surrounding tissues in CT images of the thorax, the lungs are usually larger than the trachea, and the trachea is a thin and long object.

本文中，我们提出一种肺和气管的快速分割方法，在很多不同的数据集上的非常多的测试图像上核验了本方法（据我们所知，是文献中最多的）。为此，我们使用了图像foresting变换(IFT)框架来设计一个图像处理运算快速序列，这是基于相对的形状和灰度特征的，这些图像性质非常稳健，可以顾及到非正常肺的多数外形变化，分离出气管，左肺和右肺。在本文中考虑了三个主要的特征：在胸部CT图像中，肺和气管通常比周围组织要暗，肺通常比气管要大，气管是一根细长的目标。

The proposed method is named ALTIS — automatic lung and trachea image segmentation. The segmentation comprehends the air spaces of the lungs (functional lung, including brighter tissues such as vessels and tumors) delimited by the visceral pleura (and/or fluid in pathological cases) and the trachea with the main bronchi, as a first step to detect and identify anomalies in the respiratory system. The purpose behind restricting the segmentation to the volume of air inside the lungs is to facilitate the subsequent detection of anomalies connected to the pleura, as shown in Fig. 4. Note that such detection is not performed by the proposed method, and it can be achieved by a post-processing mechanism, like in Ref. [38]. Among the contributions of ALTIS, it is worth mentioning the fast binary morphological operations (erosion and dilation) and a delineation technique specialized for the problem, both based on IFT. In ALTIS, we also design the sequence of IFTs such that extra information computed by a preceding IFT is used in the succeeding one to avoid redundant operations and make the overall process more efficient. The method is then an excellent and first example of the use of the IFT framework for this application.

提出的方法叫ALTIS，自动肺和气管图像分割。分割算法将肺的气体部分（功能性肺，包括较亮的组织，比如血管和肿瘤），这些由胸膜脏层包围（和/或病理情况下的液体），和带有主要支气管的气管，作为第一步来探测并识别呼吸系统中的异常。将分割限制在肺中的气体的目的，是促进后续的与胸膜相连的异常检测，如图4所示。注意，这样的检测并不是由提出的方法进行处理的，这可以由一个后处理机制进行，如在[38]中。在ALTIS的贡献中，值得提到的是，快速二值形态学运算（腐蚀和膨胀），和一个专门用于此问题的勾画技术，这都是基于IFT的。在ALTIS中，我们还设计来IFT的序列，这样前一个IFT计算得到的信息，用于后续的IFT，以防止额外的运算，使得总体的过程更加高效。这个方法是在这个应用中使用IFT框架的第一个非常好的例子。

Given that our focus is on the segmentation stage, this work does not present image analysis of any specific pulmonary disease. We have evaluated ALTIS for a reasonable number of CT images from public and in-house data sets. These images represent a wide variety of normal and abnormal patterns: lung nodules, tumors, pneumonia, areas of deflated/collapsed lung, pleural plaques/effusion, emphysema, scoliosis, and born/acquired thoracic/lung abnormalities. The results show that ALTIS can consistently achieve superior accuracy with respect to three other baselines based on seeded region growing and shape models, within a few seconds of processing time.

既然我们的焦点是在分割阶段，本文并不分析任何具体的肺部疾病。我们在相当数量的CT图像上评估了ALTIS算法，这些CT图像是公开和内部的数据集。这些图像代表了大量正常和非正常的模式：肺结节，肿瘤，肺炎，泄气的/坍缩的肺的区域，胸膜斑/积液，肺气肿，脊柱侧凸，先天性/后天的肺异常。结果表明，与基于种子点区域生长和形状模型的其他三种基准相比，ALTIS可以一致的获得很好的准确率，处理时间只有几秒钟。

The first contribution is ALTIS, a fast computational method for the segmentation of both lungs and trachea, entirely based on image processing operators. It shows that not every problem needs to be solved with deep learning alone. The second contribution is the fast version of binary morphological erosion and dilation based on IFT. The third and last contribution is the use of a large annotated data set for the assessment of ALTIS.

第一个贡献是ALTIS，一种双肺和气管分割的快速计算方法，完全基于图像处理算子。这表明，并不是所有问题都需要用深度学习来解决。第二个贡献是，基于IFT的二值形态学腐蚀和膨胀的快速版本。第三个贡献是，使用大型标注数据集对ALTIS进行评估。

Section 6 reviews the data sets used (including the preprocessing), the IFT framework, presents ALTIS and describe the training and testing of the baseline. Section 15 presents the evaluation protocol and experimental results, while their discussed is presented in Section 16. Section 17 presents the conclusion and future work.

第6部分回顾了使用的数据集（包括预处理），IFT框架，提出了ALTIS，描述了基准的训练集和测试集。第15部分提出了评估标准和试验结果，而其讨论是在第16部分给出。第17部分给出了结论和未来的工作。

## 2. Material and Methods

### 2.1 CT scan data sets

In this study, one in-house and four public CT-scan data sets were used, totalizing 1255 scans. More details about them are given as follows: 本研究中，使用了一个内部数据集和四个公开的CT数据集，共计1255个扫描。更多细节如下所示：

1. ASBESTOS, with 66 normal scans and 37 scans of patients with asbestos-related pleural plaques; 66个正常扫描，和37位患者的扫描，患者是与石棉相关的胸膜斑；

2. LOLA11, with 55 scans gathered from a variety of sources, and clinically common scanners and protocols. Although the annotation of the diseases are not provided, we identified some patients with scoliosis, severe pulmonary embolism, and partial/total atelectasis; 55个扫描是从各种源收集来的，是临床上常见的扫描仪和协议。虽然疾病的标注并没有提供，我们看出了一些脊柱侧凸的患者，严重的肺栓塞，和部分/全部的肺不张；

3. EXACT09, with 40 scans ranging from healthy volunteers to patients with severe abnormalities in the airways or lung parenchyma. They were acquired at different sites using different scanners, scanning protocols, and reconstruction parameters; 有40个扫描，包括正常的健康志愿者，和气道或肺薄壁有严重异常的患者。这些CT是在不同的地点使用不同的扫描仪、扫描协议和重建参数获得的；

4. VIA/ELCAP, with 50 low-dose scans including patients diagnosed with lung nodules. 有50个低剂量扫描，包括诊断出肺结节的患者。

5. LIDC-IDRI, with 1012 diagnostic and lung cancer screening thoracic scans. 1012个诊断肺癌筛查胸部扫描。

From the normal scans of the ASBESTOS data set, 50 scans were randomly selected for parameter optimization (or training) of all methods. The remaining 1205 images were used for testing.

从ASBESTOS数据集中的正常扫描中，随机选择了50个扫描，对所有方法进行参数优化（或训练）。剩下的1205幅图像用于测试。

All scans were linearly interpolated to the same voxel size, 1.25×1.25×1.25 mm^3, and reordered/rotated to a fixed patient orientation, as described in Section 2.C. Each object (left lung, right lung, and trachea) in the scans was interactively segmented by a trained expert using the differential watershed algorithm, which allowed the specialist to add and remove seeds along with multiple interventions with real-time response to define the gold standard segmentation. Note that, due to the high amount of CT scans, manual labeling is unfeasible. All computations were performed on a PC i7-3770K CPU, with 3.50 GHz processor, 8 cores, and 32 GB of RAM.

所有扫描用线性插值到相同的体素大小，1.25×1.25×1.25 mm^3，重新排序/旋转到一个固定的患者方向，如2.3节所述。扫描中的每个目标（左肺，右肺，和气管）都由一个训练有素的专家，使用差分分水岭算法进行互动分割，这使得专家可以沿着多个干涉方向增加和移除种子点，并有实时的响应，以定义金标准的分割。注意，由于CT扫描数量很大，手工标注是不可行的。所有计算都是在一台i7-3770K CPU上进行的，3.5GHz处理器，8核，32GB内存。

### 2.2 Image Foresting Transform

A gray-scale image, such as a CT scan, is a pair $\hat I = (D_I, I)$, in which $D_I ⊂ Z^3$ is the image domain and $I(p) ∈ Z$ is a mapping that assigns an intensity value to each voxel $p ∈ D_I$. An image can be interpreted as a graph $(D_I, A)$, whose nodes are the voxels and the arcs are defined by an adjacency relation $A⊂D_I×D_I$, with A(p) being the adjacent set of a voxel p. For example, a spherical adjacency relation of radius γ ≥ 1 can be defined as

一幅灰度图像，比如一个CT扫描，是一个对$\hat I = (D_I, I)$，其中$D_I ⊂ Z^3$是图像域，$I(p) ∈ Z$是一个映射，对每个体素$p ∈ D_I$指定了一个灰度值。一幅图像可以解释为一个图$(D_I, A)$，其节点是体素，弧由一个相邻关系定义，$A⊂D_I×D_I$，A(p)是一个体素p的相邻集。比如，半径γ ≥ 1的球形相邻关系可以定义为

$$A_γ: \{ (p,q)∈D_I×D_I, ||q-p||≤γ \}$$(1)

The image operators in this work use two types of adjacency relations, $A_1$ and $A_{\sqrt 3}$. Adjacency $A_1$ can better avoid leaking across object’s borders during segmentation, while $A_{\sqrt 3}$ is necessary for operators based on fast and more precise Euclidean dilation and erosion.

本文中的图像算子使用两种相邻关系，$A_1$ and $A_{\sqrt 3}$。相邻关系$A_1$可以更好的避免分割时目标边界之间的泄漏，而$A_{\sqrt 3}$是基于快速更加准确的欧式膨胀和腐蚀等算子所必须的。

The IFT is a framework to develop image processing operators based on optimum connectivity. For a given image graph $(D_I, A)$, a path-cost function f must be defined for any path $π_q = <p_1,p_2,...,p_n = q>, (p_i, p_{i+1} ∈ A, i = 1,2,..., n-1$, in the set $Π_q$ of all possible paths with terminus q, including the trivial ones $π_q = <q>$. The general IFT algorithm essentially minimizes a cost map $C(q) = min_{π_q ∈ Π_q} \{ f(π_q) \}$ by partitioning the graph into an optimum-path forest P — that is, an acyclic map that assigns either the predecessor $P(q) = p ∈ D_I$ in the optimum path $π_q^*$ or a symbol $P(q) = nil \notin D_I$ when q is a root of the map. That is, each optimum path $π_q^*$ is stored backwards in P.

IFT是基于最优连接性开发图像处理算子的框架。对于一幅给定的图像图$(D_I, A)$，必须对任意路径$π_q = <p_1,p_2,...,p_n = q>, (p_i, p_{i+1} ∈ A, i = 1,2,..., n-1$定义一个路径-损失函数f，这是以q为终点的所有可能路径的集合$Π_q$中的一条路径，包括无意义的$π_q = <q>$。一般意义上的IFT算法，本质上是通过将图分割成一个最优路径的森林P，最小化一个损失图$C(q) = min_{π_q ∈ Π_q} \{ f(π_q) \}$，即，是一个无循环图，要么在最优路径中指定节点的前序$P(q) = p ∈ D_I$，当q是一张图的根节点时，$P(q) = nil \notin D_I$。即，每条最优路径$π_q^*$在P中都是反向存储的。

The IFT algorithm is a generalization of Dijkstra’s algorithm for less restrictive path-cost functions. This algorithm starts from a predecessor map with only trivial paths. At each iteration, it extends an optimum path $π_p^*$ by an arc 〈p,q〉, namely $π_p^* · <p,q>$, whenever the cost $f(π_p^* · <p,q>)$ is less than the cost $f(π_q)$ of the current path with terminus q. In such a case, $π_q$ is substituted by $π_p^* · <p,q>$. This is done by simply setting P(q)<-p in the algorithm. In other words, the minima of an initial cost map $C_0$, where $C_0(q) = f(<q>)$ for all $q ∈ D_I$, compete among themselves by extending paths to their most closely connected nodes, as a rooted region growing process. In the end, the roots of the forest are the winner minima of the final cost map $C ≤ C_0$ — i.e., $C(q) ≤ C_0(q)$ for all $q ∈ D_I$. This process is described in Algorithm 1.

IFT算法是Dijkstra算法对没那么严格的路径损失函数的泛化。算法起始于一个前序图，只有无意义的路径。在每次迭代中，它通过一个弧〈p,q〉拓展出一个最优路径$π_p^*$，即$π_p^* · <p,q>$，只要代价值$f(π_p^* · <p,q>)$比目前以q为终点的当前路径的代价$f(π_q)$要小。在这样的情况下，$π_q$就被$π_p^* · <p,q>$替换掉。这在算法中只要设P(q)<-p就可以完成。用其他的话说，初始代价图$C_0$的最小值，其中对于所有$q ∈ D_I$，都有$C_0(q) = f(<q>)$，通过将路径拓展到其最接近的节点，并计算代价值进行竞争，作为一个有根源的区域生长过程。最后，森林的根是最后的代价图的获胜者最小值$C ≤ C_0$，即，对于所有$q ∈ D_I$，有$C(q) ≤ C_0(q)$。这个过程如算法1所描述。

Given certain conditions, $π_p$ is always optimum when p is removed from Q in Line 4. However, the resulting P is always an acyclic map and some applications do not even require C to be minimum for a path-cost function f. In most cases, the priority queue Q can be implemented such that Algorithm 1 executes in time proportional to the number of nodes (this is possible for all operations in ALTIS).

给定特定的条件，在第4行中，当p从Q中移除时，$π_p$永远是最优的。但是，得到的P永远是一个非循环图，一些应用对路径代价函数f甚至不需要C是最小的。在多数情况下，优先度队列Q可以进行实现，这样算法1的执行时间正比于节点数量（在ALTIS的所有操作中，这都是有可能的）。

The IFT-based image operators may require simple variants and/or multiple executions of the IFT algorithm followed by some local processing of the output: paths, costs, roots, root labels, etc. ALTIS is presented in the next section with examples of path-cost functions for variations of Algorithm 1.

基于IFT的图像算子可能需要IFT算法简单的变体和/或多个执行，然后加上一些输出的局部处理：路径，代价，根，根的标签，等。下一节中给出ALTIS，和路径代价函数的一些例子，作为算法1的变体。

### 2.3 ALTIS

ALTIS consists of three steps — lungs-and-trachea extraction, seed estimation for each object, and seed labeling and delineation of each lung and trachea. The method assumes that the patient orientation in the CT scan is from inferior to superior along the axial slices (z-axis), from right to left along the sagittal slices (x-axis), and from anterior to posterior along the coronal slices (y-axis). Therefore, in a coronal slice, the lungs and trachea appear in the upright position, being the right lung on the left side of the slice.

ALTIS包含三个步骤，肺和气管的提取，对每个目标的种子估计，给种子打上标签和每个肺和气管的勾画。此方法假设，CT扫描中的患者方向，沿着轴向slice(z向)是从下到上，沿着矢状slice(x向)是从右到左，冠状slice(y向)是从上到下。因此，在冠状slice中，肺部和气管在竖直的位置出现，在slice的左出现的是右肺。

#### 2.3.1 Lungs-and-trachea extraction

Let $\hat I_1 = (D_I, I_1)$ be an input CT scan. The strategy for lungs-and-trachea extraction starts by enhancing the majority of voxels inside the lungs and trachea, which appears darker than the surrounding tissues. This idea, however, does not work if applied directly to the volume since the trachea is connected to the image’s border. Therefore, it requires one execution of Algorithm 1 for each axial slice $\hat I_1^{(k)} = (D_I^{(k)}, I_1^{(k)})$ of $\hat I_1$, where $D_I^{(k)} ⊂ D_I, k∈[1,k]$, and K is the number of axial slices. This strategy is detailed next.

令$\hat I_1 = (D_I, I_1)$是一个输入CT扫描。肺和气管的提取策略，首先增强肺和气管内的主要体素，这些体素看起来比周围的组织要暗。这个思想，如果直接应用到体中，不会好用，因为气管是连接到图像的边缘的。因此，要对$\hat I_1$的每个轴向slice$ \hat I_1^{(k)} = (D_I^{(k)}, I_1^{(k)})$应用算法1，其中$D_I^{(k)} ⊂ D_I, k∈[1,k]$，其中K是轴向slice的数量。这个策略下面进行详述。

For each given $\hat I_1^{(k)}$, with $k∈[1,k]$, whenever exist dark pixels inside the lungs and trachea, they can only be reached from paths rooted at the image’s border S by passing through the brighter surrounding pixels (Fig. 5). By considering all possible paths from S to each and every voxel and choosing the one whose maximum intensity along, it is minimum, the optimum path-cost map $C^{(k)}$ will preserve as much as possible the original intensities in $I_1^{(k)}$ except for the lungs and trachea [Fig. 6(a)]. Therefore, the difference $C^{(k)}-I_1^{(k)}$ will eliminate most background pixels and preserve those dark pixels inside the lungs and trachea as bright pixels in the residual image $\hat I_2^{(k)} = (D_I^{(k)}, I_2^{(k)})$, where $I_2^{(k)} (p) = C^{(k)} (p) - I_1^{(k)} (p)$ for all $p∈D_I^{(k)}$ [Figs. 6(b) and 6(c)]. These optimal path-cost maps $C^{(k)}, ∀k∈[1,K]$ are computed by Algorithm 1 when executed in the graphs $(D_I^{(k)}, A_1)$ using path-cost function $f_{peak}$, where

对于每个给定的$\hat I_1^{(k)}$，其中$k∈[1,k]$，只要在肺和气管的内部存在暗的像素，当路径的根在图像边缘S时，要达到这些像素，必须经过周围较亮的像素（图5）。考虑从S到每个像素的所有可能路径，选择沿路径经过的最大亮度是最小的，最优路径代价图$C^{(k)}$会保留$I_1^{(k)}$中尽可能多的原始亮度，除了肺和气管（图6a）。因此，其差值$C^{(k)}-I_1^{(k)}$会去除多数背景像素，并保留肺部和气管中的那些暗的像素，成为残留图像中的亮像素$\hat I_2^{(k)} = (D_I^{(k)}, I_2^{(k)})$，其中对所有$p∈D_I^{(k)}$，有$I_2^{(k)} (p) = C^{(k)} (p) - I_1^{(k)} (p)$（图6b和图6c）。这些最佳路径代价图$C^{(k)}, ∀k∈[1,K]$都由算法1计算得到，即使用路径代价函数$f_{peak}$，在图$(D_I^{(k)}, A_1)$中执行算法1，其中

$\begin{matrix} f_{peak}(<q>) = \left \{ \begin{matrix} 0 & if q∈S \\ +∞ & otherwise \end{matrix} \right. \\ f_{peak}(π_p · <p,q>) = max\{ f_{peak}(π_p), I_1^{(k)}(q) \} \end{matrix}$(2)

Such enhancement allows the creation of a binary mask by automatic thresholding and largest connected component selection [Fig. 6(d)]. The threshold T is defined as a percentage above Otsu’s threshold, $T = κT_{otsu}$ where κ>1, and the largest component is selected in ($D_I, A_{\sqrt 3}$). The optimum value of κ is learned from a training set of normal images and desired masks (Section 2.D). The resulting mask is named $\hat I_3 = (D_I, I_3)$, where $I_3(p) = 1$ for voxels in the mask and $I_3(p) = 0$ otherwise. However, bright tissues inside the lungs appear as “holes” — that is, voxels with $I_3(p) = 0$. A morphological closing and a closing of holes operation are then necessary to eliminate them, including those bright tissues as part of the final binary mask $\hat I_4 = (D_I, I_4)$[Fig. 6(e)] — a volume of interest that represents the lungs-and-trachea as one connected component.

这样的增强就可以通过自动阈值和最大连通部分选择，创造一个二值mask（图6d）。阈值T定义为Otsu阈值上一定百分比之上的值$T = κT_{otsu}$，其中κ>1，选择($D_I, A_{\sqrt 3}$)中最大的组成部分。κ的最优值是从正常图像的训练集和期望的masks中学习到的（2.4节）。得到的mask命名为$\hat I_3 = (D_I, I_3)$，其中对于mask中的体素$I_3(p) = 1$，其他的地方$I_3(p) = 0$。但是，肺部中亮的组织会以小洞的形式出现，即$I_3(p) = 0$体素。这时就需要用形态学的闭运算来消除这些洞，包括那些亮的组织都成为了最终的二值mask的一部分$\hat I_4 = (D_I, I_4)$（图6e），这时一个感兴趣体，将肺和气管作为一个连通部分进行表示。

The morphological closing requires a dilation followed by an erosion using a disk of radius γ≥1. Since $\hat I_3$ is a binary mask, its dilation can be more efficiently computed by the Euclidean distance transform (EDT) from the internal boundary S of $\hat I_3$, defined as $S= \{ p ∈ D_I | I_3(p) = 1 and ∃q ∈ A_1(p)|I_3(q) = 0 \}$. This outputs a dilated mask (intermediate image) $\hat I'_3 = (D_I, I'_3)$ and its external boundary in a new version of S for the subsequent erosion, defined as $S= \{ p ∈ D_I | I_3(p) = 0 and ∃q ∈ A_1(p)|I_3(q) = 1 \}$. Similarly, the erosion can use the EDT to propagate background voxels from the output set S inside the mask.

形态学的闭运算需要一个膨胀和腐蚀运算，使用的圆盘半径γ≥1。由于$\hat I_3$是一个二值mask，其膨胀可以由欧式距离变换(EDT)从$\hat I_3$的内部边缘S更加高效的计算，定义为$S= \{ p ∈ D_I | I_3(p) = 1 and ∃q ∈ A_1(p)|I_3(q) = 0 \}$。其输出了一个膨胀的mask（中间图像）$\hat I'_3 = (D_I, I'_3)$，其在新版S中的外边缘进行后续的腐蚀，定义为$S= \{ p ∈ D_I | I_3(p) = 0 and ∃q ∈ A_1(p)|I_3(q) = 1 \}$。类似的，腐蚀可以使用EDT来从mask内的输出集S中传播背景体素。

In order to devise an IFT algorithm for morphological dilation (erosion), we can interpret $\hat I_3$ as a graph $(D_I, A_{\sqrt 3})$ and define the path-cost function $f_{edt}$.

为为形态学膨胀（腐蚀）设计一个IFT算法，我们可以将$\hat I_3$解释为一个图$(D_I, A_{\sqrt 3})$，定义路径-损失函数$f_{edt}$。

$$\begin{matrix} f_{edt}(<q>) = \left \{ \begin{matrix} 0 & if q∈S \\ +∞ & otherwise \end{matrix} \right. \\ f_{edt}(π_p · <p,q>) = |q-R(p)|^2 \end{matrix}$$(3)

where R(p) is the root of path $π_p$. 其中R(p)是路径$π_p$的根。

Morphological dilation by the IFT algorithm computes minimum squared distance values from an input set S of voxels formed by the mask’s ($\hat I_3$) outer border up to a given distance γ ≥ 1, while foreground voxels are propagated to obtain $\hat I'_3$. In comparison with Algorithm 1, this operation processes only voxels whose distances to the input set S are less than or equal to γ. The value of γ is also found based on training images (Section 2.D). The time complexity of the dilation is proportional to the number of voxels within distance γ from the input set S.

IFT算法进行的形态学膨胀，计算的是从由mask($\hat I_3$)外轮廓输入体素集合S，到一个给定的距离γ ≥ 1的最小平方距离值，而前景体素经过传播，得到$\hat I'_3$。与算法1相比，这种运算处理的只有其到输入集合S的距离小于或等于γ的体素。γ的值也是基于训练图像的（2.4节）。膨胀的时间复杂度与与输入集合距离小于γ的体素的数量成正比。

The morphological erosion by IFT is very similar to the dilation — that is, minimum squared distances values are computed from the set S now composed by the dilation’s ($\hat I'_3$) inner border up to a given distance γ ≥ 1, while background voxels are propagated to obtain $\hat I_4 = (D_I, I_4)$.

基于IFT的形态学腐蚀与膨胀非常类似，即，计算的是最小平方距离值，从由($\hat I'_3$)内边缘的膨胀组成的集合S，到给定距离γ ≥ 1，而背景体素经过传播，得到$\hat I_4 = (D_I, I_4)$。

The morphological closing operation is ideal to cope with veins and small structures inside the lungs. However, large structures such as advanced tumors may be left outside the binary mask [Fig. 7(b)]. To circumvent this situation, a closing of holes operation is required, similar to the one used for enhancing $\hat I_1^{(k)}$. The difference is that it is applied over the entire volume of $\hat I_4 = (D_I, I_4)$ instead of on its slices. The idea is to close all dark regions that are surrounded by brighter voxels independent of their size, which in this case are large anomalies. For that, Algorithm 1 is applied over $\hat I_4 = (D_I, I_4)$ with the seeds being voxels at the borders of the 3D image and using path-cost function $f_{peak}$. This results in $\hat I_4 = (D_I, I_4)$ without “holes” [Fig. 7(c)].

形态学闭合运算对于处理静脉和肺中的小结构非常理想。但是，大的结构，比如后期的肿瘤，可能会在二值mask之外（图7b）。为防止这种情况，需要一个孔洞的闭合操作，与用于增强$\hat I_1^{(k)}$的类似。差异在于，这是应用于$\hat I_4 = (D_I, I_4)$的整个体的，而不是其slice的。其思想是，使所有被较亮的体素包围的暗区域闭合，与其大小无关，这是较大的异常的情况。为此，算法1应用于$\hat I_4 = (D_I, I_4)$上，种子定义于3D图像的边缘体素，使用的路径代价函数为$f_{peak}$。这样处理的结果是不带孔的$\hat I_4 = (D_I, I_4)$（图7c）。

The binary mask $\hat I_4 = (D_I, I_4)$ (lungs-and-trachea object) and its boundary S are used next to estimate seeds outside and inside the lungs.

二值mask $\hat I_4 = (D_I, I_4)$（肺和气管目标）及其边缘S，下一步用于估计肺部内外的种子。

#### 2.3.2 Seed estimation

In this section, for a given volume of interest [Fig. 8(a)], we aim to estimate makers (seed set) $S_e$ outside the lungs-and-trachea object, markers $S_l$ inside each of the lungs, and markers $S_t$ inside the trachea for the subsequent object delineation by optimum seed competition.

本节中，对于一个给定的感兴趣体（图8a），我们的目标是估计肺和气管目标外的标记点（种子点集合）$S_e$，每个肺内部的标记点$S_l$，气管中的标记点$S_t$，以通过最佳种子竞争进行后续的目标勾画。

First, the voxels on the external boundary of $\hat I_4$ must be included on its boundary set S. Then, a simultaneous dilation, up to distance $γ_e ≥ 1$, and a erosion, up to distance $γ_i ≥ 1$, from that extended set must be computed to create external $S_e$ and internal $S_i$ markers, respectively [Fig. 8(b)]. This operation can also output a dilated mask $\hat I'_4$ which, in this case, must include $S_e$. The role of $\hat I'_4$ is to reduce the image region for object delineation in the next section. The two largest connected components in $(S_i, A_{\sqrt 3})$ define $S_l$ and $S_r$. $S_l$ is the one whose center voxel’s x-coordinate is higher while $S_r$ is the remaining one. The values $γ_i$ and $γ_e$ were found by training (Section 2.D), and they must be enough to eliminate the trachea and disconnect $S_i$ into connected components that fall inside the left and right lungs.

首先，在$\hat I_4$的外轮廓上的像素必须包括在其边缘集合S中。然后，对这个拓展的集合，同时需要进行膨胀，最大的距离为$γ_e ≥ 1$，还需要进行一个腐蚀，最大距离为$γ_i ≥ 1$，以计算后分别创建外部的$S_e$标记点和内部的$S_i$标记点（图8b）。这个运算也可以输出一个膨胀的mask $\hat I'_4$，在这个情况下应当包括$S_e$。$\hat I'_4$的角色是将图像区域缩小到下一节的目标勾画中。在$(S_i, A_{\sqrt 3})$中最大的两个连通区域是$S_l$和$S_r$。$S_l$是其中心体素的x坐标更大的那个，而$S_r$是剩下的那个。$γ_i$和$γ_e$是通过训练发现的（2.4节），它们需要足够大以消除气管，与落入左肺和右肺的连接部分，$S_i$要与之断开连接。

The marker $S_t$ inside the trachea can be obtained in time proportional to the size of a set $N ⊂ D_I$, whose elements are the voxels-1 (voxels with value 1 in a binary mask) in $\hat I_4$. For a graph $(N, A_{\sqrt 3})$ and geodesic path-cost function $f_{gdt}$ [Eq. 4], the lengths of the geodesic paths from $S_i$ are computed to every voxel in N by a slight modification of Algorithm 1 (i.e., $D_I$ is substituted by N in Line 1 and Line 8 is simplified to set tmp<-C(p)+w(〈p,q〉)).

气管内部的标记$S_t$获取的时间，与集合$N ⊂ D_I$的大小是成正比的，其元素是在$\hat I_4$中的体素-1（在二值mask中值为1的体素）。对于图$(N, A_{\sqrt 3})$，和测地线的路径代价函数$f_{gdt}$（式4），从$S_i$得到的测地线的路径的长度，是通过算法1的略微修改，对N中的每个体素进行计算得到的（即，在第1行中将N替换成$D_I$，第8行简化成设置tmp<-C(p)+w(〈p,q〉)）。

$$\begin{matrix} f_{gdt}(<q>) = \left \{ \begin{matrix} 0 & if q∈S_i \\ +∞ & otherwise \end{matrix} \right. \\ f_{gdt}(π_p · <p,q>) = f_{gdt}(π_p)+w(〈p,q〉) \end{matrix}$$(4)

where w(〈p,q〉) is the integer approximation of 10║q-p║ [Fig. 8(c)]. Since the algorithm works with integer values, the factor 10 that multiplies the norm ║q-p║ is needed in order to shift decimal distances into workable integer values. Given the knowledge of the patient’s orientation and the premise that the trachea is a long and thin object, we are interested in the connected component (voxels-1) with the highest z-coordinate of the center in ($N, A_{\sqrt 3}$) after thresholding the geodesic path-length map C from Algorithm 1. In other words, $p∈S_t$ if $C(p) ≥ αC_{max}$, such that $C_{max} = max_{∀p∈N} \{ C(p) \}, α ∈ (0,1)$, and p belongs to that highest component [Figs. 8(c) and 8(d)]. The value of α is found by training (Section 2.D) and $S_e ∩ S_l ∩ S_r ∩ S_t =Ø$ [Fig. 8(e)].

其中w(〈p,q〉)是10║q-p║的整数近似（图8c）。由于算法是以整数值运行的，所以范数║q-p║需要乘以系数10，以将分数值的距离转移到可用的整数值。给定患者方向的知识，和气管是一条长长细细的目标的前提，我们感兴趣的是，在将算法1中的几何路径长度图C用阈值进行处理后，($N, A_{\sqrt 3}$)的中间z值最高的连通域(voxels-1)。换句话说，如果$C(p) ≥ αC_{max}$，$p∈S_t$，这样$C_{max} = max_{∀p∈N} \{ C(p) \}, α ∈ (0,1)$，p属于最高的组成部分（图8c和8d）。α的值是通过训练得到的（2.4节），且$S_e ∩ S_l ∩ S_r ∩ S_t =Ø$（图8e）。

#### 2.3.3 Seed labeling and object delineation

In this section, the objects of interest are delineated by optimum seed competition. That is, the seeds are initially labeled by λ, such that for all $p∈S_e, λ(p)=0$; for all $p∈S_l, λ(p) = 1$; for all $p∈S_r, λ(p) = 2$; and for all $p∈S_t, λ(p) = 3$, and subsequently they compete with each other to propagate the corresponding labels to their most closely connected voxels. This process is implemented by a sequence of two IFTs (Algorithm 1).

本节中，感兴趣目标是通过最优种子竞争进行勾画的。即，种子最开始是由λ标记的，这样对于所有的$p∈S_e, λ(p)=0$，对所有的$p∈S_l, λ(p) = 1$，对所有的$p∈S_r, λ(p) = 2$，对所有的$p∈S_t, λ(p) = 3$，结果他们互相竞争，将对应的标签传播到其最紧密相连的体素中。这个过程是通过2个IFT（算法1）的序列实现的。

Let $\hat I_5 = (D_I, I_5)$ be a gradient image computed from $\hat I_2 = (D_I, I_2)$[Fig. 8(f)], such that

令$\hat I_5 = (D_I, I_5)$是从$\hat I_2 = (D_I, I_2)$计算得到的梯度图像，这样

$$I_5(p) = \frac {\sum_{∀q∈A_{\sqrt 3}(p)} |I_2(q)-I_2(p)|d(p,q)} {\sum_{∀q∈A_{\sqrt 3}(p)} d(p,q)}$$(5)

and $d(p,q)=exp(\frac {-||q-p||^2} {6})$. The first IFT uses $f_{peak}$ [Eq. 2] with $\hat I_5, S = S_e ∪ S_l ∪ S_r ∪ S_t$, and a graph $(D_I, A_{\sqrt 3})$. As a result, the labels of $S_e$, $S_l$, $S_r$, and $S_t$ are propagated along optimum paths to create a label map L [Figs. 8(g) and 8(h)]. This first object delineation can correctly segment most parts of the lungs and trachea, but the high gradient values at voxels in narrow parts of these objects may leave those parts conquered by background seeds (Fig. 9, left). The second IFT can solve the problem by using larger seed sets inside those objects and $I_1(q)-I_1(p)$ rather than $I_5(q)$ in $f_{peak}$ (Fig. 9, right). The larger seed sets are composed by the n-th ancestor of each terminal voxel in the optimum paths rooted at the original seeds. The external seeds remain the same and the new seed sets are labeled by the previous IFT.

以及$d(p,q)=exp(\frac {-||q-p||^2} {6})$。第一个IFT使用的是$f_{peak}$（式2），$\hat I_5, S = S_e ∪ S_l ∪ S_r ∪ S_t$，和一个图$(D_I, A_{\sqrt 3})$。结果是，$S_e$, $S_l$, $S_r$和$S_t$的标签沿着最佳路径传播，创建了标签图L（图8g和8h）。第一个目标勾画可以准确的分割肺和气管的大部分，但是在这些目标的很窄的部分的体素的高梯度，可能会使这些部分被背景种子征服（图9左）。第二个IFT可以通过使用在这些目标内部的更大种子集，以及使用$I_1(q)-I_1(p)$，而不是$f_{peak}$中的$I_5(q)$，来解决这个问题（图9右）。更大的种子集是由原始种子为根的最佳路径的每个最终体素的第n个祖先。外部种子维持相同，新的种子集由之前的IFT进行标记。

In CT images, the lungs may be connected by an anterior/ posterior junction line which impedes their separation. Several methods try to solve this problem through the computation of a manifold formed by that line on every slice throughout the volumetric image and then disconnecting the lungs. ALTIS circumvents this problem automatically during the optimum seed competition, eliminating the need for additional processing steps. That is, even with a weak contrast, the path cost over that line is higher or equal for the markers on both lungs. Therefore, neither marker conquers the influence region of the other, leaving the lungs disconnected in all scenarios. Figure 10 shows the optimum seed competition over that specific region.

在CT图像中，肺部可能由一个上/下节点线连接到了一起，这阻碍了其分割。几种方法通过计算在整个体中的每个slice中的线形成的流形，然后将肺分割开来，来试图来解决这个问题。ALTIS在最优种子竞争时自动避免了这个问题，这样就不用额外的步骤了。即，即使对比度比较低，在这条线上的路径代价也会比在两个肺上的标记点要高，或相等。因此，两种标记点都不会征服另一方的影响区域，在所有场景中肺都会是分离的。图10展示了在特定区域的最优种子竞争。

### 2.4 Training

ALTIS requires the choice of κ (a factor of the Otsu’s threshold), γ (a morphological closing radius), $γ_i$ (the erosion distance for $S_i$ estimation), $γ_e$ (the dilation distance for $S_e$ estimation), α (a threshold on the geodesic distance map to select $S_t$), and n (the number of predecessors for the path-cost function $f_{mean}$). The 50 normal scans were used to determine by grid search the best choice for these parameters. We found κ=1.2, γ=2.5mm, $γ_i$ = 18.75mm, $γ_e$ = 2.5mm, α = 0.7, and n = 10. These values were fixed for all experiments.

ALTIS需要选择κ（Otsu阈值的因子），γ（形态学闭合半径），$γ_i$（$S_i$估计的腐蚀半径），$γ_e$（$S_e$估计的膨胀距离），α（测地线距离图的阈值，用于选择$S_t$），和n（路径代价函数$f_{mean}$的前序数量）。用了50个扫描，通过网格搜索来确定这些参数的最佳选择。我们发现κ=1.2, γ=2.5mm, $γ_i$ = 18.75mm, $γ_e$ = 2.5mm, α = 0.7, n = 10。这些值对所有试验都是固定的。

### 2.5 Compared Methods

We have selected three methods for comparison with ALTIS: (a) SOSM-S is a segmentation method which combines object delineation based on the IFT algorithm with a probabilistic atlas; (b) MALF is a multi-atlas label fusion approach based on the atlas selection and label fusion; and (c) PTK (Pulmonary Toolkit) is a region growing method based on 3D Gaussian filtering, thresholding, and 3D connected component selection. To segment the airways, a starting point is selected in the trachea and a controlled region growing process is applied to avoid leaking inside the lungs. Once the trachea has been segmented, the left and right lungs are defined as the two connected components remaining. If necessary, a morphological erosion is used to separate the lungs.

我们选择了三种方法进行比较：(a)SOSM-S是将IFT算法与概率atlas结合起来的目标勾画方法；(b)MALF是一种多atlas标签融合的方法，基于atlas选择和标签融合；(c)PTK是一种区域生长方法，基于3D高斯滤波，阈值和3D连接区域选择。为分割气道，在气管中选择了一个起始点，应用了一个受控的区域生长过程，以避免泄漏道肺中。一旦分割了气管，左肺和右肺定义为两个剩余的连通区域。如果可能的话，就使用形态学腐蚀来分割双肺。

For the probabilistic atlas of SOSM-S, the 50 normal scans and their respective masks were registered into the reference image coordinate system using the Elastix and Transformix Softwares. The probability of a voxel v in the reference system is the percentage of times that v belonged to the object of interest in all masks. This generates a probabilistic atlas for each object. Then, for a new test image, it is registered into the reference system and the probabilistic atlas is translated over a predefined search region. At each new position, the IFT-based watershed algorithm is performed for object delineation. Among all delineations, the resulting segmentation is the one whose mean gradient value along the boundary is maximum.

对于SOSM-S的概率atlas，50个正常的扫描及其masks与参考图像坐标系系统对齐，使用的是Elastix和Transformix软件。参考系统中的一个体素v的概率是，在所有masks中，v属于感兴趣目标的次数占比。这对每个目标生成了一个概率atlas。然后，对于一幅新的测试图像，与参考系统配准，其概率atlas在一个预定义的搜索区域中进行平移。在每个新的位置，对目标勾画进行基于IFT的分水岭算法。在所有勾画中，得到的分割是其平均梯度值沿着边缘是最大的。

For the MALF method, all 50 normal scans reserved for training were registered in the coordinate system of each test image, also using the Elastix and Transformix Softwares. Then, for each voxel v of a given test image, the STAPLE algorithm is applied to measure the performance level of v in the registered training set. The final segmentation is given by weighting labels depending upon the estimated performance level of v and selecting the best one.

对于MALF方法，所有保留用于训练的50个正常的扫描，对每个测试图像的坐标系统进行配准，也使用的是Elastix和Transformix软件。然后，对于一幅给定的测试图像的每个体素v，使用STAPLE算法来度量v在配准的训练集中的性能水平。最终的分割是由标签加权得到的，这依赖于v的估计性能水平和选择的最好的那个。

PTK is a region growing method based on image processing operators. Its parameters were fixed as proposed in Ref. [26]. Thus, the training set was not used for this method.

PTK是一种区域生长方法，基于图像处理算子。其参数与[26]中建议的一样。因此，训练集在这种方法中并没有应用。

## 3. Results

The comparison among the four methods was based on two measures, DICE (the Sorensen–Dice index) and ASSD (average symmetric surface distance in mm). We also performed analysis of variance (ANOVA) with the post hoc Tukey honest significant difference (HSD) test, using the significance level 0.05 (P-value).

四种方法的比较是基于两种度量，DICE和ASSD。我们还进行了方差分析(ANOVA)，还有事后的Tukey honest significant difference (HSD)测试，使用的显著性水平为0.05（P值）。

Whenever the segmentation of the respiratory system works, the robustness of ALTIS with respect to the estimated seeds can be justified by using the concept of segmentation core. A segmentation core can be defined as a maximal connected region, wherein any voxel selected as a seed will always produce the same segmentation result. Each object may present multiple cores in the IFT framework in order to produce its gold standard segmentation. It is crucial that each core contains at least one seed voxel, such that the seeds can cover the highest number of cores as possible. This measure was obtained for each object over all available images from each data set as an indication of robustness.

不论何时呼吸系统的分割好用，ALTIS对估计的种子的稳健性，可以通过使用分割核的概念进行调整。分割核可以定义为一个最大连通区域，其中任意选择为种子的体素都可以产生相同的分割结果。每个目标在IFT框架中，会展现出多个核，以产生其金标准分割。非常关键的是，每个核都至少包含一个种子体素，这样种子可以覆盖尽可能多数量的核。对每个目标在所有可用的图像上计算这个度量，可以作为一种稳健性的指示。

Tables I and II present the DICE (higher is better) and ASSD (lower is better) measures for SOSM-S, MALF, PTK, and ALTIS, respectively. The numbers are the mean and standard deviation values of all instances of each object: left lung, right lung, and trachea. Table III presents the result of the statistical ANOVA and Tukey test. The statement A < B implies that method B is superior to method A with statistical significance. The symbol ≃ indicates statistical equivalence between methods and the statement A ≤ B indicates that method B was superior to method A in most cases, but it was not enough to grant statistical significance. Table IV presents the mean percentage of cores that had any intersection with the seeds estimated by ALTIS for each data set.

表1和表2给出了四种方法SOSM-S，MALF，PTK和ALTIS的DICE和ASSD度量。数值是每个目标的所有实例的均值和标准差值：左肺，右肺和气管。表3给出了ANOVA和Tukey测试的统计结果。A < B的意思是，方法B比方法A在统计意义上是优越的。符号≃说明这两种方法具有统计的等效性，A ≤ B说明，方法B在多数情况下是比方法A要优越的，但并不足以得出统计意义上的优越性。表4表示在每个数据集上，核心与ALTIS方法估计的种子有交集的比例。

Although LOLA11 is used in this study, the challenge’s website defines the entire space of the organ as the gold standard, which might include fluid in the parenchyma for instance. In our study, the gold standard is the functional lung with brighter structures, such as vessels and nodules. We intend to analyze abnormalities in the parenchyma and pleura after abnormal lung shape identification. Due to that difference in gold standard definition, our segmentation results achieve mean score 93.5% at the challenge’s website, leading us to the 27th position.

虽然本文中使用了LOLA11，挑战赛的网站定义了器官的整个空间作为金标准，这可能在薄壁组织中包含液体。在我们的研究中，金标准是带有明亮组织的功能肺，明亮组织有血管和结节。我们想在异常肺形状识别后，在薄壁组织和胸膜中分析出异常。由于金标准定义的不同，我们的分割结果在挑战赛网站上获得了93.5%的分数，我们得到了第27名的位置。

The mean execution times and their standard deviations for each method are as follows: 11±5 min for SOSM-S, 88±23 min for MALF, 1.75±1.55 min for PTK, and 18±4.25 seconds only for ALTIS.

每种方法的平均执行时间和标准偏差如下：SOSM-S 11±5 min，MALF 88±23 min，PTK 1.75±1.55 min，ALTIS 18±4.25s。

## 4. Discussion

According to Tables I and II, we may conclude that, for both metrics, ALTIS is more accurate than SOSM-S, MALF, and PTK with statistical significance for most data sets. Nonetheless, PTK achieved a slightly better performance than ALTIS for the trachea on the ASBESTOS data set, with respect to ASSD, and a similar performance on the LIDC-IDRI data set for both lungs, concerning the DICE.

根据表1和表2，我们可以总结出，对于两种度量，ALTIS在统计意义上在多数数据集上比SOSM-S，MALF和PTK要更加准确。尽管如此，在ASBESTOS数据集上，PTK比ALTIS在气管的ASSD度量上获得了更好的性能，在LIDC-IDRI数据集上，各种方法的双肺DICE指标类似。

According to Table IV, the percentage of overlap between the cores and the estimated seeds is close to 100% for all data sets. It indicates that the stage of seed estimation from ALTIS is robust enough to identify seeds correctly in most cases. Note that seeds outside cores do not cause segmentation errors, as long as they are inside the correct object.

根据表4，核心与估计的种子的重叠百分比，在所有数据集上接近100%。这说明，ALTIS中的种子估计阶段非常稳健，可以在多数情况中识别种子。注意，核心之外的种子不会导致分割错误，只要它们在正确的目标中。

For illustration, Fig. 11 shows the coronal slices of six segmentation results from SOSM-S, MALF, PTK, and ALTIS. In this study, a correct result must separate the left from the right lung spaces delimited by the visceral pleura, and the trachea with the main bronchi (the longest tubular object of connected darker voxels, which lies approximately at the center and the upper half of the volume). This is usually provided by ALTIS. For a patient with thoracic scoliosis (see the last four images of the first row), ALTIS performed as satisfactory as PTK but superior to SOSM-S and MALF. However, ALTIS assumes that the lungs and trachea are not connected to other dark objects, such as bowel gas, and each lung has a reasonable minimum size. Figure 11 shows a case (the first four images of the second row) where the left lung is considerably reduced, as a consequence of a large volume of fluid in the pleural cavity, compressing the lung. ALTIS managed to segment the trachea and both lungs after adjusting its parameters for this specific case. Usually, region growing methods have a difficult time separating the lungs with weak contrasted anterior/posterior junction line. ALTIS can separate the lungs even in that situation unlike PTK in some cases (first four images of the third row).

为描述，图11展示了SOSM-S, MALF, PTK, 和ALTIS的6个分割结果的冠状slices。在本研究中，正确的结果必须将左肺和右肺由胸膜脏层、气管和主支气管的间隔开的分开。通常ALTIS会给出这个结果。对于一个有脊柱侧弯的病人（见第一行的最后四幅图像），ALTIS的表现与PTK一样令人满意，但比SOSM-S和MALF要好。但是，ALTIS假设肺和气管与其他暗的目标并没有连在一起，如肠气，每个肺都有一个合理的最小大小。图11展示了一种情况（第二行的前四幅图像），其中左肺有一定的收缩，是胸膜腔中的大量液体的结果，压缩了肺。对于这种特殊的情况，ALTIS调整了参数后，成功的将气管和双肺分割了出来。通常，区域增长方法很难区分由弱对比的前/后节点线连接的双肺。ALTIS可以在多数情况下都将将肺分开。

To demonstrate the robustness of ALTIS in cases of severe pathologies, Fig. 12 shows ALTIS segmentation for cases of fibrosis, bronchiectasis/tuberculosis, multiple nodules both on the pleura and on the parenchyma, and pulmonary emphysema, respectively. Notice that ALTIS segmented the volume of air of both the lungs and the trachea.

为展现ALTIS在严重病理情况下的稳健性，图12展示了ALTIS在纤维肺，支气管扩张/肺结核，多结节情况下的分割结果。注意，ALTIS分割了肺部和气管的空气体。

Note that all methods present limitations that are very difficult to be automatically treated. For example, in the second row and second block, PTK had a leakage from the left lung to the background (yellow square). In the third row and second block, SOSM-S and MALF have included part of the abdomen as pertaining to the lung, while ALTIS and PTK have correctly detected the diaphragm. ALTIS may exclude brighter tissues from the lung, when they are in contact with the chest wall or the heart, may fail the segmentation of extremely collapsed lungs, due to pneumonia, tumors or the presence of fluid in the pleural space [Figs. 13(a) and 13(b)]. When gases from the bowel are connected to the lungs, ALTIS may include these regions in the segmentation as well [Fig. 13(c)]. Note that for images acquired with voxel size <1.25 mm, the bowel gas would be limited to the diaphragm and so disconnected to the lungs. There are also rare cases where the stretcher is segmented instead of the respiratory system [Fig. 13(d)]. This happens only when the entire stretcher appears in the image, due to the fact that the stretcher forms the largest connected component surrounded by brighter voxels (same idea as shown in Fig. 5). If the stretcher must be entirely inside the image, there is a simple solution to handle the problem. We simply compare the volumes of the connected components during the lungs-and-trachea extraction step, detect the largest one as the stretcher, and choose the second largest one as the lungs-and-trachea object (see Fig. 14).

注意，在自动方法很难处理的情况，所有方法都表现出了局限。比如，在第二行第二个模块，PTK在左肺有泄漏到背景中（黄色方块）。在第三行和第二个模块中，SOSM-S和MALF将部分腹部包括到了肺中，而ALTIS和PTK正确的检测到了隔膜。ALTIS可能会从肺中正确的排除掉了更亮的组织，当它们与胸壁或心脏相接触，对于坍缩的厉害的肺部，可能会分割失败，这是由于肺炎，肿瘤或胸膜中的液体存在造成的（图13a和13b）。当肠管中的气体与肺部连到了一起，ALTIS也会将这些区域包括在分割中（图13c）。注意对于体素大小<1.25mm的图像，肠管气体会被限制在隔膜下，因此与肺部不相连接。也有极少的病例，会将床板分割出来，而不是呼吸系统（图13d）。这种情况只在整个床板都在图像中的时候出现，这样床板会形成最大的连通区域，并且由更亮的体素包围（与在图5中展示的思想一样）。如果床板整个都在图像中，有一个简单的解决方案来处理这个问题。我们只要在肺和气管的提取步骤中，比较连通部分的体积，将最大的一个检测为床板，将第二大的部分作为肺和气管的目标（见图14）。

## 5. Conclusions

We presented a fast and effective solution, named ALTIS, for the automated segmentation of each lung and trachea in CT scans of the thorax. ALTIS showed to be statistically more accurate and considerably more efficient than three other methods, two based on shape models (SOSM-S and MALF) and one based on region growing (PTK). The complete study involved 1255 CT scans from five distinct sources and with several abnormal cases, and to the best of our knowledge, it is the largest data set used for lungs and trachea segmentation. ALTIS has shown to be very fast and robust to intensity and shape variations. However, it may fail in critical situations. This is a subject for future research as well as its extension for screening.

我们提出了一种快速有效的方案自动分割双肺和气管，名为ALTIS。ALTIS比其他三种方法在统计上更加准确更加高效，两种是基于形状模型(SOSM-S和MALF)，一种是基于区域生长(PTK)。完整的研究涉及到了1255个CT扫描，来自于5个不同的源，有几种异常情况，据我们所知，这是肺和气管分割所用到的最大的数据集。ALTIS非常快速，对灰度和形状变化非常稳健。但是，在严重的情况下可能失败。这是未来研究的一个课题，也是要进行筛查的拓展。