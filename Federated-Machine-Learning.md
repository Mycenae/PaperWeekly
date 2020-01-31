# Federated Machine Learning: Concept and Applications

Qiang Yang et al. 

Today’s AI still faces two major challenges. One is that in most industries, data exists in the form of isolated islands. The other is the strengthening of data privacy and security. We propose a possible solution to these challenges: secure federated learning. Beyond the federated learning framework first proposed by Google in 2016, we introduce a comprehensive secure federated learning framework, which includes horizontal federated learning, vertical federated learning and federated transfer learning. We provide definitions, architectures and applications for the federated learning framework, and provide a comprehensive survey of existing works on this subject. In addition, we propose building data networks among organizations based on federated mechanisms as an effective solution to allow knowledge to be shared without compromising user privacy.

今天的AI仍然面临着两个主要的挑战。一个是在大多数行业里，数据存在的形式是孤立的岛。另一个是，数据隐私性和安全性的加强。我们对这些挑战提出一种可能的解：安全联邦学习。联邦学习由谷歌在2016年首先提出，在此之外，我们提出了一种广泛的安全联邦学习框架，包括横向联邦学习，纵向联邦学习和联邦迁移学习。我们给出了联邦学习框架的定义，架构和应用，对这个主题的文章进行了广泛的回顾。另外，我们基于联邦机制提出在组织间构建数据网络，这样就可以进行有效的知识分享，而不用牺牲用户隐私。

CCS Concepts: • Security and privacy; • Computing methodologies → Artificial intelligence; Machine learning; Supervised learning;

Additional Key Words and Phrases: federated learning, GDPR, transfer learning

## 1 Introduction

2016 is the year when artificial intelligence (AI) came of age. With AlphaGo[59] defeating the top human Go players, we have truly witnessed the huge potential in artificial intelligence (AI), and have began to expect more complex, cutting-edge AI technology in many applications, including driverless cars, medical care, finance, etc. Today, AI technology is showing its strengths in almost every industry and walks of life. However, when we look back at the development of AI, it is inevitable that the development of AI has experienced several ups and downs. Will there be a next down turn for AI? When will it appear and because of what factors? The current public interest in AI is partly driven by Big Data availability: AlphaGo in 2016 used a total of 300,000 games as training data to achieve the excellent results.

2016年是人工智能成熟的年份。随着AlphaGo击败顶级人类围棋选手，我们真正见证了人工智能的巨大潜力，并开始在很多应用中期待更复杂的最尖端的AI技术，包括无人驾驶，医学护理，金融，等等。今天，AI技术几乎在所有工业领域和生活的方方面面展现出了应用。但是，当我们回顾AI的发展时，发现AI的发展不可避免的经历了几个起伏。AI的发展会有下一个转弯吗？什么时候会出现，由于什么因素出现呢？目前公众对AI的兴趣，部分是受到大数据可用性的驱动：AlphaGo在2016年使用了共计300000局博弈作为训练数据，以得到优异的结果。

With AlphaGo’s success, people naturally hope that the big data-driven AI like AlphaGo will be realized soon in all aspects of our lives. However, the real world situations are somewhat disappointing: with the exception of few industries, most fields have only limited data or poor quality data, making the realization of AI technology more difficult than we thought. Would it be possible to fuse the data together in a common site, by transporting the data across organizations? In fact, it is very difficult, if not impossible, in many situations to break the barriers between data sources. In general, the data required in any AI project involves multiple types. For example, in an AI-driven product recommendation service, the product seller has information about the product, data of the user’s purchase, but not the data that describe user’s purchasing ability and payment habits. In most industries, data exists in the form of isolated islands. Due to industry competition, privacy security, and complicated administrative procedures, even data integration between different departments of the same company faces heavy resistance. It is almost impossible to integrate the data scattered around the country and institutions, or the cost is prohibited.

随着AlphaGo的成功，人们自然的希望，大数据驱动的AI，如AlphaGo，在我们生活的所有方面都会很快得到实现。但是，真实世界的情况是有些令人失望的：除了少数产业外，多数领域只有有限的数据或很差的数据质量，使得AI技术的实现比想象的要困难。在常见的场景中，可以通过将数据在组织间传递，从而融合到一起吗？实际上，在很多情况下，为打破数据源之间的障碍，即使不是不可能的，这也是非常困难的。总体来说，任何AI项目中需要的数据都涉及到多种类型。比如，在一个AI驱动的产品推荐服务中，产品售卖者有产品的信息，有用户购买的数据，但并没有用户的购买能力的数据和支付习惯的数据。在多数产业中，数据的存在形式是孤立的岛屿。由于产业竞争，隐私安全，和复杂的行政程序，即使是同一家公司的不同部门的数据整合都面临着很大的阻力；要整合分散在不同国家和组织之间的数据，基本是不可能的，或者其代价是不可接受的。

At the same time, with the increasing awareness of large companies compromising on data security and user privacy, the emphasis on data privacy and security has become a worldwide major issue. News about leaks on public data are causing great concerns in public media and governments. For example, the recent data breach by Facebook has caused a wide range of protests [70]. In response, states across the world are strengthening laws in protection of data security and privacy. An example is the General Data Protection Regulation (GDPR)[19] enforced by the European Union on May 25, 2018. GDPR (Figure 1) aims to protect users’ personal privacy and data security. It requires businesses to use clear and plain languages for their user agreement and grants users the "right to be forgotten", that is, users can have their personal data deleted or withdrawn. Companies violating the bill will face stiff fine. Similar acts of privacy and security are being enacted in the US and China. For example, China’s Cyber Security Law and the General Principles of the Civil Law, enacted in 2017, require that Internet businesses must not leak or tamper with the personal information that they collect and that, when conducting data transactions with third parties, they need to ensure that the proposed contract follow legal data protection obligations. The establishment of these regulations will clearly help build a more civil society, but will also pose new challenges to the data transaction procedures commonly used today in AI.

同时，随着大公司越来越意识到，需要在数据安全性和用户隐私性之间折中，对数据隐私和安全性的强调，已经变成了一个世界范围内的主要问题。公共数据泄漏的新闻引起了公共媒体和政府的关心。比如，最近Facebook数据的泄漏已经导致了大范围的抗议。回应措施是，世界范围内的很多国家都加强了保护数据安全和隐私的法律。一个例子是，2018年5月25日欧盟通过的GDPR。GDPR的目的是保护用户的个人隐私和数据安全。它要求商业在其合同中使用清晰简单的语言，授予用户遗忘权，即，用户可以要求其个人数据被删除或撤回。违反这条法规的公司将会被严厉处罚。在美国和中国，也有类似的关于隐私和安全的条文。比如，中国网络空间安全法和民法中的一般条款，于2017年通过，要求互联网商业在与第三方进行数据处理时，不能泄漏收集到的个人信息，他们必须确保提出的合同要符合数据保护的法规。这些法规的确定，很明显的会建立一个更好的社会，但对于AI中经常会用到的数据处理，则提出了新的挑战。

To be more specific, traditional data processing models in AI often involves simple data transactions models, with one party collecting and transferring data to another party, and this other party will be responsible for cleaning and fusing the data. Finally a third party will take the integrated data and build models for still other parties to use. The models are usually the final products that are sold as a service. This traditional procedure face challenges with the above new data regulations and laws. As well, since users may be unclear about the future uses of the models, the transactions violate laws such as the GDPR. As a result, we face a dilemma that our data is in the form of isolated islands, but we are forbidden in many situations to collect, fuse and use the data to different places for AI processing. How to legally solve the problem of data fragmentation and isolation is a major challenge for AI researchers and practitioners today.

更具体一点，AI中传统的数据处理模型，通常涉及到的都是简单的数据处理模型，一个团体负责收集并传输数据给另一个团体，这另一个团体负责数据的清洗和融合。最后，第三方的团体会利用整合过的数据，构建模型，给其他团体来使用。最终的产品，通常是模型，作为一种服务进行销售。这种传统的过程，在上述新的数据规定和法律中，面临着挑战。由于用户可能不清楚模型未来的用处，所以这些处理就会与类似GDPR的法律相冲突。结果是，我们会面临一个两难困境，即数据是孤立岛屿的形式，但在很多情况下，我们不能收集，融合并使用这些数据，进行AI处理。怎样合法的解决数据碎片化和孤立的问题，是今天AI研究者和实践者的一个主要挑战。

In this article, we give an overview of a new approach known as federated learning, which is a possible solution for these challenges. We survey existing works on federated learning, and propose definitions, categorizations and applications for a comprehensive secure federated learning framework. We discuss how the federated learning framework can be applied to various businesses successfully. In promoting federated learning, we hope to shift the focus of AI development from improving model performance, which is what most of the AI field is currently doing, to investigating methods for data integration that is compliant with data privacy and security laws.

本文中，我们回顾了一种新方法，即联邦学习，是这种挑战的一种可能解决方案。我们总结了联邦学习的现有工作，提出了广义联邦学习框架的定义、分类和应用。我们讨论了联邦学习的框架可以怎样成功的应用于各种商业。在推广联邦学习时，我们希望AI发展的关注点，应当从改进模型性能（这是目前AI领域的主要工作），转变到研究与现有数据隐私和安全法律相关的方法上来。

## 2 An Overview of Federated Learning

The concept of federated learning is proposed by Google recently [36, 37, 41]. Their main idea is to build machine learning models based on data sets that are distributed across multiple devices while preventing data leakage. Recent improvements have been focusing on overcoming the statistical challenges [60, 77] and improving security [9, 23] in federated learning. There are also research efforts to make federated learning more personalizable [13, 60]. The above works all focus on on-device federated learning where distributed mobile user interactions are involved and communication cost in massive distribution, unbalanced data distribution and device reliability are some of the major factors for optimization. In addition, data are partitioned by user Ids or device Ids, therefore, horizontally in the data space. This line of work is very related to privacy-preserving machine learning such as [58] because it also considers data privacy in a decentralized collaborative learning setting. To extend the concept of federated learning to cover collaborative learning scenarios among organizations, we extend the original "federated learning" to a general concept for all privacy-preserving decentralized collaborative machine learning techniques. In [71], we have given a preliminary overview of the federated learning and federated transfer learning technique. In this article, we further survey the relevant security foundations and explore the relationship with several other related areas, such as multiagent theory and privacy-preserving data mining. In this section, we provide a more comprehensive definition of federated learning which considers data partitions, security and applications. We also describe a workflow and system architecture for the federated learning system.

联邦学习的概念是最近由Google提出来的。其主要思想是，基于分布在多个设备上的数据，构建机器学习的模型，同时防止数据泄漏。最近的改进聚焦在克服联邦学习的统计性挑战和改进联邦学习的安全性上。也有一些研究工作使联邦学习更加个性化。上述工作都关注的都是设备上的联邦学习，涉及到的是分布式的移动用户交互，要优化的主要因素是，大量分布、不平衡的数据分布、设备可靠性导致的通信代价。另外，数据是根据用户IDs或设备IDs分割的，因此，数据是在数据空间中横向分布的。这条线的工作与隐私保护的机器学习非常相关，如[58]，因为也考虑到了在去中心化合作学习的设置中的数据隐私。为拓展联邦学习的概念，覆盖组织间合作学习的场景，我们将原始的联邦学习的概念，拓展到广义的联邦学习，包括所有的隐私保护的去中心的合作机器学习技术。在[71]中，我们给出了联邦学习和联邦迁移学习技术的初步概览。本文中，我们进一步调查了相关的安全基础，探索了与其他几个相关领域的关系，如multi-agent理论，和隐私保护的数据挖掘。在本节中，我们给出了联邦学习的更广泛的定义，考虑到了数据分割，安全和应用。我们还描述了联邦学习系统的工作流和系统架构。

### 2.1 Definition of Federated Learning

Define N data owners {F1, ... FN}, all of whom wish to train a machine learning model by consolidating their respective data {D1, ... DN}. A conventional method is to put all data together and use D=D1∪...∪DN to train a model M_{SUM}. A federated learning system is a learning process in which the data owners collaboratively train a model M_{FED}, in which process any data owner Fi does not expose its data Di to others. In addition, the accuracy of M_{FED}, denoted as V_{FED} should be very close to the performance of M_{SUM}, V_{SUM}. Formally, let δ be a non-negative real number, if

定义N个数据拥有者{F1, ... FN}，他们都希望将各自的数据{D1, ... DN}联合到一起，训练一个机器学习模型。传统方法是，将所有数据放到一起，使用D=D1∪...∪DN来训练一个模型M_{SUM}。在联邦学习系统中，数据拥有者们共同训练得到一个模型M_{FED}，但在这个过程中，任何数据拥有者Fi都不需要将其数据Di暴露给其他人。另外，模型M_{FED}的准确率，表示为V_{FED}，应当与模型M_{SUM}的性能V_{SUM}非常接近。正式的，令δ为非负实数，如果

$$|V_{FED}−V_{SUM}|<δ$$(1)

we say the federated learning algorithm has δ-accuracy loss. 我们说联邦学习算法有δ准确率的损失。

### 2.2 Privacy of Federated Learning

Privacy is one of the essential properties of federated learning. This requires security models and analysis to provide meaningful privacy guarantees. In this section, we briefly review and compare different privacy techniques for federated learning, and identify approaches and potential challenges for preventing indirect leakage.

隐私是联邦学习的一个基本性质。这需要安全模型和分析，以提供有意义的隐私保证。本节中，我们简要回顾并比较联邦学习用到的不同的隐私技术，并给出防止间接泄漏的方法和可能挑战。

**Secure Multi-party Computation (SMC)**. SMC security models naturally involve multiple parties, and provide security proof in a well-defined simulation framework to guarantee complete zero knowledge, that is, each party knows nothing except its input and output. Zero knowledge is very desirable, but this desired property usually requires complicated computation protocols and may not be achieved efficiently. In certain scenarios, partial knowledge disclosure may be considered acceptable if security guarantees are provided. It is possible to build a security model with SMC under lower security requirement in exchange for efficiency [16]. Recently, studies [46] used SMC framework for training machine learning models with two servers and semi-honest assumptions. Ref [33] uses MPC protocols for model training and verification without users revealing sensitive data. One of the state-of-the-art SMC framework is Sharemind [8]. Ref [44] proposed a 3PC model [5, 21, 45] with an honest majority and consider security in both semi-honest and malicious assumptions. These works require participants’ data to be secretly-shared among non-colluding servers.

**安全的多方计算(SMC)**。SMC安全模型很自然的涉及到多方，在定义良好的模拟框架中给出安全证明，以确保完全的zero knowledge，即，每一方除了知道其输入和输出，其他任何都不会知道。Zero knowledge是非常理想的情况，但这种理想的性质，通常需要复杂的计算协议，可能不会很高效的得到这种性质。在特定场景中，如果有安全保证的话，部分knowledge泄漏是可以接受的。在更低的安全要求下，通过SMC是可以构建一个安全模型的，以换取效率。最近，[46]使用SMC框架使用两台服务器和semi-honest假设，训练了机器学习模型。[33]使用MPC协议进行模型训练和验证，用户不需要暴露敏感数据。一种最新的SMC框架是Sharemind[8]。[44]提出了一种3PC模型，包括一个honest majority，在semi-honest和malicious的假设下，都考虑了安全性问题。这些工作需要参与者的数据在non-colluding的服务器中进行秘密共享。

**Differential Privacy**. Another line of work use techniques Differential Privacy [18] or k-Anonymity [63] for data privacy protection [1, 12, 42, 61]. The methods of differential privacy, k-anonymity, and diversification [3] involve in adding noise to the data, or using generalization methods to obscure certain sensitive attributes until the third party cannot distinguish the individual, thereby making the data impossible to be restore to protect user privacy. However, the root of these methods still require that the data are transmitted elsewhere and these work usually involve a trade-off between accuracy and privacy. In [23], authors introduced a differential privacy approach to federated learning in order to add protection to client-side data by hiding client’s contributions during training.

**差异化隐私**。另一条研究线使用了差异化隐私的技术，或称为k-Anonymity，进行数据隐私保护。差异化隐私，k-Anonymity和diversification的方法，涉及到对数据加入噪声，或使用一般化的方法使特定敏感的属性变模糊，直到第三方不能区分每个个体，从而使得数据不可能恢复，以保护用户隐私。但是，这些方法根源上仍然需要数据传输到别处，这些工作通常都涉及到准确性与隐私的折中。在[23]中，作者提出了一种联邦学习中差异化隐私的的方法，通过隐藏训练过程中客户的贡献，以在用户端增加保护。

**Homomorphic Encryption**. Homomorphic Encryption [53] is also adopted to protect user data privacy through parameter exchange under the encryption mechanism during machine learning [24, 26, 48]. Unlike differential privacy protection, the data and the model itself are not transmitted, nor can they be guessed by the other party’s data. Therefore, there is little possibility of leakage at the raw data level. Recent works adopted homomorphic encryption for centralizing and training data on cloud [75, 76]. In practice, Additively Homomorphic Encryption [2] are widely used and polynomial approximations need to be made to evaluate non-linear functions in machine learning algorithms, resulting in the trade-offs between accuracy and privacy [4, 35].

**同态加密**。同态加密也可以用于保护用户数据隐私，即在机器学习过程中，参数的交换是通过加密机制进行的。与差异化隐私保护的方法不同，数据和模型本身并不进行传输，通过另一方的数据也不可能猜测的到。因此，原始数据级的泄漏概率是很小的。最近的文章采用同态加密进行在云上集中并训练数据。实践中，加性同态加密得到了广泛的使用，需要进行多项式近似，以评估机器学习算法中的非线性函数，得到准确率和隐私的折中。

#### 2.2.1 Indirect information leakage

Pioneer works of federated learning exposes intermediate results such as parameter updates from an optimization algorithm like Stochastic Gradient Descent (SGD) [41, 58], however no security guarantee is provided and the leakage of these gradients may actually leak important data information [51] when exposed together with data structure such as in the case of image pixels. Researchers have considered the situation when one of the members of a federated learning system maliciously attacks others by allowing a backdoor to be inserted to learn others’ data. In [6], the authors demonstrate that it is possible to insert hidden backdoors into a joint global model and propose a new "constrain-and-scale" model-poisoning methodology to reduce the data poisoning. In [43], researchers identified potential loopholes in collaborative machine learning systems, where the training data used by different parties in collaborative learning is vulnerable to inference attacks. They showed that an adversarial participant can infer membership as well as properties associated with a subset of the training data. They also discussed possible defenses against these attacks.In [62], authors expose a potential security issue associated with gradient exchanges between different parties, and propose a secured variant of the gradient descent method and show that it tolerates up to a constant fraction of Byzantine workers.

联邦学习的先驱工作，暴露的是中间结果，如优化算法（如SGD）的参数更新，但是这并没有得到任何安全保证，这些梯度的泄漏，在与图像像素这样的数据结构一起泄漏时，可能实际上泄漏重要的数据信息。研究者考虑了下面的情况，如一个联邦学习系统的成员恶意攻击其他成员，在其他系统中插入一个后门，以学习其他成员的数据。在[6]中，作者证明了，可以在联合全局模型中插入隐藏的后门，并提出一种新的constraint-and-scale的模型毒化方法，以降低数据中毒的情况。在[43]中，研究者在合作机器学习系统中识别出了可能的loopholes，其中合作学习中不同方面使用的训练数据对推理攻击来说非常脆弱。他们展示了，对抗参与者可以推断出成员属性，以及训练数据集子集的相关性质。他们还讨论了对这些攻击可能的防御措施。在[62]中，作者展示了在不同方进行梯度交换时一个可能的安全问题，并提出了一种安全的梯度下降方法，证明了其可以容忍一定部分的Byzantine workers。

Researchers have also started to consider blockchain as a platform for facilitating federated learning. In [34], researchers have considered a block-chained federated learning (BlockFL) architecture, where mobile devices’ local learning model updates are exchanged and verified by leveraging blockchain. They have considered an optimal block generation, network scalability and robustness issues.

研究者也开始考虑区块链作为平台，促进联邦学习的发展。在[34]中，研究者考虑了一种区块链联邦学习架构BlockFL，其中移动设备的局部学习模型更新通过使用区块链来进行交换和验证。他们考虑过一种最佳区块生成、网络可扩展性和稳健性问题。

### 2.3 A Categorization of Federated Learning

In this section we discuss how to categorize federated learning based on the distribution characteristics of the data.

本节中，我们讨论怎样根据数据的分布特征，来对联邦学习系统进行分类。

Let matrix Di denotes the data held by each data owner i. Each row of the matrix represents a sample, and each column represents a feature. At the same time, some data sets may also contain label data. We denote the features space as X, the label space as Y and we use I to denote the sample ID space. For example, in the financial field labels may be users’ credit; in the marketing field labels may be the user’s purchase desire; in the education field, Y may be the degree of the students. The feature X, label Y and sample Ids I constitutes the complete training dataset (I, X, Y). The feature and sample space of the data parties may not be identical, and we classify federated learning into horizontally federated learning, vertically federated learning and federated transfer learning based on how data is distributed among various parties in the feature and sample ID space. Figure 2 shows the various federated learning frameworks for a two-party scenario.

令矩阵Di表示每个数据拥有者i的数据。矩阵的每一行代表一个样本，每一列表示一个特征。同时，一些数据集还可能包含标签数据。我们将特征空间表示为X，标签空间为Y，样本ID空间表示为I。比如，在经济领域，标签可能是用户的信用；在市场领域，标签可能是用户的购买期望；在教育领域，Y可能是学生的分数。特征X，标签Y和样本ID I构成了完整的训练集(I, X, Y)。不同数据方的特征和样本空间可能不是一致的，我们将联邦学习分为三类，横向联邦学习，纵向联邦学习，和联邦迁移学习，这是基于不同数据方特征空间和样本ID空间的数据分布情况。图2展示了双方训练场景下的各种联邦学习框架。

#### 2.3.1 Horizontal Federated Learning

Horizontal federated learning, or sample-based federated learning, is introduced in the scenarios that data sets share the same feature space but different in samples (Figure 2a). For example, two regional banks may have very different user groups from their respective regions, and the intersection set of their users is very small. However, their business is very similar, so the feature spaces are the same. Ref [58] proposed a collaboratively deep learning scheme where participants train independently and share only subsets of updates of parameters. In 2017, Google proposed a horizontal federated learning solution for Android phone model updates [41]. In that framework, a single user using an Android phone updates the model parameters locally and uploads the parameters to the Android cloud, thus jointly training the centralized model together with other data owners. A secure aggregation scheme to protect the privacy of aggregated user updates under their federated learning framework is also introduced [9]. Ref [51] uses additively homomorphic encryption for model paramter aggregation to provide security against the central server.

横向联邦学习，或基于样本的联邦学习，提出的场景是，数据集有相同的特征空间，但在不同的样本中（图2a）。比如，两个区域性的银行，可能在其各自的区域内有不同的用户群体，其用户的交集也非常小。但是，它们的业务是非常类似的，所以其特征空间是相同的。[58]提出一种合作深度学习方案，其中参与者独自训练，只共享参数更新的子集。在2017年，Google提出了一种横向联邦学习方案，用于Android手机的模型更新[41]。在那个框架中，单个用户使用Android手机，在本地更新模型参数，并将参数上传至Android云，因此与其他数据拥有者联合训练了中心化的模型。提出了一种安全聚集方案，在这个联邦学习框架中，以保护聚集的用户更新的隐私。[51]使用加性同态加密进行模型参数聚集，以对抗中心服务器的安全问题。

In [60], a multi-task style federated learning system is proposed to allow multiple sites to complete separate tasks, while sharing knowledge and preserving security. Their proposed multi-task learning model can in addition address high communication costs, stragglers, and fault tolerance issues. In [41], the authors proposed to build a secure client-server structure where the federated learning system partitions data by users, and allow models built at client devices to collaborate at the server site to build a global federated model. The process of model building ensures that there is no data leakage. Likewise, in [36], the authors proposed methods to improve the communication cost to facilitate the training of centralized models based on data distributed over mobile clients. Recently, a compression approach called Deep Gradient Compression [39] is proposed to greatly reduce the communication bandwidth in large-scale distributed training.

在[60]中，提出了一种多任务类型的联邦学习系统，可以使得多个站点完成不同的任务，而同时还能共享知识，保持数据安全。他们提出的多任务学习模型，还可以处理通信代价过高的问题，stragglers的问题，和错误容忍度的问题。在[41]中，作者提出构建一种安全的客户端-服务器架构，其中联邦学习系统通过用户来分割数据，使得模型在客户端设备上就可以构建起来，并与服务器合作，构建一个全局的联邦学习模型。模型构建的过程，确保了没有任何数据泄漏。类似的，在[36]中，作者提出改进通信的消耗，以促进基于移动客户端上分布的数据的中心化模型训练。最近，[39]提出了一种压缩方法，称为深度梯度压缩[39]，可以极大的降低大规模分布式训练中的通信带宽。

We summarize horizontal federated learning as: 我们将横向联邦学习总结为：

$$X_i = X_j, Y_i = Y_j, I_i \neq I_j, ∀D_i, D_j,i \neq j$$(2)

**Security Definition**. A horizontal federated learning system typically assumes honest participants and security against a honest-but-curious server [9, 51]. That is, only the server can compromise the privacy of data participants. Security proof has been provided in these works. Recently another security model considering malicious user [29] is also proposed, posing additional privacy challenges. At the end of the training, the universal model and the entire model parameters are exposed to all participants.

**安全定义**。横向联邦学习系统一般假设参与者都是诚实的，但会有安全措施应对honest-but-curious的服务器。即，只有服务器可以对数据参与者的隐私性进行平衡折中。安全防护在这些文章中已经存在。最近另一种考虑恶意用户的安全模型也提出了，应对另外的隐私挑战。在训练的最后，全体模型和完整模型参数都对所有参与者暴露出来。

#### 2.3.2 Vertical Federated Learning

Privacy-preserving machine learning algorithms have been proposed for vertically partitioned data, including Cooperative Statistical Analysis [15], association rule mining [65], secure linear regression [22, 32, 55], classification [16] and gradient descent [68]. Recently, Ref [27, 49] proposed a vertical federated learning scheme to train a privacy-preserving logistic regression model. The authors studied the effect of entity resolution on the learning performance and applied Taylor approximation to the loss and gradient functions so that homomorphic encryption can be adopted for privacy-preserving computations.

对于纵向分割的数据，也提出了隐私保护的机器学习算法，包括合作统计分析，联合规则挖掘，安全线性回归、分类和梯度下降。最近，[27,49]提出了一种纵向联邦学习的方案，训练一个保护隐私的logistic回归模型。作者研究了实体分辨率对学习性能的影响，对损失函数和梯度函数使用了Taylor近似，这样同态加密就可以用于保护隐私的计算。

Vertical federated learning or feature-based federated learning (Figure 2b) is applicable to the cases that two data sets share the same sample ID space but differ in feature space. For example, consider two different companies in the same city, one is a bank, and the other is an e-commerce company. Their user sets are likely to contain most of the residents of the area, so the intersection of their user space is large. However, since the bank records the user’s revenue and expenditure behavior and credit rating, and the e-commerce retains the user’s browsing and purchasing history, their feature spaces are very different. Suppose that we want both parties to have a prediction model for product purchase based on user and product information.

纵向联邦学习或基于特征的联邦学习（图2b），适用的情况是，两个数据集有同样的样本ID，但特征空间却不同。如，同一城市的两个不同的公司，一个是银行，另一个是电子商务公司。他们的用户集很可能包含这个区域的大部分居民，所以其用户空间的交集是很大的。但是，由于银行记录的是用户的税收和支出行为和信用等级，而电子商务公司保留的是用户的浏览记录和购买记录，他们的特征空间是非常不一样的。假设我们希望双方基于用户和产品信息得到产品购买的预测模型。

Vertically federated learning is the process of aggregating these different features and computing the training loss and gradients in a privacy-preserving manner to build a model with data from both parties collaboratively. Under such a federal mechanism, the identity and the status of each participating party is the same, and the federal system helps everyone establish a "common wealth" strategy, which is why this system is called "federated learning.". Therefore, in such a system, we have:

纵向联邦学习是这些不同特征的聚积过程，以一种保护隐私的方式计算训练损失和梯度，双方用数据合作构建出一个模型。在这样的联邦机制中，每个参与方的身份和状态都是一样的，这个联邦系统帮助每一方确立一个共同福利的策略，这也是这个系统为什么被称作联邦学习的原因。因此，在这样的系统中，我们有：

$$X_i \neq X_j, Y_i \neq Y_j, I_i = I_j, ∀D_i, D_j,i \neq j$$(3)

**Security Definition**. A vertical federated learning system typically assumes honest-but-curious participants. In a two-party case, for example, the two parties are non-colluding and at most one of them are compromised by an adversary. The security definition is that the adversary can only learn data from the client that it corrupted but not data from the other client beyond what is revealed by the input and output. To facilitate the secure computations between the two parties, sometimes a Semi-honest Third Party (STP) is introduced, in which case it is assumed that STP does not collude with either party. SMC provides formal privacy proof for these protocols [25]. At the end of learning, each party only holds the model parameters associated to its own features, therefore at inference time, the two parties also need to collaborate to generate output.

**安全定义**。纵向联邦学习系统一般假设的是honest-but-curious的参与者。比如，在一个双方的案例中，双方是非串通的，最多有一方是由对手平衡的。安全定义是，对手只可以从属于自己的客户端学习数据，但不能从输入和输出定义的其他客户端中学习数据。为促进双方的安全计算，有时候会引入一种Semi-honest的第三方(STP)，这种情况下，STP不会与任何一方串通。SMC为这些协议提供正式的隐私保护。在学习最后，每一方只保留与其自己特征有关的模型参数，因此，在推理时，双方需要合作才能生成输出。

#### 2.3.3 Federated Transfer Learning (FTL)

Federated Transfer Learning applies to the scenarios that the two data sets differ not only in samples but also in feature space. Consider two institutions, one is a bank located in China, and the other is an e-commerce company located in the United States. Due to geographical restrictions, the user groups of the two institutions have a small intersection. On the other hand, due to the different businesses, only a small portion of the feature space from both parties overlaps. In this case, transfer learning [50] techniques can be applied to provide solutions for the entire sample and feature space under a federation (Figure2c). Specially, a common representation between the two feature space is learned using the limited common sample sets and later applied to obtain predictions for samples with only one-side features. FTL is an important extension to the existing federated learning systems because it deals with problems exceeding the scope of existing federated learning algorithms:

联邦迁移学习应用的场景是，两个数据集的不同点，不仅在于样本，而且还有特征空间。考虑两个机构，一个是位于中国的银行，另一个是在美国的电子商务公司。由于地理上的限制，两个机构的用户群体的交集会比较小。另一方面，由于不同的商业模式，双方的特征空间只有一部分是重叠的。在这种情况下，迁移学习技术可以用于为所有样本和特征空间在联邦的情况下给出解决方案（图2c）。特别的是，使用有限的共同样本集，学习到两个特征空间的共同表示，然后应用并得到只有一方特征的样本的预测结果。FTL是现有联邦学习系统的重要延伸，因为其处理的问题超出了现有的联邦学习算法范畴：

$$X_i \neq X_j, Y_i \neq Y_j, I_i \neq I_j, ∀D_i,D_j,i \neq j$$(4)

**Security Definition**. A federated transfer learning system typically involves two parties. As will be shown in the next section, its protocols are similar to the ones in vertical federated learning, in which case the security definition for vertical federated learning can be extended here.

**安全定义**。联邦迁移学习系统一般涉及到双方。下一节我们会看到，其协议与纵向联邦学习类似，纵向联邦学习的安全定义可以拓展到这里。

### 2.4 Architecture for a federated learning system

In this section, we illustrate examples of general architectures for a federated learning system. Note that the architectures of horizontal and vertical federated learning systems are quite different by design, and we will introduce them separately.

本节中，我们描述了联邦学习系统的通用框架范例。注意，横向和纵向联邦学习系统的架构在设计上是非常不同的，，我们会分别进行介绍。

#### 2.4.1 Horizontal Federated Learning

A typical architecture for a horizontal federated learning system is shown in Figure 3. In this system, k participants with the same data structure collaboratively learn a machine learning model with the help of a parameter or cloud server. A typical assumption is that the participants are honest whereas the server is honest-but-curious, therefore no leakage of information from any participants to the server is allowed [51]. The training process of such a system usually contain the following four steps:

横向联邦学习系统的典型架构如图3所示。在这个系统中，k个参与者的数据结构是一样的，它们在一组参数或云服务器的帮助下，合作学习得到一个机器学习模型。典型的假设是，参与者都是诚实的，但服务器是honest-but-curious，因此任何参与者对服务器的信息泄漏都是不允许的。这样一个系统的训练过程通常包括如下四个步骤：

- Step 1: participants locally compute training gradients, mask a selection of gradients with encryption [51], differential privacy [58] or secret sharing [9] techniques, and send masked results to server; 参与者在本地计算训练的梯度，选择部分梯度用加密、差异化隐私或秘密分享等技术进行掩膜，并将掩膜结果送给服务器；
- Step 2: Server performs secure aggregation without learning information about any participant; 服务器在不了解任何参与者信息的情况下进行安全聚积；
- Step 3: Server send back the aggregated results to participants; 服务器将聚积结果发回参与者；
- Step 4: Participants update their respective model with the decrypted gradients. 参与者用解密的梯度升级各自的模型。

Iterations through the above steps continue until the loss function converges, thus completing the entire training process. This architecture is independent of specific machine learning algorithms (logistic regression, DNN etc) and all participants will share the final model parameters.

上述步骤迭代进行，直到损失函数收敛，这样就结束了整个训练过程。这种架构与具体的机器学习算法无关（logistic回归，DNN等），所有参与者都会共享最终模型参数。

**Security Analysis**. The above architecture is proved to protect data leakage against the semi-honest server, if gradients aggregation is done with SMC [9] or Homomorphic Encryption [51]. But it may be subject to attack in another security model by a malicious participant training a Generative Adversarial Network (GAN) in the collaborative learning process [29].

**安全分析**。上述架构已经证明可以保护数据泄漏，在梯度聚积是用SMC或同态加密的情况下进行时，不会被semi-honest的服务器危害到。但在另一种安全模型中，如果一个参与者是恶意的，在合作学习过程中训练了一个GAN时，则可能会受到攻击。

#### 2.4.2 Vertical Federated Learning

Suppose that companies A and B would like to jointly train a machine learning model, and their business systems each have their own data. In addition, Company B also has label data that the model needs to predict. For data privacy and security reasons, A and B cannot directly exchange data. In order to ensure the confidentiality of the data during the training process, a third-party collaborator C is involved. Here we assume the collaborator C is honest and does not collude with A or B, but party A and B are honest-but-curious to each other. A trusted third party C is a reasonable assumption since party C can be played by authorities such as governments or replaced by secure computing node such as Intel Software Guard Extensions (SGX) [7]. The federated learning system consists of two parts, as shown in Figure 4.

假设公司A和B想要联合训练一个机器学习模型，其商业系统各自有各自的数据。另外，公司B还有模型需要预测的标签数据。由于数据隐私和安全的原因，A和B不能直接交换数据。为确保训练过程中数据的机密性，引入了第三方的合作者C。这里我们假设合作者C是诚实的，并不会与A或B进行串通，但A和B互相是honest-but-curious的。可信任的第三方C是合理的假设，因为C方可以是一些权威机构，如政府，或替换为安全计算节点，如Intel Software Guard Extensions(SGX)。联邦学习系统包括两部分，如图4所示。

Part 1. Encrypted entity alignment. Since the user groups of the two companies are not the same, the system uses the encryption-based user ID alignment techniques such as [38, 56] to confirm the common users of both parties without A and B exposing their respective data. During the entity alignment, the system does not expose users that do not overlap with each other.

第1部分。加密实体对齐。由于两个公司的用户群体并不是一样的，系统使用基于加密的用户ID对齐技术，如[38,56]，以确认双方的共同用户，而A和B都不需要暴露各自的数据。在实体对齐的过程中，系统不会暴露互相不重叠的用户。

Part 2. Encrypted model training. After determining the common entities, we can use these common entities’ data to train the machine learning model. The training process can be divided into the following four steps (as shown in Figure 4):

第2部分。加密模型训练。在确定共同实体后，我们可以使用这些共同实体的数据来训练机器学习模型。训练过程可以分解为以下四步（如图4所示）：

- Step 1: collaborator C creates encryption pairs, send public key to A and B; 合作者C生成加密对，将公钥发送给A和B；
- Step 2: A and B encrypt and exchange the intermediate results for gradient and loss calculations; A和B加密并交换梯度和损失计算的中间结果；
- Step 3: A and B computes encrypted gradients and adds additional mask, respectively, and B also computes encrypted loss; A and B send encrypted values to C; A和B分别计算加密梯度，加上额外的mask，B也计算加密损失；A和B将加密值送给C；
- Step 4: C decrypts and send the decrypted gradients and loss back to A and B; A and B unmask the gradients, update the model parameters accordingly. C解密并将解密的梯度和损失送回给A和B；A和B将梯度解掩膜，相应的更新模型参数。

Here we illustrate the training process using linear regression and homomorphic encryption as an example. To train a linear regression model with gradient descent methods, we need secure computations of its loss and gradients. Assuming learning rate η, regularization parameter λ, data set {$x_i^A$}$_{i∈D_A}$, {$x_i^B, y_i$}$_{i ∈ D_B}$, and model paramters $Θ_A, Θ_B$ corresponding to the feature space of $x_i^A, x_i^B$ respectively, the training objective is:

这里我们描述一下训练过程，使用线性回归和同态加密作为例子。为使用梯度下降方法训练一个线性回归模型，我们需要安全计算其损失和梯度。假设学习速率为η，正则化参数为λ，数据集{$x_i^A$}$_{i∈D_A}$, {$x_i^B, y_i$}$_{i ∈ D_B}$，模型参数$Θ_A, Θ_B$分别对应特征空间$x_i^A, x_i^B$，训练的目标函数为：

$$min_{Θ_A, Θ_B} \sum_i ||Θ_A x_i^A + Θ_B x_i^B − y_i||^2 + \frac{λ}{2} (||Θ_A||^2 + ||Θ_B||^2)$$(5)

let $u_i^A = Θ_A x_i^A, u_i^B = Θ_B x_i^B$, the encrypted loss is:

$$[[L]] = [[\sum_i (u_i^A + u_i^B − y_i)^2 + \frac{λ}{2} (||Θ_A||^2 + ||Θ_B||^2)]]$$(6)

where additive homomorphic encryption is denoted as $[[·]]$. Let $[[L_A]] = [[\sum_i ((u_i^A)^2) + \frac{λ}{2} Θ_A^2 ]]$, $[[L_B]] = [[\sum_i ((u_i^B - y_i)^2) + \frac{λ}{2} Θ_B^2 ]]$, and $[[L_{AB}]]=2 \sum_i ([[u_i^A]](u_i^B − y_i))$, then

$$[[L]] = [[L_A]] + [[L_B]] + [[L_{AB}]]$$(7)

Similarly, let $[[d_i]] = [[u_i^A]] + [[u_i^B − y_i]]$, then gradients are:

$$[[ \frac{∂L}{∂Θ_A} ]] = \sum_i [[d_i]]x_i^A +[[λΘ_A]]$$(8)
$$[[ \frac{∂L}{∂Θ_B} ]] = \sum_i [[d_i]]x_i^B +[[λΘ_B]]$$(9)

Table 1. Training Steps for Vertical Federated Learning : Linear Regression

| | Party A | Party B | Party C
--- | --- | --- | ---
step 1 | initialize Θ_A | initialize Θ_B | create an encryption key pair, send public key to A and B;
step 2 | compute $[[u_i^A]],[[L_A]]$ and send to B; | compute $[[u_i^B]], [[d_i^B]], [[L]]$, send $[[d_i^B]]$ to A, send $[[L]]$ to C;
step 3 | initialize R_A, compute $[[\frac{∂L}{∂Θ_A}]] + [[R_A]]$ and send to C; | initialize R_B, compute $[[\frac{∂L}{∂Θ_B}]] + [[R_B]]$ and send to C; | C decrypt L, send $\frac{∂L}{∂Θ_A} + R_A$ to A, $\frac{∂L}{∂Θ_A} + R_A$ to B;
step 4 | update Θ_A | update Θ_B
What is obtained | Θ_A | Θ_B

Table 2. Evaluation Steps for Vertical Federated Learning : Linear Regression

| | Party A | Party B | inquisitor C
--- | --- | --- | ---
step 0 | | | send user ID i to A and B;
step 1 | compute u_i^A and send to C | compute u_i^B and send to C | get result u_i^A + u_i^B

See Table 1 and 2 for the detailed steps. During entity alignment and model training, the data of A and B are kept locally, and the data interaction in training does not lead to data privacy leakage. Note potential information leakage to C may or may not be considered to be privacy violation. To further prevent C to learn information from A or B in this case, A and B can further hide their gradients from C by adding encrypted random masks. Therefore, the two parties achieve training a common model cooperatively with the help of federated learning. Because during the training, the loss and gradients each party receives are exactly the same as the loss and gradients they would receive if jointly building a model with data gathered at one place without privacy constraints, that is, this model is lossless. The efficiency of the model depends on the communication cost and computation cost of encrypted data. In each iteration the information sent between A and B scales with the number of overlapping samples. Therefore the efficiency of this algorithm can be further improved by adopting distributed parallel computing techniques.

见表1和表2的详细步骤。在实体对齐和模型训练的过程中，A和B的数据都是本地保存的，训练中的数据交互不会带来数据隐私的泄漏。注意可能的信息泄漏到C，这可以考虑为隐私冲突，也可以不考虑为隐私冲突。在这种情况下，为进一步防止C从A或B学习信息，A和B可以进一步增加加密随机mask来对C隐藏其梯度。因此，这双方在联邦学习的帮助下，可以得到合作训练一个共同模型的目的。因此在训练过程中，每一方收到的损失函数和梯度，都与将数据放到一个地方训练模型的损失函数和梯度一样，所以，模型是无损的。模型的效率取决于通信的消耗和加密数据的计算消耗。在每一次迭代中，A和B之间发送的信息随着重叠的样本数量增减。因此，算法的效率，可以通过采取分布式并行计算技术进一步改进。

**Security Analysis**. The training protocol shown in Table 1 does not reveal any information to C, because all C learns are the masked gradients and the randomness and secrecy of the masked matrix are guaranteed [16]. In the above protocol, party A learns its gradient at each step, but this is not enough for A to learn any information from B according to equation 8, because the security of scalar product protocol is well-established based on the inability of solving n equations in more than n unknowns [16, 65]. Here we assume the number of samples NA is much greater than nA, where nA is the number of features. Similarly, party B can not learn any information from A. Therefore the security of the protocol is proved. Note we have assumed that both parties are semi-honest. If a party is malicious and cheats the system by faking its input, for example, party A submits only one non-zero input with only one non-zero feature, it can tell the value of u_i^B for that feature of that sample. It still can not tell x_i^B or Θ_B though, and the deviation will distort results for the next iteration, alarming the other party who will terminate the learning process. At the end of the training process, each party (A or B) remains oblivious to the data structure of the other party, and it obtains the model parameters associated only with its own features. At inference time, the two parties need to collaboratively compute the prediction results, with the steps shown in Table 2, which still do not lead to information leakage.

**安全分析**。表1中所示的训练协议没有对C透露任何信息，因为所有C了解到的都是掩膜的梯度，而且掩膜矩阵的随机性和秘密性是得到保证的。在上述协议中，A方在每一步中学习到其梯度，但根据式8，这不足以使A从B学习到任何信息，因为标量乘法协议的安全性是非常稳固的，是基于用n个等式求解大于n个未知量的不可能性上的。这里我们假设样本的数量NA远大于nA，这里nA是特征的数量。类似的，B方也不会从A方学到任何信息。因此，协议的安全性就得到证明。注意，我们假设了，双方都是semi-honest的。如果一方是恶意的，在输入上作假，欺骗了系统，如，A方只提交了只有一个非零值特征只有一个非零值的输入，可以告诉u_i^B那个样本的特征。但它还不能告诉x_i^B或Θ_B，而且偏差会使得下一次迭代的结果扭曲，其他方会告警并终止这个学习过程。在训练过程的最后，每一方(A or B)都不知道另一方的数据结构，而且只得到与自己特征相关的模型参数。在推理时，双方需要合作计算预测结果，如表2中的步骤所示，这仍然不会带来信息泄漏。

#### 2.4.3 Federated Transfer Learning

Suppose in the above vertical federated learning example, party A and B only have a very small set of overlapping samples and we are interested in learning the labels for all the data set in party A. The architecture described in the above section so far only works for the overlapping data set. To extend its coverage to the entire sample space, we introduce transfer learning. This does not change the overall architecture shown in Figure 4 but the details of the intermediate results that are exchanged between party A and party B. Specifically, transfer learning typically involves in learning a common representation between the features of party A and B, and minimizing the errors in predicting the labels for the target-domain party by leveraging the labels in the source-domain party (B in this case). Therefore the gradient computations for party A and party B are different from that in the vertical federated learning scenario. At inference time, it still requires both parties to compute the prediction results.

假设在上面的纵向联邦学习的例子中，A方和B方的重叠样本集非常小，我们感兴趣的是对A方所有数据集学习其标签。上一节描述的结构只对重叠的数据集有用。为拓展覆盖到整个样本空间，我们引入了迁移学习。这并没有改变图4的总体架构，但A方和B方交换的中间结果的细节有所变化。具体的，迁移学习一般是在A方和B方之间学习到一个通用表示，通过利用源领域方（这里是B方）的标签，最小化预测目标领域方的预测标签误差。因此A方和B方的梯度计算与纵向联邦学习的情形不太一样。在推理时，仍然需要双方一起来计算预测结果。

#### 2.4.4 Incentives Mechanism

In order to fully commercialize federated learning among different organizations, a fair platform and incentive mechanisms needs to be developed [20]. After the model is built, the performance of the model will be manifested in the actual applications and this performance can be recorded in a permanent data recording mechanism (such as Blockchain). Organizations that provide more data will be better off, and the model’s effectiveness depends on the data provider’s contribution to the system. The effectiveness of these models are distributed to parties based on federated mechanisms and continue to motivate more organizations to join the data federation.

为使得在不同组织中联邦学习可以完全商业化，需要开发出一个公平的平台和激励的机制。在模型构建起以后，模型的性能会在实际应用中得到证明，这个性能可以用永久记录机制记录一下（如区块链）。提供更多数据的组织会更好，模型的有效性依靠数据提供者对系统的贡献。这些模型的有效性基于联邦机制分布到各方，继续鼓励更多组织加入到数据联邦中。

The implementation of the above architecture not only considers the privacy protection and effectiveness of collaboratively-modeling among multiple organizations, but also considers how to reward organizations that contribute more data, and how to implement incentives with a consensus mechanism. Therefore, federated learning is a "closed-loop" learning mechanism.

上述架构的实现不仅考虑了隐私保护和多组织合作建模的有效性，而且还考虑了怎样奖励提供更多数据的组织，以及怎样用共识的机制实现这种激励。因此，联邦学习是一种闭环学习机制。

## 3 Related works

Federated learning enables multiple parties to collaboratively construct a machine learning model while keeping their private training data private. As a novel technology, federated learning has several threads of originality, some of which are rooted on existing fields. Below we explain the relationship between federated learning and other related concepts from multiple perspectives.

联邦学习使多方可以合作构建一个机器学习模型，同时使私有的训练数据保持私有。作为一种新技术，联邦学习有几条起源线，一些源于现有的领域。下面我们从多个角度解释一下联邦学习和其他相关概念的关系。

### 3.1 Privacy-preserving machine learning

Federated learning can be considered as privacy-preserving decentralized collaborative machine learning, therefore it is tightly related to multi-party privacy-preserving machine learning. Many research efforts have been devoted to this area in the past. For example, Ref [17, 67] proposed algorithms for secure multi-party decision tree for vertically partitioned data. Vaidya and Clifton proposed secure association mining rules [65], secure k-means [66], Naive Bayes classifier [64] for vertically partitioned data. Ref [31] proposed an algorithm for association rules on horizontally partitioned data. Secure Support Vector Machines algorithms are developed for vertically partitioned data [73] and horizontally partitioned data [74]. Ref [16] proposed secure protocols for multi-party linear regression and classification. Ref [68] proposed secure multi-party gradient descent methods. The above works all used secure multi-party computation (SMC) [25, 72] for privacy guarantees.

联邦学习可以认为是保护隐私的去中心化合作机器学习，因此与多方保护隐私的机器学习密切相关。过去很多研究者都致力于这个领域。比如，[17,67]提出的算法是对纵向分割的数据进行的安全多方决策树训练。Vaidya等针对纵向分割的数据提出安全关联挖掘规则，安全k-means，Naive Bayes分类器。[31]对横向分割的数据提出了相关规则的算法。[73,74]对纵向和横向分割的数据提出了安全支持矢量机算法。[16]对多方线性回归和分类提出了安全协议。[68]提出了安全多方梯度下降方法。上述工作都使用SMC进行隐私保证。

Nikolaenko et al.[48] implemented a privacy-preserving protocol for linear regression on horizontally partitioned data using homomorphic encryption and Yao’s garbled circuits and Ref [22, 24] proposed a linear regression approach for vertically partitioned data. These systems solved the linear regression problem directly. Ref [47] approached the problem with Stochastic Gradient Descent (SGD) and they also proposed privacy-preserving protocols for logistic regression and neural networks. Recently, a follow-up work with a three-server model is proposed [44]. Aono et al.[4] proposed a secure logistic regression protocol using homomorphic encryption. Shokri and Shmatikov [58] proposed training of neural networks for horizontally partitioned data with exchanges of updated parameters. Ref [51] used the additively homomorphic encryption to preserve the privacy of gradients and enhance the security of the system. With the recent advances in deep learning, privacy-preserving neural networks inference is also receiving a lot of research interests[10, 11, 14, 28, 40, 52, 54].

Nikolaenko等[48]在横向分割的数据中，针对线性回归实现了一种保护隐私的协议，使用的是同态加密和Yao的混淆电路，[22,24]对纵向分割的数据提出了一种线性回归方法。这些系统直接解决了线性回归的问题。[47]用SGD解决logistic回归和神经网络问题，也提出了保护隐私的协议。最近，[44]提出了一种三服务器模型。Aono等[4]用同态加密提出了一种安全logistic回归协议。Shokri等[58]提出为横向分割的数据训练神经网络，交换更新的参数。[51]使用加性同态加密来保护梯度隐私，增强了系统的安全性。随着最近深度学习的发展，保护隐私的神经网络推理也得到了非常多的研究关注。

### 3.2 Federated Learning vs Distributed Machine Learning

Horizontal federated learning at first sight is somewhat similar to Distributed Machine Learning. Distributed machine learning covers many aspects, including distributed storage of training data, distributed operation of computing tasks, distributed distribution of model results, etc. Parameter Server [30] is a typical element in distributed machine learning. As a tool to accelerate the training process, the parameter server stores data on distributed working nodes, allocates data and computing resources through a central scheduling node, so as to train the model more efficiently. For horizontally federated learning, the working node represents the data owner. It has full autonomy for the local data, and can decide when and how to join the federated learning. In the parameter server, the central node always takes the control, so federated learning is faced with a more complex learning environment. Secondly, federated learning emphasizes the data privacy protection of the data owner during the model training process. Effective measures to protect data privacy can better cope with the increasingly stringent data privacy and data security regulatory environment in the future.

横向联邦学习看上去与分布式机器学习比较像。分布式机器学习覆盖了很多方面，包括训练数据的分布式存储，计算任务的分布式运算，模型结果的分布式分发，等等。参数服务器[30]是分布式机器学习的典型元素。作为加速训练过程的一个工具，参数服务器在分布式工作节点中存储数据，分配数据和计算资源是通过中央调度节点进行的，这样可以更高效的训练模型。对于横向联邦学习，工作节点代表了数据拥有者。其对本地数据有完全的自治权，可以决定何时、怎样加入联邦学习。在参数服务器中，中央节点永远拥有控制权，所以联邦学习面对的是更复杂的学习环境。第二，联邦学习强调的是数据拥有者在模型训练过程中的数据隐私保护。有效的保护数据隐私的措施，可以更好的应对未来越来越严厉的数据隐私和数据安全法规环境。

Like in distributed machine learning settings, federated learning will also need to address Non-IID data. In [77] showed that with non-iid local data, performance can be greatly reduced for federated learning. The authors in response supplied a new method to address the issue similar to transfer learning.

就像在分布式机器学习的设置下，联邦学习也需要处理非IID数据。[77]中证明了，对于non-iid本地数据，联邦学习的性能会急剧下降。作者相应的给出了一种新方法来处理这种问题，与迁移学习类似。

### 3.3 Federated Learning vs Edge Computing

Federated learning can be seen as an operating system for edge computing, as it provides the learning protocol for coordination and security. In [69], authors considered generic class of machine learning models that are trained using gradient-descent based approaches. They analyze the convergence bound of distributed gradient descent from a theoretical point of view, based on which they propose a control algorithm that determines the best trade-off between local update and global parameter aggregation to minimize the loss function under a given resource budget.

联邦学习可以视为一种边缘计算的操作系统，因为提供了协调和安全的学习规则。在[69]中，作者考虑一般类别的用梯度下降训练的机器学习模型。他们从理论上分析了分布式梯度下降的收敛边界，在这基础上，提出了一种控制算法，确定了本地更新和全局参数聚积的最佳折中，以在给定的资源预算下最小化损失函数。

### 3.4 Federated Learning vs Federated Database Systems

Federated Database Systems [57] are systems that integrate multiple database units and manage the integrated system as a whole. The federated database concept is proposed to achieve interoperability with multiple independent databases. A federated database system often uses distributed storage for database units, and in practice the data in each database unit is heterogeneous. Therefore, it has many similarities with federated learning in terms of the type and storage of data. However, the federated database system does not involve any privacy protection mechanism in the process of interacting with each other, and all database units are completely visible to the management system. In addition, the focus of the federated database system is on the basic operations of data including inserting, deleting, searching, and merging, etc., while the purpose of federated learning is to establish a joint model for each data owner under the premise of protecting data privacy, so that the various values and laws the data contain serve us better.

联邦数据库系统[57]，是整合了多个数据库单元的系统，并将整合的系统作为一个整体来管理。联邦数据库概念的提出，是为了在多个独立的数据库中进行互相操作。联邦数据系统通常对数据库单元使用分布式存储，在实践中，每个数据库单元中的数据都是异质的。因此，它在数据的类型与存储上，与分布式学习有很多相似点。但是，联邦数据库系统在相互作用的过程中，与隐私保护机制没有任何关系，所有数据库单元对管理系统都是完全可见的。另外，联邦数据库系统的焦点是数据的基本操作上，包括插入，删除，搜索，合并，等等，而联邦学习的目的是在保护数据隐私的前提下，对每个数据拥有者训练一个联合模型，所以数据包含的值与规则越多，效果会越好。

## 4 Applications

As an innovative modeling mechanism that could train a united model on data from multiple parties without compromising privacy and security of those data, federated learning has a promising application in sales, financial, and many other industries, in which data cannot be directly aggregated for training machine learning models due to factors such as intellectual property rights, privacy protection, and data security.

作为一种新型建模机制，可以从多方的数据上，训练一个联合模型，而且不用对数据的隐私和安全性进行折中，联邦学习在销售、金融和很多其他产业中都有很有前景的应用，其中数据不会直接聚积到一起训练机器学习模型，其原因比如知识产权，隐私保护，和数据安全。

Take the smart retail as an example. Its purpose is to use machine learning techniques to provide customers with personalized services, mainly including product recommendation and sales services. The data features involved in the smart retail business mainly include user purchasing power, user personal preference, and product characteristics. In practical applications, these three data features are likely to be scattered among three different departments or enterprises. For example, a user’s purchasing power can be inferred from her bank savings and her personal preference can be analyzed from her social networks, while the characteristics of products are recorded by an e-shop. In this scenario, we are facing two problems. First, for the protection of data privacy and data security, data barriers between banks, social networking sites, and e-shopping sites are difficult to break. As a result, data cannot be directly aggregated to train a model. Second, the data stored in the three parties are usually heterogeneous, and traditional machine learning models cannot directly work on heterogeneous data. For now, these problems have not been effectively solved with traditional machine learning methods, which hinder the popularization and application of artificial intelligence in more fields.

以智慧销售作为例子。其目的是使用机器学习技术，为客户提供个性化的服务，主要包括产品推荐和销售服务。与智慧零售业务相关的数据特征主要包括，用户购买能力，用户个人偏好，产品特性。在实际应用中，这三种数据特征很可能是分布在三个不同的部门或企业中的。比如，用户的购买能力可以从其银行储蓄中推断出来，其个人偏好可以从其社会网络中分析出来，而产品特性是记录在电子店铺中的。在这种场景中，我们面临两个问题。首先，为保护数据隐私和数据安全，银行、社交网络网站和电子店铺的数据屏障是很难打破的。结果是，数据不能直接聚积到一起，以训练一个模型。第二，存储在三方的数据通常是异质的，传统的机器学习模型不能直接在这些异质数据中工作。现在，这些问题用传统机器学习方法不能直接解决掉，这阻碍了人工智能在很多领域的应用和流行。

Federated learning and transfer learning are the key to solving these problems. First, by exploiting the characteristics of federated learning, we can build a machine learning model for the three parties without exporting the enterprise data, which not only fully protects data privacy and data security, but also provides customers with personalized and targeted services and thereby achieves mutual benefits. Meanwhile, we can leverage transfer learning to address the data heterogeneity problem and break through the limitations of traditional artificial intelligence techniques. Therefore federated learning provides a good technical support for us to build a cross-enterprise, cross-data, and cross-domain ecosphere for big data and artificial intelligence.

联邦学习和迁移学习是解决这些问题的关键。首先，利用联邦学习的特点，我们可以为这三方构建一个机器学习模型，而且不需要将企业数据导出，这不仅完全保护了数据隐私和数据安全，而且为客户带来了个性化和有指向性的服务，因此得到了多方获益的效果。同时，我们可以利用迁移学习来解决数据异质的问题，打破传统人工智能技术的局限。因此联邦学习为我们构建一个跨企业，跨数据和跨领域的大数据人工智能系统，提供了很好的技术支持。

One can use federated learning framework for multi-party database querying without exposing the data. For example, supposed in a finance application we are interested in detecting multi-party borrowing, which has been a major risk factor in the banking industry. This happens when certain users maliciously borrows from one bank to pay for the loan at another bank. Multi-party borrowing is a threat to financial stability as a large number of such illegal actions may cause the entire financial system to collapse. To find such users without exposing the user list to each other between banks A and B, we can exploit a federated learning framework. In particular, we can use the encryption mechanism of federated learning and encrypt the user list at each party, and then take the intersection of the encrypted list in the federation. The decryption of the final result gives the list of multi-party borrowers, without exposing the other "good" users to the other party. As we will see below, this operation corresponds to the vertical federated learning framework.

我们可以利用联邦学习框架，进行多方数据库查询，不需要暴露数据。比如，假设我们感兴趣的一个金融应用，检测多方借款，这是银行业的一个主要危险因素。这当特定用户从一家银行恶意借款，来偿还另一家银行的债务时，会发生这种情况。多方借款是金融稳定性的一个风险，因为大量的这种非法行为可能会导致整个金融系统崩溃。为找到这种用户，而不相互暴露银行A和B的用户列表，我们可以利用联邦学习的框架。特别的，我们可以使用联邦学习的加密机制，对各方的用户列表进行加密，然后对联邦中的加密列表取交集。解密的最终结果，会给出多方借款的人员列表，而不需要暴露其他的好的用户给另一方。我们下面会看到，这种操作对应着纵向联邦学习框架。

Smart healthcare is another domain which we expect will greatly benefit from the rising of federated learning techniques. Medical data such as disease symptoms, gene sequences, medical reports are very sensitive and private, yet medical data are difficult to collect and they exist in isolated medical centers and hospitals. The insufficiency of data sources and the lack of labels have led to an unsatisfactory performance of machine learning models, which becomes the bottleneck of current smart healthcare. We envisage that if all medical institutions are united and share their data to form a large medical dataset, then the performance of machine learning models trained on that large medical dataset would be significantly improved. Federated learning combining with transfer learning is the main way to achieve this vision. Transfer learning could be applied to fill the missing labels thereby expanding the scale of the available data and further improving the performance of a trained model. Therefore, federated transfer learning would play a pivotal role in the development of smart healthcare and it may be able to take human health care to a whole new level.

智慧健康是另一个从联邦学习技术中获益极大的领域。医学数据，如疾病症状，基因序列，医学报告，都是很敏感的隐私数据，但医学数据很难收集，存在于孤立的医学中心和医院。这种数据源的不足，和缺少标签，导致了机器学习模型的性能不尽如人意，这是目前智慧健康的瓶颈。我们想象一下，如果所有医学机构联合起来，共享其数据，形成一个巨大的医学数据集，那么在这个巨型医学数据集上训练得到的机器学习模型的性能，将会得到极大的提升。联邦学习与迁移学习一起，是取得这种结果的主要途径。迁移学习可以用于补充缺失的标签，因此扩展可用数据的规模，进一步改进已训练模型的性能。因此，联邦迁移学习会在智慧医疗中扮演非常重要的角色，将会使人类健康达到全新的水平。

## 5 Federated learning and Data Alliance of Enterprises

Federated learning is not only a technology standard but also a business model. When people realize the effects of big data, the first thought that occurs to them is to aggregate the data together, compute the models through a remote processor and then download the results for further use. Cloud computing comes into being under such demands. However, with the increasing importance of data privacy and data security and a closer relationship between a company’s profits and its data, the cloud computing model has been challenged. However, the business model of federated learning has provided a new paradigm for applications of big data. When the isolated data occupied by each institution fails to produce an ideal model, the mechanism of federated learning makes it possible for institutions and enterprises to share a united model without data exchange. Furthermore, federated learning could make equitable rules for profits allocation with the help of consensus mechanism from blockchain techniques. The data possessors, regardless of the scale of data they have, will be motivated to join in the data alliance and make their own profits. We believe that the establishment of the business model for data alliance and the technical mechanism for federated learning should be carried out together. We would also make standards for federated learning in various fields to put it into use as soon as possible.

联邦学习不仅是一种技术标准，而且是一种商业模型。当人们意识到大数据的效果，首先想到的是将数据聚积起来，通过一个远程处理器来计算得到模型，然后将结果下载以供进一步使用。云计算就是在这种需求下出现的。但是，在数据隐私和数据安全越来越重要时，在一个企业的利润与其数据越来越相关时，云计算模型受到了挑战。但是，联邦学习的商业模型为大数据的应用提供了新的范式。当每个组织拥有的孤立数据不能得到一个理想的模型时，联邦学习的机制使得各个组织和企业在不交换数据的情况下可以共享一个联合模型。而且，联邦学习可以在区块链技术的共识机制的帮助下，为利润分配制定公平的规则。不管数据规模的大小，数据处理器会鼓励加入数据联盟，获得其自己的利润。我们相信，为数据联盟确立的商业模型，和联邦学习的技术机制，应当一起进行。我们应当为联邦学习在各个领域中制定标准，并尽快投入使用。

## 6 Conclusion and prospects

In recent years, the isolation of data and the emphasis on data privacy are becoming the next challenges for artificial intelligence, but federated learning has brought us new hope. It could establish a united model for multiple enterprises while the local data is protected, so that enterprises could win together taking the data security as premise. This article generally introduces the basic concept, architecture and techniques of federated learning, and discusses its potential in various applications. It is expected that in the near future, federated learning would break the barriers between industries and establish a community where data and knowledge could be shared together with safety, and the benefits would be fairly distributed according to the contribution of each participant. The bonus of artificial intelligence would finally be brought to every corner of our lives.

最近几年，数据的孤立和对数据隐私的强调，越来越成为人工智能的下一个挑战，但联邦学习为我们带来了新的希望。它可以为多个企业，在本地数据得到保护的情况下，建立一个联合模型，这样企业可以在数据安全得到保证的前提下，获得共赢。本文介绍了联邦学习的基本概念，框架和技术，讨论了其在各种应用中的潜力。在不久的将来，联邦学习有望打破产业间的壁垒，形成一个团体，其中数据和知识可以进行安全的共享，而收益则会根据每个参与者的贡献，进行相对公平的分配。人工智能的红利，会最终达到我们生活的每个角落。