# Deep Learning: A Critical Appraisal

Gary Marcus New York University

## 0. Abstract

Although deep learning has historical roots going back decades, neither the term “deep learning” nor the approach was popular just over five years ago, when the field was reignited by papers such as Krizhevsky, Sutskever and Hinton’s now classic 2012 (Krizhevsky, Sutskever, & Hinton, 2012)deep net model of Imagenet.

虽然深度学习有几十年的历史，但五年前，深度学习这个术语或方法都不流行，这个领域是被Krizhevsky等的文章重新点燃的。

What has the field discovered in the five subsequent years? Against a background of considerable progress in areas such as speech recognition, image recognition, and game playing, and considerable enthusiasm in the popular press, I present ten concerns for deep learning, and suggest that deep learning must be supplemented by other techniques if we are to reach artificial general intelligence. 

这个领域在后来的五年中发现了什么呢？在语音识别，图像识别，博弈中确实有很多进展，流行媒体中也很有热情，但我提出了深度学习的十点担忧，如果要达到通用人工智能的目标，建议深度学习需要由其他技术形成补充。

## 1. Is deep learning approaching a wall?

Although deep learning has historical roots going back decades(Schmidhuber, 2015), it attracted relatively little notice until just over five years ago. Virtually everything changed in 2012, with the publication of a series of highly influential papers such as Krizhevsky, Sutskever and Hinton’s 2012 ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky, Sutskever, & Hinton, 2012), which achieved state-of-the-art results on the object recognition challenge known as ImageNet (Deng et al., ). Other labs were already working on similar work (Cireşan, Meier, Masci, & Schmidhuber, 2012). Before the year was out, deep learning made the front page of The New York Times, and it rapidly became the best known technique in artificial intelligence, by a wide margin. If the general idea of training neural networks with multiple layers was not new, it was, in part because of increases in computational power and data, the first time that deep learning truly became practical.

虽然深度学习有几十年的历史，但直到五年前，才得到关注。在2012年，一切都变了，随着一系列有影响的文章的发表，如Krizhevsky等的AlexNet，取得了2012年目标检测挑战ImageNet的最佳成绩。其他实验室也在进行类似的工作。在这一年结束之前，深度学习称为纽约时报的刊首，迅速成为人工智能的最著名技术，远超其他话题。如果训练多层神经网络的思想没那么新颖，确实是的，部分因为计算能力和数据的增加，在第一次，深度学习真的变得可行了。

Deep learning has since yielded numerous state of the art results, in domains such as speech recognition, image recognition, and language translation and plays a role in a wide swath of current AI applications. Corporations have invested billions of dollars fighting for deep learning talent. One prominent deep learning advocate, Andrew Ng, has gone so far to suggest that “If a typical person can do a mental task with less than one second of thought, we can probably automate it using AI either now or in the nearfuture.” (A, 2016). A recent New York Times Sunday Magazine article, largely about deep learning, implied that the technique is “poised to reinvent computing itself.”

深度学习因此得到非常多的目前最好的结果，领域包括语音识别，图像识别，语言翻译，并在目前的AI应用中占据重要地位。大企业为了争夺深度学习的人才花费数十亿美金。深度学习的一个主要支持者，Andrew Ng，认为“如果一个普通人花一秒钟思考来解决一个问题，我们就可以使用AI将其自动化，在现在或不远的将来。”最近一篇文章说，深度学习是一项重新定义计算的技术。

Yet deep learning may well be approaching a wall, much as I anticipated earlier, at beginning of the resurgence (Marcus, 2012), and as leading figures like Hinton (Sabour, Frosst, & Hinton, 2017) and Chollet (2017) have begun to imply in recent months.

但是深度学习可能已经遇到了墙，和我之前在复苏开始的时候进行的预测差不多，在最近几个月，领头人物如Hinton和Chollet都在做此暗示。

What exactly is deep learning, and what has its shown about the nature of intelligence? What can we expect it to do, and where might we expect it to break down? How close or far are we from “artificial general intelligence”, and a point at which machines show a human-like flexibility in solving unfamiliar problems? The purpose of this paper is both to temper some irrational exuberance and also to consider what we as a field might need to move forward.

深度学习究竟是什么，它展示了智能本质的什么呢？它能为我们做什么，在哪些地方会不好用呢？我们离通用人工智能有多远？在解决不熟悉的问题时，机器能表现出多少像人类这样的灵活性呢？本文的目的是使一些不理性的想法得到缓和，同时作为一个领域，应当如何继续发展。

This paper is written simultaneously for researchers in the field, and for a growing set of AI consumers with less technical background who may wish to understand where the field is headed. As such I will begin with a very brief, nontechnical introduction aimed at elucidating what deep learning systems do well and why (Section 2), before turning to an assessment of deep learning’s weaknesses (Section 3) and some fears that arise from misunderstandings about deep learning’s capabilities (Section 4), and closing with perspective on going forward (Section 5).

本文是为这个领域的研究者所写的，也是为没有很多技术背景的AI消费者所写的，可以理解这个领域的发展方向。我会以一个简要的非技术式的介绍开始，说明深度学习系统擅长哪些，为什么擅长（第2部分），然后评估一下深度学习的弱点（第3部分），以及对深度学习能力的误解产生的恐惧（第4部分），最后展望将来的发展方向（第5部分）。

Deep learning is not likely to disappear, nor should it. But five years into the field’s resurgence seems like a good moment for a critical reflection, on what deep learning has and has not been able to achieve.

深度学习是不太可能消失的，也不应当。但5年的发展似乎可以进行一个批评性的回顾，关于深度学习已经取得什么成就，以及不能取得什么成就。

## 2. What deep learning is, and what it does well

Deep learning, as it is primarily used, is essentially a statistical technique for classifying patterns, based on sample data, using neural networks with multiple layers.

深度学习的基本用处是进行模式分类，其实是基于样本数据的一种统计技术，使用的是多层神经网络。

Neural networks in the deep learning literature typically consist of a set of input units that stand for things like pixels or words, multiple hidden layers (the more such layers, the deeper a network is said to be) containing hidden units (also known as nodes or neurons), and a set output units, with connections running between those nodes. In a typical application such a network might be trained on a large sets of handwritten digits (these are the inputs, represented as images) and labels (these are the outputs) that identify the categories to which those inputs belong (this image is a 2, that one is a 3, and so forth).

很多深度学习文献中的神经网络，一般包括输入层，如像素或语句，多个隐藏层（这样的层越多，就称这个网络越深），有隐含节点（或神经元），以及输出层，还有这些节点间的连接。典型的应用中，网络在大型数据集上进行训练，如手写数字，这些是输入，表示为图像，还有标签，识别这些输入属于哪个类别（这个图像是2，那个是3，等等）。

Over time, an algorithm called back-propagation allows a process called gradient descent to adjust the connections between units using a process, such that any given input tends to produce the corresponding output.

随着时间的发展，反向传播算法采用梯度下降调整神经元之间的连接，这样给定输入后，可以得到对应的输出。

Collectively, one can think of the relation between inputs and outputs that a neural network learns as a mapping. Neural networks, particularly those with multiple hidden layers (hence the term deep) are remarkably good at learning input-output mappings.

我们可以认为，神经网络学习到输入和输出之间的映射。神经网络，尤其是那些有多个隐藏层的（所以称之为深度）特别擅长于输入-输出间的映射。

Such systems are commonly described as neural networks because the input nodes, hidden nodes, and output nodes can be thought of as loosely analogous to biological neurons, albeit greatly simplified, and the connections between nodes can be thought of as in some way reflecting connections between neurons. A longstanding question, outside the scope of the current paper, concerns the degree to which artificial neural networks are biologically plausible.

这样的系统通常称之为神经网络，因为输入节点，隐藏节点和输出节点与生理上的神经元非常类似，虽然进行了极度的简化，节点间的连接可以认为反映了神经元之间的连接。长久以来的问题是，人工神经网络在生物上的多大程度上是可行的，本文没有讨论这个话题。

Most deep learning networks make heavy use of a technique called convolution (LeCun, 1989), which constrains the neural connections in the network such that they innately capture a property known as translational invariance. This is essentially the idea that an object can slide around an image while maintaining its identity; a circle in the top left can be presumed(even absent direct experience) to be the same as a circle in the bottom right.

多数深度神经网络都使用了很多卷积，这将神经元之间的连接进行了限制，这样天生就有了平移不变的性质。这实际上反映了，目标可以在图像中进行平移，同时保持其目标性；左上角的圆与右下角的圆可以认为是一样的。

Deep learning is also known for its ability to self-generate intermediate representations, such as internal units that may respond to things like horizontal lines, or more complex elements of pictorial structure.

深度学习还可以生成中间表示，如内部节点可以对水平线段这样的东西作出响应，或更复杂的图像结构元素。

In principle, given infinite data, deep learning systems are powerful enough to represent any finite deterministic “mapping” between any given set of inputs and a set of corresponding outputs, though in practice whether they can learn such a mapping depends on many factors. One common concern is getting caught in local minima, in which a systems gets stuck on a suboptimal solution, with no better solution nearby in the space of solutions being searched. (Experts use a variety of techniques to avoid such problems, to reasonably good effect). In practice, results with large data sets are often quite good, on a wide range of potential mappings.

原则上来说，只要有无限的数据，深度学习系统就可以表示任意有限的确定性映射，将输入集映射到输出集，但是实际上是否可以学习到这样的映射，依赖于很多因素。一个常见的担忧是陷在局部极小值上，这样系统会停留在一个次优解上，而在更好的解附近的空间则没有进行搜索。（专家会使用各种技术来阻止这种问题，达到足够好的效果）。实际上，大型数据集得到的结果通常是很好的，会有很广的潜在映射。

In speech recognition, for example, a neural network learns a mapping between a set of speech sounds, and set of labels (such as words or phonemes). In object recognition, a neural network learns a mapping between a set of images and a set of labels (such that, for example, pictures of cars are labeled as cars). In DeepMind’s Atari game system (Mnih et al., 2015), neural networks learned mappings between pixels and joystick positions.

比如，在语音识别上，神经网络学习到的映射，是从语音声音上，到一些标签集上（比如语句或音素）。在目标识别上，神经网络学习的映射是从图像集合到标签集合的（如，车的图像标注为车）。在DeepMind的Atari博弈系统中，神经网络学习的映射是从像素到操纵杆的位置。

Deep learning systems are most often used as classification system in the sense that the mission of a typical network is to decide which of a set of categories (defined by the output units on the neural network) a given input belongs to. With enough imagination, the power of classification is immense; outputs can represent words, places on a Go board, or virtually anything else.

深度学习系统通常用于分类系统，典型网络的任务是给定的输入是属于哪个类别的（由神经网络的输出单元确定）。只要有足够的想象，分类可以做的事情很多；输出可以代表单词，围棋板上的位置，或任何其他东西。

In a world with infinite data, and infinite computational resources, there might be little need for any other technique. 在有无限数据、无限计算能力上，可能基本就不需要其他技术了。

## 3. Limits on the scope of deep learning

Deep learning’s limitations begin with the contrapositive: we live in a world in which data are never infinite. Instead, systems that rely on deep learning frequently have to generalize beyond the specific data that they have seen, whether to a new pronunciation of a word or to an image that differs from one that the system has seen before, and where data are less than infinite, the ability of formal proofs to guarantee high-quality performance is more limited.

深度学习的局限在于，我们生活的世界，其数据不是无限的。深度学习实现的系统则要泛化到常见数据之外，要么是单词的新发音上，要么与系统看到的图像不太一样，由于数据不是无限的，那么网络的性能就很可能是有限的。

As discussed later in this article, generalization can be thought of as coming in two flavors, interpolation between known examples, and extrapolation, which requires going beyond a space of known training examples (Marcus, 1998a).

后面本文会讨论，泛化可以认为是两种，已知样本的内插，和在已知训练样本以外的外插。

For neural networks to generalize well, there generally must be a large amount of data, and the test data must be similar to the training data, allowing new answers to be interpolated in between old ones. In Krizhevsky et al’s paper (Krizhevsky, Sutskever, & Hinton, 2012), a nine layer convolutional neural network with 60 million parameters and 650,000 nodes was trained on roughly a million distinct examples drawn from approximately one thousand categories.

要神经网络泛化效果好，必须要有很多数量的样本，测试数据必须要与训练数据类似，使得新的答案是已经的答案的内插。在Krizhevsky等的文章中，使用9层CNN，有六千万参数，65000个节点，在大约一百万不同的训练样本上训练，分属于大约1000个类别。

This sort of brute force approach worked well in the very finite world of ImageNet, into which all stimuli can be classified into a comparatively small set of categories. It also works well in stable domains like speech recognition in which exemplars are mapped in constant way onto a limited set of speech sound categories, but for many reasons deep learning cannot be considered (as it sometimes is in the popular press) as a general solution to artificial intelligence.

这种暴力方法在非常有限的ImageNet集上效果很好，所有激励都可以归类到相对较小的类别中。在比较稳定的领域，如语音识别上也效果较好，其中样本映射到有限的语音类别中，但由于很多原因，深度学习不能作为人工智能的通用解决方案。

Here are ten challenges faced by current deep learning systems: 这里是深度学习系统面临的十个挑战：

### 3.1. Deep learning thus far is data hungry

Human beings can learn abstract relationships in a few trials. If I told you that a schmister was a sister over the age of 10 but under the age of 21, perhaps giving you a single example, you could immediately infer whether you had any schmisters, whether your best friend had a schmister, whether your children or parents had any schmisters, and so forth. (Odds are, your parents no longer do, if they ever did, and you could rapidly draw that inference, too.)

人类可以几次就学习到抽象的关系。如果告诉你，schmister是10-21岁的，可能只要给你一个样本，你就可以立刻推断出是否有schmister，你最好的朋友是否有schmister，你的孩子或父母有没有schmister，等等。（奇怪的是，你的父母不会有的，但可能曾经有过，你可以立刻推断出这一点）

In learning what a schmister is, in this case through explicit definition, you rely not on hundreds or thousands or millions of training examples, but on a capacity to represent abstract relationships between algebra-like variables.

在学习什么是schmister的过程中，在本例中是通过显式的定义学到的，你不需要成百上千或上百万个训练样本进行学习，而是用表示在代数变量之间抽象关系的能力学到的。

Humans can learn such abstractions, both through explicit definition and more implicit means (Marcus, 2001). Indeed even 7-month old infants can do so, acquiring learned abstract language-like rules from a small number of unlabeled examples, in just two minutes (Marcus, Vijayan, Bandi Rao, & Vishton, 1999). Subsequent work by Gervain and colleagues (2012) suggests that newborns are capable of similar computations.

人类可以学习到这种抽象，从显式的定义和更隐式的方法都可以。甚至7个月的婴儿都可以，从少数几个未标注的样本中学习抽象的语言类的规则，在两分钟以内学会。还有文章表明，新生儿可以进行类似的计算。

Deep learning currently lacks a mechanism for learning abstractions through explicit, verbal definition, and works best when there are thousands, millions or even billions of training examples, as in DeepMind’s work on board games and Atari. As Brenden Lake and his colleagues have recently emphasized in a series of papers, humans are far more efficient in learning complex rules than deep learning systems are (Lake, Salakhutdinov, & Tenenbaum, 2015; Lake, Ullman, Tenenbaum, & Gershman, 2016). (See also related work by George et al (2017), and my own work with Steven Pinker on children’s overregularization errors in comparison to neural networks (Marcus et al., 1992).)

深度学习现在缺少从显式的语言定义中学习抽象的机制，只在有数千，数百万，甚至数十亿的训练样本中才能得到很好的结果，就像DeepMind在博弈游戏和Atari上一样。就像Brenden Lake及其同事在最近的一系列文章中强调的那样，人类比深度学习系统学习复杂规则的能力要高的多。

Geoffrey Hinton has also worried about deep learning’s reliance on large numbers of labeled examples, and expressed this concern in his recent work on capsule networks with his coauthors (Sabour et al., 2017) noting that convolutional neural networks (the most common deep learning architecture) may face “exponential inefficiencies that may lead to their demise. A good candidate is the difficulty that convolutional nets have in generalizing to novel viewpoints [ie perspectives on object in visual recognition tasks]. The ability to deal with translation[al invariance] is built in, but for the other ... [common type of] transformation we have to chose between replicating feature detectors on a grid that grows exponentially ... or increasing the size of the labelled training set in a similarly exponential way.”

Geoffrey Hinton也担心深度学习对大量标注样本的依赖，并在最近capsule网络的工作中表达了担忧，写到CNNs可能面临“指数级增加的低效可能会导致其消亡，卷积网络的一个困难是很难泛化新的观点上（即视觉识别任务中的目标观察新视角）。处理平移的能力是内在的，但其他常见的变换，我们要么选择在网格上重复特征检测器，这会成指数增加，或以类似的量级增加标注训练样本的规模”。

In problems where data are limited, deep learning often is not an ideal solution. 在数据是有限的情况下，深度学习通常并不是理想解决方案。

### 3.2.Deep learning thus far is shallow and has limited capacity for transfer

Although deep learning is capable of some amazing things, it is important to realize that the word “deep” in deep learning refers to a technical, architectural property (the large number of hidden layers used in a modern neural networks, where there predecessors used only one) rather than a conceptual one (the representations acquired by such networks don’t, for example, naturally apply to abstract concepts like “justice”, “democracy” or “meddling”).

虽然深度学习可以做很多神奇的事情，但要意识到，深度学习中的深，指的是一种架构上的性质（神经网络中隐含层的数量很大，之前的技术都只用一层），而并不是概念上的深（这样的网络得到的表示并不会应用到抽象概念上，如正义，民主或干预）。

Even more down-to-earth concepts like “ball” or “opponent” can lie out of reach. Consider for example DeepMind’s Atari game work (Mnih et al., 2015) on deep reinforcement learning, which combines deep learning with reinforcement learning (in which a learner tries to maximize reward). Ostensibly, the results are fantastic: the system meets or beats human experts on a large sample of games using a single set of “hyperparameters” that govern properties such as the rate at which a network alters its weights, and no advance knowledge about specific games, or even their rules. But it is easy to wildly overinterpret what the results show. To take one example, according to a widely-circulated video of the system learning to play the brick-breaking Atari game Breakout, “after 240 minutes of training, [the system] realizes that digging a tunnel through the wall is the most effective technique to beat the game”.

更实际的概念，如“球”或“反对者”，也可能无法解决。考虑DeepMind的Atari博弈游戏的例子，采用的强化学习的方法，将深度学习与强化学习结合到一起（强化学习是学习者试图最大化回报函数）。表面上来说，结果非常好：系统在很多博弈样本的基础上，使用一组超参数，控制了如网络改变权重的速度这样的性质，达到或击败了人类专家，不需要关于博弈的高级知识，甚至不需要知道规则。但很容易就过度解读结果的表现。举一个例子，根据那个学习进行打砖块游戏的Atari游戏Breakout的视频，“在240分钟的训练后，系统意识到在墙上打一个隧道是最有效通关的方法”。

But the system has learned no such thing; it doesn’t really understand what a tunnel, or what a wall is; it has just learned specific contingencies for particular scenarios. Transfer tests — in which the deep reinforcement learning system is confronted with scenarios that differ in minor ways from the one ones on which the system was trained show that deep reinforcement learning’s solutions are often extremely superficial. For example, a team of researchers at Vicarious showed that a more efficient successor technique, DeepMind’s Atari system [Asynchronous Advantage Actor-Critic; also known as A3C], failed on a variety of minor perturbations to Breakout (Kansky et al., 2017) from the training set, such as moving the Y coordinate (height) of the paddle, or inserting a wall midscreen. These demonstrations make clear that it is misleading to credit deep reinforcement learning with inducing concept like wall or paddle; rather, such remarks are what comparative (animal) psychology sometimes call overattributions. It’s not that the Atari system genuinely learned a concept of wall that was robust but rather the system superficially approximated breaking through walls within a narrow set of highly trained circumstances.

但系统并没有学习到这个东西；系统并不真正理解什么是一个隧道，什么是一堵墙；它只是学习到了特定场景下的特别应对措施。迁移测试中，将深度学习系统放入与训练系统略微有差异的场景中，会发现深度强化学习的解决方案通常是非常肤浅的。比如，Vicarious的一个研究小组发现，DeepMind的Atari系统的一个更高效的后继技术(A3C)在对Breakout的一些很小的扰动下失败了，比如移动paddle的Y轴（高度），或加入一堵墙的中间屏障。这清楚的说明了，如果说深度强化学习学习到了墙或paddle的概念，是非常误导人的；这种评论与一种动物心理很像，称为overatrribution。并不是Atari系统真的学习到了墙的概念，而是系统在一个很窄的高度训练的情况下肤浅的近似了破墙。

My own team of researchers at a startup company called Geometric Intelligence (later acquired by Uber) found similar results as well, in the context of a slalom game, In 2017, a team of researchers at Berkeley and OpenAI has shown that it was not difficult to construct comparable adversarial examples in a variety of games, undermining not only DQN (the original DeepMind algorithm) but also A3C and several other related techniques (Huang, Papernot, Goodfellow, Duan, & Abbeel, 2017).

我们在一个初创公司的研究小组（称为Geometric Intelligence，后来被Uber收购）也发现了类似的结果，在slalom游戏中，在2017年，一组Berkeley研究者和OpenAI证明了，在很多游戏中都不难构建类似的对抗样本，可以使用不止DQN，还有A3C算法，以及其他几种相关的技术。

Recent experiments by Robin Jia and Percy Liang (2017) make a similar point, in a different domain: language. Various neural networks were trained on a question answering task known as SQuAD (derived from the Stanford Question Answering Database), in which the goal is to highlight the words in a particular passage that correspond to a given question. In one sample, for instance, a trained system correctly, and impressively, identified the quarterback on the winning of Super Bowl XXXIII as John Elway, based on a short paragraph. But Jia and Liang showed the mere insertion of distractor sentences (such as a fictional one about the alleged victory of Google’s Jeff Dean in another Bowl game) caused performance to drop precipitously. Across sixteen models, accuracy dropped from a mean of 75% to a mean of 36%.

最近Robin等的试验在不同的领域得到了类似的结论：语言。在一个问题回答任务，SQuAD(Stanford Question Answering Database)上训练了各种神经网络，其中的目标是在特定段落中划出对应的单词，对应着给定问题的答案。比如，在一个样本中，一个训练好的系统，正确的，令人印象深刻的，基于一个短的段落，识别出了Super Bowl XXXIII获胜队伍的quarterback是John Elway。但Robin等展示了，只需要插入一个混淆语句，就会导致性能急剧下降。在16个模型中，准确率从平均75%降到了36%。

As is so often the case, the patterns extracted by deep learning are more superficial than they initially appear. 通常都是这种情况，深度学习提取出的模式比初始看起来的要肤浅很多。

### 3.3.Deep learning thus far has no natural way to deal with hierarchical structure

To a linguist like Noam Chomsky, the troubles Jia and Liang documented would be unsurprising. Fundamentally, most current deep-learning based language models represent sentences as mere sequences of words, whereas Chomsky has long argued that language has a hierarchical structure, in which larger structures are recursively constructed out of smaller components. (For example, in the sentence the teenager who previously crossed the Atlantic set a record for flying around the world, the main clause is the teenager set a record for flying around the world, while the embedded clause who previously crossed the Atlantic is an embedded clause that specifies which teenager.)

对于Noam Chomsky这样的语言学家来讲，Robin遇到的问题不会令人吃惊。根本上来说，多数目前的基于深度学习的语言模型表示的语句只是单词序列，而Chomsky很早就指出语言有着层次化的结构，其中更大的结构是从更小的部件中构建得到的。（比如，在语句。。。中，主要的句子是。。，而嵌套的句子。。指定了是这个青年）。

In the 80’s Fodor and Pylyshyn (1988)expressed similar concerns, with respect to an earlier breed of neural networks. Likewise, in (Marcus, 2001), I conjectured that single recurrent neural networks (SRNs; a forerunner to today’s more sophisticated deep learning based recurrent neural networks, known as RNNs; Elman, 1990) would have trouble systematically representing and extending recursive structure to various kinds of unfamiliar sentences (see the cited articles for more specific claims about which types).

在80年代时，Fodor等对早一代的神经网络表达了同样的担忧。类似的，2001年我推测SRNs在将递归结构表示和拓展成各种不熟悉的语句时会遇到系统性的麻烦。

Earlier this year, Brenden Lake and Marco Baroni (2017) tested whether such pessimistic conjectures continued to hold true. As they put it in their title, contemporary neural nets were “Still not systematic after all these years”. RNNs could “generalize well when the differences between training and test ... are small [but] when generalization requires systematic compositional skills, RNNs fail spectacularly”.

今年更早的时候，Breden等测试了这样悲观的推测是否仍然是真的。就像在标题中描述的那样，目前的神经网络在这么多年过去后仍然是非系统性的。RNNs在训练和测试集的差异很小的时候可以泛化，但当泛化需要系统性的组成技巧时，RNNs就失败了。

Similar issues are likely to emerge in other domains, such as planning and motor control, in which complex hierarchical structure is needed, particular when a system is likely to encounter novel situations. One can see indirect evidence for this in the struggles with transfer in Atari games mentioned above, and more generally in the field of robotics, in which systems generally fail to generalize abstract plans well in novel environments.

类似的问题在其他领域中也很可能会出现，比如计划和马达控制，其中都需要复杂的层次结构，尤其是一个系统很可能会遇到新的情况时。在Atari游戏的迁移的困难中，也可以得到间接的证明，在机器人技术中也可以看到更一般的证据，其中系统在将抽象计划泛化到新环境中，一般都会失败。

The core problem, at least at present, is that deep learning learns correlations between sets of features that are themselves “flat” or nonhierachical, as if in a simple, unstructured list, with every feature on equal footing. Hierarchical structure (e.g., syntactic trees that distinguish between main clauses and embedded clauses in a sentence) are not inherently or directly represented in such systems, and as a result deep learning systems are forced to use a variety of proxies that are ultimately inadequate, such as the sequential position of a word presented in a sequences.

至少现在的核心问题是，深度学习学习到了集合特征之间的关联性，这些特征是扁平的，非层次化的，就像是在一个简单的没有结构的列表中一样，每个特征都是相同的关系。层次化的结构（如，句法树，区分一个句子中的主要句子和嵌套句子）在这种系统中不能直接进行表示，所以深度学习系统必须使用很多代理，最终是不足以表示序列中单词的序列化位置的。

Systems like Word2Vec (Mikolov, Chen, Corrado, & Dean, 2013) that represent individuals words as vectors have been modestly successful; a number of systems that have used clever tricks try to represent complete sentences in deep-learning compatible vector spaces (Socher, Huval, Manning, & Ng, 2012). But, as Lake and Baroni’s experiments make clear, recurrent networks continue limited in their capacity to represent and generalize rich structure in a faithful manner.

像Word2Vec的系统，将单个单词表示为向量，有了一些成功；一些系统使用了技巧来表示完整的句子，用了与深度学习兼容的向量空间。但是，就像Lake等的试验表明的那样，RNN在富结构的表达和泛化上的能力仍然是有限的。

### 3.4.Deep learning thus far has struggled with open-ended inference

If you can’t represent nuance like the difference between “John promised Mary to leave” and “John promised to leave Mary”, you can’t draw inferences about who is leaving whom, or what is likely to happen next. Current machine reading systems have achieved some degree of success in tasks like SQuAD, in which the answer to a given question is explicitly contained within a text, but far less success in tasks in which inference goes beyond what is explicit in a text, either by combining multiple sentences (so called multi-hop inference) or by combining explicit sentences with background knowledge that is not stated in a specific text selection. Humans, as they read texts, frequently derive wide-ranging inferences that are both novel and only implicitly licensed, as when they, for example, infer the intentions of a character based only on indirect dialog.

如果不能表达这样的细微差别，“John promised Mary to leave” 和 “John promised to leave Mary”，就不能推理出谁在离开谁，或将要发生什么。目前的机器阅读系统在一些任务如SQuAD任务上有了一定程度的成功，其中对给定问题的回答是包含在一段文本中的，但如果推理结果不在文本中，那么成功的概率就少的多。人类在阅读文本的时候，经常推理出很广泛的结论，通常都是新的且隐含的意思，比如，经常会从间接的对话推出角色的意图。

Altough Bowman and colleagues (Bowman, Angeli, Potts, & Manning, 2015; Williams, Nangia, & Bowman, 2017) have taken some important steps in this direction, there is, at present, no deep learning system that can draw open-ended inferences based on real-world knowledge with anything like human-level accuracy.

尽管Bowman等在这方面有了很重要的一些步骤，但目前尚没有深度学习系统能够基于真实世界的知识，以人类式的准确率，作出开放式的推理。

### 3.5.Deep learning thus far is not sufficiently transparent

The relative opacity of “black box” neural networks has been a major focus of discussion in the last few years (Samek, Wiegand, & Müller, 2017; Ribeiro, Singh, & Guestrin, 2016). In their current incarnation, deep learning systems have millions or even billions of parameters, identifiable to their developers not in terms of the sort of human interpretable labels that canonical programmers use (“last_character_typed”) but only in terms of their geography within a complex network (e.g., the activity value of the ith node in layer j in network module k). Although some strides have been in visualizing the contributions of individuals nodes in complex networks (Nguyen, Clune, Bengio, Dosovitskiy, & Yosinski, 2016), most observers would acknowledge that neural networks as a whole remain something of a black box.

神经网络的不透明黑箱性过去几年一直是主要的讨论焦点。在目前的形态中，深度学习系统有数百万甚至数十亿参数，开发者能够辨识的并不是人类可解读的标签，而是在一个复杂网络中的几何分布。虽然在一些复杂网络中的一些节点的贡献已经通过可视化技术进行了一些展示，但大多数观察者承认，神经网络大体上还是一个黑盒子的。

How much that matters in the long run remains unclear (Lipton, 2016). If systems are robust and self-contained enough it might not matter; if it is important to use them in the context of larger systems, it could be crucial for debuggability.

这会有多大问题，长期看来仍然不太清楚。如果系统足够稳健，那么可能没有太大问题；但如果在更大的系统中进行很重要的使用，那么可以调试的能力是非常重要的。

The transparency issue, as yet unsolved, is a potential liability when using deep learning for problem domains like financial trades or medical diagnosis, in which human users might like to understand how a given system made a given decision. As Catherine O’Neill (2016) has pointed out, such opacity can also lead to serious issues of bias.

透明性的问题，由于尚未解决，所以对于在像金融交易或医学诊断领域中的应用来说，是一个潜在的负担，这样的应用中用户很想知道给定的系统是如何作出这样的决策的。如同Catherine指出的那样，这样的不透明型会带来严重的歧视问题。

### 3.6.Deep learning thus far has not been well integrated with prior knowledge

The dominant approach in deep learning is hermeneutic, in the sense of being self-contained and isolated from other, potentially usefully knowledge. Work in deep learning typically consists of finding a training database, sets of inputs associated with respective outputs, and learn all that is required for the problem by learning the relations between those inputs and outputs, using whatever clever architectural variants one might devise, along with techniques for cleaning and augmenting the data set. With just a handful of exceptions, such as LeCun’s convolutional constraint on how neural networks are wired(LeCun, 1989), prior knowledge is often deliberately minimized.

深度学习的主要方法是hermeneutic，意思是独立的，或与其他可能有用的知识隔绝的。深度学习相关的工作一般包括找到一个训练数据集，包含输入的集合，与相关的输出的集合，从这些输入和输出的关系中，学习到问题所需要的知识，可以随意使用任何设计出来的优秀架构，还有各种清洗、扩充数据集的技术。先验知识的引入通常很少。

Thus, for example, in a system like Lerer et al’s (2016) efforts to learn about the physics of falling towers, there is no prior knowledge of physics (beyond what is implied in convolution). Newton’s laws, for example, are not explicitly encoded; the system instead (to some limited degree) approximates them by learning contingencies from raw, pixel level data. As I note in a forthcoming paper in innate (Marcus, in prep) researchers in deep learning appear to have a very strong bias against including prior knowledge even when (as in the case of physics) that prior knowledge is well known.

因此，比如，在Lerer等的系统中，学习了有关falling towers的物理原理，并没有任何物理的先验知识（除了卷积中所内涵的除外）。比如，牛顿定律，并没有显式的包含其中；系统通过从原始的像素级的数据，学习到了一些安排，作为近似。我正在准备的一篇文章中提到，深度学习的研究者似乎对于加入先验知识非常有偏见，即使是在这些先验知识广为所知的情况下。

It is also not straightforward in general how to integrate prior knowledge into a deep learning system, in part because the knowledge represented in deep learning systems pertains mainly to (largely opaque) correlations between features, rather than to abstractions like quantified statements (e.g. all men are mortal), see discussion of universally-quantified one-to-one-mappings in Marcus (2001), or generics (violable statements like dogs have four legs or mosquitos carry West Nile virus (Gelman, Leslie, Was, & Koch, 2015)).

怎样将先验知识集成到深度学习系统中，一般也是不太容易的，部分因为深度学习表示的知识，主要存在于（不透明的）特征之间的关系中，而不是在量化的语句的抽象中的（如，所有人都是凡人）。

A related problem stems from a culture in machine learning that emphasizes competition on problems that are inherently self-contained, without little need for broad general knowledge. This tendency is well exemplified by the machine learning contest platform known as Kaggle, in which contestants vie for the best results on a given data set. Everything they need for a given problem is neatly packaged, with all the relevant input and outputs files. Great progress has been made in this way; speech recognition and some aspects of image recognition can be largely solved in the Kaggle paradigm.

还有一个相关的问题，是源于机器学习的，强调的是孤立问题的竞争，而不需要广泛的通用知识。这个趋势在机器学习比赛Kaggle上有明显体现，在这个平台上，竞争者在给定数据集上比赛得到最好的结果。对于给定的问题，他们所需要的东西都是简洁的打包好的，包括所有相关的输入和输出文件。用这种方式，我们取得了很大的进展；语音识别和图像识别的一部分可以很大程度上以Kaggle的模式得到解决。

The trouble, however, is that life is not a Kaggle competition; children don’t get all the data they need neatly packaged in a single directory. Real-world learning offers data much more sporadically, and problems aren’t so neatly encapsulated. Deep learning works great on problems like speech recognition in which there are lots of labeled examples, but scarcely any even knows how to apply it to more open-ended problems. What’s the best way to fix a bicycle that has a rope caught in its spokes? Should I major in math or neuroscience? No training set will tell us that.

但是问题是，生活不是一场Kaggle比赛；孩子们不会一次得到所有的打包好的数据。真实世界的学习得到的数据是更加零星的，而且问题也不会这么简单的包装好的。深度学习在语音识别这样的问题中表现很好，这种问题中有很多标注好的样本，但大家基本都不知道，怎样应用到更加开放的问题中。怎样修理一辆自行车，绳子缠到了辐条上？我应该选择数学还是神经科学作为专业？不会有训练集告诉我们这些知识。

Problems that have less to do with categorization and more to do with commonsense reasoning essentially lie outside the scope of what deep learning is appropriate for, and so far as I can tell, deep learning has little to offer such problems. In a recent review of commonsense reasoning, Ernie Davis and I (2015) began with a set of easily-drawn inferences that people can readily answer without anything like direct training, such as Who is taller, Prince William or his baby son Prince George? Can you make a salad out of a polyester shirt? If you stick a pin into a carrot, does it make a hole in the carrot or in the pin? As far as I know, nobody has even tried to tackle this sort of thing with deep learning.

与分类关系不大，而与常识推理更多相关的问题，基本上深度学习是不太适合的，据我所知，深度学习对这种问题帮助不大。在最近关于常识推理的一篇总结中，Ernie等从很容易得到的推理集合（人们不需要直接的训练就可以回答的）开始，比如谁更高，用衬衫是否可以做沙拉，如果把针插入胡萝卜，那么是胡萝卜中有孔还是针上有孔？据我所知，还没有人试图用深度学习来解决这个问题。

Such apparently simple problems require humans to integrate knowledge across vastly disparate sources, and as such are a long way from the sweet spot of deep learning-style perceptual classification. Instead, they are perhaps best thought of as a sign that entirely different sorts of tools are needed, along with deep learning, if we are to reach human-level cognitive flexibility.

这么明显的问题，需要人整合很多信息源的知识，但这离深度学习适合解决的问题非常远，即感知分类。我们应当这样认为，即这需要不同类型的工具，如果要达到人类认知水平的灵活性，需要将这些工具与深度学习一起使用。

### 3.7.Deep learning thus far cannot inherently distinguish causation from correlation

If it is a truism that causation does not equal correlation, the distinction between the two is also a serious concern for deep learning. Roughly speaking, deep learning learns complex correlations between input and output features, but with no inherent representation of causality. A deep learning system can easily learn that height and vocabulary are, across the population as a whole, correlated, but less easily represent the way in which that correlation derives from growth and development (kids get bigger as they learn more words, but that doesn’t mean that growing tall causes them to learn more words, nor that learning new words causes them to grow). Causality has been central strand in some other approaches to AI (Pearl, 2000) but, perhaps because deep learning is not geared towards such challenges, relatively little work within the deep learning tradition has tried to address it.

众所周知，因果关系与关联关系是不同的，两者之间的区别，也是深度学习要考虑的问题。大致来说，深度学习学习到的是输入和输出特征之间的复杂相关性，但并与因果关系没有什么关系。深度学习系统可以很容易的学习到，在整个人口中，高度和词汇量是相关的，但不太容易学习到，从增长怎样推理到发展（孩子长大的过程会学到更多词汇量，但并不是说，长高是学到更多词汇的原因，或者说学到新词汇需要人长高）。因果关系是AI其他部分的中心，但是，深度学习并没有这种能力，很少有深度学习的工作试图去解决这个问题。

### 3.8.Deep learning presumes a largely stable world, in ways that may be problematic

The logic of deep learning is such that it is likely to work best in highly stable worlds, like the board game Go, which has unvarying rules, and less well in systems such as politics and economics that are constantly changing. To the extent that deep learning is applied in tasks such as stock prediction, there is a good chance that it will eventually face the fate of Google Flu Trends, which initially did a great job of predicting epidemological data on search trends, only to complete miss things like the peak of the 2013 flu season (Lazer, Kennedy, King, & Vespignani, 2014).

深度学习的逻辑是，在非常稳定的世界里，会得到很好的结果，如围棋对弈，有着不变的规则，但在其他系统，如政治和经济，这样持续变动的系统中，则不太好。在深度学习处理其他任务，如股票预测，很可能最终遇到Google Flu Trends这样的命运，开始在流行病学这样的搜索趋势上表现很好，但只是为了补全丢失的东西，如2013 flu季节的峰值。

### 3.9. Deep learning thus far works well as an approximation, but its answers often cannot be fully trusted

In part as a consequence of the other issues raised in this section, deep learning systems are quite good at some large fraction of a given domain, yet easily fooled.

深度学习对于给定领域的应用，大部分都是蛮擅长的，但却很容易被愚弄，这是其他问题的一个结果。

An ever-growing array of papers has shown this vulnerability, from the linguistic examples of Jia and Liang mentioned above to a wide range of demonstrations in the domain of vision, where deep learning systems have mistaken yellow-and-black patterns of stripes for school buses (Nguyen, Yosinski, & Clune, 2014) and sticker-clad parking signs for well-stocked refrigerators (Vinyals, Toshev, Bengio, & Erhan, 2014) in the context of a captioning system that otherwise seems impressive.

越来越多的文章证实了这种脆弱性，从上文提到的Jia等的语言的例子，到视觉领域的很多例子，其中深度学习系统将黄黑的条形图像误认为是校车，将贴了sticker的停车标志误认为是冰箱，这都是典型的例子。

More recently, there have been real-world stop signs, lightly defaced, that have been mistaken for speed limit signs (Evtimov et al., 2017) and 3d-printed turtles that have been mistake for rifles (Athalye, Engstrom, Ilyas, & Kwok, 2017). A recent news story recounts the trouble a British police system has had in distinguishing nudes from sand dunes.

最近，有真实世界的停车标志，有轻微的损害，但是被误识为限速标志，3d打印的乌龟误识别为手枪。最近还发现了，一个英国警务系统很难区分沙堡与裸体图片。

The “spoofability” of deep learning systems was perhaps first noted by Szegedy et al(2013). Four years later, despite much active research, no robust solution has been found.

深度学习系统的防伪能力首先由Szegedy等提及。四年以后，尽管研究上很积极，但没有发现什么稳健的解决方案。

### 3.10. Deep learning thus far is difficult to engineer with

Another fact that follows from all the issues raised above is that is simply hard to do robust engineering with deep learning. As a team of authors at Google put it in 2014, in the title of an important, and as yet unanswered essay (Sculley, Phillips, Ebner, Chaudhary, & Young, 2014), machine learning is “the high-interest credit card of technical debt”, meaning that is comparatively easy to make systems that work in some limited set of circumstances (short term gain), but quite difficult to guarantee that they will work in alternative circumstances with novel data that may not resemble previous training data (long term debt, particularly if one system is used as an element in another larger system).

上述问题造成的另一个事实是，很难采用深度学习方法进行稳健的工程实现。就像Google的一组工程师在2014年所说的，其题目非常重要，但仍然没有答案，机器学习是“高利率的科技债信用卡”，意思是，在一些有限的环境中很容易工作，但到了另外的环境中，新数据与之前的训练数据不太相像，那么就很难得到很好的结果。

In an important talk at ICML, Leon Bottou (2015) compared machine learning to the development of an airplane engine, and noted that while the airplane design relies on building complex systems out of simpler systems for which it was possible to create sound guarantees about performance, machine learning lacks the capacity to produce comparable guarantees. As Google’s Peter Norvig (2016) has noted, machine learning as yet lacks the incrementality, transparency and debuggability of classical programming, trading off a kind of simplicity for deep challenges in achieving robustness.

在ICML 2015中，Leon等将机器学习与飞机引擎的开发相比较，指出，飞行器设计依靠从简单系统中构建复杂系统，这样可以得到有保证的性能，但机器学习缺少得到可比的保证的能力。就如同Google的Peter Norvig指出的，机器学习缺少传统编程中的递增能力，透明性和可调试能力，得到了对深度挑战的简单性，但是以稳健性为代价的。

Henderson and colleagues have recently extended these points, with a focus on deep reinforcement learning, noting some serious issues in the field related to robustness and replicability (Henderson et al., 2017).

Henderson和同事拓展了这些观点，他们关注的是深度强化学习，指出了这个领域中稳健性和可复制性的一些严重问题。

Although there has been some progress in automating the process of developing machine learning systems (Zoph, Vasudevan, Shlens, & Le, 2017), there is a long way to go.

虽然在自动机器学习系统的研究上有一些进展，但还有很长的路要走。

### 3.11. Discussion

Of course, deep learning, is by itself, just mathematics; none of the problems identified above are because the underlying mathematics of deep learning are somehow flawed. In general, deep learning is a perfectly fine way of optimizing a complex system for representing a mapping between inputs and outputs, given a sufficiently large data set.

当然，深度学习本身只是数学；上述所有问题，没有哪个是因为深度学习的数学有什么缺陷。总体上，深度学习是一种非常好的优化方式，对表示输入输出之间的复杂系统进行了很好的优化，只要有足够的大型数据集。

The real problem lies in misunderstanding what deep learning is, and is not, good for. The technique excels at solving closed-end classification problems, in which a wide range of potential signals must be mapped onto a limited number of categories, given that there is enough data available and the test set closely resembles the training set.

真正的问题在于，对于深度学习是什么，不是什么，有所误解。这项技术在解决有预定限度的分类问题是非常好的，其中大量信号要映射到有限的几个类别，只要有足够数量的数据，测试集与训练集的相似程度很高。

But deviations from these assumptions can cause problems; deep learning is just a statistical technique, and all statistical techniques suffer from deviation from their assumptions.

但与这些假设不符会导致问题；深度学习只是一种统计技术，而且所有统计技术都会在与假设不符的时候表现不好。

Deep learning systems work less well when there are limited amounts of training data available, or when the test set differs importantly from the training set, or when the space of examples is broad and filled with novelty. And some problems cannot, given real- world limitations, be thought of as classification problems at all. Open-ended natural language understanding, for example, should not be thought of as a classifier mapping between a large finite set of sentences and large, finite set of sentences, but rather a mapping between a potentially infinite range of input sentences and an equally vast array of meanings, many never previously encountered. In a problem like that, deep learning becomes a square peg slammed into a round hole, a crude approximation when there must be a solution elsewhere.

当训练数据的数量有限时，或当测试集与训练集的差异太大时，或样本空间很广且都是新的，深度学习系统的效果就没那么好了。一些问题无法给出真实世界的限制，从而解释成一个分类的问题。比如，没有限制的自然语言理解，就不能认为是大型有限的语句集合到大型有限语句的分类的映射，而要认为是可能是无限输入的语句和同样大型的意义阵列的映射，其中的很多可能都没有遇到过。在这样的问题中，深度学习就像是方形的钉子楔入了圆形的孔，所以是一个粗糙的近似，但是可能有其他类型的解。

One clear way to get an intuitive sense of why something is amiss to consider a set of experiments I did long ago, in 1997, when I tested some simplified aspects of language development on a class of neural networks that were then popular in cognitive science. The 1997-vintage networks were, to be sure, simpler than current models — they used no more than three layers (inputs nodes connected to hidden nodes connected to outputs node), and lacked Lecun’s powerful convolution technique. But they were driven by backpropagation just as today’s systems are, and just as beholden to their training data.

在1997年，我进行了一些试验，在一类神经网络中测试了一些语言发展的简化方面，那时候这在认知科学中是非常流行的。1997年的古老网络肯定是比现在的模型要更简单的，当时使用的网络不超过3层（输入节点与隐含层连接，然后连接到输出层），也没有Lecun的卷积技术。但使用的反向传播技术和今天的系统类似，而且也需要相当的训练数据。

In language, the name of the game is generalization — once I hear a sentence like John pilked a football to Mary, I can infer that is also grammatical to say John pilked Mary the football, and Eliza pilked the ball to Alec; equally if I can infer what the word pilk means, I can infer what the latter sentences would mean, even if I had not hear them before.

在语言中，游戏的名称是泛化，我听到了一个句子，John pilked a football to Mary，我会根据语法推导出，John pilked Mary the football，Eliza pilked the ball to Alec；如果我能推理出单词pilk的意思，我也会推导出后面的句子是什么意思，即使我之前没听说过。

Distilling the broad-ranging problems of language down to a simple example that I believe still has resonance now, I ran a series of experiments in which I trained three-layer perceptrons (fully connected in today’s technical parlance, with no convolution) on the identity function, f(x) = x, e.g, f(12)=12.

将涵盖范围广泛的语言问题简化成一个简单的例子，我觉得今天仍然会有很多共鸣，我进行了一系列试验，对恒等函数f(x)=x训练了一些三层感知机（全连接层），，如f(12)=12。

Training examples were represented by a set of input nodes (and corresponding output nodes) that represented numbers in terms of binary digits. The number 7 for example, would be represented by turning on the input (and output) nodes representing 4, 2, and 1. As a test of generalization, I trained the network on various sets of even numbers, and tested it all possible inputs, both odd and even.

训练样本由输入节点进行表示（及对应的输出节点），以二值数字来表示数字。以7为例，就表示为打开输入节点4，2，1。作为泛化的测试，我在很多偶数集合上进行网络训练，测试了所有可能的输入，包括奇数和偶数。

Every time I ran the experiment, using a wide variety of parameters, the results were the same: the network would (unless it got stuck in local minimum) correctly apply the identity function to the even numbers that it had seen before (say 2, 4, 8 and 12), and to some other even numbers (say 6 and 14) but fail on all the odds numbers, yielding, for example f(15) = 14.

每次我进行试验，使用很多参数，结果都是一样的：网络每次都会对之前遇到过的偶数产生正确的结果（如，2，4，8，12），对其他偶数也会产生正确的结果（如6，14），对所有奇数则会产生不正确的结果，如f(15)=14。

In general, the neural nets I tested could learn their training examples, and interpolate to a set of test examples that were in a cloud of points around those examples in n-dimensional space (which I dubbed the training space), but they could not extrapolate beyond that training space.

总体上，我测试的神经网络可以学习到其训练样本，并可以在训练样本的n维空间中，对于可以用训练样本插值出的样本，也可以学到，但无法从训练空间中外插出其他样本。

Odd numbers were outside the training space, and the networks could not generalize identity outside that space.Adding more hidden units didn’t help, and nor did adding more hidden layers. Simple multilayer perceptrons simply couldn’t generalize outside their training space (Marcus, 1998a; Marcus, 1998b; Marcus, 2001). (Chollet makes quite similar points in the closing chapters of his his (Chollet, 2017) text.)

奇数是在训练空间之外的，网络无法泛化到训练空间之外的数字。增加隐藏单元是没有用的，增加更多的隐含层也是没用的。简单的多层感知机就是无法泛化到训练空间之外。Chollet在其教材中也表达了类似的观点。

What we have seen in this paper is that challenges in generalizing beyond a space of training examples persist in current deep learning networks, nearly two decades later. Many of the problems reviewed in this paper — the data hungriness, the vulnerability to fooling, the problems in dealing with open-ended inference and transfer — can be seen as extension of this fundamental problem. Contemporary neural networks do well on challenges that remain close to their core training data, but start to break down on cases further out in the periphery.

我们在本文中看到的是，泛化到训练样本空间之外的挑战，在二十年后，在目前的深度学习网络中仍然是存在的。本文中回顾的很多问题，对数据的渴望，很容易愚弄这个网络，要处理开放的推理到迁移的问题，可以视为这些基本问题的延伸。目前的神经网络对于与核心训练数据相近的数据表现较好，但对于此外的数据，就开始不起作用了。

The widely-adopted addition of convolution guarantees that one particular class of problems that are akin to my identity problem can be solved: so-called translational invariances, in which an object retains its identity when it is shifted to a location. But the solution is not general, as for example Lake’s recent demonstrations show. (Data augmentation offers another way of dealing with deep learning’s challenges in extrapolation, by trying to broaden the space of training examples itself, but such techniques are more useful in 2d vision than in language).

广泛采用的卷积相加保证了，如果特定的问题与训练的问题接近，那么就可以得到解决：这也就是所谓的平移不变性，尽管目标变到了另外的位置上，仍然会保持其目标性。但解决方案并不是一般性的，就像Lake最近给出的结果所显示的。（数据扩充给出了另一种深度学习处理外插的方式，即增加训练样本的空间，但这种技术在2d视觉中更有用，语言中则没有那么有用）。

As yet there is no general solution within deep learning to the problem of generalizing outside the training space. And it is for that reason, more than any other, that we need to look to different kinds of solutions if we want to reach artificial general intelligence.

因为目前深度学习泛化到训练空间之外尚没有一般的解决方案。由于这个原因，我们想要达到通用人工智能，我们需要探索其他类型的解决方案。

## 4. Potential risks of excessive hype 过度炒作的可能风险

One of the biggest risks in the current overhyping of AI is another AI winter, such as the one that devastated the field in the 1970’s, after the Lighthill report (Lighthill, 1973), suggested that AI was too brittle, too narrow and too superficial to be used in practice. Although there are vastly more practical applications of AI now than there were in the 1970s, hype is still a major concern. When a high-profile figure like Andrew Ng writes in the Harvard Business Review promising a degree of imminent automation that is out of step with reality, there is fresh risk for seriously dashed expectations. Machines cannot in fact do many things that ordinary humans can do in a second, ranging from reliably comprehending the world to understanding sentences. No healthy human being would ever mistake a turtle for a rifle or parking sign for a refrigerator.

目前AI的过度炒作的最大风险是另一个AI寒冬，就像在1970‘s摧毁这个领域的一样，在Lighthill报告之后，建议AI太脆弱了，太狭窄了，太肤浅了，不能在实际中得到应用。虽然现在AI的应用比1970s要多的多，但炒作仍然是一个主要的问题。当高调的人物，如Andrew Ng在哈佛商业评论中给出了很有希望的前景，有些不切实际，那么就会有很严重的不切实际的希望。机器实际上很难作出人类在一秒钟就做到的事，包括可靠的理解这个世界到理解语句。没有哪个健康的人类，会把乌龟识别成手枪，或把停车符号识别成冰箱。

Executives investing massively in AI may turn out to be disappointed, especially given the poor state of the art in natural language understanding. Already, some major projects have been largely abandoned, like Facebook’s M project, which was launched in August 2015 with much publicity as a general purpose personal assistant, and then later downgraded to a significantly smaller role, helping users with a vastly small range of well-defined tasks such as calendar entry.

在AI上投资很多的人，会逐渐变得很失望，尤其是在NLP中得到并不太好的结果。而且，一些主要的项目已经大致被抛弃了，如Facebook的M项目，这是在2015年8月启动的，得到了很多关注，希望成为通用目标的个人助理，后来降级到了一个明显更小的角色，帮助用户进行更小的定义的很好的任务，如日程表。

It is probably fair to say that chatbots in general have not lived up to the hype they received a couple years ago. If, for example, driverless car should also, disappoint, relative to their early hype, by proving unsafe when rolled out at scale, or simply not achieving full autonomy after many promises, the whole field of AI could be in for a sharp downturn, both in popularity and funding. We already may be seeing hints of this, as in a just published Wired article that was entitled “After peak hype, self-driving cars enter the trough of disillusionment.”

这样说应当很公平，聊天机器人没有达到前些年吹嘘的水平。无人驾驶也证明了没有达到之前的宣传，在一些情况下会比较危险，或者在很多承诺后，还是达不到完全的自动驾驶，整个AI领域可能会有一个急转弯，包括受欢迎程度和投资。我们已经看到了其中的一些线索，如在Wired杂志上已经发表了这样的文章，题目是“在炒作过度的风险后，自动驾驶汽车进入了幻灭的低谷”。

There are other serious fears, too, and not just of the apocalyptic variety (which for now to still seem to be stuff of science fiction). My own largest fear is that the field of AI could get trapped in a local minimum, dwelling too heavily in the wrong part of intellectual space, focusing too much on the detailed exploration of a particular class of accessible but limited models that are geared around capturing low-hanging fruit — potentially neglecting riskier excursions that might ultimately lead to a more robust path.

还有其他的严重的恐惧，不止是那种大灾变似的（现在看来仍然像是科幻一样）。我自己最大的恐惧是，AI可能会陷入到局部极小值中，陷入到智慧空间的错误部分，过度聚焦在特定类别的模型中，而忽略了其他可能的更稳健的道路上。

I am reminded of Peter Thiel’s famous (if now slightly outdated) damning of an often too-narrowly focused tech industry: “We wanted flying cars, instead we got 140 characters”. I still dream of Rosie the Robost, a full-service domestic robot that take of my home; but for now, six decades into the history of AI, our bots do little more than play music, sweep floors, and bid on advertisements.

我想起了Peter Thiel著名的科技企业诅咒论：我们想要飞行汽车，而我们得到的却是140个字符。我仍然会梦到Rosie the Robust，全功能的家用机器人；但现在，进入了AI历史的60年后，我们的机器人能做的比弹奏音乐，扫地，和在广告上投标多不了多少。

If didn’t make more progress, it would be a shame. AI comes with risk, but also great potential rewards. AI’s greatest contributions to society, I believe, could and should ultimately come in domains like automated scientific discovery, leading among other things towards vastly more sophisticated versions of medicine than are currently possible. But to get there we need to make sure that the field as whole doesn’t first get stuck in a local minimum.

如果没有得到更多进展，那么就很丢人。AI是有风险的，但也有很大的回报。AI对社会最大的贡献，我相信最终是自动科学发现，带头研发出复杂的多的药。但在到达那一步之前，我们首先要确保这个领域整体上别陷入到局部最小值中。

## 5. What would be better?

Despite all of the problems I have sketched, I don’t think that we need to abandon deep learning. 虽然我列出了很多问题，但我觉得还是不能抛弃深度学习。

Rather, we need to reconceptualize it: not as a universal solvent, but simply as one tool among many, a power screwdriver in a world in which we also need hammers, wrenches, and pliers, not to mentions chisels and drills, voltmeters, logic probes, and oscilloscopes.

但是，我们需要重新确立其概念：不是作为一个统一的解决方法，但只是作为一个工具，就像一个厉害的螺丝刀，但我们还是需要锤子，扳手和钳子，更不要说凿子和钻子，电压表，逻辑探头和振荡器。

In perceptual classification, where vast amounts of data are available, deep learning is a valuable tool; in other, richer cognitive domains, it is often far less satisfactory.

对于概念分类，在有大量数据存在可用的情况下，深度学习是一个非常好的工具；但在其他的，更丰富的认知领域中，其效果通常远不如意。

The question is, where else should we look? Here are four possibilities. 问题是，我们应当看向哪里呢？这里有四种可能性。

### 5.1.Unsupervised learning

In interviews, deep learning pioneers Geoff Hinton and Yann LeCun have both recently pointed to unsupervised learning as one key way in which to go beyond supervised, data- hungry versions of deep learning.

在采访中，深度学习的先驱Geoff Hinton和Yann LeCun最近都指出，无监督学习是一种关键的方式，比有监督的，对数据要求很高的深度学习要好。

To be clear, deep learning and unsupervised learning are not in logical opposition. Deep learning has mostly been used in a supervised context with labeled data, but there are ways of using deep learning in an unsupervised fashion. But there is certainly reasons in many domains to move away from the massive demands on data that supervised deep learning typically requires.

为说清楚，深度学习和无监督学习在逻辑上并不是对立的。深度学习大多数只是用于有监督的上下文，而且有标注的数据，但有很多无监督使用深度学习的方法。但在很多领域中，有一些原因不能对数据提出太多的要求，而这是深度学习所需要的。

Unsupervised learning, as the term is commonly used, tends to refer to several kinds of systems. One common type of system “clusters” together inputs that share properties, even without having them explicitly labeled. Google’s cat detector model (Le et al., 2012) is perhaps the most publicly prominent example of this sort of approach.

无监督学习，由于这个术语使用的很多，一般是指几类系统。一种常见的系统将性质相同的输入进行聚类，而且甚至不需要显式的标注。Google的猫检测模型，可能是这类方法中最著名的例子。

Another approach, advocated researchers such as Yann LeCun (Luc, Neverova, Couprie, Verbeek, & LeCun, 2017), and not mutually exclusive with the first, is to replace labeled data sets with things like movies that change over time. The intuition is that systems trained on videos can use each pair of successive frames as a kind of ersatz teaching signal, in which the goal is to predict the next frame; frame t becomes a predictor for frame t1, without the need for any human labeling.

另一种方法与第一种方法不是互相排斥的，是将标注的数据集替换成随着时间变化的电影，Yann LeCun等研究者支持这个思路。直觉是这样的，在视频上训练的系统可以使用成对的连续的帧作为人造的教学信号，其目的是预测下一帧；帧t成为帧t1的预测者，而且不需要任何人类的标注。

My view is that both of these approaches are useful (and so are some others not discussed here), but that neither inherently solve the sorts of problems outlined in section 3. One is still left with data hungry systems that lack explicit variables, and I see no advance there towards open-ended inference, interpretability or debuggability.

我的想法是，这些想法都是有用的（还有一些这里没有讨论的），但这没有但这都没有解决第3节提出的问题。对数据要求很多的系统，缺少显式的变量，对于开放问题的推理，我没有看到什么进展，可解释性和调试性都没有什么进展。

That said, there is a different notion of unsupervised learning, less discussed, which I find deeply interesting: the kind of unsupervised learning that human children do. Children often set themselves a novel task, like creating a tower of Lego bricks or climbing through a small aperture, as my daughter recently did in climbing through a chair, in the space between the seat and the chair back. Often, this sort of exploratory problem solving involves (or at least appears to involve) a good deal of autonomous goal setting (what should I do?) and high level problem solving (how do I get my arm through the chair, now that the rest of my body has passed through?), as well the integration of abstract knowledge (how bodies work, what sorts of apertures and affordances various objects have, and so forth). If we could build systems that could set their own goals and do reasoning and problem-solving at this more abstract level, major progress might quickly follow.

这就是说，无监督学习有不同的概念，讨论的很少，这我发现很有趣：也就是人类的小孩所拥有的无监督学习。孩子通常会给自己一个新的任务，如创建一个Lego积木塔，或爬过一个小的孔，就像我的女儿，最近在爬过一个椅子，就在座位和椅子后之间的空间。通常，这种解释性的问题的解决包含（或至少看起来包含）自动目标设定的很多元素（我应该做什么？），和很多高层次问题的解决（我怎么把胳膊伸到椅子里，现在我身体的剩下部分怎么过去？），以及抽象知识的整合（身体怎么工作？各种物体有什么样的孔和支撑物，等等）。如果我们可以构建一个系统，可以设定自己的目标，在这种更抽象的层次上进行推理和问题解决，可能很快就会出现主要的进展。

### 5.2.Symbol-manipulation, and the need for hybrid models

Another place that we should look is towards classic, “symbolic” AI, sometimes referred to as GOFAI (Good Old-Fashioned AI). Symbolic AI takes its name from the idea, central to mathematics, logic, and computer science, that abstractions can be represented by symbols. Equations like f = ma allow us to calculate outputs for a wide range of inputs, irrespective of whether we have seen any particular values before; lines in computer programs do the same thing (if the value of variable x is greater than the value of variable y, perform action a).

我们应当关注的另一个地方是更经典的符号系统AI，有时候称之为GOFAI(Good Old-Fashioned AI)。符号AI从其思想中得名，是以数学、逻辑和计算机科学为中心的，抽象可以用符号来表示。等式如f=ma允许我们计算很广范围的输出，与我们是否曾经观察到特定的值没有关系；计算机程序做的同样的事（如果x值大于y的值，那么就执行动作a）。

By themselves, symbolic systems have often proven to be brittle, but they were largely developed in era with vastly less data and computational power than we have now. The right move today may be to integrate deep learning, which excels at perceptual classification, with symbolic systems, which excel at inference and abstraction. One might think such a potential merger on analogy to the brain; perceptual input systems, like primary sensory cortex, seem to do something like what deep learning does, but there are other areas, like Broca’s area and prefrontal cortex, that seem to operate at much higher level of abstraction. The power and flexibility of the brain comes in part from its capacity to dynamically integrate many different computations in real-time. The process of scene perception, for instance, seamlessly integrates direct sensory information with complex abstractions about objects and their properties, lighting sources, and so forth.

符号系统通常比较脆弱，但是在数据很少，计算能力也很弱的时代提出的。现在正确的事，可能是整合深度学习和符号系统，一个擅长感知分类，一个擅长推理和抽象。我们可能会将这种合并与人脑类；输入的感知系统，就像初级的传感器皮质，做的事就像深度学习做的一样，但还有其他区域，如Broca区域和prefrontal cortex，似乎是负责更高层的抽象的。大脑的能力和灵活性，似乎部分是由其动态实时整合很多不同的计算的能力。比如，场景感知的过程，将直接的感官信息与复杂的抽象无缝整合到一起，包括目标及其性质，光源，以及等等。

Some tentative steps towards integration already exist, including neurosymbolic modeling (Besold et al., 2017) and recent trend towards systems such as differentiable neural computers (Graves et al., 2016), programming with differentiable interpreters (Bošnjak, Rocktäschel, Naradowsky, & Riedel, 2016), and neural programming with discrete operations (Neelakantan, Le, Abadi, McCallum, & Amodei, 2016). While none of this work has yet fully scaled towards anything like full-service artificial general intelligence, I have long argued (Marcus, 2001) that more on integrating microprocessor- like operations into neural networks could be extremely valuable.

整合方面有一些试验性的步骤，包括神经符号建模和最近的像微分神经网络系统的趋势，可微分解释器的编程，和用离散运算的神经编程。随着这些工作都还没有成为通用人工智能，我很早之前就提出，能将类似微处理器的运算整合到神经网络中将是非常有用的。

To the extent that the brain might be seen as consisting of “a broad array of reusable computational primitives—elementary units of processing akin to sets of basic instructions in a microprocessor—perhaps wired together in parallel, as in the reconfigurable integrated circuit type known as the field-programmable gate array”, as I have argued elsewhere(Marcus, Marblestone, & Dean, 2014), steps towards enriching the instruction set out of which our computational systems are built can only be a good thing.

曾经有这种想法，大脑是由很多可复用的计算原语单元组成的，与微处理器的基本指令类似，可能是并行连接的，就像可以重新配置的FPGA一样，我在其他地方提出过，丰富我们的计算系统的指令集，会是一件很好的事。

### 5.3.More insight from cognitive and developmental psychology

Another potential valuable place to look is human cognition (Davis & Marcus, 2015; Lake et al., 2016; Marcus, 2001; Pinker & Prince, 1988). There is no need for machines to literally replicate the human mind, which is, after all, deeply error prone, and far from perfect. But there remain many areas, from natural language understanding to commonsense reasoning, in which humans still retain a clear advantage; learning the mechanisms underlying those human strengths could lead to advances in AI, even the goal is not, and should not be, an exact replica of human brain.

另一个可能非常有用的地方是人类认知。机器完全没有必要复制人类思想，毕竟人的思想是很容易出错的，很不完美。但有很多领域，从NLP到常识推理，人类仍然有明显的优势；学习这之下的机制可以带来AI的发展，即使其目标不是，也不应该是复制人类大脑。

For many people, learning from humans means neuroscience; in my view, that may be premature. We don’t yet know enough about neuroscience to literally reverse engineer the brain, per se, and may not for several decades, possibly until AI itself gets better. AI can help us to decipher the brain, rather than the other way around.

对于很多人来说，从人类学习就是神经科学；从我来看，这可能是不太成熟的。我们对神经科学知道的还不够多，不足以对人脑进行逆向工程，在最近几十年内都不可能，可能得直到AI本身变得更好。AI可以帮助我们对大脑进行解码，而不是反过来，我们依靠大脑来推动AI的发展。

Either way, in the meantime, it should certainly be possible to use techniques and insights drawn from cognitive and developmental and psychology, now, in order to build more robust and comprehensive artificial intelligence, building models that are motivated not just by mathematics but also by clues from the strengths of human psychology.

同时，不管哪条路，都当然可能利用认知科学、发展心理学中的思想和技术，来构建更稳健的更广泛的人工智能，构建不止由数学推动的模型，还受到人类心理学的线索推动的模型。

A good starting point might be to first to try understand the innate machinery in humans minds, as a source of hypotheses into mechanisms that might be valuable in developing artificial intelligences; in companion article to this one (Marcus, in prep) I summarize a number of possibilities, some drawn from my own earlier work (Marcus, 2001) and others from Elizabeth Spelke’s (Spelke & Kinzler, 2007). Those drawn from my own work focus on how information might be represented and manipulated, such as by symbolic mechanisms for representing variables and distinctions between kinds and individuals from a class; those drawn from Spelke focus on how infants might represent notions such as space, time, and object.

一个好的起点可能是，首先试图理解人类思想的内在机理，作为一种假设的来源，这对于发展人工智能是非常宝贵的；在另一篇文章中，我总结了几个可能性，一些是我之前文章中的，另一些是Spelke的。我自己文章中的，关注的是信息是怎样表示和处理；从Spelke中关注的是婴儿是怎样表示概念的，如空间，时间和物体。

A second focal point might be on common sense knowledge, both in how it develops (some might be part of our innate endowment, much of it is learned), how it is represented, and how it is integrated on line in the process of our interactions with the real world (Davis & Marcus, 2015). Recent work by Lerer et al (2016), Watters and colleagues (2017), Tenenbaum and colleagues(Wu, Lu, Kohli, Freeman, & Tenenbaum, 2017) and Davis and myself (Davis, Marcus, & Frazier-Logue, 2017) suggest some competing approaches to how to think about this, within the domain of everyday physical reasoning.

第二个关注点可能是常识知识，包括常识是怎样产生的（多数是学习得到的，一些也可能是内在的），怎样表示，以及怎样整合到我们与真实世界的交互过程的。最近的一些工作给出了一些这些方法的思考，都是在日常推理的领域中的。

A third focus might be on human understanding of narrative, a notion long ago suggested by Roger Schank and Abelson (1977) and due for a refresh (Marcus, 2014; Kočiský et al., 2017).

第三个关注点可能是人类对描述的理解。

### 5.4.Bolder challenges

Whether deep learning persists in current form, morphs into something new, or gets replaced altogether, one might consider a variety of challenge problems that push systems to move beyond what can be learned in supervised learning paradigms with large datasets. Drawing in part of from a recent special issue of AI Magazine devoted to moving beyond the Turing Test that I edited with Francesca Rossi, Manuelo Veloso (Marcus, Rossi, Veloso - AI Magazine, & 2016, 2016), here are a few suggestions:

深度学习是否以目前的形式发展，或发展成新的形式，或被其他方法替代，我们可能会遇到很多挑战，推动现在的系统的发展，超越监督学习的范式，即需要大型数据集。从最近的一本AI杂志得来的一些建议如下：

- A comprehension challenge (Paritosh & Marcus, 2016; Kočiský et al., 2017)] which would require a system to watch an arbitrary video (or read a text, or listen to a podcast) and answer open-ended questions about what is contained therein. (Who is the protagonist? What is their motivation? What will happen if the antagonist succeeds in her mission?) No specific supervised training set can cover all the possible contingencies; infererence and real-world knowledge integration are necessities.

一个综合挑战是，让系统观看任意的视频（或阅读文本，或听podcast），然后回答开放式的问题，关于其中包含了什么。（谁是主角？其动机是什么？如果对手的任务成功了，那么会发生什么？）没有哪个特定的监督训练集可以覆盖所有可能的情况；推理和真实世界知识的整合是必须的。

- Scientific reasoning and understanding, as in the Allen AI institute’s 8th grade science challenge (Schoenick, Clark, Tafjord, P, & Etzioni, 2017; Davis, 2016). While the answers to many basic science questions can simply be retrieved from web searches, others require inference beyond what is explicitly stated, and the integration of general knowledge.

科学推理和理解，就像在Allen AI研究所第8级科学挑战上的。对很多基本的科学问题的回答都可以从网络搜索中检索到，其他的需要一些推理，没有显式的描述，还需要与通用知识的整合。

- General game playing (Genesereth, Love, & Pell, 2005), with transfer between games (Kansky et al., 2017), such that, for example, learning about one first-person shooter enhances performance on another with entirely different images, equipment and so forth. (A system that can learn many games, separately, without transfer between them, such as DeepMind’s Atari game system, would not qualify; the point is to acquire cumulative, transferrable knowledge).

通用博弈，和博弈之间的迁移，比如从第一视角射击游戏学到的经验，可以增强另一种游戏的性能，而画面、设备完全不同等等。（一个系统可以分立的学到很多博弈，在这些博弈中没有迁移，比如DeepMind的Atari博弈系统，可能不太合格；关键是要能够得到可累积的、可迁移的知识）。

- A physically embodied test an AI-driven robot that could build things (Ortiz Jr, 2016), ranging from tents to IKEA shelves, based on instructions and real-world physical interactions with the objects parts, rather than vast amounts trial-and-error.

有物理实体的AI驱动的机器人可以构建物体，从IKEA的架子，基于指令和真实世界的与物体其他部分的物理互动，而不是基于大量的试错得到。

No one challenge is likely to be sufficient. Natural intelligence is multi-dimensional (Gardner, 2011), and given the complexity of the world, generalized artificial intelligence will necessarily be multi-dimensional as well.

没有哪个挑战是充分的。自然智慧是多维的，因为世界的复杂性，通用人工智能也必须是多维的。

By pushing beyond perceptual classification and into a broader integration of inference and knowledge, artificial intelligence will advance, greatly.

从感知分类，到更广的推理和知识的整合，人工智能会得到更大的发展。

## 6. Conclusions

As a measure of progress, it is worth considering a somewhat pessimistic piece I wrote for The New Yorker five years ago, conjecturing that “deep learning is only part of the larger challenge of building intelligent machines” because “such techniques lack ways of representing causal relationships (such as between diseases and their symptoms), and are likely to face challenges in acquiring abstract ideas like “sibling” or “identical to.” They have no obvious ways of performing logical inferences, and they are also still a long way from integrating abstract knowledge, such as information about what objects are, what they are for, and how they are typically used.”

五年前我为New Yorker写过一篇文章，观点有些悲观，文章推测到，“深度学习只是更大的挑战的一部分，即构建智能机器的一部分”，因为“这种技术缺少表示因果关系的方法（如症状与疾病之间的关系），而且在获取抽象观点时会遇到困难，如sibling或identical to。没有明显的进行逻辑推理的方式，而且在与抽象知识整合的目标还有很长的路要走，比如关于什么是物体的信息，是用来做什么的，以及典型的应用方式。”

As we have seen, many of these concerns remain valid, despite major advances in specific domains like speech recognition, machine translation, and board games, and despite equally impressive advances in infrastructure and the amount of data and compute available.

正如我们看到的，许多这些担忧都是对的，尽管在一些特定领域，如语音识别，机器翻译，博弈游戏，有了很大发展，而且有一些基础设施，和数据量，计算量，都有了很大的发展。

Intriguingly, in the last year, a growing array of other scholars, coming from an impressive range of perspectives, have begun to emphasize similar limits. A partial list includes Brenden Lake and Marco Baroni (2017), François Chollet (2017), Robin Jia and Percy Liang (2017), Dileep George and others at Vicarious (Kansky et al., 2017) and Pieter Abbeel and colleagues at Berkeley (Stoica et al., 2017).

有趣的是，在去年，越来越多的其他学者，从很多其他的角度，开始强调类似的局限性。

Perhaps most notably of all, Geoff Hinton has been courageous enough to reconsider has own beliefs, revealing in an August interview with the news site Axios16 that he is “deeply suspicious” of back-propagation, a key enabler of deep learning that he helped pioneer, because of his concern about its dependence on labeled data sets.

可能最著名的，Geoff Hinton也开始重新思考他的信仰，在8月份的一个采访中说到，对于反向传播非常担心，这是深度学习的基础技术，而且对于标注的数据集的严重依赖性。

Instead, he suggested (in Axios’ paraphrase) that “entirely new methods will probably have to be invented.” 他表示，可能需要发明全新的方法。

I share Hinton’s excitement in seeing what comes next. 我很期待看到未来会出现什么。