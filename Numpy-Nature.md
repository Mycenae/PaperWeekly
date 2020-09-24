# Array Programming with NumPy

Charles R. Harris et. al. 

Array programming provides a powerful, compact and expressive syntax for accessing, manipulating and operating on data in vectors, matrices and higher-dimensional arrays. NumPy is the primary array programming library for the Python language. It has an essential role in research analysis pipelines in fields as diverse as physics, chemistry, astronomy, geoscience, biology, psychology, materials science, engineering, finance and economics. For example, in astronomy, NumPy was an important part of the software stack used in the discovery of gravitational waves and in the first imaging of a black hole. Here we review how a few fundamental array concepts lead to a simple and powerful programming paradigm for organizing, exploring and analysing scientific data. NumPy is the foundation upon which the scientific Python ecosystem is constructed. It is so pervasive that several projects, targeting audiences with specialized needs, have developed their own NumPy-like interfaces and array objects. Owing to its central position in the ecosystem, NumPy increasingly acts as an interoperability layer between such array computation libraries and, together with its application programming interface (API), provides a flexible framework to support the next decade of scientific and industrial analysis.

阵列编程为向量、矩阵和更高维度的阵列的数据访问、操作提供了一个强力的、紧凑的和有表达力的语法。NumPy是Python语言首选的阵列编程库。在研究分析过程中有很关键的角色，领域包括物理，化学，天文学，地球科学，生物学，心理学，材料科学，工程学，金融学和经济学。比如，在天文学中，NumPy是软件栈的一部分，用于发现引力波，和对黑洞的第一次成像。这里，我们回顾了阵列的几个基本概念怎样带来简单强力的编程范式，对科学数据进行组织、探索和分析。NumPy是科学Python生态构建的基础。它非常普遍，几个面向特殊需求的观众的项目，开发出了其自己的类似NumPy的接口和阵列目标。由于其在生态系统中的中心位置，NumPy越来越成为这样的阵列计算库中的互操作层，与其API一起，为下一个十年的科学和工业分析提供了一个灵活的框架。

Two Python array packages existed before NumPy. The Numeric package was developed in the mid-1990s and provided array objects and array-aware functions in Python. It was written in C and linked to standard fast implementations of linear algebra. One of its earliest uses was to steer C++ applications for inertial confinement fusion research at Lawrence Livermore National Laboratory. To handle large astronomical images coming from the Hubble Space Telescope, a reimplementation of Numeric, called Numarray, added support for structured arrays, flexible indexing, memory mapping, byte-order variants, more efficient memory use, flexible IEEE 754-standard error-handling capabilities, and better type-casting rules. Although Numarray was highly compatible with Numeric, the two packages had enough differences that it divided the community; however, in 2005 NumPy emerged as a ‘best of both worlds’ unification — combining the features of Numarray with the small-array performance of Numeric and its rich C API.

在NumPy之前，有过两个Python阵列包。在1990s中期，开发过Numeric包，以Python语言给出了阵列目标。这是以C语言写的，给出了线性代数的标准快速实现。其最早的一个用处是，在Lawrence Livermore国家实验室为惯性局限融合运行C++应用。为处理从Hubble空间望远镜传来的很大的太空图像，Numarray重新实现了Numeric，增加了对以下功能的支持：结构化阵列，灵活索引，内存映射，位顺序变体，更高效的内存使用，灵活的IEEE 754标准的错误处理能力，和更好的类别转换规则。虽然Numarray与Numeric高度兼容，但这两个包有足够的差异，使得团体分裂；但是，在2005年，NumPy出现了，作为两者联合的最佳者，结合了Numarray的特征与Numeric的小型阵列性能的特征，及其丰富的C API。

Now, 15 years later, NumPy underpins almost every Python library that does scientific or numerical computation, including SciPy, Matplotlib, pandas, scikit-learn and scikit-image. NumPy is a community-developed, open-source library, which provides a multidimensional Python array object along with array-aware functions that operate on it. Because of its inherent simplicity, the NumPy array is the de facto exchange format for array data in Python.

现在，15年后，NumPy加强了几乎每个Python库，进行科学或数值计算的库，包括SciPy，Matplotlib，pandas，scikit-learn和scikit-image。NumPy是一个团体开发，开源的库，提供了一个多维Python阵列对象，和在其上进行运算的阵列函数。由于其内在的简单性，NumPy阵列是事实上的Python阵列数据交换格式。

NumPy operates on in-memory arrays using the central processing unit (CPU). To utilize modern, specialized storage and hardware, there has been a recent proliferation of Python array packages. Unlike with the Numarray–Numeric divide, it is now much harder for these new libraries to fracture the user community — given how much work is already built on top of NumPy. However, to provide the community with access to new and exploratory technologies, NumPy is transitioning into a central coordinating mechanism that specifies a well defined array programming API and dispatches it, as appropriate, to specialized array implementations.

NumPy用CPU在内存中的阵列进行运算。为利用现代的，专用的存储和硬件，最近又涌现出了一些Python阵列包。与Numarray-Numeric的分割不同，现在这些新库很难再去分裂用户团体了 - 因为在NumPy的基础上已经构建了很多工作。但是，为使团体能够接触到新的和有探索性的技术，NumPy正转变成一种中央协调机制，指定一种良好定义的阵列编程API并分派到专用的阵列实现中。

## 1. NumPy arrays

The NumPy array is a data structure that efficiently stores and accesses multidimensional arrays (also known as tensors), and enables a wide variety of scientific computation. It consists of a pointer to memory, along with metadata used to interpret the data stored there, notably ‘data type’, ‘shape’ and ‘strides’ (Fig. 1a).

NumPy阵列是一个数据结构，对多维阵列进行高效的存储和访问（也称为张量），并使得大量科学计算成为可能。它包含对内存的一个指针，以及用于解释存储这儿的数据的元数据，即数据类型，形状和步长（图1a）。

The data type describes the nature of elements stored in an array. An array has a single data type, and each element of an array occupies the same number of bytes in memory. Examples of data types include real and complex numbers (of lower and higher precision), strings, timestamps and pointers to Python objects.

数据类型描述了元素在阵列中存储的本质。一个阵列只有一个数据类型，一个阵列的每个元素在内存中占据了相同数量的比特数。数据类型的例子包括实数和复数（精度或低或高），字符串，时间戳和指针，以及Python的目标。

The shape of an array determines the number of elements along each axis, and the number of axes is the dimensionality of the array. For example, a vector of numbers can be stored as a one-dimensional array of shape N, whereas colour videos are four-dimensional arrays of shape (T, M, N, 3).

一个阵列的形状确定了元素沿着每个轴的数量，轴的数量是阵列的维度。比如，一个向量可以存储为一维阵列，形状为N，而彩色视频是四维阵列，形状为(T,M,N,3)。

Strides are necessary to interpret computer memory, which stores elements linearly, as multidimensional arrays. They describe the number of bytes to move forward in memory to jump from row to row, column to column, and so forth. Consider, for example, a two-dimensional array of floating-point numbers with shape (4, 3), where each element occupies 8 bytes in memory. To move between consecutive columns, we need to jump forward 8 bytes in memory, and to access the next row, 3 × 8 = 24 bytes. The strides of that array are therefore (24, 8). NumPy can store arrays in either C or Fortran memory order, iterating first over either rows or columns. This allows external libraries written in those languages to access NumPy array data in memory directly.

步长对于解释计算机内存是必须的，因为多维阵列的元素是线性存储的。这描述了在内存中从一行跳到下一行，从一列跳到另一列，等等，所需要移动的bytes数量。比如，考虑一个二维浮点数阵列，形状为(4,3)，其中每个元素占据了内存中的8个bytes。为在连续列中移动，我们需要在内存中条约8个bytes，要跳到下一行，我们需要在内存中移动3 × 8 = 24 bytes。这个阵列的步长因此就是(24, 8)。NumPy可以以C或Fortran的内存顺序存储阵列，以行或列来进行迭代。这允许以这些语言所写的外部的库，直接在内存中访问NumPy阵列数据。

Users interact with NumPy arrays using ‘indexing’ (to access subarrays or individual elements), ‘operators’ (for example, +, − and × for vectorized operations and @ for matrix multiplication), as well as ‘array-aware functions’; together, these provide an easily readable, expressive, high-level API for array programming while NumPy deals with the underlying mechanics of making operations fast.

用户与NumPy阵列互动使用索引（访问子阵列或单个元素），或运算符（比如，向量运算的+, -和×，和矩阵相乘的@），以及以阵列为参数的函数；这些一起，为阵列编程提供了一种可以容易阅读的，有表达力的，高层API，而NumPy处理的是使这些运算更加快速。

Indexing an array returns single elements, subarrays or elements that satisfy a specific condition (Fig. 1b). Arrays can even be indexed using other arrays (Fig. 1c). Wherever possible, indexing that retrieves a subarray returns a ‘view’ on the original array such that data are shared between the two arrays. This provides a powerful way to operate on subsets of array data while limiting memory usage.

阵列索引返回一个元素，子阵列或满足一定条件的元素（图1b）。阵列可以用其他阵列进行索引（图1c）。只要可能，返回一个子阵列的索引，返回的是原始阵列的一个视图，其数据是两个阵列所共有的。这在内存有限的情况下，提供了操作子阵列的很强的方法。

To complement the array syntax, NumPy includes functions that perform vectorized calculations on arrays, including arithmetic, statistics and trigonometry (Fig. 1d). Vectorization—operating on entire arrays rather than their individual elements—is essential to array programming. This means that operations that would take many tens of lines to express in languages such as C can often be implemented as a single, clear Python expression. This results in concise code and frees users to focus on the details of their analysis, while NumPy handles looping over array elements near-optimally — for example, taking strides into consideration to best utilize the computer’s fast cache memory.

为补充阵列的语法，NumPy中包含了对阵列进行向量计算的函数，包括代数运算，统计运算和三角运算（图1d）。向量化，即在整个阵列中进行运算，而不是对单个元素中进行运算，这对于阵列编程是极为关键的。这意味着，用C语言这样以几十行才能表达的运算，可以用一行简洁的Python表达式实现。这形成了简洁的代码，使用户可以关注其分析的细节，而NumPy处理对阵列元素以几乎最优的效率进行循环 - 比如，将步长纳入考虑的话，最好利用计算机的快速缓存。

When performing a vectorized operation (such as addition) on two arrays with the same shape, it is clear what should happen. Through ‘broadcasting’ NumPy allows the dimensions to differ, and produces results that appeal to intuition. A trivial example is the addition of a scalar value to an array, but broadcasting also generalizes to more complex examples such as scaling each column of an array or generating a grid of coordinates. In broadcasting, one or both arrays are virtually duplicated (that is, without copying any data in memory), so that the shapes of the operands match (Fig. 1d). Broadcasting is also applied when an array is indexed using arrays of indices (Fig. 1c).

当在两个相同形状的阵列中进行向量化的运算（如相加）时，很清楚会发生什么。通过广播NumPy允许维度不一样，产生吸引直觉的结果。一个不重要的例子是，标量值与阵列的相加，但广播还会泛化到更复杂的样本中，比如对一个阵列中的每一列进行缩放，或生成坐标网格。在广播中，一个或两个阵列被虚拟复制（即，没有在内存中复制任何数据），这样操作数的形状进行匹配（图1d）。当一个阵列是用索引阵列进行索引时，广播也在进行应用。

Other array-aware functions, such as sum, mean and maximum, perform element-by-element ‘reductions’, aggregating results across one, multiple or all axes of a single array. For example, summing an n-dimensional array over d axes results in an array of dimension n − d (Fig. 1f).

其他阵列函数，比如求和，平均和最大值，进行的是元素级的缩减，在一个阵列的一个、多个或所有轴上聚积得到结果。比如，对一个n维阵列，在d个轴上求和，得到的阵列维度为n-d。

NumPy also includes array-aware functions for creating, reshaping, concatenating and padding arrays; searching, sorting and counting data; and reading and writing files. It provides extensive support for generating pseudorandom numbers, includes an assortment of probability distributions, and performs accelerated linear algebra, using one of several backends such as OpenBLAS or Intel MKL optimized for the CPUs at hand (see Supplementary Methods for more details).

NumPy还包含了一些阵列函数，以创建阵列，改变阵列的形状，拼接阵列和对阵列补充一定的值；搜索，排序，对数据进行计数；读写文件。对生成伪随机数还提供了广泛的支持，包括各种概率分布，和进行加速的线性代数运算，使用一种或几种后台，比如OpenBLAS或Intel MKL对CPU进行的优化。

Altogether, the combination of a simple in-memory array representation, a syntax that closely mimics mathematics, and a variety of array-aware utility functions forms a productive and powerfully expressive array programming language.

总而言之，简单的内存中阵列表示，密切模仿数学的语法，和很多阵列工具函数的组合，形成了一种强力的阵列编程语言。

## 2. Scientific Python ecosystem

Python is an open-source, general-purpose interpreted programming language well suited to standard programming tasks such as cleaning data, interacting with web resources and parsing text. Adding fast array operations and linear algebra enables scientists to do all their work within a single programming language—one that has the advantage of being famously easy to learn and teach, as witnessed by its adoption as a primary learning language in many universities.

Python是一种开源的通用目标的解释编程语言，与标准编程任务很好的适配，如数据清洗，与网络资源互动并解析文字。增加快速阵列运算和线性代数，使科学家可以使用单一一种编程语言完成其任务，很容易学习和教授，在很多大学都被采用作为首选的学习语言。

Even though NumPy is not part of Python’s standard library, it benefits from a good relationship with the Python developers. Over the years, the Python language has added new features and special syntax so that NumPy would have a more succinct and easier-to-read array notation. However, because it is not part of the standard library, NumPy is able to dictate its own release policies and development patterns.

虽然NumPy并不是Python的标准库的一部分，但与Python开发者的关系很好。过去这些年，Python语言加入了新的特征和特殊的语法，这样NumPy可以有更简洁，更容易阅读的阵列表示。但是，由于其不是标准库的一部分，NumPy可以保持自己的发行政策和开发模式。

SciPy and Matplotlib are tightly coupled with NumPy in terms of history, development and use. SciPy provides fundamental algorithms for scientific computing, including mathematical, scientific and engineering routines. Matplotlib generates publication-ready figures and visualizations. The combination of NumPy, SciPy and Matplotlib, together with an advanced interactive environment such as IPython or Jupyter, provides a solid foundation for array programming in Python. The scientific Python ecosystem (Fig. 2) builds on top of this foundation to provide several, widely used technique-specific libraries, that in turn underlie numerous domain-specific projects. NumPy, at the base of the ecosystem of array-aware libraries, sets documentation standards, provides array testing infrastructure and adds build support for Fortran and other compilers.

SciPy和Matplotlib与NumPy紧密结合，在历史上，开发上和使用上都是。SciPy提供了科学计算的基本算法，包括数学的，科学的和工程上的。Matplotlib可以生成供发表的图表和可视化。NumPy, SciPy和Matplotlib的组合，与高级互动环境一起，如IPython或Jupyter，为使用Python进行阵列编程提供了坚实的基础。在这个基础上构建的Python科学生态（图2），给出了几个广泛使用的专用技术库，是几个专用领域工程的基础。NumPy是这个生态的阵列库的基础，设定了文档上的标准，给出了阵列测试的基础设施，为Fortran和其他编译器提供了构建的支持。

Many research groups have designed large, complex scientific libraries that add application-specific functionality to the ecosystem. For example, the eht-imaging library, developed by the Event Horizon Telescope collaboration for radio interferometry imaging, analysis and simulation, relies on many lower-level components of the scientific Python ecosystem. In particular, the EHT collaboration used this library for the first imaging of a black hole. Within eht-imaging, NumPy arrays are used to store and manipulate numerical data at every step in the processing chain: from raw data through calibration and image reconstruction. SciPy supplies tools for general image-processing tasks such as filtering and image alignment, and scikit-image, an image-processing library that extends SciPy, provides higher-level functionality such as edge filters and Hough transforms. The ‘scipy.optimize’ module performs mathematical optimization. NetworkX, a package for complex network analysis, is used to verify image comparison consistency. Astropy handles standard astronomical file formats and computes time–coordinate transformations. Matplotlib is used to visualize data and to generate the final image of the black hole.

很多研究小组都设计了大型复杂的科学计算库，为这个生态系统加入了应用专属的功能。比如，eht-imging库，由Event Horizon Telescope合作开发，用于辐射干扰量度成像，分析与仿真，依赖于更多更底层的Python科学计算生态的组件。特别是，EHT将这个库用于第一次黑洞成像。在eht-imaging中，NumPy阵列用于在整个处理链中的每个步骤中进行数值数据的存储和处理：从原始数据到标定到图像重建。SciPy提供了通用图像处理任务的工具，比如滤波和图像对齐，而拓展了SciPy的专用图像处理库scikit-image则给出更高层的功能，如边缘滤波和Hough变换。scipy.optimize模块进行数学优化。NetworkX，这是进行复杂网络分析的库，用于核实图像对比的一致性。Astropy处理标准的天文文件格式，计算时间协调的变换。Matplotlib用于数据可视化，以及生成最终的黑洞图像。

The interactive environment created by the array programming foundation and the surrounding ecosystem of tools—inside of IPython or Jupyter—is ideally suited to exploratory data analysis. Users can fluidly inspect, manipulate and visualize their data, and rapidly iterate to refine programming statements. These statements are then stitched together into imperative or functional programs, or notebooks containing both computation and narrative. Scientific computing beyond exploratory work is often done in a text editor or an integrated development environment (IDE) such as Spyder. This rich and productive environment has made Python popular for scientific research.

阵列编程基础创造的互动环境，和周围的工具生态（在IPython或Jupyter内），与探索数据分析非常吻合。用户可以流畅的检查，操作并可视化数据，迅速迭代以改进编程语句。这些语句然后放到一起，形成命令式的或函数式的程序，或notebooks，包含计算和叙述。探索性工作之外的科学计算，通常是在文本编辑器或IDE中完成，如Spyder。这种丰富的有产出的环境，使得Python在科学研究中非常流行。

To complement this facility for exploratory work and rapid prototyping, NumPy has developed a culture of using time-tested software engineering practices to improve collaboration and reduce error. This culture is not only adopted by leaders in the project but also enthusiastically taught to newcomers. The NumPy team was early to adopt distributed revision control and code review to improve collaboration on code, and continuous testing that runs an extensive battery of automated tests for every proposed change to NumPy. The project also has comprehensive, high-quality documentation, integrated with the source code.

为对这些探索性的工作和快速成型进行补充，NumPy形成了一种使用经过时间检验的软件工程实践文化来改进合作并减少错误。这种文化不仅仅被工程中的领导所采用，而且会被教授给新来人。NumPy团队很早就采用了分布式版本控制和代码检查来改进代码的合作，并持续测试。这个工程有综合的高质量的文档，与源代码整合到了一起。

This culture of using best practices for producing reliable scientific software has been adopted by the ecosystem of libraries that build on NumPy. For example, in a recent award given by the Royal Astronomical Society to Astropy, they state: “The Astropy Project has provided hundreds of junior scientists with experience in professional-standard software development practices including use of version control, unit testing, code review and issue tracking procedures. This is a vital skill set for modern researchers that is often missing from formal university education in physics or astronomy”. Community members explicitly work to address this lack of formal education through courses and workshops.

这种使用最佳实践来产生可靠的科学软件的文化，已经被构建与NumPy之上的生态系统的库所采用。比如，在皇家天文学会给Astropy最近的奖励中，他们说：Astropy工程使数百个初级科学家有了专业标准的软件开发实践经验，包括使用版本控制，单元测试，代码评审和问题追踪过程。这对于现代研究者来说是很关键的技能，而这些人通常缺少正规大学在物理或天文学的教育。团队成员通过课程和研讨会来解决缺少正规教育的问题。

The recent rapid growth of data science, machine learning and artificial intelligence has further and dramatically boosted the scientific use of Python. Examples of its important applications, such as the eht-imaging library, now exist in almost every discipline in the natural and social sciences. These tools have become the primary software environment in many fields. NumPy and its ecosystem are commonly taught in university courses, boot camps and summer schools, and are the focus of community conferences and workshops worldwide. NumPy and its API have become truly ubiquitous.

最近数据科学、机器学习和人工智能的快速增长，已经进一步急剧的提升了Python的科学用处。其重要应用的例子，比如eht-imaging库，现在在自然和社会科学的每个学科中。这些工具已经在很多领域中成为首选的软件环境。NumPy及其生态系统一般在大学课程、训练营和暑期学校中都会教授，是世界范围内团体会议和研讨会的焦点。NumPy及其API已经变得真正的无处不在。

## 3. Array proliferation and interoperability

NumPy provides in-memory, multidimensional, homogeneously typed (that is, single-pointer and strided) arrays on CPUs. It runs on machines ranging from embedded devices to the world’s largest supercomputers, with performance approaching that of compiled languages. For most its existence, NumPy addressed the vast majority of array computation use cases.

NumPy提供了内存中的多维同质类型（即，单指针单步长）的CPU阵列。它可以运行在从嵌入式设备到世界上最大的超级计算机的设备上，性能接近编译后的语言。对于多数存在，NumPy解决了多数阵列计算使用情况中的问题。

However, scientific datasets now routinely exceed the memory capacity of a single machine and may be stored on multiple machines or in the cloud. In addition, the recent need to accelerate deep-learning and artificial intelligence applications has led to the emergence of specialized accelerator hardware, including graphics processing units (GPUs), tensor processing units (TPUs) and field-programmable gate arrays (FPGAs). Owing to its in-memory data model, NumPy is currently unable to directly utilize such storage and specialized hardware. However, both distributed data and also the parallel execution of GPUs, TPUs and FPGAs map well to the paradigm of array programming: therefore leading to a gap between available modern hardware architectures and the tools necessary to leverage their computational power.

但是，科学数据集现在通常都超过了单个机器的内存容量，可能存储在多个机器上或在云上。另外，最近加速深度学习和人工智能应用的需求，带来了专用加速器硬件的崛起，包括GPU，TPU和FPGA等。由于其内存中的数据模型，NumPy目前不能直接利用这样的存储和专用硬件。但是，分布式数据和GPU、TPU、FPGA的并行执行与阵列编程的范式对应的很好：因此在可用的现代硬件架构和利用其计算能力的必须工具的间隙。

The community’s efforts to fill this gap led to a proliferation of new array implementations. For example, each deep-learning framework created its own arrays; the PyTorch , Tensorflow , Apache MXNet and JAX arrays all have the capability to run on CPUs and GPUs in a distributed fashion, using lazy evaluation to allow for additional performance optimizations. SciPy and PyData/Sparse both provide sparse arrays, which typically contain few non-zero values and store only those in memory for efficiency. In addition, there are projects that build on NumPy arrays as data containers, and extend its capabilities. Distributed arrays are made possible that way by Dask, and labelled arrays—referring to dimensions of an array by name rather than by index for clarity, compare x[:, 1] versus x.loc[:, 'time']—by xarray.

弥补这个空白的努力，带来了新的阵列实现的涌现。比如，每个深度学习的框架都会创建其本身的阵列；PyTorch，Tensorflow，Apache MXNet和JAX的阵列都有在CPUs和GPUs上以分布式方式运行的能力，使用懒评估来允许额外的性能优化。SciPy和PyData/Sparse都提供了稀疏阵列，一般包含很少的非零值，一般只将非零值存储在内存中，以提高效率。另外，有一些在NumPy的基础上构建的工程，作为数据容器，并拓展了其能力。Dask开发了分布式阵列，标记阵列xarray - 指用名称来指代阵列的维度，而不是用索引，比较一下x[:, 1]与x.loc[:, 'time']。

Such libraries often mimic the NumPy API, because this lowers the barrier to entry for newcomers and provides the wider community with a stable array programming interface. This, in turn, prevents disruptive schisms such as the divergence between Numeric and Numarray. But exploring new ways of working with arrays is experimental by nature and, in fact, several promising libraries (such as Theano and Caffe) have already ceased development. And each time that a user decides to try a new technology, they must change import statements and ensure that the new library implements all the parts of the NumPy API they currently use.

这样的库通常会模仿NumPy的API，因为这降低了新人进入的门槛，给更多的团体提供了一个稳定的阵列编程接口。这又防止了破坏性的分裂，如Numeric和Numarray之间的分歧。但探索新的利用阵列进行工作的方法是试验性的，实际上，几个很有希望的库（比如Theano和Caffe）已经停止了开发。每次一个用户决定去尝试一个新的技术，他们必须变换import语句，确保新的库实现了他们目前使用的NumPy API的所有部分。

Ideally, operating on specialized arrays using NumPy functions or semantics would simply work, so that users could write code once, and would then benefit from switching between NumPy arrays, GPU arrays, distributed arrays and so forth as appropriate. To support array operations between external array objects, NumPy therefore added the capability to act as a central coordination mechanism with a well specified API (Fig. 2).

理想情况下，使用NumPy函数或语义在专用阵列上运算，会好用的，这样用户可以写一遍代码，就可以在NumPy阵列，GPU阵列，分布式阵列等等中无缝切换。为在外部阵列目标之间支持阵列运算，NumPy因此加入了作为中央协调机制的能力，包含定义很好的API（图2）。

To facilitate this interoperability, NumPy provides ‘protocols’ (or contracts of operation), that allow for specialized arrays to be passed to NumPy functions (Fig. 3). NumPy, in turn, dispatches operations to the originating library, as required. Over four hundred of the most popular NumPy functions are supported. The protocols are implemented by widely used libraries such as Dask, CuPy, xarray and PyData/Sparse. Thanks to these developments, users can now, for example, scale their computation from a single machine to distributed systems using Dask. The protocols also compose well, allowing users to redeploy NumPy code at scale on distributed, multi-GPU systems via, for instance, CuPy arrays embedded in Dask arrays. Using NumPy’s high-level API, users can leverage highly parallel code execution on multiple systems with millions of cores, all with minimal code changes.

为促进这种互操作性，NumPy提供了一些协议，允许专用阵列传入NumPy函数（图3）。NumPy，将运算转发到原始的库。在NumPy最流行的几百个函数中，都支持这种协议。这些协议在广泛使用的库中进行了实现，如Dask，CuPy，xarray和PyData/Sparse。多亏了这些开发，用户现在可以将其计算从单个机器使用Dask放大到分布式系统中。这些协议的组成很好，可以使用户将NumPy代码在分布式，多GPU系统上重新部署，使用嵌入到Dask阵列中的CuPy阵列。使用NumPy的高层API，用户可以在多个系统中，有数百万个核心上，利用高度并行的代码执行，而代码变化可以做到最少。

These array protocols are now a key feature of NumPy, and are expected to only increase in importance. The NumPy developers — many of whom are authors of this Review — iteratively refine and add protocol designs to improve utility and simplify adoption.

这些阵列协议现在是NumPy的一个关键特征，其重要性预计只会变强。NumPy开发者，很多是这篇文章的作者，迭代的提炼并增加协议设计，以改进工具并简化采用的过程。

## 4. Discussion

NumPy combines the expressive power of array programming, the performance of C, and the readability, usability and versatility of Python in a mature, well tested, well documented and community-developed library. Libraries in the scientific Python ecosystem provide fast implementations of most important algorithms. Where extreme optimization is warranted, compiled languages can be used, such as Cython Numba and Pythran; these languages extend Python and transparently accelerate bottlenecks. Owing to NumPy’s simple memory model, it is easy to write low-level, hand-optimized code, usually in C or Fortran, to manipulate NumPy arrays and pass them back to Python. Furthermore, using array protocols, it is possible to utilize the full spectrum of specialized hardware acceleration with minimal changes to existing code.

NumPy综合了阵列编程的表达能力，C的性能和可读性，可用性和Python的全面性，成为一个成熟的，测试良好的，文档丰富的团体开发的库。科学Python生态系统的库，给出了多数重要算法的快速实现。其中有很好的优化，编译的语言可以使用，比如Cython，Numba和Pythran；这些语言拓展了Python，对瓶颈进行了明显的加速。由于NumPy简单的内存模型，很容易写出低层的手工优化的代码，通常是用C或Fortran，以操作NumPy阵列并将其传回到Python。而且，使用阵列协议，可能利用所有专用硬件加速，而现有代码的变化可以做到最少。

NumPy was initially developed by students, faculty and researchers to provide an advanced, open-source array programming library for Python, which was free to use and unencumbered by license servers and software protection dongles. There was a sense of building something consequential together for the benefit of many others. Participating in such an endeavour, within a welcoming community of like-minded individuals, held a powerful attraction for many early contributors.

NumPy开始时是由学生、教职员工和研究者开发的，为Python提供一个高级的开源的阵列编程库，免费使用，不受限制。有一种为很多其他人构建基础的感觉。参与这样一个努力，在一个热情的团里中，满是热心的人，为很多早期的贡献者提供了很强的吸引力。

These user–developers frequently had to write code from scratch to solve their own or their colleagues’ problems—often in low-level languages that preceded Python, such as Fortran and C. To them, the advantages of an interactive, high-level array library were evident. The design of this new tool was informed by other powerful interactive programming languages for scientific computing such as Basis, Yorick, R and APL, as well as commercial languages and environments such as IDL (Interactive Data Language) and MATLAB.

这些用户-开发者不断的要从头写代码，解决其本身或同事的问题-通常要用Python之前的低层语言，比如Fortran和C。对于他们来说，一个互动的高层阵列库的好处是明显的。这个新工具的设计是通过用于科学计算的其他强力的互动编程语言来了解的，如Basis，Yorick，R和APL，以及商业语言和环境，比如IDL和MATLAB。

What began as an attempt to add an array object to Python became the foundation of a vibrant ecosystem of tools. Now, a large amount of scientific work depends on NumPy being correct, fast and stable. It is no longer a small community project, but core scientific infrastructure.

为Python加入阵列目标的尝试，成为了一个生态环境的基础。现在，大量科学工作依赖于NumPy进行正确，快速，稳定的计算。这不再是一个小团体的项目，而是科学基础设施的核心。

The developer culture has matured: although initial development was highly informal, NumPy now has a roadmap and a process for proposing and discussing large changes. The project has formal governance structures and is fiscally sponsored by NumFOCUS, a nonprofit that promotes open practices in research, data and scientific computing. Over the past few years, the project attracted its first funded development, sponsored by the Moore and Sloan Foundations, and received an award as part of the Chan Zuckerberg Initiative’s Essentials of Open Source Software programme. With this funding, the project was (and is) able to have sustained focus over multiple months to implement substantial new features and improvements. That said, the development of NumPy still depends heavily on contributions made by graduate students and researchers in their free time (see Supplementary Methods for more details). 

开发者的文化已经很成熟了：虽然初始的开发是高度非正式的，NumPy现在已经有了一个路线图，和提出并讨论大的变化的过程。这个工程有正式的治理结构，由NumFOCUS财政上支持，这是一个非盈利机构，推动在研究，数据和科学计算中的开放行为。在过去这些年，这个项目吸引了其第一个投资的开发，由Moore and Sloan Foundations投资，并获得了奖项。有了资金支持，项目得以继续，可以用几个月时间聚焦，实现真正的新功能和改进。这就是说，NumPy的开发仍然非常依赖于毕业的学生和研究者在其自由时间的贡献。

NumPy is no longer merely the foundational array library underlying the scientific Python ecosystem, but it has become the standard API for tensor computation and a central coordinating mechanism between array types and technologies in Python. Work continues to expand on and improve these interoperability features.

NumPy并不只是科学Python生态环境的基础阵列库，而是已经成为张量计算的标准API，成为了Python的阵列类型和技术的中央协调机制。在改进这些互操作特征上，仍然有持续的工作。

Over the next decade, NumPy developers will face several challenges. New devices will be developed, and existing specialized hardware will evolve to meet diminishing returns on Moore’s law. There will be more, and a wider variety of, data science practitioners, a large proportion of whom will use NumPy. The scale of scientific data gathering will continue to increase, with the adoption of devices and instruments such as light-sheet microscopes and the Large Synoptic Survey Telescope (LSST). New generation languages, interpreters and compilers, such as Rust, Julia and LLVM, will create new concepts and data structures, and determine their viability.

在过去这十年，NumPy开发者会面临几种挑战。新的设备持续开发，现有的专用硬件会演化，达到Moore定律逐渐消失的程度。会有更多的，更广泛的数据科学实践者，其中很大一部分，都要使用NumPy。科学数据收集的规模会持续增加，采用的设备和仪器，比如light-sheet microscopes and the Large Synoptic Survey Telescope (LSST)。新的语言的生成，解释器和编译器，如Rust，Julia和LLVM，会增加新的概念和数据结构，确定其生存能力。

Through the mechanisms described in this Review, NumPy is poised to embrace such a changing landscape, and to continue playing a leading part in interactive scientific computation, although to do so will require sustained funding from government, academia and industry. But, importantly, for NumPy to meet the needs of the next decade of data science, it will also need a new generation of graduate students and community contributors to drive it forward.

通过本文描述的机制，NumPy已经准备好去拥抱这样一种变化的场景，持续在互动科学计算中扮演一个领导角色，虽然这样做会需要持续的资金支持，从政府、学术组织和工业圈。但重要的是，NumPy要满足下一个十年数据科学的需求，它也会需要新一代学生和团体贡献者来不断向前推进。