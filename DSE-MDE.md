# Design-Space Exploration in Model Driven Engineering – An Initial Pattern Catalogue

Ken Vanherpen et. al. @ University of Antwerp, Belgium

## 0. Abstract

A designer often has to evaluate alternative designs during the development of a system. A multitude of Design-Space Exploration (DSE) techniques exist in the literature. Integration of these techniques into the modelling paradigm is needed when a model-driven engineering approach is used for designing systems. To a greater or lesser extent, the integration of those different DSE techniques share characteristics with each other. Inspired by software design patterns, we introduce an initial pattern catalogue to categorise the embedding of different DSE techniques in an MDE context. We demonstrate their use by a literature survey and discuss the consequences of each pattern.

设计者通常在系统开发的过程中，需要评估不同的设计。文献中存在一些DSE技术。当设计系统时使用模型驱动的工程方法，需要将将这些技术集成到建模范式中。在或多或少的程度上，这些不同的DSE技术的集成，有着共同的特征。受软件设计模式启发，我们提出了初始的模式目录，将不同的DSE技术嵌入到MDE上下文进行分类。我们通过文献综述展示了其应用，讨论了每个模式的结果。

**Keywords**: Constraint/Rule-Based, Evolutionary Algorithm, DSE, MDE, MILP

## 1. Introduction

Model-Driven Engineering (MDE) uses abstraction to bridge the cognitive gap between the problem space and the solution space in complex software system problems. To bridge this gap, MDE uses models to describe complex systems at multiple levels of abstraction, using appropriate modelling formalisms. Model transformations play a key role in MDE to manipulate models. Transformations are used for code synthesis, mapping between models at the same or multiple levels of abstraction, etc. Model transformation is even regarded as the “heart and soul of model-driven software and system development” [1].

模型驱动的工程(MDE)使用抽象来弥合复杂软件系统问题中问题空间和解空间之间的认知空白。为填补这个空白，MDE使用模型在多个层次的抽象上描述复杂的系统，使用了合适的建模形式。模型变换在MDE中操作模型起了关键作用。变换用于代码综合，在同样或多个级别抽象层次上的模型之间的映射，等。模型变换甚至被当做是“模型驱动的软件和系统开发的灵魂”。

While designing a system, the need often arises to explore different design alternatives for a specific problem. Design Space Exploration (DSE) is an automatic process where possible alternatives of a particular design problem are explored. The exploration is guided with imposed constraints and optimality criteria on the different candidate solutions. In the literature a multitude of design-space exploration techniques are available, for example (Mixed Integer) Linear Programming, evolutionary algorithms and constraint satisfaction.

在设计一个系统时，通常会有对一个特定的问题探索不同的设计选择的需求。设计空间探索(DSE)是一个探索特定设计问题的可能替代品的自动过程。这个探索过程是用施加的约束和最优性准则在不同的候选解上进行引导探索的。在文献中，有一些设计空间探索的技术，比如，混合整数线性规划，进化算法，和约束满足。

In our experience with embedding DSE in a model-driven engineering context and a survey of the literature, we observed the use of different models, expressed using different formalisms, for both design, exploration and the modelling of goal functions. Combining the different models, using transformations, with the multitude of techniques available for searching design-spaces revealed similarities between the models and transformations of the different exploration techniques. To consolidate this knowledge, we organise these methods into an initial pattern catalogue, inspired by software design patterns. The goal of this effort is to create a more complete pattern catalogue for model-driven engineering approaches for design-space exploration with the support of the community.

在我们的经验中，将DSE嵌入到模型驱动的工程的上下文中，通过对文献进行调查，我们观察到，使用不同的模型，以不同的形式来表述，用于设计、探索和目标函数的建模。结合不同的模型，使用变换，使用不同的搜索设计空间的技术，说明不同的探索技术的模型和变换是有相似性的。为巩固这些知识，我们将这些方法组织成了一个初始的模式类别，这是受到软件设计模式的启发的。这些努力的目标是为设计空间探索的模型驱动的工程方法创建更完整的模式目录。

The remainder of this paper is structured as follows. Related work is elaborated in Section 2. Section 3 introduces the Initial Pattern Catalogue. In Section 4, we discuss other useful techniques for DSE in an MDE context. Finally, Section 5 concludes our contributions and elaborates on future work. Due to space limitations, we would like to refer to our technical report [2] where we elaborate our work in more detail and apply our initial pattern catalogue to some case studies.

本文余下组织如下。第2部分是相关的工作，第3部分介绍了初始模式目录，第4部分我们讨论了DSE在MDE上下文中的其他有用的技术，最后在第5部分给出了总结和未来工作。由于空间限制，我们推荐读者参考我们的技术报告[2]，其中更加详细的描述了我们的工作，将我们的初始模式目录应用到了一些案例研究中。

## 2. Related Work

The concept of patterns is widely used in Software Engineering. They provide generalized solutions to common software problems in the form of templates. The templates can be used by software developers to tackle the complexity in a larger software problem. One of the most highly cited contributions to pattern catalogues in the field of software is the work of the “Gang of Four” [3], which presents various design patterns with respect to object-oriented programming. Inspired by the Gang of Four, Amrani et al. [4] presents a model transformation intent catalogue which identifies and describes the intents and properties that they may or must possess. Their catalogue can be used for several purposes such as requirements analysis for transformations, identification of transformation properties and model transformation language design. Their presented catalogue is a first attempt to introduce the concept of patterns in MDE.

软件工程中广泛使用了模式的概念，以模板的形式为通用的软件问题提供了一般化的解。软件开发者使用模板，来解决更大的软件问题中的复杂度。在软件领域，模式目录的一个引用最多的贡献，是Gang of Four的工作，给出了OOP中的各种设计模式。受到Gang of Four的启发，Amrani等[4]给出了一个模型变换目的的目录，识别和描述了它们可能或必须占据的目的和性质。他们的目录可以用于几个目的，比如变换的需求分析，识别变换的性质，和建模变换语言设计。他们给出的目录是MDE中第一次提出模式的概念。

A more in-depth literature study is integrated in Section 3 such that each pattern is illustrated by known uses. This motivates one to the application of the introduced patterns.

更深入的文献研究在第3部分中，每个模式用已知的使用来进行描述。这促使提出的模式进行应用。

## 3. Initial Pattern Catalogue for DSE

In this section we first discuss the need for a pattern catalogue specific to the Design Space Exploration domain. Next, our proposed pattern structure is described by analogy with the seminal work of the “Gang of Four” [3]. Finally, Subsections 3.2 and 3.3 will elaborate our initial pattern catalogue.

本部分中，我们首先讨论了设计空间探索领域对模式目录的需求。然后，我们提出的模式结构与Gang of Four的工作进行类比描述。最后，3.2节和3.3节描述了我们的初始模式目录。

### 3.1 The need for patterns

By definition design patterns are used to formalise a problem which recur repeatedly. They help a designer to evaluate alternatives for a given design problem in order to choose the most appropriate design. The usefulness of such patterns has already been proven in the Software Engineering domain where the “Gang of Four” [3] gave impetus to the creation of a widely accepted software design patterns catalogue. The successful impact of its widespread use is undoubtedly the well defined structure of each pattern. More specifically, each pattern is typed by: (1) Pattern Name and Classification, (2) Intent, (3) Also Known as, (4) Motivation, (5) Applicability, (6) Structure, (7) Participants, (8) Collaborations, (9) Consequences, (10) Implementation, (11) Sample Code, (12) Known Uses and (13) Related Patterns. Each of these sections is textually described and where necessary graphically supported using Class Diagrams, describing structure, and/or Activity Diagrams, describing the workflow of the pattern. At least one case study demonstrates how the patterns can be applied in practice.

从定义上来说，设计模式是用于对重复出现的问题进行形式化的。它们帮助设计者对给定的设计问题来评估替代品，以选择最合适的设计。这种模式的有用性已经在软件工程中被证明了，其中Gang of Four创建了广泛接受的软件设计模式目录。其广泛使用的成功影响，毫无疑问是每种模式的定义明确的结构。更具体的，每种模式的类型定义是：(1)模式名称和分类，(2)目的，(3)也被称为，(4)动机，(5)可应用性，(6)结构，(7)参与者，(8)合作，(9)结果，(10)实现，(11)代码样本，(12)已知的使用，(13)相关的模式。这些部分的每个都进行了文字描述，也有必须的图形支持，比如使用Class Diagram来描述结构，以及Activity Diagram描述模式的workflow。至少有一个案例研究来展示模式怎样在实践中应用。

In accordance to software design patterns, we define the format of each proposed pattern as follows. **Intent**: Gives a short explanation of the intention of the pattern. **Structure**: Describes the general structure of the pattern. **Consequences**: Describes the trade-offs in using the pattern. **Known Uses**: Lists the applications of the pattern in the literature. While this is not meant to be an exhaustive literature review of all the applications of the pattern, one can draw inspiration from these examples to apply the pattern. **Application**: Gives a short description in which cases this pattern can be useful and how it can be implemented.

与软件设计模式对应，我们定义每个提出的模式的格式如下。**目的**：给出模式的目的的简短解释。**结构**：描述模式的一般结构。**结果**：描述使用这个模式的trade-offs。**已知的用处**：列出文献中模式的应用。这并不是模式所有应用的详尽文献回顾，只是可以从这些例子中得到应用模式的启发。**应用**：给出简短描述，在哪些情况中这个模式是有用的，怎样进行实现。

The *Structure* is graphically supported by the Formalism Transformation Graph and Process Model (FTG+PM). The left side of the FTG+PM clearly shows all involved formalisms (boxes) and their relations using model transformations (circles). The right side shows the process with the involved models (boxes), transformed by a model transformation (roundtangle). Complex dataflow (dashed line) and control-flow (full line) relations can exist in the process part of the FTG+PM. This can be summarized as a legend, which is shown in Figure 1. The reason behind this latter supported formalism is threefold: (1) It clearly represents the structure of the approach by connecting the different formalisms with transformations on the left side of the FTG+PM. The FTG+PM also shows the workflow of combining the different models and transformations in a process on the right side. (2) The FTG+PM can be used to (semi-)automatically execute the defined transformation chains (yellow coloured). Manual operations are also possible that allow for experience based optimisation and design (grey coloured). (3) Different patterns described in this formalisms are easily connected to each other. This enables the embedding of DSE within the MDE design of systems.

*结构*由Formalism Transformation Graph和Process Model进行图形化支持(FTG+PM)。FTG+PM的左边清楚的展示了所有涉及到的formalisms（框），以及用模型变换表示的之间的关系（圆）。右边展示了涉及的模型的process（框），由模型变换（圆矩形）来进行变换。复杂的数据流（虚线）和控制流（实线）关系可以在FTG+PM的process部分存在。这可以由图1中的图例来总结。后者支持的formalism的原因有3个方面：(1)方法的结构可以很清楚的得到表示，只要将不同的formalism用用变换连接起来，在FTG+PM的左边。FTG+PM还展示了将不同的模型和变换在右边的process中结合到一起的workflow。(2)FTG+PM可以用于(半)自动的执行定义的变换链（黄色）。手工操作也是可能的，基于经验进行优化和设计（灰色）。(3)在这个formalism中描述的不同的模式，很容易的连接到彼此。这使得DSE可以嵌入到MDE设计系统中。

As mentioned in section 1, we would like to refer to our technical report [2] where we apply our initial pattern catalogue to some case studies.

第1部分中提到过，读者可以参考我们的技术报告[2]，将我们的初始模式目录应用到了一些案例研究中。

### 3.2 Exploring Design Spaces

Performing design-space exploration in a model-driven engineering context can be abstracted in some steps: (1) A meta-model defines the structural constraints of a valid solution. (2) A DSE-tool generates valid candidate solutions conforming to the meta-model. An initial model adds other structural constraints to the set of candidate solutions. (3) A transformation transforms the set of candidate solutions to an analysis formalism to check the feasibility of the solution with respect to a set of constraints. (4) If necessary, a second transformation generates a model in a performance formalism to check the optimality of the solution with respect to certain optimisation goals. (5) Depending on the optimisation technique, the process is iterated multiple times. Information from feasibility and performance models is used to guide the exploration.

在模型驱动工程的上下文中进行设计空间探索，可以抽象成几个步骤：(1)一个元模型，定义了一个有效解的结构约束；(2)一个DSE工具，生成有效的候选解，符合元模型。一个初始模型，为候选解的集合加入了其他结构约束。(3)一个变换，将候选解的集合，变换到一个分析formalism，检查解对约束集合的的可行性；(4)如果需要的话，另一个变换生成一个性能formalism的模型，来检查解对特定优化目标的最优性；(5)依赖于优化技术，整个过程迭代多次。可行性和性能模型的信息，用于引导探索过程。

Depending on the exploration technique, we classify different model-driven engineering approaches to solve this generic design-space exploration strategy.

视探索技术不同，我们将不同的模型驱动的工程方法分类，以求解这个通用的设计空间探索策略。

*Model Generation Pattern*

**Intent**: This pattern transforms the meta-model of a problem space together with constraints to a constraint-satisfaction problem. The exploration of the design consists of the generation of a set of models that satisfy the structural constraints imposed by the meta-model and the other constraints provided using a constraint formalism.

目的：这种模式将一个问题空间的元模型与约束一起，变换成一个约束满足问题。设计的探索，就是生成模型的集合，这些模型满足元模型加入的结构约束，和使用约束formalism提供的其他约束。

**Structure**: The pattern, shown in Figure 2, starts with a meta-model and some constraints. A transformation transforms these models into a constraint satisfaction problem. By invoking a solver, an exploration of the design space generates candidate solutions. Each candidate solution is transformed into an analysis representation. The analysis produces traces of each candidate solution. Based on the goal function model, the optimal trace is transformed to a solution model. This solution model can either be expressed in the exploration formalism, the original model formalism or a specific solution formalism.

结构：图2中所示的模式，从元模型和一些约束开始。一个变换将这些模型变换成一个约束满足问题。通过调用一个solver，设计空间的探索生成了候选解。每个候选解变换到一个分析表示。分析生成每个候选解的迹。基于目标函数模型，最优迹变换到一个解模型。这个解模型可以在探索formalism中进行表述，也可以在原始的模型formalism，或具体的解formalism中表述。

**Consequences**: Depending on the used solver, this method might be computationally and memory intensive because an exhaustive search of the design space is executed. A transformation is needed to translate the meta-model with constraints to a model that is usable by the DSE-tool. Domain knowledge can be introduced by adding constraints to the meta-model. Note that adding extra constraints helps the search for a solution. An initial model, where some choices are predetermined, adds extra constraints. A less generic alternative is to add the initial model when evaluating candidate solutions.

结果：依赖于使用的solver，这种方法在计算量和存储使用上会耗费很大，因为对设计空间进行了穷举式搜索。需要一个变换，将带有约束的元模型翻译成DSE工具可用的模型。通过对元模型加入约束，可以引入领域知识。注意，加入额外的约束可以帮助搜索解。初始模型中的一些选项是预先确定的，加入了额外的约束。一个没那么通用的替代品，是在评估候选解时，加入初始模型。

**Known Uses**: Neema et al. [5] present the DESERT framework used for Model-Driven constraint-based DSE. It implements an automated tool which abstracts the Simulink design space to generate candidate solutions. In [6] the FORMULA tool is presented, where candidate solutions are generated from a meta-model. A similar tool called Alloy is used by Sen et al. [7] to automatically generate test models. Saxena and Karsai [8] present an MDE framework for generalized design-space exploration. A DSE problem is constructed out of a generalized constraint meta-model combined with a domain specific meta-model.

已知的使用：Neema等[5]提出DESERT框架，用于模型驱动的基于约束的DSE，实现了一个自动化工具，抽象了Simulink设计空间，以生成候选解。[6]提出了FORMULA工具，可以从一个元模型中生成候选解。Sen等[7]使用了一个类似的工具，称为Alloy，自动生成测试模型。Saxena和Karsai[8]提出一个MDE框架，进行通用的设计空间探索。由一个通用约束元模型，和领域专用元模型结合到一起，就构建出一个DSE问题。

**Application**: The pattern is not recommended when one searches for an optimal solution out of a large search space without a lot of constraints. On the other hand, this pattern is very useful to rapidly obtain candidate solutions conforming to the meta-model.

应用：当从一个大型搜索空间中搜索最优解，并没有很多约束时，并不推荐用这种模式。另一方面，在快速得到符合元模型的候选解方面，这个模式还是很有用的。

*Model Adaptation Pattern*

**Intent**: This pattern transforms the model or a population of models to a generic search model used in (meta-) heuristic searches. Depending on the problem and search algorithm, different search representations can be used.

目的：这个模式将一个模型或一族模型变换到一个通用搜索模型，用在元启发式搜索中。依赖于问题和搜索算法，可以使用不同的搜索表示。

**Structure**: A model or population of models expressed in a certain formalism is transformed to a specific exploration formalism. Based on the guidance of a goal function, an algorithm creates new candidate solutions. A (set of) candidate solutions are transformed to an analysis model in order to evaluate. Finally, the result is transformed to a solution model. This solution model can either be expressed in the exploration formalism, the original model formalism or a specific solution formalism.

结构：一个模型，或一族模型，以特定formalism进行表示，变换到特定的探索formalism。基于目标函数的引导，算法会创建新的候选解。候选解的集合变换到一个分析模型，以进行评估。最后，结果变换到一个解模型。解模型可以表达为探索formalism，原始模型formalism，或特定的解formalism。

**Consequences**: A dedicated search representation has to be created as well as manipulation functions to create alternative designs. This requires an adequate understanding of the problem and domain knowledge. A translation from the problem domain to the search representation and vice-versa is required. An initial model, as a constraint, can be added by fixing the generated solution or by rewriting the functions to create new solutions (cross-over, mutation, etc.).

结果：要创建一个专用的搜索表示，以及操作函数，以创建其他的设计。这需要对问题和领域知识有充分的理解。需要一个从问题域到搜索表示的翻译，以及相反的翻译。一个初始模型，作为一个约束，可以通过固定生成的解，或重写函数来创建新的解，来加入。

**Known Uses**: Williams et al. searched for game character behaviour using a mapping to a genetic algorithm [9]. Burton et al. solve acquisition problems using MDE [10]. Genetic algorithms are used to create a Pareto front of solutions. A stochastic model transformation creates an initial population. Finally, Kessentini and Wimmer propose a generic approach to searching models using Genetic Algorithms [11]. The proposed method is very similar to the described pattern.

已知的使用：Williams等使用到遗传算法的映射来搜索游戏角色行为。Borton等使用MDE来求解acquisition问题。使用遗传算法来生成解的Pareto front。统计模型变换创建了一个初始族群。最后，Kessentini和Wimmer提出了一个通用方法来使用遗传算法来搜索模型。提出的方法与描述的模式非常类似。

**Application**: This pattern is recommended when a design problem can easily be transformed to an optimal search representation, e.g. a list or tree representation. Different operations on this new representation are implemented in the solution space (usually a generic programming language). Well-known algorithms, like genetic algorithms and hill-climbing, implement the search.

应用：这个模式在一个设计问题可以很容易的变换到一个最优搜索表示时非常适用，如，列表或树的表示。在这个新表示上的不同运算，在解空间上实现（通常是一个通用编程语言）。用很有名的算法来实现这个搜索，如遗传算法和爬山算法。

*Model Transformation Pattern*

**Intent**: This pattern uses the original model to explore a design-space. Model transformations encode the knowledge to create alternative models. Guidance to the search can be given by selecting the most appropriate next transformation or by adding meta-heuristics to the model transformation scheduling language.

目的：这个模式使用原始模型来探索一个设计空间。模型变换编码了知识，以创建不同的模型。对搜索的引导，可以通过选择最合适的下一个变换，或加入元启发式到模型变换调度语言中进行。

**Structure**: A model combined with a goal function is used to create a set of candidate solutions that are expressed in the original model formalism. These are transformed to an analysis representation to gather some metrics that are expressed by a trace. Using (meta-)heuristics, a new set of candidate solutions can be generated according to a goal function. Finally, if required, the most optimal solution or set of solutions can be transformed into a solution model.

结构：模型与目标函数一起，用于创建候选解的集合，在原始模型formalism中进行表达。这变换到一个分析表示中，收集了一些由迹表达的度量。使用元启发式，可以根据目标函数生成候选解的新集合。最后，如果需要的话，最优解或最优解集可以变换到解模型中。

**Consequences**: A high degree of domain knowledge about the problem is required to design the transformation rules. On the other hand, the rules encode domain knowledge to guide the exploration. Model-to-model or model-to-text transformations are required to evaluate a candidate solution. An initial model as a constraint can be added by adjusting the meta-model with variation tags. Similarly to the Model Adaptation Pattern, the initial conditions can also be implemented as fix operations using model transformations. Model transformations to create new candidate solutions are computationally expensive because of the subgraph isomorphism problem.

结果：需要大量关于问题的领域知识，来设计变换规则。另一方面，规则编码了领域知识，来引导探索。需要模型到模型，或模型到文本的变换，来评估一个候选解。初始模型作为约束，可以通过用变化tags调整元模型来加入。与Model Adaptation Pattern类似，初始条件也可以使用模型变换实现为固定的操作。模型变换来创建新的候选解，其计算量很大，因为有子图的isomorphism问题。

**Known Uses**: In [12] a model-driven framework is presented for guided design space exploration using graph transformations. The exploration is characterised by a so-called exploration strategy which uses hints to identify dead-end states and to order exploration rules. This way the number of invalid alternatives is reduced. Denil et al. [13] demonstrates how search-based optimization (SBO) techniques can be included in rule-based model transformations.

已知的使用：在[12]中提出了一个模型驱动的框架，使用图变换来进行引导的设计空间探索。探索的特点是一个所谓的探索策略，使用线索来识别dead-end状态，以排序探索规则。这样，无效替代的数量就减少了。Denil等[13]证明了，基于搜索的优化(SBO)可以怎样包括在基于规则的模型变换中。

**Application**: The pattern is used when it is hard to obtain a generic search representation. Model transformation rules, expressed in the natural language of the engineer, are implemented using current model transformation tools. Guidance is implemented through the scheduling of the model transformation rules.

应用：当很难得到通用搜索表示时，就可以使用这个模式。模型变换规则，以工程师的自然语言表述，使用目前的模型变换工具来实现。引导是通过调度模型变换规则来实现的。

### 3.3 Exploration Chaining Pattern

In order to prune the design space more efficiently, multiple of the proposed patterns can be chained. This technique is called “Divide and Conquer” and may as well be described by a pattern. To represent the chaining of multiple FTG+PMs, this pattern is graphically supported by means of a principle representation.

为更高效的修剪设计空间，可以将多个提出的模式进行链接。这个技术称为Divide and Conquer，也可以描述为一个模式。为表示多个FTG+PMs的链接，这个模式由主表示的方法来进行图形支持。

**Intent**: This pattern adds multiple abstraction layers in the exploration problem where candidate solutions can be pruned. High-level estimators are used to evaluate the candidate solutions and prune out non-feasible solutions and solutions that can never become optimal with respect to the evaluated properties. Figure 5 shows the overall approach of this pattern.

目的：这个模式在探索问题中加入多个抽象层，在这些地方候选解可以被修剪掉。高层次估计器用于评估候选解，然后修剪掉不可行的解，和相对于被评估的性质不可能是最优的解。图5给出了这种模式的总体方法。

**Structure**: At each of the abstraction layers an exploration pattern is used to create and evaluate candidate solutions. Non-pruned solutions are explored further in the next exploration step.

结构：在每个抽象层，都用一个探索模式来创建和评估候选解。没有被修剪掉的解在下一个探索步骤中进一步被探索。

**Consequences**: Domain knowledge about the problem is required to add levels of abstraction. High-level estimators are needed at each of the abstraction layers to evaluate a candidate solution. Because more information is introduced at each of the abstraction layers, the evaluation of a single candidate solution becomes more complex and usually more computationally intensive. Finally, a pruning strategy is required to decide what solutions have to be pruned at each of the abstraction layers.

结果：需要关于问题的领域知识以增加抽象层次。在每个抽象层，都需要高层次估计器来评估候选解。因为在每个抽象层都引入了更多信息，单个候选解的评估变得更复杂，通常计算量会更大。最后，需要一个修剪策略来决定什么样的解在每个抽象层需要被修剪掉。

**Known Uses**: Sen and Vangheluwe add different levels of abstraction in the design of a multi-domain physical model [14]. This numerically constraints the modeller to create only valid models. Kerzhener and Paredis introduce multiple levels of fidelity in [15]. Finally, multiple levels of abstractions for an automotive allocation and scheduling problem are introduced in [16].

已知的使用。[14]在设计多领域物理模型时，加入了不同级别的抽象，这在数值上约束了建模者只生成有效的模型。[15]中引入了多级保真度。最后，[16]中为汽车分配和调度问题引入了多级抽象。

**Application**: This pattern provides a solution when memory and time complexity are an issue during the exploration of the design space. It tackles the complexity by its layered pruning approach. Therefore, this pattern is preferred when searching for (an) optimal solution(s) in a large search space. Different exploration patterns are chained to create solutions.

应用。在探索设计空间时，当内存和时间复杂度是问题时，这种模式也可以给出一个解。对复杂度的应对，是靠分层的剪枝方法。因此，当在一个大规模搜索空间中搜索最优解时，会倾向于采用这种模式。不同的探索模式链接到一起，来给出解。

## 4 Discussion

In this section we describe some other techniques that are useful for design-space exploration in a model-driven engineering context. Some techniques could potentially become a pattern in a new version of the catalogue.

在这一部分，我们描述了模型驱动的工程学上下文中，设计空间探索的其他一些技术。一些技术可能在新版本的目录中可能成为一个模式。

*Dealing with Multiple Objectives*: Multi-objective optimisation deals with the decision making process in the presence of trade-offs between multiple goal functions. Certain DSE and search algorithms can deal with multi-objective functions by construction. However, some techniques do not have this features. Here we give two ways of dealing with the problem.

处理多目标：多目标优化在存在多个目标函数的折中时，处理决策过程。特定的DSE和搜索算法可以处理多目标函数。但是，一些技术就没有这个特征。这里我们给出两种处理这个问题的方法。

**Scalarize the Objective-Function**: When scalarizing a multi-objective optimisation problem, the problem is reformulated as a single-objective function. The goal function model becomes a combination of individual objective functions. A model defines how the combination of the different individual goal function models is done, for example in a linear fashion, or other more complex functions.

目标函数标量化：当多目标优化问题标量化时，这个问题重新表述为单目标函数。目标函数模型成为单个目标函数的组合。一个模型会定义不同目标函数模型怎样组合，比如线性模式，或其他更复杂的函数。

**Create Variants**: In certain cases the designer would like to compare the different trade-offs using a Pareto curve. We use the scalarizing pattern to create multiple variants of the combined objective function. Intermediate results of the exploration are used to select an appropriate recombination that could potentially add a new Pareto solution.

创建变体：在特定的情况下，设计者可能希望使用Pareto曲线比较不同的trade-offs。我们使用标量化模式来创建组合目标函数的多个变体。探索的中间结果用于选择一种合适的重组合，可能在增加一个新的Pareto解。

*Meta-model reduction*: By using sensitivity analysis of the involved modelling elements and parameters, the meta-model can be reduced with the elements and parameters that have a small influence on the result of the goal function. An example of this technique can be found in [17].

元模型缩减：使用涉及到的建模元素和参数的敏感性分析，那些对目标函数结果的影响较小的元素和参数，可以用于对元模型降维。这种技术的一个例子可以见[17]。

## 5. Conclusions and Future Work

Resulting from our own experiences with DSE and a literature survey, we presented an initial pattern catalogue which categorizes different approaches of Model-Driven Design Space Exploration. We described the patterns by the use of the FTG+PM to visualise the involved formalisms and their relations using model transformations.

从我们对DSE的经验和文献的调查中，我们提出了一个初始模式目录，将不同的模型驱动的设计空间探索进行了分类。我们描述模式使用的是FTG+PM，使用模型变换对涉及到的formalism及其相互关系进行了可视化。

With the support of the community, it is our ambition to extend this towards a more complete this initial pattern catalogue, similar to the widely available software design patterns used in software engineering. Finally, we would like to investigate the parts of patterns that can be fully or partially automated.

我们的目标是拓展这个成为更完整的初始模式目录，与在软件工程中广泛使用的设计模式类似。最后，我们还希望研究那些可能完全或部分自动化的模式。