# Toward Federated Learning At Scale: System Design

Keith Bonawitz et al. Google Inc.

## Abstract

Federated Learning is a distributed machine learning approach which enables model training on a large corpus of decentralized data. We have built a scalable production system for Federated Learning in the domain of mobile devices, based on TensorFlow. In this paper, we describe the resulting high-level design, sketch some of the challenges and their solutions, and touch upon the open problems and future directions.

联邦学习是一种分布式机器学习方法，可以在大量去中心化的数据上进行模型训练。我们在移动设备的领域中，基于TensorFlow构建了一种可扩展的联邦学习生产系统。在本文中，我们描述了得到的高层设计，概述了一些挑战及其解决方案，并讨论了一些开放性的问题和未来的方向。

## 1 Introduction

Federated Learning (FL) (McMahan et al., 2017) is a distributed machine learning approach which enables training on a large corpus of decentralized data residing on devices like mobile phones. FL is one instance of the more general approach of “bringing the code to the data, instead of the data to the code” and addresses the fundamental problems of privacy, ownership, and locality of data. The general description of FL has been given by McMahan & Ramage (2017), and its theory has been explored in Konecny et al. (2016a); McMahan et al. (2017; 2018).

联邦学习是一种分布式机器学习方法，可以在大量去中心化的数据上进行训练，这些数据可以在移动手机这样的设备上。FL是更一般性方法的例子，即“将代码送到数据那里去，而不是将数据送到代码里”，处理的是隐私、归属和数据本地性这样的根本问题。FL的一般性描述由McMahan等给出，其理论也有文章进行了探索。

A basic design decision for a Federated Learning infrastructure is whether to focus on asynchronous or synchronous training algorithms. While much successful work on deep learning has used asynchronous training, e.g., Dean et al. (2012), recently there has been a consistent trend towards synchronous large batch training, even in the data center (Goyal et al., 2017; Smith et al., 2018). The Federated Averaging algorithm of McMahan et al. (2017) takes a similar approach. Further, several approaches to enhancing privacy guarantees for FL, including differential privacy (McMahan et al., 2018) and Secure Aggregation (Bonawitz et al., 2017), essentially require some notion of synchronization on a fixed set of devices, so that the server side of the learning algorithm only consumes a simple aggregate of the updates from many users. For all these reasons, we chose to focus on support for synchronous rounds, while mitigating potential synchronization overhead via several techniques we describe subsequently. Our system is thus amenable to running large-batch SGD-style algorithms as well as Federated Averaging, the primary algorithm we run in production; pseudo-code is given in Appendix B for completeness.

FL基础架构的一个基本设计决定是，是否聚焦在异步或同步训练算法上。很多成功的深度学习工作都使用了异步训练，如Dean等，但最近有很多同步的大批量训练的趋势，甚至是在数据中心中。McMahan等的联邦平均算法采用了类似的方法。而且，几种增强隐私性的方法可以为FL进行保证，包括差异化隐私，安全聚积，基本都是需要在某一固定设备集上进行同步措施，这样服务器端的学习算法只需要简单的聚积很多用户的更新。由于这些所有原因，我们选择关注支持同步回合，同时通过下面叙述的几种技术，弥补同步的潜在代价。我们的系统因此是可以运行大批次SGD类型的算法的，也可以运行联邦平均算法，这是在生产中运行的基本算法；附录B中给出了算法的伪代码。

In this paper, we report on a system design for such algorithms in the domain of mobile phones (Android). This work is still in an early stage, and we do not have all problems solved, nor are we able to give a comprehensive discussion of all required components. Rather, we attempt to sketch the major components of the system, describe the challenges, and identify the open issues, in the hope that this will be useful to spark further systems research.

本文中，我们给出了在移动手机(Android)领域运行的算法的系统设计。本文仍然处在早期阶段，我们并没有解决所有问题，我们也不能给出所有需要的组件的综合讨论。我们只是试图对系统的主要组件进行概述，描述了遇到的挑战，写出开放的问题，希望这对于激发未来的系统研究会有帮助。

Our system enables one to train a deep neural network, using TensorFlow (Abadi et al., 2016), on data stored on the phone which will never leave the device. The weights are combined in the cloud with Federated Averaging, constructing a global model which is pushed back to phones for inference. An implementation of Secure Aggregation (Bonawitz et al., 2017) ensures that on a global level individual updates from phones are uninspectable. The system has been applied in large scale applications, for instance in the realm of a phone keyboard.

我们的系统可以训练一个深度神经网络，使用的是TensorFlow，使用的数据存储在手机上，数据不会离开设备。权重在云端结合到一起，使用的是Federated Averaging，构建的全局模型，然后推送到手机上进行推理。安全聚积的一个实现，确保了从手机上发送的个体的更新，在全局层次上是不可检视的。系统已经在大规模应用中得到了应用，比如在手机键盘的领域。

Our work addresses numerous practical issues: device availability that correlates with the local data distribution in complex ways (e.g., time zone dependency); unreliable device connectivity and interrupted execution; orchestration of lock-step execution across devices with varying availability; and limited device storage and compute resources. These issues are addressed at the communication protocol, device, and server levels. We have reached a state of maturity sufficient to deploy the system in production and solve applied learning problems over tens of millions of real-world devices; we anticipate uses where the number of devices reaches billions.

我们的工作处理的是很多实际的问题：设备可用性，这与本地数据分布有复杂的关系（如，时区的依赖性）；不可靠的设备连接，和被打断的执行；不同设备在不同的可用性下，锁步执行的协调；有限的设备存储和计算资源。这些问题都是在通信协议、设备和服务器层次进行解决的。我们已经达到了一定的成熟度，足够将系统部署在生产阶段，在百万、千万台实际的设备上解决应用的学习问题；我们期待使用的设备数量达到十亿的级别。

## 2 Protocol

To understand the system architecture, it is best to start from the network protocol. 为理解系统架构，最好先从网络协议开始。

### 2.1 Basic Notions

The participants in the protocol are devices (currently Android phones) and the FL server, which is a cloud-based distributed service. Devices announce to the server that they are ready to run an FL task for a given FL population. An FL population is specified by a globally unique name which identifies the learning problem, or application, which is worked upon. An FL task is a specific computation for an FL population, such as training to be performed with given hyperparameters, or evaluation of trained models on local device data.

协议的参与者是设备（目前都是Android手机）和FL服务器，这是一个基于云的分布式服务。设备向服务器宣称，设备已经准备好了运行一个FL任务，这是在给定的FL群体里的。一个FL群体是由一个全局唯一的名称指定的，明确了学习的问题，或应用。一个FL任务是一个FL群体的一种特定的计算，比如训练后用给定的超参数进行运行，或在本地设备数据上评估训练好的模型。

From the potential tens of thousands of devices announcing availability to the server during a certain time window, the server selects a subset of typically a few hundred which are invited to work on a specific FL task (we discuss the reason for this subsetting in Sec. 2.2). We call this rendezvous between devices and server a round. Devices stay connected to the server for the duration of the round.

在特定时间窗口内，可能有上万个设备向服务器宣称可用性，服务器一般会选择几百台，邀请其在特定FL任务中进行工作（我们会在2.2节中讨论这种选择子集的原因）。我们称这种设备与服务器的集结为一轮。设备在这一轮的时间内，保持连接到服务器的状态。

The server tells the selected devices what computation to run with an FL plan, a data structure that includes a TensorFlow graph and instructions for how to execute it. Once a round is established, the server next sends to each participant the current global model parameters and any other necessary state as an FL checkpoint (essentially the serialized state of a TensorFlow session). Each participant then performs a local computation based on the global state and its local dataset, and sends an update in the form of an FL checkpoint back to the server. The server incorporates these updates into its global state, and the process repeats.

服务器告诉选择的设备，给定的FL计划和数据结构（包含一个TesnorFlow图和怎样执行的指南），要进行什么样的计算。一旦确定了一轮，服务器下一步就给每个参与者发送目前的全局模型参数和任何其他必要的状态，作为一个FL检查点（其实就是一个TensorFlow session的序列化状态）。每个参与者然后基于全局状态和其本地数据集，进行本地计算，然后将其更新以FL检查点的形式发回服务器。服务器将这些更新纳入到其全局状态中，然后这个过程重复进行。

### 2.2 Phases

The communication protocol enables devices to advance the global, singleton model of an FL population between rounds where each round consists of the three phases shown in Fig. 1. For simplicity, the description below does not include Secure Aggregation, which is described in Sec. 6. Note that even in the absence of Secure Aggregation, all network traffic is encrypted on the wire.

通信协议使得设备可以在一轮一轮时间中推进更新一个FL群的全局的单个模型，其中每一轮包含三个阶段，如图1所示。简化起见，下面的描述没有包含安全聚积，这在第6部分会进行阐述。注意，即使在没有安全聚积的情况下，所有网络流量也是进行了加密的。

Fig. 1 Federated Learning Protocol

1. Devices check-in with the FL server, rejected ones are told to come back later
2. Server reads model checkpoint from persistent storage
3. Model and configuration are sent to selected devices
4. On-device training is performed, model update is reported back
5. Server aggregates updates into the global model as they arrive
6. Server writes global model checkpoint into persistent storage

**Selection**. Periodically, devices that meet the eligibility criteria (e.g., charging and connected to an unmetered network; see Sec. 3) check in to the server by opening a bidirectional stream. The stream is used to track liveness and orchestrate multi-step communication. The server selects a subset of connected devices based on certain goals like the optimal number of participating devices (typically a few hundred devices participate in each round). If a device is not selected for participation, the server responds with instructions to reconnect at a later point in time(In the current implementation, selection is done by simple reservoir sampling, but the protocol is amenable to more sophisticated methods which address selection bias).

**选择**。符合选择条件的设备（如，正在充电，并连接到了unmetered网络，见第3部分），会周期性的到服务器处报道，开启双向数据流。数据流用于跟踪设备生存性，并协调多步骤的通信。服务器根据一定的目标，如最佳参与设备量（典型的如每轮几百个参与设备），选择一部分连接的设备。如果设备没有被选择参与，服务器会进行回应，告诉设备后面某个时间再进行连接（在目前的实现中，选择是通过简单的蓄水池采样进行的，但协议可以包含更复杂的方法，处理选择的偏差）。

**Configuration**. The server is configured based on the aggregation mechanism selected (e.g., simple or Secure Aggregation) for the selected devices. The server sends the FL plan and an FL checkpoint with the global model to each of the devices.

**配置**。服务器是基于对所选择设备的聚积机制的选择进行配置的（如，简单聚积，或安全聚积）。服务器向每个设备发送FL计划和一个FL检查点，包含全局模型。

**Reporting**. The server waits for the participating devices to report updates. As updates are received, the server aggregates them using Federated Averaging and instructs the reporting devices when to reconnect (see also Sec. 2.3). If enough devices report in time, the round will be successfully completed and the server will update its global model, otherwise the round is abandoned.

**报告**。服务器等待参与的设备将更新汇报过来。当接收到更新时，服务器使用联邦平均算法将其聚积起来，并告诉汇报的设备，什么时候重新连接（见2.3节）。如果足够多设备都及时汇报了，那么这一轮就成功结束了，服务器更新其全局模型，否则这一轮的结果就抛弃掉。

As seen in Fig. 1, straggling devices which do not report back in time or do not react on configuration by the server will simply be ignored. The protocol has a certain tolerance for such drop-outs which is configurable per FL task.

如图1所示，掉队的设备，也就是没有及时汇报回来结果的，或对服务器的配置没有回应的，就会被忽略掉。协议对这种drop-out有一定的容忍度，这是在每个FL任务中进行配置的。

The selection and reporting phases are specified by a set of parameters which spawn flexible time windows. For example, for the selection phase the server considers a device participant goal count, a timeout, and a minimal percentage of the goal count which is required to run the round. The selection phase lasts until the goal count is reached or a timeout occurs; in the latter case, the round will be started or abandoned depending on whether the minimal goal count has been reached.

选择和汇报阶段，是由于一组参数指定的，这会产生灵活的时间窗口。如，对于选择阶段，服务器考虑的是一个设备参与者的目标计数，超时，和要运行这一轮计算，所需的最小比例的目标计数；在后者的情况下，看是否达到了最小的目标计数，来决定这一轮是否开始或抛弃。

### 2.3 Pace Steering 节奏控制

Pace steering is a flow control mechanism regulating the pattern of device connections. It enables the FL server both to scale down to handle small FL populations as well to scale up to very large FL populations.

节奏控制是一种流控制机制，规范了设备连接的模式。它使得FL服务器可以缩小规模，以处理小型FL群的任务，或增大规模，以处理大型FL群的任务。

Pace steering is based on the simple mechanism of the server suggesting to the device the optimum time window to reconnect. The device attempts to respect this, modulo its eligibility.

节奏控制是基于服务器的一种简单机制，即为设备建议一个最佳的重新连接的时间窗口。 设备尝试按照这个，安排自己的可选择性。

In the case of small FL populations, pace steering is used to ensure that a sufficient number of devices connect to the server simultaneously. This is important both for the rate of task progress and for the security properties of the Secure Aggregation protocol. The server uses a stateless probabilistic algorithm requiring no additional device/server communication to suggest reconnection times to rejected devices so that subsequent checkins are likely to arrive contemporaneously.

在小FL群的情况下，节奏控制用于确保有足够多的设备同时连接到服务器中。这对于任务过程的执行速度和安全聚积协议的安全性质都很重要。服务器使用的是无状态的概率算法，不需要额外的设备/服务器通信来推荐重新连接的次数，来拒绝设备，这样后续的报道很可能是同时到达的。

For large FL populations, pace steering is used to randomize device check-in times, avoiding the “thundering herd” problem, and instructing devices to connect as frequently as needed to run all scheduled FL tasks, but not more.

对于大FL群的情况，节奏控制用于使得设备报道次数随机化，避免“thundering herd”的问题，指示设备按照需要的频率进行连接，以运行所有规划好的FL任务，而不是连接的越多越好。

Pace steering also takes into account the diurnal oscillation in the number of active devices, and is able to adjust the time window accordingly, avoiding excessive activity during peak hours and without hurting FL performance during other times of the day.

节奏控制还在活跃设备的数量中考虑了每日的波动，可以相应的调节时间窗口，在高峰时间段内防止过多活跃节点，在每天的其他时间内还不会降低FL的性能。

## 3 Device

This section describes the software architecture running on a device participating in FL. This describes our Android implementation but note that the architectural choices made here are not particularly platform-specific.

本节描述了参与FL的设备中运行的软件架构。现在描述的是在我们的Android设备中的实现，但注意，这种架构选择并不是针对特定平台的。

The device’s first responsibility in on-device learning is to maintain a repository of locally collected data for model training and evaluation. Applications are responsible for making their data available to the FL runtime as an example store by implementing an API we provide. An application’s example store might, for example, be an SQLite database recording action suggestions shown to the user and whether or not those suggestions were accepted. We recommend that applications limit the total storage footprint of their example stores, and automatically remove old data after a pre-designated expiration time, where appropriate. We provide utilities to make these tasks easy. Data stored on devices may be vulnerable to threats like malware or physical disassembly of the phone, so we recommend that applications follow the best practices for on-device data security, including ensuring that data is encrypted at rest in the platform-recommended manner.

在设备上学习任务中，设备的首要责任是维护本地收集的数据的仓库，以进行模型训练和评估。应用负责使其数据对于FL运行时是可用的，包装成一个example store，我们是实现一个API。一个应用的example store可以是比如一个SQLite数据库，记录的是展示给用户看的行为建议，以及是否这些建议被接受。我们推荐应用要限制它们的example store的总计存储消耗，在合适的情况下，在预先指定的过期时间后，自动将老的数据删除掉。我们提供工具，可以使得这些任务非常简单。存储在设备上的数据，对于一些威胁来说，可能很脆弱，如恶意软件，或手机的物理拆解，所以我们推荐，应用要按照设备上数据安全的最佳实践进行，包括确保数据是加密的。

The FL runtime, when provided a task by the FL server, accesses an appropriate example store to compute model updates, or evaluate model quality on held out data. Fig. 2 shows the relationship between the example store and the FL runtime. Control flow consists of the following steps:

当FL服务器提供了一个任务时，FL运行时访问合适的example store，以计算模型更新，或在保留的数据上评估模型质量。图2展示了example store和FL运行时的关系。控制流包括下列步骤：

**Programmatic Configuration**. An application configures the FL runtime by providing an FL population name and registering its example stores. This schedules a periodic FL runtime job using Android’s JobScheduler. Possibly the most important requirement for training machine learning (ML) models on end users’ devices is to avoid any negative impact on the user experience, data usage, or battery life. The FL runtime requests that the job scheduler only invoke the job when the phone is idle, charging, and connected to an unmetered network such as WiFi. Once started, the FL runtime will abort, freeing the allocated resources, if these conditions are no longer met.

**程序性配置**。一个应用配置FL运行时，只要给出FL群的名称，并注册其example stores。这使用Android的JobScheduler安排周期性的FL运行时。在终端用户设备上训练ML模型，最重要的要求可能是，避免对用户体验、数据使用，或电池寿命有负面影响。FL运行时要求，job scheduler只在手机闲置、充电并且连接到不限流量的网络（如WiFi）的时候才调用这个任务。一旦开启，如果这些条件不满足时，FL运行时会放弃任务，释放已经分配的资源。

**Job Invocation**. Upon invocation by the job scheduler in a separate process, the FL runtime contacts the FL server to announce that it is ready to run tasks for the given FL population. The server decides whether any FL tasks are available for the population and will either return an FL plan or a suggested time to check in later.

**任务调用**。在job scheduler在一个分立的进程中调用任务时，FL运行时与FL服务器联系，宣称已经准备好为给定的FL群运行任务了。服务器决定是否任意FL任务是可用于这个群体的，然后要么返回一个FL计划，或返回一个建议的后面再来报道的时间。

**Task Execution**. If the device has been selected, the FL runtime receives the FL plan, queries the app’s example store for data requested by the plan, and computes plan-determined model updates and metrics.

**任务执行**。如果设备被选中，FL运行时接收FL计划，查询app的example store，检索计划要求的数据，并计算计划确定的模型更新和度量。

**Reporting**. After FL plan execution, the FL runtime reports computed updates and metrics to the server and cleans up any temporary resources.

**汇报**。在FL计划运行完后，FL运行时向服务器汇报计算的更新和度量，并清理使用的临时资源。

As already mentioned, FL plans are not specialized to training, but can also encode evaluation tasks - computing quality metrics from held out data that wasn’t used for training, analogous to the validation step in data center training.

上面已经提到，FL计划并不是专用于训练的，也可以包含评估任务，即从保留的数据中（未用于训练）计算质量度量，与在数据中心训练的验证步骤是类似的。

Our design enables the FL runtime to either run within the application that configured it or in a centralized service hosted in another app. Choosing between these two requires minimal code changes. Communication between the application, the FL runtime, and the application’s example store as depicted in Fig. 2 is implemented via Android’s AIDL IPC mechanism, which works both within a single app and across apps.

我们的设计使得FL运行时可以运行在应用配置的流程下，或以在另外一个app中主导的中心化服务中运行。在这两者之中选择，需要的代码改变非常少。应用、FL运行时和应用的example store之间的通信如图2所示，这是通过Android的AIDL IPC机制实现的，这可以在单个app之间运行，也可以跨app运行。

**Multi-Tenancy**. Our implementation provides a multi-tenant architecture, supporting training of multiple FL populations in the same app (or service). This allows for coordination between multiple training activities, avoiding the device being overloaded by many simultaneous training sessions at once.

**多租期**。我们的实现给出了一个多租期的架构，在相同的app（或服务）中支持多FL群体。这可以在多个训练活动中进行协调，避免了设备在同时进行多个训练sessions的时候出现过载的情况。

**Attestation**. We want devices to participate in FL anonymously, which excludes the possibility of authenticating them via a user identity. Without verifying user identity, we need to protect against attacks to influence the FL result from non-genuine devices. We do so by using Android’s remote attestation mechanism (Android Documentation), which helps to ensure that only genuine devices and applications participate in FL, and gives us some protection against data poisoning (Bagdasaryan et al., 2018) via compromised devices. Other forms of model manipulation – such as content farms using uncompromised phones to steer a model – are also potential areas of concern that we do not address in the scope of this paper.

**证据**。我们希望设备匿名的参与FL任务，这就避免了需要通过用户ID来验证的可能性。不需要验证用户ID，我们就需要保护以免收到攻击，进而从非真的设备上影响FL结果。我们通过Android的远程attestation机制来实现这个功能，这可以帮助确保只有真正的设备和应用参与到FL中，在一定程度上可以保护数据不会受到非真实设备的破坏。其他形式的模型操作，比如content farms，使用的真实的手机来操作模型，这也是可能的考虑的领域，但我们在本文中并没有涉及到这个问题。

## 4 Server

The design of the FL server is driven by the necessity to operate over many orders of magnitude of population sizes and other dimensions. The server must work with FL populations whose sizes range from tens of devices (during development) to hundreds of millions, and be able to process rounds with participant count ranging from tens of devices to tens of thousands. Also, the size of the updates collected and communicated during each round can range in size from kilobytes to tens of megabytes. Finally, the amount of traffic coming into or out of any given geographic region can vary dramatically over a day based on when devices are idle and charging. This section details the design of the FL server infrastructure given these requirements.

FL服务器的设计，是受到要在很多量级的群体规模上进行操作这个因素和其他因素驱动的。服务器需要根据FL群体规模来工作，而这个规模大小从数十个设备（在开发时）到成百万上千万，并还要可以处理一轮一轮的计算，每一轮的参与者数量从数十个设备到上万个设备。同时，每一轮中更新的收集和通信，其大小从KB到数十MB。最后，任意给定地理区域中，流入或流出的通信量在一天中的波动会很大，这主要依赖于设备什么时候是闲置并在充电。本节详述在这些要求下，FL服务器架构的设计。

### 4.1 Actor Model

The FL server is designed around the Actor Programming Model (Hewitt et al., 1973). Actors are universal primitives of concurrent computation which use message passing as the sole communication mechanism.

FL服务器是基于Actor Programming Model设计的。Actors是并发计算的统一原语，使用message passing作为唯一的通信机制。

Each actor handles a stream of messages/events strictly sequentially, leading to a simple programming model. Running multiple instances of actors of the same type allows a natural scaling to large number of processors/machines. In response to a message, an actor can make local decisions, send messages to other actors, or create more actors dynamically. Depending on the function and scalability requirements, actor instances can be co-located on the same process/machine or distributed across data centers in multiple geographic regions, using either explicit or automatic configuration mechanisms. Creating and placing fine-grained ephemeral instances of actors just for the duration of a given FL task enables dynamic resource management and load-balancing decisions.

每个actor对一个信息/事件流进行严格的顺序处理，这就带来了一个简单的编程模型。运行同一类型actor的多个实例，可以很自然的缩放到大量处理器/机器中。一个actor可以对信息作出响应，作出本地决策，向其他actors发送消息，或动态创建更多的actors。依据需求的功能和可扩展性，actor实例可以共同存在于同样的处理器/机器中，或分布在不同地理位置中的数据中心，使用的要么是显式的配置机制或自动的配置机制。对给定FL任务在一段事件内，临时actors实例的创建和放置，使得动态资源管理和负载均衡的决策成为可能。

### 4.2 Architecture

The main actors in the system are shown in Fig. 3. 系统的主要actor如图3所示。

**Coordinators** are the top-level actors which enable global synchronization and advancing rounds in lockstep. There are multiple Coordinators, and each one is responsible for an FL population of devices. A Coordinator registers its address and the FL population it manages in a shared locking service, so there is always a single owner for every FL population which is reachable by other actors in the system, notably the Selectors. The Coordinator receives information about how many devices are connected to each Selector and instructs them how many devices to accept for participation, based on which FL tasks are scheduled. Coordinators spawn Master Aggregators to manage the rounds of each FL task.

**Coordinators**是最高层级的actors，负责全局同步并在lockstep中推进一轮一轮数据处理的。有多个Coordinators，每个负责很多设备构成的FL群体。一个Coordinator在一个共享的锁服务中注册其地址和其管理的FL群体，所以对于每个FL群体永远都有单一的所有者，系统中的其他actors（尤其是Selectors）可以接触到这个FL群体。Coordinator接收一些信息，包括每个Selector连接了多少个设备，并基于安排了哪个FL任务，指导其接受几个设备进行参与。Coordinator生成很多Master Aggregators来管理每个FL任务的每轮计算。

**Selectors** are responsible for accepting and forwarding device connections. They periodically receive information from the Coordinator about how many devices are needed for each FL population, which they use to make local decisions about whether or not to accept each device. After the Master Aggregator and set of Aggregators are spawned, the Coordinator instructs the Selectors to forward a subset of its connected devices to the Aggregators, allowing the Coordinator to efficiently allocate devices to FL tasks regardless of how many devices are available. The approach also allows the Selectors to be globally distributed (close to devices) and limit communication with the remote Coordinator.

**Selector**负责接收并转发设备连接。它们周期性的从Coordinator接收信息，如每个FL群体需要多少设备，Selector用这个信息来作出本地决策，即是否接受每个设备。在Master Aggregator和Aggregator集产生以后，Coordinator指示Selectors转发其连接设备的子集给Aggregators，使得Coordinator可以高效的给FL任务分配设备，而不需要关心有多少设备是可用的。这种方法还允许Selectors分布在全球（离设备很近），并限制其与远程Coordinator的通信。

**Master Aggregators** manage the rounds of each FL task. In order to scale with the number of devices and update size, they make dynamic decisions to spawn one or more Aggregators to which work is delegated.

**Master Aggregators**管理每个FL任务的每轮计算。为随着设备数量和更新大小而扩展，需要进行动态决策，产生一个或多个Aggregators，代理某项任务。

No information for a round is written to persistent storage until it is fully aggregated by the Master Aggregator. Specifically, all actors keep their state in memory and are ephemeral. Ephemeral actors improve scalability by removing the latency normally incurred by distributed storage. In-memory aggregation also removes the possibility of attacks within the data center that target persistent logs of per-device updates, because no such logs exist.

每一轮的计算信息，直到Master Aggregator完全聚积了之后，才会写到固定存储去。具体的，所有的actors将其状态保存到内存中，都是临时的。临时actors通过消除掉分布式存储带来的延迟，改进了可扩展性。在内存中的聚积也消除掉了数据中心内攻击的可能性，有的攻击目标是每个设备更新的日志，但现在这种日志都是不存在的。

### 4.3 Pipelining

While Selection, Configuration and Reporting phases of a round (Sec. 2) are sequential, the Selection phase doesn’t depend on any input from a previous round. This enables latency optimization by running the Selection phase of the next round of the protocol in parallel with the Configuration/Reporting phases of a previous round. Our system architecture enables such pipelining without adding extra complexity, as parallelism is achieved simply by the virtue of Selector actors running the selection process continuously.

每一轮的Selection, Configuration和Reporting阶段是顺序化的，但Selection阶段并不依靠之前轮的任何输入。这使得延迟优化成为可能，具体是将协议下一轮的Selection阶段与前一轮的Configuration/Reporting阶段并行进行。我们的系统架构使得可以在不增加额外的复杂度的情况下形成这样的流程，因为并行化只需要Selector连续不断的进行选择过程就可以了。

### 4.4 Failure Modes

In all failure cases the system will continue to make progress, either by completing the current round or restarting from the results of the previously committed round. In many cases, the loss of an actor will not prevent the round from succeeding. For example, if an Aggregator or Selector crashes, only the devices connected to that actor will be lost. If the Master Aggregator fails, the current round of the FL task it manages will fail, but will then be restarted by the Coordinator. Finally, if the Coordinator dies, the Selector layer will detect this and respawn it. Because the Coordinators are registered in a shared locking service, this will happen exactly once.

在所有的失败情况下，系统会持续运行下去，要么是完成目前这一轮，要么是从之前一轮的结果重启。在很多情况下，一个actor的损失不会防止这一轮成功运行结束。比如，如果一个Aggregator或Selector崩溃了，只有连接到那个actor的设备会丢失。如果Master Aggregator坏掉了，它管理的FL任务的当前轮会运行失败，但会立刻由Coordinator重启。最后，如果Coordinator死掉了，Selector层会检测到，并重新生成之。因为Coordinators是注册在共享锁服务中的，这就精确的重启一次。

## 5 Analytics

There are many factors and failsafes in the interaction between devices and servers. Moreover, much of the platform activity happens on devices that we neither control nor have access to.

在设备和服务器的互动中，有很多因素和失效保护。而且，平台活动很多发生在设备上的，我们既没有控制权，也没有访问权限。

For this reason, we rely on analytics to understand what is actually going on in the field, and monitor devices’ health statistics. On the device side we perform computation-intensive operations, and must avoid wasting the phone’s battery or bandwidth, or degrading the performance of the phone. To ensure this, we log several activity and health parameters to the cloud. For example: the device state in which training was activated, how often and how long it ran, how much memory it used, which errors where detected, which phone model / OS / FL runtime version was used, and so on. These log entries do not contain any personally identifiable information (PII). They are aggregated and presented in dashboards to be analyzed, and fed into automatic time-series monitors that trigger alerts on substantial deviations.

因为这个，我们依靠分析来理解在服务器之外到底在发生什么，并监控设备的健康统计数据。在设备端，我们进行的是计算量很大的操作，并且要避免浪费手机的电池或带宽，或降低手机的性能。为确保这个，我们记录几种行为和健康参数到云中。比如，激活训练的时候设备的状态，运行的频繁度和时长，使用了多少内存，检测到了哪些错误，怎样检测到的，使用的哪个手机模型/OS/FL运行时版本，等等。这些日志不包含任何识别个人的信息(Personally Identifiable Information, PII)。它们聚积并显示在仪表盘上，以便分析，并送入自动的时间序列监控中，在有显著偏移的时候触发报警。

We also log an event for every state in a training round, and use these logs to generate ASCII visualizations of the sequence of state transitions happening across all devices (see Table 1 in the appendix). We chart counts of these sequence visualizations in our dashboards, which allows us to quickly distinguish between different types of issues.

我们还记录了训练过程中的每个状态，使用这些日志生成ASCII可视化的在所有设备上发生的状态迁移序列（见附录中的表1）。我们将这些可视化序列的计数做成表格，放到仪表盘中，使我们可以快速辨别这些问题的不同类别。

For example, the sequence “checking in, downloaded plan, started training, ended training, starting upload, error” is visualized as “-v[]+* ”, while the shorter sequence “checking in, downloaded plan, started training, error” is “-v[* ”. The first indicates that a model trained successfully but the results upload failed (a network issue), whereas the second indicates that a training round failed right after loading the model (a model issue).

比如，序列“登记，下载计划，开始训练，训练结束，开始上传，错误”可视化为“-v[]+* ”，而短一点的序列“登记，下载计划，开始训练，错误”则为“-v[* ”。第一个序列说明，一个模型成功进行了训练，但结果上传失败（网络问题），而第二个序列说明，模型装载后训练的一轮失败了（模型问题）。

Server side, we similarly collect information such as how many devices where accepted and rejected per training round, the timing of the various phases of the round, throughput in terms of uploaded and downloaded data, errors, and so on.

服务器端，我们类似的也收集信息，比如每一轮训练中接受和拒绝了多少设备，每轮中各种阶段的计时，上传数据和下载数据的吞吐量，错误，等等。

Since the platform’s deployment, we have relied on the analytics layer repeatedly to discover issues and verify that they were resolved. Some of the incidents we discovered were device health related, for example discovering that training was happening when it shouldn’t have, while others were functional, for example discovering that the drop out rates of training participants were much higher than expected.

自从平台的部署，我们就反复的依靠分析层来发现问题，核实问题被解决了。我们发现的一些意外是与设备健康相关的，比如发现在不应该开始的时候，训练就开始了，另外一些问题是功能性的，如发现训练参与者的drop out率比期望的要高很多。

Federated training does not impact the user experience, so both device and server functional failures do not have an immediate negative impact. But failures to operate properly could have secondary consequences leading to utility degradation of the device. Device utility to the user is mission critical, and degradations are difficult to pinpoint and easy to wrongly diagnose. Using accurate analytics to prevent federated training from negatively impacting the device’s utility to the user accounts is a substantial part of our engineering and risk mitigation costs.

联邦训练不会影响用户的体验，所以设备和服务器的功能性问题不会有什么立刻的负面影响。但不能适当的操作，可能会有次级后果，导致设备的工具质量降低。设备工具对于用户来说是很关键的，质量降低很难查实，很容易得到错误诊断。使用精确的分析来防止联邦训练对设备的工具有负面影响，从而影响用户账户，这是我们工程的重要一部分，也是威胁弥补代价的一部分。

## 6 Secure Aggregation

Bonawitz et al. (2017) introduced Secure Aggregation, a Secure Multi-Party Computation protocol that uses encryption to make individual devices’ updates uninspectable by a server, instead only revealing the sum after a sufficient number of updates have been received. We can deploy Secure Aggregation as a privacy enhancement to the FL service that protects against additional threats within the data center by ensuring that individual devices’ updates remain encrypted even in-memory. Formally, Secure Aggregation protects from “honest but curious” attackers that may have access to the memory of Aggregator instances. Importantly, the only aggregates needed for model evaluation, SGD, or Federated Averaging are sums(e.g.,w_t and n_t in Appendix 1)(It is important to note that the goal of our system is to provide the tools to build privacy preserving applications. Privacy is enhanced by the ephemeral and focused nature of the FL updates, and can be further augmented with Secure Aggregation and/or differential privacy — e.g., the techniques of McMahan et al. (2018) are currently implemented. However, while the platform is designed to support a variety of privacy-enhancing technologies, stating specific privacy guarantees depends on the details of the application and the details of how these technologies are used; such a discussion is beyond the scope of the current work).

Bonawitz等提出了Secure Aggregation，这是一种安全的多方计算协议，使用加密来使得个体设备的更新无法由于服务器检视，而只能在收到足够数量的更新并相加后才能看到结果。我们可以部署Secure Aggregation，作为FL服务的一种隐私增强服务，通过确保个体设备的更新即使在内存中也是加密的，保护了在数据中心内的额外威胁。正式的，Secure Aggregation防止那些honest-but-curious攻击者，它们可能可以访问Aggregator实例的内存。重要的是，模型评估所需要的唯一aggregates，SGD，或联邦平均，是求和的关系。（我们要注意的重要一点是，我们系统的目标是，为构建隐私保护的应用提供工具。隐私的加强，是通过FL更新的临时性和聚焦性实现的，可以进一步用Secure Aggregation和/或differential privacy增强，如现在实现的是McMahan等的技术。但是，平台的设计是支持很多不同的隐私增强技术的，而描述具体的隐私保证依靠的是应用的细节，以及这些技术是怎样使用的细节；这样的讨论不在本文的范围之内。）

Secure Aggregation is a four-round interactive protocol optionally enabled during the reporting phase of a given FL round. In each protocol round, the server gathers messages from all devices in the FL round, then uses the set of device messages to compute an independent response to return to each device. The protocol is designed to be robust to a significant fraction of devices dropping out before the protocol is complete. The first two rounds constitute a Prepare phase, in which shared secrets are established and during which devices who drop out will not have their updates included in the final aggregation. The third round constitutes a Commit phase, during which devices upload cryptographically masked model updates and the server accumulates a sum of the masked updates. All devices who complete this round will have their model update included in the protocol’s final aggregate update, or else the entire aggregation will fail. The last round of the protocol constitutes a Finalization phase, during which devices reveal sufficient cryptographic secrets to allow the server to unmask the aggregated model update. Not all committed devices are required to complete this round; so long as a sufficient number of the devices who started to protocol survive through the Finalization phase, the entire protocol succeeds.

Secure Aggregation是一个四轮交互协议，是在给定FL计算轮中，在reporting阶段可以选择开启的。在每个协议轮中，服务器在一个FL轮中，从所有设备收集信息，然后使用这些设备信息来计算得到独立的响应，返回到每个设备中。协议的设计，是可以容忍相当一部分设备在协议结束前drop out的。前两轮包含一个Prepare阶段，这里会确定共享的秘密，在这期间设备掉线的话，其更新不会包括在最后的聚积中。第三轮包含一个Commit阶段，在这期间，设备上传加密的掩膜的模型更新，服务器将这些掩膜的更新叠加聚积。所有完成了这一轮的设备，会将其模型更新累加到协议最终的聚积更新中，否则整个聚积会失败。协议的最后一轮是一个Finalization阶段，在这期间会将足够的加密秘密进行揭开，使服务器对聚积的模型更新去掩膜。并不是所有committed设备都需要来完成这一轮；只要有足够多开始了这个协议的设备完成到了Finalization阶段，整个协议就成功了。

Several costs for Secure Aggregation grow quadratically with the number of users, most notably the computational cost for the server. In practice, this limits the maximum size of a Secure Aggregation to hundreds of users. So as not to constrain the number of users that may participate in each round of federated computation, we run an instance of Secure Aggregation on each Aggregator actor (see Fig. 3) to aggregate inputs from that Aggregator’s devices into an intermediate sum; FL tasks define a parameter k so that all updates are securely aggregated over groups of size at least k. The Master Aggregator then further aggregates the intermediate aggregators’ results into a final aggregate for the round, without Secure Aggregation.

安全聚积的几种代价随着用户数量呈四次方增长，最主要的是服务器的计算代价。在实践中，这限制了安全聚积的最大规模，最多只能到数百个用户。只要不限制可能参与到联邦计算的每一轮的用户数量，我们在每个Aggregator actor上都运行一个安全聚积实例，从那些Aggregator的设备上聚积输入，到一个中间累加结果中；FL任务定义了一个参数k，所有更新都在至少k个群体上进行安全聚积。Master Aggregator然后进一步将中间aggregator的结果聚积到该轮的最终agrregate中，这时候不需要安全聚积。

## 7 Tools and Workflow

Compared to the standard model engineer workflows on centrally collected data, on-device training poses multiple novel challenges. First, individual training examples are not directly inspectable, requiring tooling to work with proxy data in testing and simulation (Sec. 7.1). Second, models cannot be run interactively but must instead be compiled into FL plans to be deployed via the FL server (Sec. 7.2). Finally, because FL plans run on real devices, model resource consumption and runtime compatibility must be verified automatically by the infrastructure (Sec. 7.3).

与在中心化收集的数据的情况下，标准的模型工程师工作流程相比，在设备上训练提出了多个新的挑战。第一，个体训练样本并不是直接可检视的，需要工具来对代理数据进行测试和模拟。第二，模型不能进行交互式运行，而是要编译到FL计划里，通过FL服务器进行部署。最后，因为FL计划是运行在真实设备中的，模型资源消耗和运行时兼容性必须通过基础设施自动验证。

The primary developer surface of model engineers working with the FL system is a set of Python interfaces and tools to define, test, and deploy TensorFlow-based FL tasks to the fleet of mobile devices via the FL server. The workflow of a model engineer for FL is depicted in Fig. 4 and described below.

模型工程化与FL系统的主要开发者界面是一组Python接口和工具，要进行定义，测试并通过FL服务器部署基于TensorFlow的FL任务到移动设备上。模型工程化的工作流如图4所示，描述如下。

### 7.1 Modeling and Simulation

Model engineers begin by defining the FL tasks that they would like to run on a given FL population in Python. Our library enables model engineers to declare Federated Learning and evaluation tasks using engineer-provided TensorFlow functions. The role of these functions is to map input tensors to output metrics like loss or accuracy. During development, model engineers may use sample test data or other proxy data as inputs. When deployed, the inputs will be provided from the on-device example store via the FL runtime.

模型工程师首先开始定义FL任务，该任务在给定的FL群体上以Python语言运行。我们的库使得模型工程师可以使用工程师提供的TensorFlow函数来声明FL任务和评估任务。这些函数的任务是将输入张量映射到输出度量中，如损失函数或精确度。在开发阶段，模型工程师可能要用样本测试数据或其他代理数据作为输入。在部署时，输入将由在设备上的example store通过FL运行时提供。

The role of the modeling infrastructure is to enable model engineers to focus on their model, using our libraries to build and test the corresponding FL tasks. FL tasks are validated against engineer-provided test data and expectations, similar in nature to unit tests. FL task tests are ultimately required in order to deploy a model as described below in Sec. 7.3.

建模基础设施的角色是使模型工程师聚焦关注其模型，使用我们的库来构建并测试对应的FL任务。FL任务由工程师提供的测试数据和期望来确认，与单元测试的本质类似。FL任务测试是绝对需要的，以使模型可以如7.3所述的进行部署。

The configuration of tasks is also written in Python and includes runtime parameters such as the optimal number of devices in a round as well as model hyperparameters like learning rate. FL tasks may be defined in groups: for example, to evaluate a grid search over learning rates. When more than one FL task is deployed in an FL population, the FL service chooses among them using a dynamic strategy that allows alternating between training and evaluation of a single model or A/B comparisons between models.

任务的配置也是用Python写的，包括运行时的参数，如在一轮中的最佳设备数量，以及模型的超参数，如学习率。FL任务可以以组来定义：比如，为评估学习率的网格搜索。当在一个FL群体中部署来多余一个FL任务时，FL服务在其中使用动态策略进行选择，可以在一个任务或A/B模型的比较的训练和评估中进行切换。

Initial hyperparameter exploration is sometimes done in simulation using proxy data. Proxy data is similar in shape to the on-device data but drawn from a different distribution – for example, text from Wikipedia may be viewed as proxy data for text typed on a mobile keyboard. Our modeling tools allow deployment of FL tasks to a simulated FL server and a fleet of cloud jobs emulating devices on a large proxy dataset. The simulation executes the same code as we run on device and communicates with the server using simulated FL populations. Simulation can scale to a large number of devices and is sometimes used to pre-train models on proxy data before it is refined by FL in the field.

初始的超参数的探索有时候是用代理数据进行模拟的。代理数据在形状上与设备上的数据类似，但是服从不同的分布的，比如，Wikipedia上的文本可以理解为移动手机键盘上打印文本的一种代理数据。我们的建模工具使FL任务可以部署到一个模拟的FL服务器上，很多云上的任务模拟设备在大型代理数据集上进行工作。这种模拟执行的是与运行在设备上相同的代码，与服务器的通信也是使用的模拟FL群。模拟可以缩放到很大数量的设备上，有时候用于在代理数据上预训练模型，然后再用真实的FL进行精炼。

### 7.2 Plan Generation

Each FL task is associated with an FL plan. Plans are automatically generated from the combination of model and configuration supplied by the model engineer. Typically, in data center training, the information which is encoded in the FL plan would be represented by a Python program which orchestrates a TensorFlow graph. However, we do not execute Python directly on the server or devices. The FL plan’s purpose is to describe the desired orchestration independent of Python.

每个FL任务都与一个FL计划是关联在一起的。计划是从模型和配置的组合中自动生成的，这都是由模型工程师提供的。一般的，在数据中心的训练中，在FL计划里的信息都由一个Python程序运行，表示的是一个TensorFlow图。但是，我们没有在服务器或设备上直接执行Python代码。FL计划的目的是，描述想要的与Python无关的表示。

An FL plan consists of two parts: one for the device and one for the server. The device portion of the FL plan contains, among other things: the TensorFlow graph itself, selection criteria for training data in the example store, instructions on how to batch data and how many epochs to run on the device, labels for the nodes in the graph which represent certain computations like loading and saving weights, and so on. The server part contains the aggregation logic, which is encoded in a similar way. Our libraries automatically split the part of a provided model’s computation which runs on device from the part that runs on the server (the aggregation).

一个FL计划包括两个部分：一个是用于设备的，一个是用于服务器的。FL计划设备的那部分包括：TensorFlow图，example store中训练数据的选择规则，在设备上怎样成批打包数据和训练多少轮的说明，图中节点的标签，表示的是特定的计算，如权重的loading和saving，等等。服务器部分包括，聚积逻辑，编码的形式类似。我们的库自动将提供的模型的计算分解为两部分，一部分在设备上运行，另一部分在服务器端（聚积）。

### 7.3 Versioning, Testing, and Deployment

Model engineers working in the federated system are able to work productively and safely, launching or ending multiple experiments per day. But because each FL task may potentially be RAM-hogging or incompatible with version(s) of TensorFlow running on the fleet, engineers rely on the FL system’s versioning, testing, and deployment infrastructure for automated safety checks.

模型工程师建构的联邦系统，可以很有成效并安全的工作，每天开启或结束多个试验。因为每个FL任务都可能是很消耗内存的，或与运行在fleet上的TensorFlow版本不兼容，工程师要进行自动的安全检查，就要依靠FL系统的版本、测试和部署基础架构。

An FL task that has been translated into an FL plan is not accepted by the server for deployment unless certain conditions are met. First, it must have been built from auditable, peer reviewed code. Second, it must have bundled test predicates for each FL task that pass in simulation. Third, the resources consumed during testing must be within a safe range of expected resources for the target population. And finally, the FL task tests must pass on every version of the TensorFlow runtime that the FL task claims to support, as verified by testing the FL task’s plan in an Android emulator.

一个已经翻译为FL计划的FL任务，必须要满足一些条件，才会被服务器接受，进行部署。首先，必须是由可审计的、同行评议的代码构建起来的。第二，必须针对每个FL任务进行了测试，在模拟测试中通过。第三，测试时消耗的资源对于目标群体来说必须在期望资源的范围内。最后，FL任务测试必须在每个FL任务支持的TensorFlow运行时的版本上都通过，这些版本在一个Android模拟器中由FL任务的计划进行了验证。

Versioning is a specific challenge for on-device machine learning. In contrast to data-center training, where the TensorFlow runtime and graphs can generally be rebuilt as needed, devices may be running a version of the TensorFlow runtime that is many months older than what is required by the FL plan generated by modelers today. For example, the old runtime may be missing a particular TensorFlow operator, or the signature of an operator may have changed in an incompatible way. The FL infrastructure deals with this problem by generating versioned FL plans for each task. Each versioned FL plan is derived from the default (unversioned) FL plan by transforming its computation graph to achieve compatibility with a deployed TensorFlow version. Versioned and unversioned plans must pass the same release tests, and are therefore treated as semantically equivalent. We encounter about one incompatible change that can be fixed with a graph transformation every three months, and a slightly smaller number that cannot be fixed without complex workarounds.

版本是设备上机器学习的一个特殊挑战。与数据中心的训练对比，其中TensorFlow运行时和图一般都可以根据需要进行rebuilt，但设备上运行的TensorFlow运行时的版本，可能比今天建模者生成的FL计划所需要的版本老好几个月。比如，老的运行时可能缺少一个特别的TensorFlow算子，或一个算子的签字可能以一种不兼容的方式进行了改变。FL基础设施通过对每个任务生成不同的FL计划，来处理这个问题。每个版本的FL任务都是由默认的（无版本的）FL计划推导而来，将其计算图变形，以与已经部署的TensorFlow版本兼容。有版本的和无版本的计划必须通过相同的测试，因为可以认为在语义上是等价的。我们每三个月，都会遇到一种不兼容的变化，可以通过图变换进行修补，略少一点的问题可以无需复杂的变通就可以解决。

### 7.4 Metrics 度量

As soon as an FL task has been accepted for deployment, devices checking in may be served the appropriate (versioned) plan. As soon as an FL round closes, that round’s aggregated model parameters and metrics are written to the server storage location chosen by the model engineer.

一旦FL任务被接受进行部署，签到的设备可能可能运行的是合适的（有版本的）计划。一旦一个FL轮结束了，这一轮聚积的模型参数和度量就写入模型工程师选择的服务器存储的位置。

Materialized model metrics are annotated with additional data, including metadata like the source FL task’s name, FL round number within the FL task, and other basic operational data. The metrics themselves are summaries of device reports within the round via approximate order statistics and moments like mean. The FL system provides analysis tools for model engineers to load these metrics into standard Python numerical data science packages for visualization and exploration.

物质化的模型度量是用额外的数据进行标注的，包括元数据，如源FL任务的名称，FL任务中FL轮的数量，以及其他基本的操作数据。度量本身是这一轮内设备报告的总结，采用的是近似的阶的统计和动量，如均值。FL系统为模型工程师提供分析工具，可以将这些度量load为标准Python数值数据科学包进行可视化和探索。

## 8 Applications

Federated Learning applies best in situations where the on-device data is more relevant than the data that exists on servers (e.g., the devices generate the data in the first place), is privacy-sensitive, or otherwise undesirable or infeasible to transmit to servers. Current applications of Federated Learning are for supervised learning tasks, typically using labels inferred from user activity (e.g., clicks or typed words).

联邦学习最好的应用场景中，设备上的数据比在服务器上的数据更有相关性（如，设备首先生成的数据），是隐私敏感的数据，或否则不好或不可能传输到服务器端。目前FL的应用是对于有监督的学习任务，一般使用的标签是从用户行为推断出来的（如，点击或打印的文本）。

**On-device item ranking**. A common use of machine learning in mobile applications is selecting and ranking items from an on-device inventory. For example, apps may expose a search mechanism for information retrieval or in-app navigation, for example settings search on Google Pixel devices (ai.google, 2018). By ranking these results on-device, expensive calls to the server (in e.g., latency, bandwidth or power consumption dimensions) are eliminated, and any potentially private information from the search query and user selection remains on the device. Each user interaction with the ranking feature can become a labeled data point, since it’s possible to observe the user’s interaction with the preferred item in the context of the full ranked list.

**设备上的item排序**。移动应用中机器学习的一个常见用处是，从一个设备上的仓库中选择并排序items。比如，apps可能会暴露一个搜索机制，进行信息检索，或app内导航，比如设置在Google Pixel设备上的设置搜索。通过这些在设备上的排序结果，对服务器的耗费较高的呼叫（如，延迟，带宽或能量消耗等维度）就不用了，搜索请求和用户选择得到的任何可能的隐私信息也都一直保留在设备上。每个用户与排序特征的交互，都能变成一个标记的数据点，因为观察用户与喜欢的item的交互，在完全排序的列表的上下文下，是可能的。

**Content suggestions for on-device keyboards**. On-device keyboard implementations can add value to users by suggesting relevant content – for example, search queries that are related to the input text. Federated Learning can be used to train ML models for triggering the suggestion feature, as well as ranking the items that can be suggested in the current context. This approach has been taken by Google’s Gboard mobile keyboard team, using our FL system (Yang et al., 2018).

**对于设备上键盘的内容建议**。设备上键盘的实现可以对用户增值的，主要通过建议相关的内容，比如，搜索与输入文本相关的检索。FL可以用于训练触发建议特征的ML模型，以及在当前上下文下对建议items的排序。这种方法已经为Google的Gboard移动键盘小组采用，使用的是我们的FL系统。

**Next word prediction**. Gboard also used our FL platform to train a recurrent neural network (RNN) for next-word-prediction (Hard et al., 2018). This model, which has about 1.4 million parameters, converges in 3000 FL rounds after processing 6e8 sentences from 1.5e6 users over 5 days of training (so each round takes about 2–3 minutes). It improves top-1 recall over a baseline n-gram model from 13.0% to 16.4%, and matches the performance of a server-trained RNN which required 1.2e8 SGD steps. In live A/B experiments, the FL model outperforms both the n-gram and the server-trained RNN models.

**下一个单词的预测**。Gboard还使用我们的FL平台，训练了一个RNN模型，进行下一个单词的预测。这个模型有大约140万参数，在3000FL轮后，处理了1.5e6个用户的6e8个句子后，经过5天的训练（所以每一轮大约耗时2-3min）。它在n-gram模型的基准上，将top-1召回率从13.0%改进到了16.4%，与一个在服务器端训练的RNN性能相似，而这个RNN需要1.2e8个SGD步骤。在live A/B试验中，FL模型超过了n-gram和服务器端训练的RNN模型。

## 9 Operational Profile

In this section we provide a brief overview of some key operational metrics of the deployed FL system, running production workloads for over a year; Appendix A provides additional details. These numbers are examples only, since we have not yet applied FL to a diverse enough set of applications to provide a complete characterization. Further, all data was collected in the process of operating a production system, rather than under controlled conditions explicitly for the purpose of measurement. Many of the performance metrics here depend on the device and network speed (which can vary by region); FL plan, global model and update sizes (varies per application); number of samples per round and computational complexity per sample.

本节中，我们简要概览一下部署的FL系统的一些关键operational metrics，运行生产级workloads已经超过了一年；附录A有更多的细节。这些数字只是例子，因为我们尚未将FL应用到一个足够多元化的应用集中，以给出完整的特点描述。更多的，所有数据的收集都是在操作生产系统的过程中的，而不是在受控的环境下的。很多性能度量这里依靠的是设备和网络速度（这会随着区域变化不同）；FL计划，全局模型和更新大小（每个应用都不一样）；每一轮的样本数量和每个样本的计算复杂度。

We designed the FL system to elastically scale with the number and sizes of the FL populations, potentially up into the billions. Currently the system is handling a cumulative FL population size of approximately 10M daily active devices, spanning several different applications.

我们设计的FL系统可以随着FL群体的数量大小弹性扩展，可能最大到十亿的数量级。目前系统处理的累积FL群体大小为大约每天10M活跃设备，支持几个不同的应用。

As discussed before, at any point in time only a subset of devices connect to the server due to device eligibility and pace steering. Given this, in practice we observe that up to 10k devices are participating simultaneously. It is worth noting that the number of participating devices depends on the (local) time of day (see Fig. 5). Devices are more likely idle and charging at night, and hence more likely to participate. We have observed a 4× difference between low and high numbers of participating devices over a 24 hours period for a US-centric population.

就像之前讨论的那样，在任意时间点上，只有一部分设备连接到服务器上，这是由于设备的可选择性和节奏控制的需要。在此情况下，实践中，我们观察到了，最多10k设备同时参与。值得注意的是，参与设备的数量依靠的是本地时间（见图5）。设备在夜间更可能是闲置充电的，因此更可能参与。在以美国为中心的群体中，我们在24小时内观察到参与设备数量最高与最低达到4x的差异。

Based on the previous work of McMahan et al. (2017) and experiments we have conducted on production FL populations, for most models receiving updates from a few hundred devices per FL round is sufficient (that is, we see diminishing improvements in the convergence rate from training on larger numbers of devices). We also observe that on average the portion of devices that drop out due to computation errors, network failures, or changes in eligibility varies between 6% and 10%. Therefore, in order to compensate for device drop out as well as to allow stragglers to be discarded, the server typically selects 130% of the target number of devices to initially participate. This parameter can be tuned based on the empirical distribution of device reporting times and the target number of stragglers to ignore.

基于McMahan等之前的工作和试验，我们在生产中进行了FL群体的试验，对于多数模型，每FL轮从几百个设备上接收更新是足够的（即，在更多数量的设备上进行训练时，我们观察到的收敛速度改进逐渐消失）。我们还观察到，由于计算错误，网络故障，或可用性改变，导致掉线的部分设备，平均为6%-10%。因此，为弥补设备掉线，并允许抛弃迷路的设备，服务器一般选择130%目标数量的设备来初始化参与。这个参数可以基于设备汇报次数的经验分布和要忽视的迷途设备数量来进行调整。

## 10 Related Work

**Alternative Approaches**. To the best of our knowledge,the system we described is the first production-level Federated Learning implementation, focusing primarily on the Federated Averaging algorithm running on mobile phones. Nevertheless, there are other ways to learn from data stored on mobile phones, and other settings in which FL as a concept could be relevant.

据我们所知，我们所叙述的系统，是第一个生产级的FL实现，主要聚焦在运行在移动手机上的联邦平均算法。尽管如此，还有其他方法来从存储在移动手机上的数据进行学习，以及其他设置上，都与FL这个概念相关的。

In particular, Pihur et al. (2018) proposes an algorithm that learns from users’ data without performing aggregation on the server and with additional formal privacy guarantees. However, their work focuses on generalized linear models, and argues that their approach is highly scalable due to avoidance of synchronization and not requiring to store updates from devices. Our server design described in Sec. 4, rebuts the concerns about scalability of the synchronous approach we are using, and in particular shows that updates can be processed online as they are received without a need to store them. Alternative proposals for FL algorithms include Smith et al. (2017); Kamp et al. (2018), which would be on the high-level compatible with the system design described here.

特别的，Pihur等提出了一个算法，从用户的数据上学习，而不需要在服务器上进行聚积，并有额外的正式的隐私保护。但是，他们的工作聚焦的是一般化的线性模型，并声称其方法，由于避免了同步，不需要从设备上存储更新，从而是高度可扩展的。我们的服务器设计，在第4部分进行了阐述，不需要担心我们使用的同步方法的可扩展性，特别是更新可以在线处理，因为其接收后不需要进行存储。其他的FL算法的建议包括Smith等，Kamp等的算法，其在高层次上与我们描述的系统是兼容的。

In addition, Federated Learning has already been proposed in the context of vehicle-to-vehicle communication (Samarakoon et al., 2018) and medical applications (Brisimi et al., 2018). While the system described in this work as a whole does not directly apply to these scenarios, many aspects of it would likely be relevant for production application.

另外，FL已经在vehicle-to-vehicle通信的上下文中已经提出来了，以及医学应用。本文中描述的系统整体上不能直接应用到这些场景中，但很多方面很可能对于生产性应用是相关的。

Nishio & Yonetani (2018) focuses on applying FL in different environmental conditions, namely where the server can reach any subset of heterogeneous devices to initiate a round, but receives updates sequentially due to cellular bandwidth limit. The work offers a resource-aware selection algorithm maximizing the number of participants in a round, which is implementable within our system.

Nishio等聚焦在将FL应用到不同的环境条件下，即服务器可以接触到任意异质设备，来开始一轮计算，但接收到的更新是顺序的，因为有蜂窝带宽的限制。文章给出了对资源敏感的选择算法，可以最大化一轮中的参与数量，也是可以用我们的系统实现的。

**Distributed ML**. There has been significant work on distributed machine learning, and large-scale cloud-based systems have been described and are used in practice. Many systems support multiple distribution schemes, including model parallelism and data parallelism, e.g., Dean et al. (2012) and Low et al. (2012). Our system imposes a more structured approach fitting to the domain of mobile devices, which have much lower bandwidth and reliability compared to datacenter nodes. We do not allow for arbitrary distributed computation but rather focus on a synchronous FL protocol. This domain specialization allows us, from the system viewpoint, to optimize for the specific use case.

在分布式机器学习中有很多工作，大规模云端系统得到了很多应用。很多系统支持多个分布式方案，包括模型并行及数据并行，如Dean等和Low等的文章。我们的系统提出了更为结构化的方法，更适合移动设备的领域，其带宽更低，与数据中心节点相比可靠性更低。我们不允许任意的分布式计算，而是聚焦在同步的FL协议。这个领域的专门化使得我们可以针对具体的使用场景进行优化。

A particularly common approach in the datacenter is the parameter server, e.g., Li et al. (2014); Dean et al. (2012); Abadi et al. (2016), which allows a large number of workers to collaborate on a shared global model, the parameter vector. Focus in that line of work is put on an efficient server architecture for dealing with vectors of the size of 10^9 to 10^12. The parameter server provides global state which workers access and update asynchronously. Our approach inherently cannot work with such a global state, because we require a specific rendezvous between a set of devices and the FL server to perform a synchronous update with Secure Aggregation.

在数据中心中，一个特别的常用方法是参数服务器，如Li等Dean等Abadi等人的文章，使大量workers在一个共享的全局模型上进行合作，即参数向量。聚焦这条线的工作，是用高效的服务器架构，处理规模在10^9到10^12的向量。参数服务器给出了哪个workers进行异步访问并更新的全局状态。我们的方法不能用这种全局状态进行工作，因为我们需要对设备集合和FL服务器进行特定的集结，以采用安全聚积进行同步更新。

**MapReduce**. For datacenter applications, it is now commonly accepted that MapReduce (Dean & Ghemawat, 2008) is not the right framework for ML training. For the problem space of FL, MapReduce is a close relative. One can interpret the FL server as the Reducer, and FL devices as Mappers. However, there are also fundamental technical differences compared to a generic MapReduce framework. In our system, FL devices own the data on which they are working. They are fully self-controlled actors which attend and leave computation rounds at will. In turn, the FL server actively scans for available FL devices, and brings only selected subsets of them together for a round of computation. The server needs to work with the fact that many devices drop out during computation, and that availability of FL devices varies drastically over time. These very specific requirements are better dealt with by a domain specific framework than a generic MapReduce.

对于数据中心的应用，大家普遍接受MapReduce并不是ML训练的正确框架。对于FL的问题空间，MapReduce是一个接近的相关。可以将FL服务器解释为Reducer，将FL设备解释为Mappers。但是，与一个通用MapReduce框架比较，总还是有基本的技术区别的。在我们的系统中，FL设备拥有要处理的数据。它们是完全自己控制的actors，可以随意加入及离开计算任务。而FL服务器主动扫描可用的FL设备，并只将选择的子集带回到一起，进行一轮计算。服务器需要在很多设备计算时掉线的情况下工作，而且设备的可用性随着时间变化极大。这些都是非常具体的需求，最好由一个领域相关的框架进行处理，而不是通用的MapReduce。

## 11 Future Work

**Bias**. The Federated Averaging (McMahan et al., 2017) protocol assumes that all devices are equally likely to participate and complete each round. In practice, our system potentially introduces bias by the fact that devices only train when they are on an unmetered network and charging. In some countries the majority of people rarely have access to an unmetered network. Also, we limit the deployment of our device code only to certain phones, currently with recent Android versions and at least 2 GB of memory, another source of potential bias.

**偏移**。联邦平均协议假设所有设备参与并完成一轮计算的可能性是一样的。在实际中，我们的系统是可以引入偏置的，因为事实上设备只在连入不计流量的网络并在充电的时候才进行训练。在一些国家，大多数人都很少可以访问不计流量的网络。而且，我们限制了只部署到特定手机的设备码，目前是最新的Android版本，并至少有2GB内存的，这也是另一种偏置的原因。

We address this possibility in the current system as follows: During FL training, the models are not used to make user- visible predictions; instead, once a model is trained, it is evaluated in live A/B experiments using multiple application-specific metrics (just as with a datacenter model). If bias in device participation or other issues lead to an inferior model, it will be detected at this point. So far, we have not observed this to be an issue in practice, but this is likely application and population dependent. Further quantification of these possible effects across a wider set of applications, and if needed algorithmic or systems approaches to mitigate them, are important directions for future work.

我们在现有的系统中处理这种可能性如下：在FL训练时，模型并不用户进行用户可见的预测；而一旦模型训练好，就在live A/B实验中进行评估，使用的多个专属应用的度量（就像在数据中心模型中一样）。如果设备参与或其他问题的偏移，导致了模型并不是最优的，将会在这一点上被检测到。迄今为止，我们在实践中还没有观察到这是一个问题，但这很可能是依赖于应用和群体的。这些可能的效果在更多应用中的进一步量化，如果需要的话，使用算法或系统方法进行缓和，这些都是未来工作的重要方向。

**Convergence Time**. We noted in Sec. 8 that we currently observe a slower convergence time for Federated Learning compared to ML on centralized data where training is backed by the power of a data center. Current FL algorithms such as Federated Averaging can only efficiently utilize 100s of devices in parallel, but many more are available; FL would greatly benefit from new algorithms that can utilize increased parallelism.

**收敛时间**。我们在第8部分提到了，我们现在观察到的FL的收敛时间，是比在中心化的数据上（有数据中心的支持）进行训练是要慢的。目前的FL算法，比如联邦平均，只能高效的进行100s设备的并行，但还有更多的是可用的；新的可以增加并行性的算法，可以使FL得到极大的获益。

On the operational side, there is also more which can be done. For example, the time windows to select devices for training and wait for their reporting is currently configured statically per FL population. It should be dynamically adjusted to reduce the drop out rate and increase round frequency. We should ideally use online ML for tuning this and other parameters of the protocol configuration, bringing in e.g. time of the day as context.

在操作方面，也有很多事情可以去做。比如，选择设备去训练和等待其汇报的时间窗口，目前是每个FL群体进行静态配置的。可以进行动态调整，以降低drop out率，提高每一轮的效率。我们应当理想化的使用在线ML以调节协议配置的这个和其他参数。将比如每天的时间纳入到上下文中。

**Device Scheduling**. Currently, our multi-tenant on-device scheduler uses a simple worker queue for determining which training session to run next (we avoid running training sessions on-device in parallel because of their high resource consumption). This approach is blind to aspects like which apps the user has been frequently using. It’s possible for us to end up repeatedly training on older data (up to the expiration date) for some apps, while also neglecting training on newer data for the apps the user is frequently using. Any optimization here, though, has to be carefully evaluated against the biases it may introduce.

**设备安排计划**。目前，我们的多租户设备上的scheduler使用的是一个简单的worker队列，来确定下面要运行哪个训练会话（我们避免在设备上并行运行训练会话，因为其高资源消耗）。这种方法对于一些方面是盲目的，如哪些apps是频繁使用的。但是，这里的任何优化，都必须很小心的进行评估，可能会带来哪些偏置。

**Bandwidth**. When working with certain types of models, for example recurrent networks for language modeling, even small amounts of raw data can result in large amounts of information (weight updates) being communicated. In particular, this might be more than if we would just upload the raw data. While this could be viewed as a tradeoff for better privacy, there is also much which can be improved. To reduce the bandwidth necessary, we implement compression techniques such as those of Konecny et al. (2016b) and Caldas et al. (2018). In addition to that, we can modify the training algorithms to obtain models in quantized representation (Jacob et al., 2017), which will have synergetic effect with bandwidth savings and be important for efficient deployment for inference.

**带宽**。当操作特定类型的模型时，比如语言建模的RNN模型，即使很少的原始数据，都可以得到大量信息（权重更新）需要进行通信。特别是，这可能会比仅仅上传原始数据要更多。为降低需要的带宽，我们实现了一些压缩技术，如Konecny等和Caldas等的文章。除了这个，我们可以对训练算法进行修改，得到量化表示的模型，这与节约带宽一起有协同作用，这对于推理的高效部署很重要。

**Federated Computation**. We believe there are more applications besides ML for the general device/server architecture we have described in this paper. This is also apparent from the fact that this paper contains no explicit mentioning of any ML logic. Instead, we refer abstractly to ’plans’, ’models’, ’updates’ and so on.

**联邦计算**。我们相信除了ML，本文描述的一般性的设备/服务器架构会有更多的应用。本文也没有提及任何特定的ML逻辑，也可以看得出来。我们只是抽象的采用了计划，模型，更新等概念。

We aim to generalize our system from Federated Learning to Federated Computation, which follows the same basic principles as described in this paper, but does not restrict computation to ML with TensorFlow, but general MapReduce like workloads. One application area we are seeing is in Federated Analytics, which would allow us to monitor aggregate device statistics without logging raw device data to the cloud.

我们的目标是将我们的系统从联邦学习泛化到联邦计算，遵循的是本文描述的同样原则，但并不限制是采用TensorFlow的ML计算，而是一般性的MapReduce类的workloads。我们看到的一个应用领域是联邦分析，这使我们可以监控聚积设备的统计数据，而不需要将原始的设备数据记录到云端。

## A Operational Profile Data