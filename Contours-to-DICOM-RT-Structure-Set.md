# Exporting Contours to DICOM-RT Structure Set

Subrahmanyam Gorthi et. al. 

## 0. Abstract

This paper presents an ITK [1] implementation for exporting the contours of the automated segmentation results to DICOM-RT Structure Set format. The “radiotherapy structure set” (RTSTRUCT) object of the DICOM standard is used for the transfer of patient structures and related data, between the devices found within and outside the radiotherapy department. It mainly contains the information of regions of interest (ROIs) and points of interest (E.g. dose reference points). In many cases, rather than manually drawing these ROIs on the CT images, one can indeed benefit from the automated segmentation algorithms already implemented in ITK. But at present, it is not possible to export the ROIs obtained from ITK to RTSTRUCT format. In order to bridge this gap, we have developed a framework for exporting contour data to RTSTRUCT. We provide here the complete implementation of RTSTRUCT exporter and present the details of the pipeline used. Results on a 3-D CT image of the Head and Neck (H&N) region are presented.

本文提出了一种将自动分割的轮廓导出成DICOM-RT Structure Set格式的ITK实现。放射治疗structure set，即DICOM标准的RTSTRUCT object，是用于在放射治疗部门内外的设备之间，传递患者的结构和相关的数据的。其主要包括了感兴趣区域和感兴趣点的信息（如，剂量参考点）。在很多病例中，可以不用在CT图像中手动画出ROIs，而是利用ITK中已经实现的自动分割算法。但目前，ITK得到的ROIs不可能导出到RTSTRUCT格式。为弥补这个空白，我们提出了一个框架，将轮廓数据导出到RTSTRUCT格式。我们这里给出RTSTRUCT导出的完整实现，给出使用的流程细节，并给出了在头颈部区域的3D CT图像的结果。

## 1 Introduction

“Digital Imaging and Communications in Medicine” (DICOM) is a standard for handling, storing, printing and transmitting information in medical imaging. DICOM enables the integration of scanners, servers, workstations, printers and network hardware from multiple manufacturers into a picture archiving and communication system (PACS). It supports a wide variety of modules like CT, MR, nuclear medicine, ultrasound, X-ray, PET and radiotherapy (RT).

DICOM是处理，存储，打印和传输医学图像信息的一种标准。DICOM将不同厂商的扫描仪、服务器、工作站、打印机和网络硬件整合成一个图像归档和通信系统(PACS)。其支持很多模块，如CT，MR，核医学，超声，X射线，PET和放射治疗(RT)。

There are several RT-specific modules in DICOM standard; the key DICOM-RT modules are RT Image, RT Dose, RTSTRUCT, RT Plan and RT Treatment Record. DICOM-RT, through these modules, is found to effectively describe most of the data needed in radiotherapy. Hence, major manufacturers have already adopted this standard for their applications. This paper deals with exclusively the RTSTRUCT module, in the context of writing contour data.

在DICOM标准中有几个RT专用的模块，关键的DICOM-RT模块有RT图像，RT剂量，RTSTRUCT，RT计划和RT治疗记录。DICOM-RT通过这些模块，可以有效的描述放射治疗中所需的多数数据。因此，主要的厂商都已经对其应用采用了这个标准。本文专门处理RTSTRUCT模块在写入轮廓数据中的应用。

The purpose of the RTSTRUCT module is to address the requirements for transfer of patient structures and related data defined on CT scanners, virtual simulation workstations, treatment planning systems and similar devices. RTSTRUCT module is described in more detail in the next Section.

RTSTRUCT模块的目的，是处理在CT扫描仪，虚拟仿真工作站，治疗计划系统和类似设备中定义的患者结构和相关数据的传输需求。RTSTRUCT模块在下一节中详述。

The rest of the paper is organized as follows: The next Section describes various modules contained by the RTSTRUCT and their attributes. The implementation details are presented in Section 3. Results on a 3-D CT image of the H&N region are presented in Section 4. Finally, conclusions and future work are presented in Section 5.

本文如下部分组织如下：下一节描述了RTSTRUCT包含各种模块及其属性。实现细节在第3部分中给出。第4部分给出在头颈部区域的3D CT图像。最后，第5部分给出结论和未来工作。

## 2 RTSTRUCT

[2] contains a comprehensive documentation of the DICOM standard; this documentation is vast (spread over 14 documents with more than 2000 pages!) and covers all the modules of the DICOM in detail. Our present work, however, is limited to only the RTSTRUCT and its related modules; further, we use only those attributes of the modules that are necessary for creating a valid and meaningful RTSTRUCT file. Hence, for an easy reference, we present here a summary of the RTSTRUCT.

[2]是DICOM标准的综合文档，这个文档很大（有14个文件，超过2000页），包含了DICOM各个模块的细节。但是，我们目前的工作，只局限于RTSTRUCT及其相关的模块；而且，我们使用的模块的属性，只是那些对于创建有效有意义的RTSTRUCT文件所必须的。因此，为进行简单的参考，我们给出了RTSTRUCT的总结。

The mandatory modules contained by the RTSTRUCT are presented in Table 1. These modules are grouped based on the type of information entity that they represent. We now present a brief description of each of these modules and mention how their attributes are obtained for creating the RTSTRUCT file:

RTSTRUCT必须包含的模块如表1所示。这些模块是基于其表示的信息实体的类型进行分组的。我们现在给出每个模块的简要描述，以及怎样得到相关的属性以创建RTSTRUCT文件：

Table 1: Mandatory modules of RTSTRUCT.

Information Entity (IE) | Mandatory Modules
--- | ---
Patient | (i) Patient
Study | (ii) General Study
Series | (iii) RT Series
Equipment | (iv) General Equipment 
Structure Set | (v) Structure Set (vi) ROI Contour (vii) RT ROI Observations (viii) SOP Common

(i) Patient Module: This module specifies the attributes of the patient that describe and identify the patient who is the subject of a diagnostic study. This module contains attributes of the patient that are needed for diagnostic interpretation of the image and are common for all studies performed on the patient. Table 2 presents the attributes that we use for creating the RTSTRUCT file. A brief description of each attribute along with its DICOM tag are presented in the Table. Patient module is common to both CT image and RT-STRUCT; hence, we extract this information from an input DICOM CT image for creating the RTSTRUCT file.

(i)患者模块：这个模块指定了患者的属性，对患者进行了描述和识别，是诊断study的对象。这个模块包含病人的属性，是图像诊断解释所必须的，是在这个患者上所进行的所有studies通用的。表2给出了我们用于创建RTSTRUCT文件的属性。每个属性的简要描述也在表中给出。患者模块对于CT图像和RT-STRUCT是通用的；因此，我们从输入的DICOM CT图像中提取信息，以用于创建RTSTRUCT文件。

Table 2: Patient module attributes that we use for writing the RTSRTUCT file.

S. No. | Attribute Name | Tag | Attribute Description
--- | --- | --- | ---
1 | Patient’s Name | (0010, 0010) | Patient’s full name.
2 | Patient ID | (0010, 0020) | Hospital ID number for the patient.
3 | Patient’s Birth Date | (0010, 0030) | Birth date of the patient.
4 | Patient’s Sex | (0010, 0040) | Sex of the named patient.

(ii) General Study Module: This module specifies the attributes that describe and identify the study performed upon the patient. Table 3 presents the attributes that we use for creating the RTSTRUCT file.

总体研究模块：这个模块指定了一些属性，描述和识别在病人上做的study。表3给出了我们用于创建RTSTRUCT文件的属性。

A small note on the notation followed: Certain Tables describe sequences of items by using the symbol: ‘>’. The symbol ‘>’ precedes the attribute name of the members of an item. All marked attributes belong to the generic description of an item which may be repeated to form a sequence of items. This sequence of items is nested in the attribute which precedes in the table the first member marked with a ‘>’. For an easy tracing of sequences, a sub-numbering is used in the S.No. column of the Tables.

符号说明：有的项使用了符号>。在属性名称之前会有>符号。有这个标记的属性属于一个项目的通用描述，会在图像序列中重复。有>标记的表格中的项目，其序列项目在属性中是嵌套的。为容易追踪序列，在表格的S.No.列中，有一个数字排序。

General study module is also common to both CT image and RTSTRUCT and hence, we extract this information from an input DICOM CT image for creating the RTSTRUCT file.

通用study模块对CT图像和RTSTRUCT也是通用的，因此，我们从输入DICOM CT图像中提取这个信息，以创建RTSTRUCT文件。

Table 3: General study module attributes that we use for writing the RTSRTUCT file.

S. No | Attribute Name | Tag | Attribute Description
--- | --- | --- | ---
1 | Study Instance UID | (0020, 000d) | Unique identifier for the Study.
2 | Study Date | (0008, 0020) | Date the Study started.
3 | Study Time | (0008, 0030) | Time the Study started.
4 | Referring Physician’s Name | (0008, 0090) | Name of the patient’s ref. physician.
5 | Study ID | (0020, 0010) | Study identifier.
6 | Accession Number | (0008, 0050) | A number to identify the order for the Study.
7 | Study Description | (0008, 1030) | Institution-generated description.
8 | Physician(s) of Record | (0008, 1048) | Names of the physician(s) who are responsible for overall patient care at time of Study
9 | Referenced Study Sequence | (0008, 1110) | A sequence that provides reference to a Study SOP Class/Instance pair.
9.1 | >Referenced SOP Class UID | (0008, 1150) | Uniquely identifies the referenced SOP Class.
9.2 | >Referenced SOP Instance UID | (0008, 1155) | Uniquely identifies the referenced SOP Instance.

(iii) RT Series Module: This module has been created to satisfy the requirements of the standard DICOM Query/Retrieve model while including only those attributes relevant to the identification and selection of radiotherapy objects. Table 4 presents the attributes that we use for creating the RTSTRUCT file.

RT序列模块：这个模块用于满足标准DICOM查询/检索模型的需求，同时包含与放射治疗对象的识别与选择相关的属性。表4给出了我们用于创建RTSTRUCT文件的属性。

Table 4: RT series module attributes that we use for writing the RTSRTUCT file.

S. No | Attribute Name | Tag | Attribute Description
--- | --- | --- | ---
1 | Modality | (0008, 0060) | Type of equipment that originally acquired the data. It contains enumerated values like RTSTRUCT, RTIMAGE, RTDOSE and RTPLAN.
2 | Series Instance User ID | (0020, 000e) | Unique identifier of the series.
3 | Series Number | (0020, 0011) | A number that identifies this series.
4 | Series Description | (0008, 103e) | User provided description of the series.

This module primarily contains the information of the modality (“RTSTRUCT” in this case) and a series instance user ID that uniquely identifies the series. This ID is created using GDCM [3], a library used by ITK for reading and writing DICOM files.

这个模块主要包含模态的信息，和序列实例user ID，可以唯一指定这个序列。这个ID是使用GDCM生成的，ITK使用这个库对DICOM文件进行读写。

(iv) General Equipment Module: It specifies the attributes that identify and describe the piece of equipment that produced a series of composite instances. Table 5 presents the attributes that we use for creating the RTSTRUCT file. These attributes are created by the RTSTRUCT-exporter program.

通用设备模块：这指定了识别和描述产生复合实例序列的设备的属性。表5给出了我们用于创建RTSTRUCT文件的属性。这些属性是由RTSTRUCT导出程序创建的。

Table 5: General equipment module attributes that we use for writing the RTSRTUCT file.

S. No | Attribute Name | Tag | Attribute Description
--- | --- | --- | ---
1 | Manufacturer | (0008, 0070) | Manufacturer of the equipment that produced the composite instances.
2 | Station Name | (0008, 1010) | User defined name identifying the machine that produced the composite instances.

(v) Structure Set Module: This module defines a set of areas of significance. Each area can be associated with a frame of reference and zero or more images. Information which can be transferred with each ROI includes geometrical and display parameters, and generation technique. Table 6 presents the attributes that we use for creating the RTSTRUCT file. These attributes are created by the RTSTRUCT-exporter program.

(v) Structure Set模块：这个模块定义了显著区域的集合。每个区域与一个参考帧和零幅或多幅图像相关。与每个ROI一起可以传递的信息包括，几何参数，和显示参数，和生成技术。表6给出了我们用于创建RTSTRUCT文件的属性。这些属性是用RTSTRUCT导出程序创建的。

(vi) ROI Contour Module: In general, an ROI can be defined by either a sequence of overlays or a sequence of contours. This module is used to define the ROI as a set of contours. Each ROI contains a sequence of one or more contours, where a contour is either a single point (for a point ROI) or more than one point (representing an open or closed polygon). Table 7 presents the attributes that we use for creating the RTSTRUCT file. These attributes are created by the RTSTRUCT-exporter program.

(vi)ROI轮廓模块：一般来说，一个ROI可以通过一个重叠序列或一个轮廓序列来定义。这个模块用于将ROI定义为轮廓集合。每个ROI包含一个序列，即一个或多个轮廓，其中一个轮廓要么是单个点（对于点ROI），或多于一个点（表示一个开放的或闭合的多边形）。表7给出了我们用于创建RTSTRUCT文件的属性。这些属性是通过RTSTRUCT导出程序创建的。

(vii) RT ROI Observations Module: The RT ROI Observations module specifies the identification and interpretation of an ROI specified in the Structure Set and ROI Contour modules. Table 8 presents the attributes that we use for creating the RTSTRUCT file. These attributes are created by the RTSTRUCT-exporter program.

(vii) RT ROI观察模块：这个模块指定了一个ROI的识别和解释，这是在Structure Set和ROI Contour模块中指定的。表8给出了我们用于创建RTSTRUCT文件的属性。这些属性是由RTSTRUCT导出程序来创建的。

(viii) SOP Common Module: The SOP (Service-Object Pair) Common Module is mandatory for all DICOM “Information Object Definitions” (IOD). It defines the attributes which are required for proper functioning and identification of the associated SOP Instances. They do not specify any semantics about the Real-World Object represented by the IOD. Table 9 presents the attributes that we use for creating the RT-STRUCT file. These attributes are created by the RTSTRUCT-exporter program.

(viii) SOP通用模块：这个模块对于所有DICOM IOD都是必须有的。其定义了相关的SOP实例的正常工作和识别所必须的属性。它们没有指定与IOD表示的真实世界目标的语义。表9给出了我们用于创建RTSTRUCT文件的属性。这些属性是由RTSTRUCT程序创建的。

In the next Section, we present the implementation details of our RTSTRUCT-exporter.

下一节中，我们给出了RTSTRUCT导出程序的实现细节。

## 3 Implementation

Fig. 1 illustrates the pipeline that we use for exporting the automated segmentation results to RTSTRUCT format. It mainly contains three steps: (1) Automated Segmentation, (2)Mask to Contour Conversion and (3) RTSTRUCT-Exporter. Below is a detailed description of these steps:

图1给出了我们导出自动分割的结果到RTSTRUCT格式的流程。主要包括3个步骤：(1)自动分割，(2)掩膜到轮廓的转换，(3) RTSTRUCT-Exporter。下面是这些步骤的详述：

### 3.1 Automated Segmentation

The input DICOM CT images are converted into a convenient image format (if required) and an automated segmentation is performed using ITK or similar tools. The output ROIs from this tool should be a mask. There can be multiple masks corresponding to different structures of interest and the current program indeed allows to export multiple masks. It is also possible to export the ROIs obtained on images that are cropped along z-axis; in such cases, the information of starting-slice-number and the number of slices used should be later passed to the RTSTRUCT-exporter module.

输入的DICOM CT图像转换成方便的图像格式，然后用ITK或类似的工具进行自动分割。这些工具输出的ROIs应当是一个mask。可能有不同的masks，对应不同的感兴趣结构，目前的程序也可以导出多个masks。也可以将图像沿着z轴进行剪切，导出这些图像的ROIs，也是可能的；在这样的情况中，开始的slice数值的信息和slice数量，应当传递到RTSTRUCT导出模块中。

The output of this module is passed to the “mask to contour converter”. 这个模块的输出，传入到“mask到轮廓转换器”中。

### 3.2 Mask to Contour Conversion

We first extract axial slices (in case of a 3-D image) of the mask using ExtractImageFilter of ITK. We then use ContourExtractor2DImageFilter [4] of ITK, for obtaining contours on each of these slices. We finally create an output text file containing the information of total number of contours, coordinates of each contour-point along with the corresponding slice number, number of contour points for each contour and type of geometry of each contour (open or closed). We implemented the code for this module in “mask2contour.cxx”. A screen shot of a sample output text file is shown in Fig. 2. For more details, please refer to the sample file submitted with the code.

我们首先使用ITK的ExtractImageFilter提取横断面的mask。然后我们使用ITK的ContourExtractor2DImageFilter，来得到这些slices上的轮廓。我们最后创建一个输出的文本文件，包含轮廓总数的信息，每个轮廓点的坐标值，以及对应的slice索引，每个轮廓的轮廓点数量，和每个轮廓的几何类型（开启的或闭合的）。我们在mask2contour.cxx中实现了这个模块的代码。图2是一个输出文本文件的例子。细节可以查看代码文件。

While writing the contour data, the values of the attributes: “Image Position” & “Image Orientation” in the original DICOM CT images have to be carefully taken into account. It is because, when a DICOM image is converted to other image formats for performing the segmentation, some of the formats (like raw format) do not represent the absolute coordinates of the voxels. Rather, they represent the offset values and thus result in a translation of the contour coordinates compared to the original DICOM image. Here is a brief description of the above mentioned attributes: The Image Position attribute (tag:(0020, 0032)) specifies the x, y and z coordinates of the upper left hand corner of the image; it is the center of the first voxel transmitted. Image Orientation attribute (tag:(0020, 0037)) specifies the direction cosines of the first row and the first column with respect to the patient. These attributes are provided as a pair. Row value for the x, y and z axes are respectively followed by the column value for the x, y and z axes. The direction of the axes is defined fully by the patient’s orientation. The x-axis is increasing to the left hand side of the patient. The y-axis is increasing to the posterior side of the patient. The z-axis is increasing toward the head of the patient. mask2contour.cxx program facilitates the specification of translation parameters in x, y and z directions to address this issue.

在写入轮廓数据时，下列属性的值：原始DICOM CT中的图像位置，图像方向，得到的仔细的考虑。这是因为，当DICOM文件转换成其他图像格式，进行分割时，一些格式（如raw格式）并没有表示体素的绝对坐标。它们表示的是偏移值，因此与原始DICOM图像相比，得到的是轮廓坐标的平移。这里是上面提到的属性的简要描述：图像位置属性，tag:(0020, 0037)，指定的是相对患者的第一行和第一列的方向cosine。这些属性是成对提供的。对于x，y和z轴的行值后面就是x，y和z轴的列值。轴的方向是完全由病人的方向定义的。x轴沿着病人的左手方向是增加的方向。y轴沿着病人向后的方向是增加的方向。z轴朝着病人的头的方向是增加的方向。mask2contour.cxx程序可以很方便的指定x，y和z方向的平移值，以处理这些问题。

Another minor point to be noted is that the vertices values given by the ContourExtractor2DImageFilter are in terms of indexes but not in mm. Hence, these values are multiplied with their respective voxel spacings in the x, y and z directions in the mask2contour.cxx program.

需要说明的另一个小点是，顶点值可以由ContourExtractor2DImageFilter计算出，是以索引为单位的，而不是mm。因此，这些值需要乘以mask2contour.cxx程序里的x，y和z轴相应的体素大小。

The output of this module is passed to the “RTSTRUCT-Exporter” module. 这个模块的输出传入到RTSTRUCT导出模块。

### 3.3 RTSTRUCT-Exporter

RTSTRUCT-Exporter is the main contribution of our work. In this subsection, we first present the design details and features of the newly implemented classes. Then we particulary mention the approach that we use for writing sequences because this may be useful in other contexts as well. We then describe the inputs and outputs to the Exporter. Finally, we summarize the list of files that we added/modified, and their potential locations within the ITK directory structure.

RTSTRUCT-Exporter是我们工作的主要贡献。在这一小节中，我们首先给出设计细节，和新实现的类的特征。然后我们特别提到了，我们用来写序列的方法，因为这在其他上下文中也有用。我们然后描述Exporter的输入和输出。最后，我们总结了我们加入/修改的文件列表，及其在ITK文件夹结构中的可能位置。

Exporting of the contours to RTSTRUCT format requires the implementation of RTSTRUCT-Writer. We implemented the RTSTRUCT-Writer in the “RTSTRUCTIO” class. For creating instances of RTSTRUCTIO objects using an object factory, “RTSTRUCTIOFactory” class is also implemented. Fig. 3 shows the inheritance diagrams for both RTSTRUCTIO & RTSTRUCTIOFactory classes.

将轮廓输出到RTSTRUCT格式，需要实现RTSTRUCT-Writer。我们在RTSTRUCTIO类中实现了RTSTRUCT-Writer。为使用object factory创建RTSTRUCTIO对象的实例，还实现了RTSTRUCTIOFactory类。图3展示了RTSTRUCTIO & RTSTRUCTIOFactory类的继承关系。

Note that these new classes are similar to the existing GDCMImageIO & GDCMImageIOFactory classes. The main difference between GDCMImageIO & RTSTRUCTIO classes is that GDCMImageIO, as the name suggests, is specific to image-based DICOM files whereas the later one is for writing non-image DICOM files. Thus, RTSTRUCTIO class do not contain any image-specific member functions or member variables.

注意，这些新类与现有的GDCMImageIO & GDCMImageIOFactory类是类似的。GDCMImageIO & RTSTRUCTIO类之间的主要差别是，GDCMImageIO就像名字一样，是基于图像的DICOM文件专用的，而后者是用于非图像的DICOM文件的写入用的。因此，RTSTRUCTIO类没有包含任何图像专用的成员函数或成员变量。

Another important difference between GDCMImageIO & RTSTRUCTIO classes is that GDCMImageIO currently cannot write sequence data whereas RTSTRUCTIO can write sequences. Unlike for DICOM image files, many mandatory attributes of the RTSTRUCT have sequence data. Thus it is essential to handle the sequence data for RTSTRUCT files. Writing sequence data to an RTSTRUCT file is a challenging problem for the following reasons: ITK-GDCM layer does not provide any direct interface for transmitting the sequence data. Further, a sequence can contain other sequences and also it can have multiple items; but still, a single MetaDataDictionary has to be passed to the RTSTRUCT-Writer.

GDCMImageIO & RTSTRUCTIO之间另一个重要的区别是，GDCMImageIO目前还不能写序列数据，而RTSTRUCTIO可以写入序列数据。与DICOM图像文件不同的是，RTSTRUCT的很多必须的属性都有序列数据。因此，对于RTSTRUCT文件处理序列数据是非常关键的。

Here is the approach that we use for writing the sequence data: Each sequence is encapsulated inside a MetaDataDictionary, and this MetaDataDictionary is further encapsulated inside the final MetaDataDictionary which is passed to the RTSTRUCT-Writer. As we mentioned above, each sequence can contain multiple items. So for separating the items within a sequence, they are also encapsulated inside a MetaDataDictionary. But for distinguishing the MetaDataDictionary of sequence from the MetaDataDictionary of item, a predefined string-prefix is used for encapsulating items. For instance, the following prefix is used for distinguishing the encapsulation of an item of DICOM sequence from the encapsulation of the DICOM sequence itself.

这里是我们用于写入序列数据的方法：每个序列都封装在一个MetaDataDictionary中，而这个MetaDataDictionary又封装在最后的MetaDataDictionary中，今儿传入到RTSTRUCT-Writer中。就像我们前面提到的，每个序列可以包含多个项目。所以为了将一个序列中的项目分开，他们也封装在一个MetaDataDictionary中。但为了区分序列的MetaDataDictionary和项目的MetaDataDictionary，封装的项目都用了预定义的字符串前缀。比如，下面的前缀用于区分DICOM序列的一个项目的封装和DICOM序列本身的封装：

const std::string ITEM_ENCAPSULATE_STRING("DICOM_ITEM_ENCAPSULATE");

While parsing the final MetaDataDictionary in RTSTRUCT-Writer, whenever the MetaDataDictionary entry is another MetaDataDictionary, it is recursively traversed; items within the sequence are identified using the above defined prefix and the sequences are finally written to the RTSTRUCT-file by calling the GDCM library functions in RTSTRUCT-Writer.

在RTSTRUCT-Writer中将最终的MetaDataDictionary解析时，只要MetaDataDictionary的入口是另外一个MetaDataDictionary，那么就递归的进行traversed；序列中的项目使用上面的预定义前缀进行识别，最后这个序列通过调用RTSTRUCT-Writer中的GDCM库函数写入到RTSTRUCT文件中。

We illustrate the above mentioned approach using a simple example of writing Structure Set ROI Sequence (tag: (3006, 0020)), mentioned in Table 6, to RTSTRUCT. A sample code describing the encapsulation this sequence in export2RTSTRUCT.cxx is presented in Listing 1. A sample code describing the recursive sub-routine that parses a MetaDataDictionary containing a sequence, and writes to RTSTRUCT file is presented in Listing 2.

我们使用一个简单的例子，即将Structure Set ROI序列(tag: (3006, 0020))写入到RTSTRUCT中，描述上述方法。export2RTSTRUCT.cxx中封装这个序列的代码例子如Listing 1所示。解析包含序列的MetaDataDictionary，并写入RTSTRUCT的递归子程序的代码例子，如Listing 2所示。

The inputs to the RTSTRUCT exporter are: RTSTRUCT导出的输入为：

- An axial slice of the DICOM CT image of the patient (for extracting the information that is common to both CT image & RTSTRUCT, as described in Section 2). 患者的轴项slice的DICOM CT图像（为提取CT图像和RTSTRUCT通用的信息，如第2部分描述）；
- Output(s) of the Mask to Contour Converter. (Multiple contours can be exported, as described in Section 2) 掩膜到轮廓转换器的输出（可以输出多个轮廓，如第2部分所述）
- Few additional inputs like starting slice number with respect to the original image, total number of slices to be considered, ROI interpreted types and the colors to be assigned to each ROI. 一些额外的输入，比如相对于原始图像的开始slice数值，要考虑的总slices数量，ROI解义类型和对每个ROI指定的颜色。

All this information is passed to export2RTSTRUCT executable through a parameter-file. A screen shot of a sample parameter-file is shown in Fig. 4. Please refer to the submitted code for more details. 所有这些信息都通过一个参数文件传入到export2RTSTRUCT程序中。图4给出了参数文件的例子的截图。请参考提交的代码的细节。

Finally, we summarize here the list of files that we added/modified, and their potential locations within the ITK directory structure: 最后，我们总结了一下，我们增加/修改的文件列表，以及其在ITK目录结构中可能的位置：

- code/IO/ (added) – itkRTSTRUCTIO.h – itkRTSTRUCTIO.cxx – itkRTSTRUCTIOFactory.h – itkRTSTRUCTIOFactory.cxx
- Utilities/gdcm/src/ (modified) – gdcmFile.cxx – gdcmFileHelper.h – gdcmFileHelper.cxx
- Other executables (added) – mask2contour.cxx – export2RTSTRUCT.cxx

## 4 Example

The DICOM CT image used in this paper is acquired during routine clinical practice at Divisions of Radiotherapy, Fribourg Hospital (HFR), Switzerland. The image is acquired on GE Medical System (Model: LightSpeedRT16). The size of each slice is 512 × 512 pixels with a spacing of 1.269531 × 1.269531 mm; the inter-slice distance is 2.5 mm. There are 116 slices in total.

本文中使用的DICOM CT图像，是在临床日常实践中获得的，在瑞士Fribourg医院放射科。图像是在GE医学系统中获得的，型号LightSpeedRT16。每个slice的大小为512 × 512像素，间距为1.269531 × 1.269531 mm；层厚为2.5mm。总共有116个slices。

Since we are interested only in the first 83 slices of the patient’s image, the original DICOM image is cropped in the z-direction to contain only these slice and a new image file (with .mhd extension) is created. The image is then thresholded in selected regions for removing the bed and other immobilization devices. Fig5 shows the thresholded image. We created separate masks for the external-contour and bones, through simple windowing of the image shown in Fig. 5. These masks are showin in Fig. 6 & Fig. 7.

由于我们只关心患者的前83个slices图像，原始的DICOM图像在z方向剪切，以只包含这些slice，然后创建一个新的图像文件，扩展名为mhd。在选择的区域中，用阈值将床板和固定装置去除。图5展示了阈值处理过后的图像。我们通过简单的窗操作，对外轮廓和骨骼分别创建了masks，如图5所示。这些掩膜如图6和7所示。

The contour data of both masks is obtained using mask2contour. Note that, as mentioned in Section 3.2, we perform a translation of the contour coordinates based on the Image Position attribute (tag:(0020, 0032)). The contours of the masks along with a slice of the DICOM CT image and and other information is passed to export2RTSTRUCT, using the parameter-file submitted along with the code. The correctness of the RTSTRUCT file is verified by inspecting all the tags of the RTSTRUCT file in DicoRView [5], a viewer for DICOM-RT files. DicoRView is a free-tool that provides a very useful tree-view of the attributes. Another free-tool, DICOM Compare [6], is also used for comparing the contents of different DICOM files. The resultant RTSTRUCT file is also superposed over the original DICOM CT Image in a proprietary software tool, and is shown in Fig. 8. The Exporter is found to correctly create the RTSTRUCT file.

两个掩膜的轮廓数据是用mask2contour得到的。注意，如3.2节所示，我们基于图像位置属性（tag:(0020, 0032)），对轮廓坐标系进行平移。掩膜的轮廓，与DICOM CT图像的slice一起，以及其他信息，一起传入export2RTSTRUCT，与代码一起使用参数文件。RTSTRUCT文件的正确性，通过在DicoView中检视RTSTRUCT文件的所有tags进行验证，DicoView是一个DICOM-RT文件的查看器。DicoView是一个免费工具，提供属性的树形视图。另一个免费工具，DICOM Compare，也用于比较不同的DICOM文件的内容。得到的RTSTRUCT文件，与原始DICOM CT图像叠加在一起，如图8所示。我们可以发现，Exporter正确的创建了RTSTRUCT文件。

The command lines for running the example submitted with the code are as follows:
```
mask2contour.exe mask1.mhd contour1.txt 256 256 0
mask2contour.exe mask2.mhd contour2.txt 256 256 0
export2RTSTRUCT.exe parameter_file.txt
```

## 5 Conclusions & Future Work

An ITK implementation of the RTSTRUCT-Exporter is presented. A summary of the RTSTRUCT format, details of the pipeline used and description of each module in the pipeline are presented. The implementation is validated on a 3-D H&N CT image, by exporting the ROIs of the external-contour and bones to RTSTRUCT format.

我们给出了RTSTRUCT-Exporter的ITK实现。RTSTRUCT格式的总结，流程的细节和每个模块的描述，都在本文中给出了。其实现在一个3D头颈部CT图像中得到的验证，将其外轮廓和骨骼的ROIs导出到RTSTRUCT格式。

Some of the possible extensions of the current work are listed below: 目前工作的可能的拓展列在下面：

- RTSTRUCT-Exporter is tested only on the DICOM CT images acquired from a GE Medical System (Model: LightSpeedRT16) at Fribourg Hospital, Switzerland. A thorough testing on more images, acquired from various manufacturers and models, will make it more robust. RTSTRUCT-Exporter只在瑞士Fribourg医院的GE医疗系统(Model: LightSpeedRT16)上获得的DICOM CT图像进行了测试。对各种厂商和模型获取的图像的完全测试，会使其更加稳健。

- At present, only the RTSTRUCT-Writer is implemented but not the RTSTRUCT-Reader. Development of the RTSTRUCT-Reader will be of interest for many general applications. 目前，只有RTSTRUCT-Writer进行了实现，但RTSTRUCT-Reader还没有。RTSTRUCT-Reader的开发会是很多通用应用的兴趣。

- RTSTRUCTIO currently inherits ImageIOBase class; however, this does not seem to be the ideal choice since RTSTRUCT is not an image as such. RTSTRUCTIO只继承了ImageIO类；但是，这可能不是很理想的选择，因为RTSTRUCT不是一幅图像。

- There is no direct/simple interface at ITK-layer level, for handling the DICOM sequences. Such interface will be very useful for those who are using ITK for processing DICOM files. 在ITK层的层次，没有直接的/简单的接口，以处理DICOM序列。这样的接口对于那些使用ITK处理DICOm文件的会非常有用。

- A complete checker that confirms/warns the user about the presence/correctness of the mandatory attributes of the RTSTRUCT file will be definitely a good contribution to the DICOM/ITK community. 对于RTSTRUCT文件的强制属性的存在性/正确性的完全检查，并确认和警告，对于DICOM/ITK团体会是一个非常好的贡献。