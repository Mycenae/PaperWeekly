# What Is Camera Calibration?

Geometric camera calibration, also referred to as camera resectioning, estimates the parameters of a lens and image sensor of an image or video camera. You can use these parameters to correct for lens distortion, measure the size of an object in world units, or determine the location of the camera in the scene. These tasks are used in applications such as machine vision to detect and measure objects. They are also used in robotics, for navigation systems, and 3-D scene reconstruction.

相机几何标定，也称为相机resectioning，估计相机或摄像机的镜片和图像传感器的参数。你可以用这些参数来修正镜片失真，以世界单位来衡量目标的大小，或确定场景中相机的位置。这些任务在机器视觉应用中有应用，以检测和衡量目标；也在机器人中有应用，进行导航和3D场景重建。

Examples of what you can do after calibrating your camera: 相机标定后可以做的事的例子如下：

![](https://ww2.mathworks.cn/help/vision/ug/calibration_applications.png)

Camera parameters include intrinsics, extrinsics, and distortion coefficients. To estimate the camera parameters, you need to have 3-D world points and their corresponding 2-D image points. You can get these correspondences using multiple images of a calibration pattern, such as a checkerboard. Using the correspondences, you can solve for the camera parameters. After you calibrate a camera, to evaluate the accuracy of the estimated parameters, you can:

相机参数包括内部参数，外部参数和形变参数。为估计这些相机参数，需要有3D世界的点及其对应的2D图像中的点。你可以用一种校准模式（如棋盘格）的多幅图像得到这些对应性。使用这些对应性，就可以对相机参数进行求解。在标定了相机后，为评估估计参数的准确性，你可以：

- Plot the relative locations of the camera and the calibration pattern. 画出相机和标定模式的相对位置；

- Calculate the reprojection errors. 计算重投影误差；

- Calculate the parameter estimation errors. 计算参数估计的误差；

Use the **Camera Calibrator** to perform camera calibration and evaluate the accuracy of the estimated parameters. 使用**Camera Calibrator**来进行相机标定，以及评估估计参数的准确性。

## 1. Camera Model

The Computer Vision Toolbox™ calibration algorithm uses the camera model proposed by Jean-Yves Bouguet [3]. The model includes: 计算机视觉工具箱的标定算法使用的是Jean-Yves Bouguet [3]提出的相机模型，这个模型包括：

- The pinhole camera model [1]. 针孔相机模型；

- Lens distortion [2]. 镜片变形；

The pinhole camera model does not account for lens distortion because an ideal pinhole camera does not have a lens. To accurately represent a real camera, the full camera model used by the algorithm includes the radial and tangential lens distortion.

针孔相机模型没有考虑到镜片形变，因为理想的针孔相机并没有镜片。为准确的表达一个真正的相机，算法使用的完整相机模型包含镜片径向和切向形变。

## 2. Pinhole Camera Model

A pinhole camera is a simple camera without a lens and with a single small aperture. Light rays pass through the aperture and project an inverted image on the opposite side of the camera. Think of the virtual image plane as being in front of the camera and containing the upright image of the scene.

针孔相机是一个简单的相机，没有镜片，只有一个很小的小孔。光线通过小孔，并在相机另外一边投影出一个倒立的图像。设想虚拟图像平面是在相机前面，包含场景的直立图像。

![](https://ww2.mathworks.cn/help/vision/ug/camera_calibration_focal_point.png)

The pinhole camera parameters are represented in a 4-by-3 matrix called the camera matrix. This matrix maps the 3-D world scene into the image plane. The calibration algorithm calculates the camera matrix using the extrinsic and intrinsic parameters. The extrinsic parameters represent the location of the camera in the 3-D scene. The intrinsic parameters represent the optical center and focal length of the camera.

针孔相机的参数表示为一个4✖3矩阵，称为相机矩阵；这个矩阵将3-D世界的场景映射到图像平面上。标定算法用内部参数和外部参数来计算相机矩阵。外部参数表示相机在3D场景中的位置，内部参数表示光学中心和相机的焦距。

![](https://ww2.mathworks.cn/help/vision/ug/calibration_camera_matrix.png)

The world points are transformed to camera coordinates using the extrinsics parameters. The camera coordinates are mapped into the image plane using the intrinsics parameters.

世界坐标系中的点通过外部参数变到相机坐标系中，相机中的坐标通过内部参数映射到图像平面。

![](https://ww2.mathworks.cn/help/vision/ug/calibration_cameramodel_coords.png)

## 3. Camera Calibration Parameters

The calibration algorithm calculates the camera matrix using the extrinsic and intrinsic parameters. The extrinsic parameters represent a rigid transformation from 3-D world coordinate system to the 3-D camera’s coordinate system. The intrinsic parameters represent a projective transformation from the 3-D camera’s coordinates into the 2-D image coordinates.

标定算法使用外部参数和内部参数来计算相机矩阵。外部参数表示的是，从3D世界坐标系，到3D相机坐标系的刚性变换。内部参数表示的是，从3D相机坐标系到2D图像坐标系的投影变换。

![](https://ww2.mathworks.cn/help/vision/ug/calibration_coordinate_blocks.png)

**Extrinsic Parameters**

The extrinsic parameters consist of a rotation, R, and a translation, t. The origin of the camera’s coordinate system is at its optical center and its x- and y-axis define the image plane.

外部参数由一个旋转矩阵R，平移矢量t组成。相机的坐标系统的原点是在其光学中心的，其x轴和y轴定义了图像平面。

![](https://ww2.mathworks.cn/help/vision/ug/calibration_rt_coordinates.png)

**Intrinsic Parameters**

The intrinsic parameters include the focal length, the optical center, also known as the principal point, and the skew coefficient. The camera intrinsic matrix, K, is defined as:

内部参数包含焦距、光学中心（也称为主点）和倾斜系数。相机的内部矩阵K，定义为：

$$\left[ \begin{matrix} f_x & 0 & 0 \\ s & f_y & 0 \\ c_x & c_y & 1 \end{matrix} \right]$$

The pixel skew is defined as: 像素的倾斜定义为：

![](https://ww2.mathworks.cn/help/vision/ug/calibration_skew.png)

[$c_x, c_y$] - Optical center (the principal point), in pixels. 光学中心（主点），单位为像素。

($f_x, f_y$) - Focal length in pixels. 以像素为单位的焦距。

f_x = F/p_x

f_y = F/p_y

F - Focal length in world units, typically expressed in millimeters. 以世界度量为单位的焦距，一般是毫米。

($p_x, p_y$) - Size of the pixel in world units. 以世界度量为单位的像素大小。

s - Skew coefficient, which is non-zero if the image axes are not perpendicular. 倾斜系数，如果图像轴不是垂直的话，就是非零的。

$s = f_x tanα$

## 4. Distortion in Camera Calibration

The camera matrix does not account for lens distortion because an ideal pinhole camera does not have a lens. To accurately represent a real camera, the camera model includes the radial and tangential lens distortion.

相机矩阵并没有考虑到镜片形变，因为理想的针孔相机并没有镜片。为准确的表示一个真正的相机，相机模型需要包含径向和切相镜片形变。

**Radial Distortion**

Radial distortion occurs when light rays bend more near the edges of a lens than they do at its optical center. The smaller the lens, the greater the distortion. 当光线在镜片的边缘处，比在其光学中心弯曲的更多时，就会发生径向形变。镜头越小，形变越大。

![](https://ww2.mathworks.cn/help/vision/ug/calibration_radial_distortion.png)

The radial distortion coefficients model this type of distortion. The distorted points are denoted as ($x_{distorted}, y_{distorted}$):

径向形变系数对这种形变进行建模。形变点表示为($x_{distorted}, y_{distorted}$)：

$$x_{distorted} = x(1 + k_1*r^2 + k_2*r^4 + k_3*r^6)$$

$$y_{distorted} = y(1 + k_1*r^2 + k_2*r^4 + k_3*r^6)$$

- x, y — Undistorted pixel locations. x and y are in normalized image coordinates. Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels. Thus, x and y are dimensionless. 未形变的像素位置。x和y是在归一化的图像坐标系中的。归一化的图像坐标系，是将像素坐标系平移到光学中心，并除以以像素为单位的焦距得到的。因此，x和y都是没有维度的。

- $k_1, k_2$, and $k_3$ — Radial distortion coefficients of the lens. 镜片的径向形变系数。

- $r^2 = x^2 + y^2$.

Typically, two coefficients are sufficient for calibration. For severe distortion, such as in wide-angle lenses, you can select 3 coefficients to include $k_3$.

一般，两个系数就足够进行标定了。对于严重的形变，比如广角镜片，可以选择3个系数，将$k_3$也包含进来。

**Tangential Distortion**

Tangential distortion occurs when the lens and the image plane are not parallel. The tangential distortion coefficients model this type of distortion.

切向形变，当镜片和图像平面并不平行的时候发生。切向形变系数对这种形变进行建模。

![](https://ww2.mathworks.cn/help/vision/ug/calibration_tangentialdistortion.png)

The distorted points are denoted as ($x_{distorted}, y_{distorted}$): 形变的点表示为($x_{distorted}, y_{distorted}$)：

$$x_{distorted} = x + [2 * p_1 * x * y + p_2 * (r^2 + 2 * x^2)]$$

$$y_{distorted} = y + [p_1 * (r^2 + 2 * y^2) + 2 * p_2 * x * y]$$

- x, y — Undistorted pixel locations. x and y are in normalized image coordinates. Normalized image coordinates are calculated from pixel coordinates by translating to the optical center and dividing by the focal length in pixels. Thus, x and y are dimensionless. 为形变的像素位置。x和y是在归一化的图像坐标系中的。归一化的图像坐标系，是将像素坐标平移到光学中心，然后除以以像素为单位的焦距得到的。因此，x和y都是没有维度的。

- $p_1$ and $p_2$ — Tangential distortion coefficients of the lens. 镜片的切向形变系数。

- $r^2 = x^2 + y^2$.

## 5. References

[1] Zhang, Z. “A Flexible New Technique for Camera Calibration.” IEEE Transactions on Pattern Analysis and Machine Intelligence. Vol. 22, No. 11, 2000, pp. 1330–1334.

[2] Heikkila, J., and O. Silven. “A Four-step Camera Calibration Procedure with Implicit Image Correction.” IEEE International Conference on Computer Vision and Pattern Recognition.1997.

[3] Bouguet, J. Y. “Camera Calibration Toolbox for Matlab.” Computational Vision at the California Institute of Technology. Camera Calibration Toolbox for MATLAB.

[4] Bradski, G., and A. Kaehler. Learning OpenCV: Computer Vision with the OpenCV Library. Sebastopol, CA: O'Reilly, 2008.