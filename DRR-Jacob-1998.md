# A fast algorithm to calculate the exact radiological path through a pixel or voxel space

Filip Jacobs, et. al. University of Ghent

## 0. Abstract

Calculating the exact radiological path through a pixel or voxel space is a frequently encountered problem in medical image reconstruction from projections and greatly influences the reconstruction time. Currently, one of the fastest algorithms designed for this purpose was published in 1985 by Robert L. Siddon [1]. In this paper, we propose an improved version of Siddon's algorithm, resulting in a considerable speedup.

Keywords: reconstruction, projection, radiological path

## 1. Introduction

In image reconstruction from projections we calculate an image from a given set of measurements which are related to line-integrals over the image [2,3]. Examples of such images can be found in PET, CT and MRI studies where the images represent the distribution of an administere radio-active tracer, the distribution of the linear attenuation coefficients of tissue and the distribution of protons, respectively, in cross-sections of a patient.

In order to simplify the notations, we will restrict ourselves to a 2D-description of the algorithm. The theory can easily be extended to rays lying in a 3D space. The results presented in the last section of this paper, however, are based on both a 2D- and 3D- implementation of the algorithms.

Prior to the reconstruction, the image is discretized into pixels and the line-integrals into weighted sums. The weighting factor for the value $ρ(i,j)$ of pixel (i,j) is denoted by l(i,j) and equals the intersection length of the ray with pixel (i,j). Hence, the line-integral from point $p(p_{1x},p_{1y})$ to $p(p_{2x}, p_{2y})$, over the discretized image, can be approximated by the following weighted sum:

$$d_{12} = \sum_{(i,j)} l(i,j)ρ(i,j)$$(1)

Because of the huge amount of measurements given by a medical scanner and the large number of pixels, it is impossible to store all weighting factors l(i,j) in a file prior to the reconstruction. Therefore, they have to be calculated on the fly which greatly limits the reconstruction time. A fast algorithm to evaluate equation (1) is a necessity to obtain acceptable reconstruction times.

Currently, one of the fastest algorithms designed for this purpose was published in 1985 by Robert L. Siddon[1]. We improved his algorithm in a way that the time spent in the inner loop is reduced considerably, and the reconstruction time accordingly (also see [4]). In the first section we introduce the used notations. In the following section, we review Siddon's algorithm for rays lying in a 2D plane and give the basic formulas needed to explain the improved algorithm in the subsequent section. In the final section, we compare the two algorithms for rays lying in a 2D plane as well as for rays lying in a 3D space by comparing the obtained reconstruction times for 3D (i.e a stack of several 2D planes) and fully 3D (i.e. oblique raysums are also available) PET images.

## 2. Notations

The pixel space is determined by the intersection of two sets of equally spaced parallel planes, perpendicular to an x- and an y-axis (see Figure 1). The x- and y-axis are perpendicular to each other. The two sets will be referred to as the x- and y-planes, respectively. The number of x-planes equals $N_x$ and the distance between them is denoted by $d_x$. The x-planes are numbered from 0 to $N_x-1$. Similar notations hold for the y-planes. Hence, the pixel values $ρ(i,j)$ have indices running from (0,0) to ($N_x-2, N_y-2$). The lowest left corner of the pixel space, i.e. the intersection point of x-plane 0 and y-plane 0, has coordinates ($b_x, b_y$). Each ray goes from a point denoted by $p_1 = p(p_{1x}, p_{1y})$ to another point denoted by $p_2 = p(p_{2x}, p_{2y})$.

## 3. Siddon's Algorithm

We could evaluate equation (1) by summing over all (i,j). This would be very inefficient, as pointed by Siddon, because most l(i,j) are zero. It is more efficient to follow the ray through the pixel space. Therefore, we use a parametrical representation of the ray,

$$p_{12} = \{ \begin{matrix} p_x(α)=p_{1x}+α(p_{2x}-p_{1x}) \\ p_y(α)=p_{1y}+α(p_{2y}-p_{1y}) \end{matrix}$$(2)

with α ∈ [0, 1] for points between $p_1$ and $p_2$ and α $\notin$ [0, 1] for all other points. In what follows, we will assume that the ray is generic, i.e. that $p_{1x} \neq p_{2x}$ and $p_{1y} \neq p_{2y}$. Non-generic rays are trivial to handle and will not be discussed further. Following Siddon's algorithm, we first determine the entry point ($α = α_{min}$) and exit point ($α = α_{max}$) of the ray (see Fig. 1). Equation (9) calculates the α parameter corresponding to the intersection of the i-th x-plane and the line going through $p(p_{1x}, p_{2x})$ and $p(p_{1y}, p_{2y})$, hence, these values are not restricted to the interval [0, 1].

$$α_{min} = max(α_{xmin}, α_{ymin})$$(3)

$$α_{max} = min(α_{xmax}, α_{ymax})$$(4)

with

$$α_{xmin} = min(α_x(0), α_x(N_x-1))$$(5)

$$α_{xmax} = max(α_x(0), α_x(N_x-1))$$(6)

$$α_{ymin} = min(α_y(0), α_y(N_y-1))$$(7)

$$α_{ymax} = max(α_y(0), α_y(N_y-1))$$(7)

and

$$α_x(i) = \frac {(b_x+id_x)-p{1x}} {p_{2x}-p{1x}}$$(9)

$$α_y(j) = \frac {(b_y+jd_y)-p{1y}} {p_{2y}-p{1y}}$$(10)

Given that the ray does intersect the pixel space, i.e. $α_{min}<α_{max}$, we calculate the number of the first intersected x-plane $i_f$ after the ray entered the pixel space and the number of the last intersected x-plane $i_l$ including the outer plane. We will use the variables $i_{min} = min(i_f, i_l)$ and $i_{max} = max(i_f, i_l)$ to simplify the following formulas. Similar definitions hold for $j_{min}$ and $j_{max}$ concerning the y-planes. Whenever $p_{1x}<p_{2x}$ we calculate $i_{min}$ and $i_{max}$ with equations (11)-(14) and otherwise with equations (15)-(18). The definition of $ϕ_x(α)$ is given by equation (19). Similar formulas hold for $j_{min}$ and $j_{max}$.

$$α_{min} = α_{xmin} -> i_{min} = 1$$(11)

$$α_{min} \neq α_{xmin} -> i_{min} = ⌈ϕ_x(α_{min})⌉$$(12)

$$α_{max} = α_{xmax} -> i_{max} = N_x - 1$$(13)

$$α_{max} \neq α_{xmax} -> i_{max} = ⌊ϕ_x(α_{max})⌋$$(14)

$$α_{min} = α_{xmin} -> i_{max} = N_x - 2$$(15)

$$α_{min} \neq α_{xmin} -> i_{max} = ⌊ϕ_x(α_{min})⌋$$(16)

$$α_{max} = α_{xmax} -> i_{min} = 0$$(17)

$$α_{max} \neq α_{xmax} -> i_{min} = ⌈ϕ_x(α_{max})⌉$$(18)

$$ϕ_x(α) = \frac {p_x(α)-b_x}{d_x}$$(19)

Furthur, we calculate two arrays $α_x[⋅]$ and $α_y[⋅]$ holding the parametric values of the intersection points of the ray with the x- resp. y-planes, after the ray enter the pixel space. If $p_{1x} < p_{2x}$ the first array is given by equation (20) and otherwise by equation (21). Similar formulas are used to calculate the second array.

$$α_x[i_{min}⋅⋅⋅i_{max}] = (α_x(i_{min}), α_x(i_{min}+1), ⋅⋅⋅, α_x(i_{max}))$$(20)

$$α_x[i_{max}⋅⋅⋅i_{min}] = (α_x(i_{max}), α_x(i_{max}-1), ⋅⋅⋅, α_x(i_{min}))$$(21)

Subsequently, we sort the elements of ($α_{min}, α_x[⋅], α_y[⋅]$) in ascending order and replace all values that occur twice by one copy of the value, resulting in the array $α_{xy}[0⋅⋅⋅N_v]$ holding the parametric values of all intersected points. The occurence of dual α-values is due to the simultaneous intersection of an x-plane, a y-plane and the ray. Given the $α_{xy}[⋅]$ array, we calculate the coordinates ($i_m, j_m$) of the intersected pixels with equations (22)-(23) and their intersection lengths $l(i_m, j_m)$ with equation (24) for all m ∈ [1⋅⋅⋅N_v]. The variable d_{conv} equals the Euclidean distance between the points $p_1$ and $p_2$. The pixels (i,j) which do not correspond to a certain ($i_m, j_m$) are not intersected.

$$i_m = ⌊ϕ_x(\frac {α_{xy}[m]+α_{xy}[m-1]}{2})⌋$$(22)
$$j_m = ⌊ϕ_y(\frac {α_{xy}[m]+α_{xy}[m-1]}{2})⌋$$(23)

$$l(i_m,j_m) = (α_{xy}[m]- α_{xy}[m-1])d_{conv}$$(24)

After implementing and profiling Siddon's algorithms, we found that its speed is greatly limited by the frequent use of equations (22) and (23) where floating point values are converted into integer values. In the following section we present an altered algorithm, based on Siddon's algorithm, which restricts the use of these equations to once for each ray.

## 4. Improved algorithm

As pointed out in the above section, frequent use of equations (22) and (23) limits the speed of Siddon's algorithm. In this section we propose an improved algorithm which restricts the use of these equations to once for each ray. It also obviates the need to allocate memory for the different α-arrays.

We follow Siddon's approach until the values of $α_{min}$ and $α_{max}$ are calculated. Starting from here, our approach differs from the one used by Siddon. Instead of calculating the arrays $α_x[⋅]$ and $α_y[⋅]$, we only calculate the values $α_x = α_x[0]$ and $α_y = α_y[0]$, i.e. the parametric value of the first intersection point of the ray with the x- resp. y-planes, after the ray entered the pixel space.

We also calculate the values $i_{min}$ and $i_{max}$ given by equations (11)-(18) and $j_{min}$ and $j_{max}$ given by similar equations. These values are used to calculate the number of planes $N_p$ crossed by the ray when it runs through the pixel space, after it entered the pixel space, i.e.

$$N_p = (i_{max}-i_{min}+1) + (j_{max}-j_{min}+1)$$(25)

Note that $N_p>N_v$ because the simultaneous crossing of an x-plane, a y-plane and the ray is subdivided into two separate events, i.e. the crossing of an x-plane and then the crossing of a y-plane, or vice versa. We only use equations (22) and (23) to calculate the indices (i,j) of the first intersected pixel, i.e.

$$i = ⌊ϕ_x(\frac {min(α_x, α_y)+α_{min}} {2})⌋$$(26)

$$j = ⌊ϕ_y(\frac {min(α_x, α_y)+α_{min}} {2})⌋$$(27)

Following the ray through the pixel space, we have to update the values of $α_x, α_y, i$ and j according to whether we cross an x- or y-plane. Whenever $α_x<α_y$ the next intersected plane is an x-plane and we increase $α_x$ and i with $α_{xu}$ resp. $i_u$, given by equations (28) and (29). Similar equations hold for updating $α_y$ and j when the ray crosses a y-plane, i.e. $α_y<α_x$. If $α_x≡α_y$, then we can use either case to update the variables.

$$α_{xu} = \frac {d_x} {|p_{2x}-p_{1x}|}$$(28)
$$i_u = \{ \begin{matrix} 1 & if \space p_{1x}<p_{2x} \\ -1 & else \end{matrix}$$(29)

Finally, after initializing $d_{12}$ to 0 and $α_c$ to $α_{min}$, we are able to calculate the radiological path by running $N_v$ times through the following algorithm: if $α_x<α_y$ then calculate l(i,j) with (30) and update $d_{12}, i, α_c$ and $α_x$ with equations (31)-(34), else calculate l(i,j) with (35) and update $d_{12}, j, α_c$ and $α_y$ with equations (36)-(39).

$$l(i,j) = (α_x-α_c)d_{conv}$$(30)
$$d_{12} = d_{12} + l(i,j)ρ(i,j)$$(31)
$$i = i+i_u$$(32)
$$α_c = α_x$$(33)
$$α_x = α_x + α_{xu}$$(34)

$$l(i,j) = (α_y-α_c)d_{conv}$$(35)
$$d_{12} = d_{12} + l(i,j)ρ(i,j)$$(36)
$$j = j+j_u$$(37)
$$α_c = α_y$$(38)
$$α_y = α_y + α_{yu}$$(39)

Besides the calculation of raysums, some algorithms also need the exact intersection lengths l(i,j) to calculate something else, e.g. the backprojection of an image. It is for these algorithms that we formulated (30-39). Algorithms which do not need the explicit calculation of intersection lengths should incorporate (30) and (35) into (31) resp. (36) without the multiplication with $d_{conv}$. The multiplication of $d_{12}$ can be done afterwards.

## 5. Evaluation and Results

In order to compare the two algorithms in realistic situations, we chose the reconstruction of 3D and fully 3D Positron Emission Tomography (PET) images with the Maximum Likeihood Expectation Maximization (MLEM) algorithm [5].

3D PET data consists of a set of sinograms. Each sinogram corresponds to a spatial plane through the patient. Each element of a sinogram corresponds to a raysum of a ray through the spatial plane and is determined by its angle with respect to the x-axis of a 2D Cartesian xy-coordinate system and its distance to the origin. Because each sinogram corresponds to a 2D image, 3D PET is actually a 2D problem. For the evaluation, we used a data set obtained with an ECAT 951 PET-scanner consisting of 31 sinograms of 256 angles and 192 distances each. The 31 reconstructed images have dimensions 192 by 192.

Fully 3D PET data consists of a set of data planes. Each data plane corresponds to a spatial plane through the origin of a 3D Cartesian xyz-coordinate system and is determind by its tilt with respect to the xy-plane and its angle with respect to the xz-plane. Each element of a data plane correspond to a raysum of a ray perpendicular to the spatial plane and is determined by two Cartesian coordinates. Because raysums are available for rays with different tilts, fully 3D PET is a real 3D problem. For the evaluation we used a data set calculated by the software package eval3dpet[6,7]. The data planes in the data set correspond to 15 tilts and 96 angles and have dimensions of 90 by 128. The reconstructed image has dimensions 64 by 128 by 128.

The MLEM algorithms have been implemented in C on a Sun Ultra 2 Creator with 2 Ultra-SPARC processors. In Table 1 we compare the reconstruction times for 5 iterations for the 3D PET case and 1 iteration for the fully 3D PET case. We observe that speedups between 3 and 5 are obtained. The speedup for fully 3D PET is smaller than the one for 3D PET because only a smaller fraction of the total reconstruction time is used to calculate the radiological paths, i.e. 37% instead of 57%.

We found that the improvement of Siddon's algorithm resulted in a speedup of 7.5 for the calculation of radiological paths and in a speedup 5.0 for the total reconstruction time in the case of 3D. The time used to calculate the raysums was reduced from 89% to 57% which emphasises the importance of reducing the time spent on calculating the raysums and/or the intersection lengths.

Fig. 1 A schematic overview of the used notations. The pixel values are denoted by $ρ(i,j)$ and a point in the 2D plane is referred to as p(x,y). The α-variable holds the relative distance of a point on the line through $p(p_{1x}, p_{1y})$ and the point $p(p_{2x}, p_{2y})$ and the point $p(p_{1x}, p_{1y})$. The α-variable equals 1 for the point $p(p_{2x}, p_{2y})$ and any value between 0 and 1 for points between $p(p_{1x}, p_{1y})$ and $p(p_{2x}, p_{2y})$. The other variables hold real distances.

Table 1. A comparison of the algorithm of Siddon and the improved algorithm by comparing the reconstruction times (in min.) of a 3D PET image after 5 iterations and a fully 3D PET image after 1 iteration.