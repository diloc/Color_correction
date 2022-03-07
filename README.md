# ColorBayes Bayesian color constancy


Our goal is to correct the local color distortion on plant phenotyping images caused by non-uniform illumination. The corrected image will show the colors of individual plants as if they were taken under the same standard illuminant (D65). This color constancy approach has two main steps. The first step is to estimate an unknown illuminant's color and spatial distribution that causes the local color distortion. For this step, it is required a training dataset (ground truth), observed image data. Also, it is used the Bayes' rule and the maximum a posteriori (MAP). The second step is to transform the observed image using the chromaticity adaptation method.


<figure>
  <img src="https://github.com/diloc/Color_correction/blob/main/images/Figure_2_ColorLight_distribution4.png">
  <figcaption>
  Figure 1 Light environment characteristics in the HTPP system. (a) A top-view image capturing one hundred and forty-eight pots and twelve Macbeth ColorCheckers, illustrating non-uniform illumination. (b) Spatial distribution of the illumination color on the plant phenotyping scene. (c) Illumination color distribution on the Chromaticity diagram using the 1931 CIE colorimetric system coordinates. (d) Spectral Irradiance at (x = 20 cm and y = 20 cm) and (x = 0 cm and y = 6 cm), the wavelength range is from 325 nm to 785 nm on the horizontal axis and the spectral irradiance [μW cm-2 nm-1] on the vertical axis. Two peaks represent the primary emission LED lights at blue (450 nm) and red (630 nm) and a curve plateau between these peaks (550 - 600 nm).
  </figcaption>
</figure>

## Description

Our methods relies on the following assumptions:

- An observed image (5105 x 3075 pixels) is made up of three independent color channels <(k = {R,G,B})> and divided into areas that correspond to individual pot areas.
- A pixel class is assigned individually to segmented objects in a pot area such as plant and soil pixels. It means that a pixel class is a collection of pixels ![equation](https://github.com/diloc/Color_correction/blob/main/equations/pixelClass.png) within the same spatial neighborhood and similar color values. The pixel value ![equation](https://github.com/diloc/Color_correction/blob/main/equations/pixel.png), is a random variable at location i=0,1,2,…,n.
- The reflectance of a pixel class is a collection of reflectance ![equation](https://github.com/diloc/Color_correction/blob/main/equations/reflectanceClass.png), where ![equation](https://github.com/diloc/Color_correction/blob/main/equations/reflect.png), is a random variable representing the reflectance at the location i=0,1,2,…,n. Two adjacent reflectance are independent of each other, and the joint probability of is given by ![equation](https://github.com/diloc/Color_correction/blob/041befc0d5f053fd72862480c8d15fc4a464f010/equations/twoReflect.png). Based on the same assumption, all reflectance in a pixel class are independent events with joint probability ![equation](https://github.com/diloc/Color_correction/blob/041befc0d5f053fd72862480c8d15fc4a464f010/equations/allReflect.png).
- The illuminant of a pixel class is a collection of illuminants ![equation](https://github.com/diloc/Color_correction/blob/main/equations/illumClass.png),. However, it is assumed that the illuminant is constant for all pixels in a class, meaning, ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/lk_lki.png). Then, the probability distribution of the illuminant is uniform, ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/prob_illum.png), being ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/constant.png) a constant value. 
- The illumination and the reflectance are statistically independent of each other ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/prob_refl_illum.png).
- The value of a pixel ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/pixel.png) is a function of the reflectance ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/reflect.png), the illuminant ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/illumin.png) and the Gaussian noise w_ki with a mean equal to zero and variance ![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/var_noise.png) (Eq. 1). <br/>



![equation](https://github.com/diloc/Color_correction/blob/c42493778cf3d81cb2be0e80d740012cee213f03/equations/pixelFunc.png)	              Eq. 1 <br/>

The multivariable function described in Eq. 1 can be statistically represented using the likelihood function. It is equivalent to Gaussian noise probability distribution (Eq. 2). <br/>

![equation](https://github.com/diloc/Color_correction/blob/64641311ebcfd22add59b5f9db0430e8ccd500d0/equations/likelihood.png)               Eq. 2 <br/>


### Priors: Reflectance & Illuminant: 
We created an **image dataset** to get the reflectance and illuminant prior distributions. It has images of green fabric pieces on pots and Macbeth colorChecker charts. They were illuminated using D65 standard illuminant. <br/>

![equation](https://github.com/diloc/Color_correction/blob/67d3eb7d24be12a07f11351454e3983ae2ba2498/equations/priorReflect.png)               Eq. 3 <br/>
As the illumation is uniform over a pixel class the probability distribution is given by:<br/>
![equation](https://github.com/diloc/Color_correction/blob/b99b19f530fbd06e94a81c221153e5a50614ace0/equations/priorIllum.png)               Eq. 4 <br/>

### Maximum a posteriori 
We estimate the illumination value when the posterior distribution reaches the highest value. <br/>

![equation](https://github.com/diloc/Color_correction/blob/d017b24942fae440ff7fccf4500cfc8ea158f8c6/equations/MAP.png)               Eq. 5 <br/>
## Resources


* Source Repository (prior): https://github.com/diloc/Color_correction/blob/8dea8b92ac3cea5e5c198348c04a50d10c2f8adb/Color_Constancy/prior.ipynb
* Source Repository (main): https://github.com/diloc/Color_correction/blob/919477408e1679f0c3c715a99ab9bff2afca433f/Color_Constancy/main.ipynb

## Dependencies
* Python (3. 7 or higher).
* Pandas (1.0.3 or higher).
* OpenCV (4.2.0 or higher).
* Datetime
* Scipy (1.4.1 or higher).
* Matplotlib (1.18.1 or higher).



# Results
The ColorBayes algorithm improved the accuracy of plant color on images taken by an indoor plant phenotyping system. Compared with existing approaches, it gave the most accurate metric results when correcting images from a dataset of Arabidopsis thaliana images.




