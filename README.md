# ColorBayes Bayesian color constancy


We aim to correct the local colour distortion on plant phenotyping images caused by non-uniform illumination. The corrected image will show the colours of individual plants as if they were taken under the same standard illuminant (D65). This colour constancy approach has two main steps. The first step is to estimate an unknown illuminant's colour and spatial distribution that causes the local colour distortion. For this step, it is required a training dataset (ground truth), and observed image data. Also, it is used the Bayes' rule and the maximum a posteriori (MAP). The second step is transforming the observed image using the chromaticity adaptation method.


<figure>
  <img src="https://github.com/diloc/Color_correction/blob/main/images/Figure_2_ColorLight_distribution4.png">
  <figcaption>
  Figure 1 Light environment characteristics in the HTPP system. (a) A top-view image capturing one hundred and forty-eight pots and twelve Macbeth ColorCheckers, illustrating non-uniform illumination. (b) Spatial distribution of the illumination colour on the plant phenotyping scene. (c) Illumination colour distribution on the Chromaticity diagram using the 1931 CIE colourimetric system coordinates. (d) Spectral Irradiance at (x = 20 cm and y = 20 cm) and (x = 0 cm and y = 6 cm), the wavelength range is from 325 nm to 785 nm on the horizontal axis and the spectral irradiance [μW cm-2 nm-1] on the vertical axis. Two peaks represent the primary emission LED lights at blue (450 nm) and red (630 nm), and a curve plateau between these peaks (550 - 600 nm).
  </figcaption>
</figure>

## Description

Our method relies on the following assumptions:
1. An observed image (5105 x 3075 pixels) comprises three independent colour channels (𝑐 = 𝑅, 𝐺, 𝐵).
2. The reflectance $𝑟_𝑐𝑗$ is a random variable at the location 𝑗 = 0,1,2, …,𝑚, and colour channel c. Two adjacent reflectances are independent of each other, and the joint probability is given by $𝑝(𝑟_{𝑐𝑗}, 𝑟_{𝑐𝑙}) = 𝑝(𝑟_{𝑐𝑗})𝑝(𝑟_{𝑐𝑙})$. Based on the same assumption, all reflectance are independent events with joint probability $$p(R_{c}) = \prod_{j=1}^{m} 𝑝(𝑟_{𝑐𝑗})$$.
3. The illuminant 𝑙𝑐𝑗 is also a random variable at the location 𝑗 = 0,1,2, …, 𝑛, and colour channel c.
4. The illumination and the reflectance are statistically independent of each other $𝑝(𝐿_{𝑐𝑗}, 𝑅_{𝑐𝑗}) = 𝑝(𝐿_{𝑐𝑗})𝑝(𝑅_{𝑐𝑗})$.
5. An image is divided into an m number of small images corresponding to individual pot areas 𝐴𝑝 where the index 𝑝 = 0,1,2, …, 𝑚 indicates the number of pot areas. It means that each pot area 𝐴𝑝 has a predetermined n number of pixels $𝑍_{𝑐𝑝} = \{ 𝑧_{𝑐𝑝ℎ} \}$ at the location ℎ = 0,1,2, …, 𝑞. Also, the reflectance $𝑅_{𝑐𝑝} = {𝑟𝑐𝑝ℎ}$ and illuminant $𝐿_{𝑐𝑝} = \{ 𝑙_{𝑐𝑝ℎ}\}$ associated with each pixel within a pot area share the same location ℎ.
6. The illuminant is constant for all pixels within a pot area $𝐴_𝑝$, meaning, $𝑙_{𝑐𝑝} = 𝑙_{𝑐𝑝ℎ}$ and 𝐿𝑐𝑝 = {𝑙𝑐𝑝}. Then, the probability distribution of the illuminant is uniform, $𝑝(𝑙_{𝑐𝑝}) = 𝑢_{𝑐𝑝}$, where $𝑢_{𝑐𝑝}$ is a constant value. However, two adjacent pot area illuminants are independent of each other $𝑝(𝑙_{𝑐𝑝}, 𝑙_{𝑐𝑞}) = 𝑝(𝑙_{𝑐𝑝})𝑝(𝑙_{𝑐q})$.

### Likelihood: 
The likelihood of the pixel class is given the illuminant and reflectance and follows a normal distribution. <br/>
$$p(z_{cki}│l_{cki},r_{cki})= \frac{1}{\sqrt{2\pi \sigma_{ck}^2}}  exp⁡\biggl(-\frac{(z_{cki}-l_{ck}r_{cki})^2}{2\sigma_{ck}^2}\biggr)$$  


### Priors: Reflectance & Illuminant: 
We created an **image dataset** to get the reflectance and illuminant prior distributions. It has images of green fabric pieces on pots and Macbeth colorChecker charts. They were illuminated using D65 standard illuminant.. <br/>

$$P(r_{cki})= \frac{1}{\sqrt{2\pi \tau_{ck}^2}}  exp⁡\biggl(-\frac{(r_i-\mu_{ck})^2}{2\tau_{ck}^2}\biggr)$$ <br/>

As the illumination is uniform over a pixel class, the probability distribution is given by:
$$p(l_{ck})=u_{ck}$$ <br/>

### Posterior
It is possible to analytically calculate the posterior distribution using the Bayes' rule as the prior is a conjugate prior for the likelihood function. The posterior distribution is given by:
$$P(L_{ck}|Z_{ck})=\prod_{i=1}^{n} \int \frac{p(z_{cki}|l_{lck}, r_{cki})p(z_{cki}) pl(l_{ck})}  {p(z_{cki})} dr_{cki} $$ <br/>


### Maximum a posteriori 
We estimate the illumination value when the posterior distribution reaches the highest value.

$$  \hat{l}_{MAP} =\underset{l_{ck}}{\operatorname{argmax}}  P(L_{ck}|Z_{ck})$$

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




