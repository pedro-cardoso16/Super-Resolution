Metrics
#######

PSNR
====

The PSNR (*Peak Signal to Noise Ratio*) is a metric used to measure the quality 
of a reconstructed or compressed image compared to its original version. It is 
expressed in decibels (dB) and is defined as:

.. math::

   PSNR = 10 \cdot \log_{10} \left(\frac{MAX_I^2}{MSE} \right)

where :math:`MAX_I` is the maximum possible pixel value of the image (e.g., 255 for an 8-bit image), 
and *MSE* is the Mean Squared Error. A higher PSNR value generally indicates 
better image quality, as it suggests a lower level of distortion. However, PSNR 
does not always align well with human perception of image quality.

MSE
===

The MSE (*Mean Squared Error*) measures the average squared difference between 
corresponding pixels in two images. It is given by:

.. math::

   MSE = \frac{1}{m \cdot n} \sum_{i=1}^{m} \sum_{j=1}^{n} (I(i,j) - K(i,j))^2

where *I(i,j)* and *K(i,j)* are the pixel values of the original and 
reconstructed images, respectively, and *m* and *n* are the dimensions of the images. 
A lower MSE value indicates that the images are more similar. However, like PSNR, 
MSE does not always correspond to perceived visual quality.

L1 Loss
=======

L1 Loss, also known as *Mean Absolute Error (MAE)*, calculates the absolute 
differences between the pixel values of two images:

.. math::

   L1 = \frac{1}{m \cdot n} \sum_{i=1}^{m} \sum_{j=1}^{n} |I(i,j) - K(i,j)|

Unlike MSE, which squares the error terms, L1 Loss treats all errors linearly, 
making it less sensitive to large differences. It is often used in optimization 
tasks where robustness to outliers is important.

Structural Similarity (SSIM)
============================

SSIM (*Structural Similarity Index Measure*) is a perceptual metric that 
evaluates the structural similarity between two images by considering luminance, 
contrast, and structural components. It is defined as:

.. math::

   SSIM(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}

where:

- :math:`\mu_x` and :math:`\mu_y` are the mean intensities,
- :math:`\sigma_x^2` and :math:`\sigma_y^2` are the variances,
- :math:`\sigma_{xy}` is the covariance,
- :math:`C_1` and :math:`C_2` are small constants to stabilize the division.

SSIM ranges from -1 to 1, where 1 indicates identical images. It is often preferred 
over PSNR and MSE because it better aligns with human perception of visual quality.

