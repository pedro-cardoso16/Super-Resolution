Home
====

Convolutional Neural Networks
`Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel 
Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_
They call their model an ESPCN (Efficient Sub-Pixel Convolutional Neural-Network)

In order to have a Low-Resolution image we convolve by a gaussian filter and then 
we downsample the image by a 'r' factor (upscaling factor)

.. math::
   I^{HR} \to I^{LR}

.. math::
   f^l (\mathbf{I}^{LR}; W_{1:l}, b_{1:l}) = \phi \left( W_l*f^{l-1}(\mathbf{I}^{LR}) + b_l \right)


.. note::
   In the final layer the 

Single Image Super-Resolution (SISR)
------------------------------------

We will try to recover from a single image the 
In the paper the upscalling part is only done at the end of the NN

- Since the operations are done in the LR image the computational complexity is 
  significantly reduced
- The NN learns more upscalling filters rather than only one


Comparison 
----------

.. code-block:: python
   
   crop_size = target.shape - (target.shape % upscale_factor)

