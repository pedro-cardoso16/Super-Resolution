Home
====

Convolutional Neural Networks
-----------------------------

The following implementation is based on the article  
*Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel  
Convolutional Neural Network* [`arXiv <https://arxiv.org/abs/1609.05158>`_]  
as well as the `PyTorch super-resolution example <https://github.com/pytorch/examples/tree/main/super_resolution>`_.  

The authors refer to their model as **ESPCN** (Efficient Sub-Pixel Convolutional Neural Network).  

To generate a **low-resolution (LR)** image, we first apply a **Gaussian filter** and then downsample it by a factor of *r* (the upscaling factor).  

.. math::
   I^{HR} \to I^{LR}

The network follows a convolutional transformation:

.. math::
   f^l (\mathbf{I}^{LR}; W_{1:l}, b_{1:l}) = \phi \left( W_l * f^{l-1}(\mathbf{I}^{LR}) + b_l \right)

.. note::
   In the final layer, the upscaling operation is performed.

Single Image Super-Resolution (SISR)
------------------------------------

Single Image Super-Resolution (SISR) aims to recover high-resolution details from a single low-resolution input image.  
In the ESPCN model, **the upscaling operation is performed only at the final layer** of the neural network.

Key Advantages:
^^^^^^^^^^^^^^^
- Since most operations are performed on the **low-resolution (LR) image**, computational complexity is significantly reduced.  

Key Disadvantages:
^^^^^^^^^^^^^^^^^^
- The network needs to learn multiple **upscaling filters**, rather than relying on a single predefined training.  
