Architecture
============
The proposed neural network is based on the article https://arxiv.org/abs/1609.05158
The proposed neural network consists of three primary layers:

ESPCN *Efﬁcient Sub-Pixel Convolutional Neural Network*
-------------------------------------------------------

In this neural network, we use only the channel ``Y`` of ``YCbCr`` color space 
and to get the original color image we do an interpolation using the color channels 
``Cb`` and ``Cr``. 

.. image:: images/SVSRnet.jpg
   :alt: SVSRnet

1. **Feature Extraction Layer:**
   The first L-1 convolutional layers extract feature maps from the LR input:

   .. math::

      f^l( I_{LR} ; W_1:l, b_1:l ) = \phi(W_l * f^{l-1}( I_{LR} ) + b_l)
   
   where :math:`W_l` and :math:`b_l` are the learnable weights and biases, and 
   :math:`\phi` is the activation function for example :math:`\tanh(x)` or 
   :math:`\sigma(x)` the sigmoid function.

1. **Propagation Inside The Neural Network**
   Then, we do successives 2D convolutions and apply ``ReLU`` after each convolution.
   The number of channels may vary, in the implemented mode we change the number of channels 
   to 32 and 64.

3. **Sub-Pixel Convolution Layer:**
   The last layer performs upscaling using an efficient sub-pixel convolution 
   operation:

   .. math::

      I_{SR} = PS(W_L * f^{L-1}(I_{LR}) + b_L)
   
   where :math:`PS` is the periodic shuffling operator that rearranges elements 
   to increase spatial resolution. In the final layer the number of channels is always
   :math:`r^2` where :math:`r` is the upscaling-factor.

4. **Output Layer:**
   Produces the final high-resolution image from the learned upscaling filters by
   applying the 

.. literalinclude:: summary.txt

Training
''''''''

The training of the model is specific for each upscalling factor and the target images
must have a dimension that is dividable by :math:`r` so a crop is applied to, we use 
the ``crop_size = crop_size - (crop_size % upscale_factor)``

