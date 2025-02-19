Architecture
============
The proposed neural network consists of three primary layers:

1. **Feature Extraction Layer:**
   The first L-1 convolutional layers extract feature maps from the LR input:

   .. math::

      f^l( I_{LR} ; W_1:l, b_1:l ) = \phi(W_l * f^{l-1}( I_{LR} ) + b_l)
   
   where :math:`W_l` and :math:`b_l` are the learnable weights and biases, and :math:`\phi` is the activation function.
   
2. **Sub-Pixel Convolution Layer:**
   The last layer performs upscaling using an efficient sub-pixel convolution operation:

   .. math::

      I_{SR} = PS(W_L * f^{L-1}(I_{LR}) + b_L)
   
   where :math:`PS` is the periodic shuffling operator that rearranges elements to increase spatial resolution.

3. **Output Layer:**
   Produces the final high-resolution image from the learned upscaling filters.
