# Image-to-Image-Translation-with-cGAN

This repository implements pix2pix, a conditional GAN (cGAN) that learns a mapping from paired input and output images using a U-Net generator and PatchGAN discriminator 
youtube.com
+8
tensorflow.org
+8
machinelearningmastery.com
+8
. It supports standard training loops with combined adversarial and L1 losses, works out of the box on datasets like facades or maps, and includes checkpointing for inference and fine-tuning. Simply prep your data, train for ~100 epochs on a GPU, then apply the generator to new inputs.
