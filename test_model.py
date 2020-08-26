import torch
import numpy as np
import os
import torchvision
from pro_gan_pytorch import PRO_GAN as pg


device = 'cuda'


#-----------------preModel-------------------
# netG = torch.nn.DataParallel(pg.Generator(depth=9))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 

# netD = torch.nn.DataParallel(pg.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netD.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))

# #print(netD)
# x = torch.randn(1,3,1024,1024)
# z = netD(x,height=7,alpha=1)
# print(z.shape)

#print(gen)
# depth=0
# z = torch.randn(4,512)
# x = (gen1(z,depth=depth,alpha=1)+1)/2
# torchvision.utils.save_image(x, './face_dp%d.jpg'%depth, nrow=4)
# del x
# x = (gen2(z,depth=depth,alpha=1)+1)/2
# torchvision.utils.save_image(x, './face-shadow%d.jpg'%depth, nrow=4)


from pro_gan_pytorch import encoder

netD = encoder.encoder_v1(height=9, feature_size=512)

#print(netD.final_block)
x = torch.randn(1,3,1024,1024)
z = netD(x,height=8,alpha=1)
print(z.shape)
