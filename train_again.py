import torch
import numpy as np
import os
import torchvision
from pro_gan_pytorch import PRO_GAN as pg
from pro_gan_pytorch import encoder
from torch.optim import Adam
from torch.nn import DataParallel
from pro_gan_pytorch.CustomLayers import update_average
import torch.nn as nn
from torch.nn import AvgPool2d
from torch.nn.functional import interpolate
from pro_gan_pytorch.DataTools import DatasetFromFolder
import copy
import time
import timeit

device = 'cuda'
resultPath = "./result_trainAG_1.1"
if not os.path.exists(resultPath):
    os.mkdir(resultPath)

# netG = torch.nn.DataParallel(pg.Generator(depth=9))# in: [-1,512], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netG.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
# netD1 = torch.nn.DataParallel(encoder.Discriminator(height=9, feature_size=512))# in: [-1,3,1024,1024],out:[], depth:0-4,1-8,2-16,3-32,4-64,5-128,6-256,7-512,8-1024
# netD1.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))

# netD2 = torch.nn.DataParallel(encoder.encoder_v1(height=9, feature_size=512).to(device))

# def toggle_modelGrad(model,flag):
#     for i in model.parameters():
#         i.requires_grad_(flag)

# toggle_modelGrad(netD1,False)
# toggle_modelGrad(netD2,False)

# paraDict = dict(netD1.named_parameters()) # pre_model weight dict
# for i,j in netD2.named_parameters():
#     if i in paraDict.keys():
#         w = paraDict[i]
#         j.copy_(w)

# toggle_modelGrad(netD2,True)

#-------------------------training again-------------------------
class ProGAN:
    """ Wrapper around the Generator and the Discriminator """
    def __init__(self, depth=7, latent_size=512, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", use_ema=True, ema_decay=0.999,
                 device=torch.device("cpu")):
        # Create the Generator and the Discriminator
        self.gen = torch.nn.DataParallel(pg.Generator(depth=depth))
        self.dis = torch.nn.DataParallel(encoder.Discriminator(height=depth))
        self.gen.load_state_dict(torch.load('./pre-model/GAN_GEN_SHADOW_8.pth',map_location=device)) #shadow的效果要好一些 
        self.dis.load_state_dict(torch.load('./pre-model/GAN_DIS_8.pth',map_location=device))
        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.device = device
        self.drift = drift
        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,betas=(beta_1, beta_2), eps=eps)
        self.loss = nn.MSELoss()

        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __progressive_downsampling(self, real_batch, depth, alpha):
        """
        private helper for downsampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fader alpha
        :return: real_samples => modified real batch of samples
        """

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_downsample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_downsample_factor)(real_batch),
                                                scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True)

    def train(self, dataSet, epochs, batch_sizes,
              fade_in_percentage, num_samples=16,
              start_depth=0, num_workers=3, feedback_factor=100,
              log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=20):

        assert self.depth == len(batch_sizes), "batch_sizes not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth):
            print("\n\nCurrently working on Depth: ", current_depth)
            current_res = np.power(2, current_depth + 2)
            print("Current resolution: %d x %d" % (current_res, current_res))
            data = torch.utils.data.DataLoader(dataset=dataSet,batch_size=batch_sizes[current_depth],shuffle=True,numworks=num_workers, pin_memory=True)
            ticker = 1
            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch
                print("\nEpoch: %d" % epoch)
                total_batches = len(iter(data))
                fader_point = int((fade_in_percentage[current_depth] / 100)* epochs[current_depth] * total_batches)
                step = 0  # counter for number of iterations
                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1
                    images = batch.to(self.device)
                    z = self.dis(images,height=current_depth,alpha=alpha)
                    real_samples = self.__progressive_downsampling(images, current_depth, alpha)
                    fake_samples = self.gen(z, current_depth, alpha).detach()
                    loss = self.loss(real_samples, fake_samples)
                    self.dis_optim.zero_grad()
                    loss.backward()
                    self.dis_optim.step()
                    dis_loss += loss.item()
# provide a loss feedback
                    if i % int(total_batches / feedback_factor) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"% (elapsed, i, dis_loss, gen_loss))
                        # also write the losses to the log file:
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) +"\t" + str(gen_loss) + "\n")
                        # create a grid of samples and save it
                        os.makedirs(sample_dir, exist_ok=True)
                        gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +"_" + str(epoch) + "_" +str(i) + ".png")
                        # this is done to allow for more GPU space
                        with torch.no_grad():
                            self.create_grid(
                                samples=self.gen(
                                    fixed_input,
                                    current_depth,
                                    alpha
                                ).detach() if not self.use_ema
                                else self.gen_shadow(
                                    fixed_input,
                                    current_depth,
                                    alpha
                                ).detach(),
                                scale_factor=int(np.power(2, self.depth - current_depth - 1)),
                                img_file=gen_img_file,
                            )
                            torchvision.utils.save_image(real_samples, resultPath+'/recons-%d-%d.jpg'%(epoch,i), nrow=8)
                            torchvision.utils.save_image(fake_samples, resultPath+'/face-%d-%d.jpg'%(epoch,i), nrow=8)
                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1
                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))
                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + ".pth")
                    gen_optim_save_file = os.path.join(save_dir,"GAN_GEN_OPTIM_" + str(current_depth)+ ".pth")
                    dis_optim_save_file = os.path.join(save_dir,"GAN_DIS_OPTIM_" + str(current_depth)+ ".pth")
                    torch.save(self.gen.state_dict(), gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)
                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_" +str(current_depth) + ".pth")
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
        # put the gen, shadow_gen and dis in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()
        print("Training completed ...")

if __name__ == '__main__':
    # some parameters:
    depth = 9 # 4-->8-->16-->32-->64-->128-->256-->512-->1024 ，0开始,8结束,所以depth是9
    # hyper-parameters per depth (resolution)
    num_epochs = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    fade_ins = [50, 50, 50, 50, 50, 50, 50, 50, 50]
    batch_sizes = [128, 128, 128, 64, 64, 64, 64, 32, 32]
    latent_size = 512

    # ======================================================================
    # This line creates the PRO-GAN
    # ======================================================================
    pro_gan = ProGAN(depth=depth, latent_size=latent_size, device=device)

    #data_path='/home/disanda/Desktop/dataSet/CelebAMask-HQ/CelebA-HQ-img'
    data_path='/_yucheng/dataSet/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img'
    #data_path='F:/dataSet2/CelebAMask-HQ/CelebA-HQ-img'
    trans = torchvision.transforms.ToTensor()
    dataset = DatasetFromFolder(data_path,transform=trans)

    # This line trains the PRO-GAN
    pro_gan.train(
        dataSet = dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes,
        sample_dir="./result/celeba1024_rc/",
        log_dir="./result/celeba1024_rc/", 
        save_dir="./result/celeba1024_rc/",
        num_workers=0
    )
    # ======================================================================