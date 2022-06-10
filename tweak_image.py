import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torchvision.transforms as transforms

def imshow(img, norm_mean, norm_std, transpose = True):
    for i in range(3):
        img[i] = img[i] * norm_std[i] + norm_mean[i]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def generate_tweak(img_size=32, verbose=True):
    # Our 2-dimensional distribution will be over variables X and Y
    N = img_size #32 #40
    # r1 = [random.uniform(1, 3) for _ in range(4)]
    X = np.linspace(-1, 1, N) #np.linspace(-r1[0], r1[1], N) #np.linspace(-2, 2, N)
    Y = np.linspace(-1, 1, N) #np.linspace(-r1[2], r1[3], N) #np.linspace(-2, 2, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    muxy = [random.randint(-1, 1) for _ in range(2)]
    mu = np.array([muxy[0], muxy[1]]) #np.array([0., 0.])

    sigxy = [random.uniform(0.1,0.5) for _ in range(2)]
    cov = random.uniform(0,0.1)
    while sigxy[0]*sigxy[1]<cov**2 :
        sigxy = [random.uniform(0.1,0.5) for _ in range(2)]
        cov = random.uniform(0,0.1)

    Sigma = np.array([[ sigxy[0] , cov], [cov,  sigxy[1]]]) #np.array([[ 1. , -0.5], [-0.5,  1.]])

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    if verbose == True:
        
        # plot using subplots
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1,projection='3d')

        ax1.plot_surface(-X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                        cmap=cm.viridis)
        ax1.view_init(55,90)  #view_init(55,-70)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.set_xlabel(r'$x_1$')
        ax1.set_ylabel(r'$x_2$')

        ax2 = fig.add_subplot(2,1,2,projection='3d')
        ax2.contourf(-X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
        ax2.view_init(90,90) #view_init(90, 270)

        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.set_xlabel(r'$x_1$')
        ax2.set_ylabel(r'$x_2$')

        plt.show()

        print("deformation maxi:",Z.max())
        print("sigX",sigxy[0])
        print("sigY",sigxy[1])
        print("cov",cov)
    
    return Z

def apply_tweak(img,norm_mean,norm_std, verbose=True):
#     print(img.min(),img.max())
    Z = generate_tweak(verbose=verbose)
#     print(Z.shape)
    Z_tensor = torch.Tensor(Z)
    Z_tensor_3dim = torch.unsqueeze(Z_tensor, 0)
    noise_tensor = torch.cat((Z_tensor_3dim,Z_tensor_3dim,Z_tensor_3dim),0)

    tweaked_img = img - noise_tensor
    tweaked_img_double = tweaked_img.double()

    tweaked = torch.where(tweaked_img_double < -1., -1., tweaked_img_double)

    final_tweaked = tweaked.float()
    
    if verbose == True:
        imshow(final_tweaked,norm_mean, norm_std)
        
    return final_tweaked

#     noise3 = np.expand_dims(Z, axis=2)
#     new_noise = np.append(noise3, noise3, axis=2)
#     final_noise = np.append(new_noise,noise3, axis=2)
    
#     transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std) ])
    
#     final_noise_tensor = transform(final_noise)
#     print(final_noise_tensor.min(), final_noise_tensor.max(), final_noise_tensor.shape,final_noise_tensor.type())
    
#     print(img.min(), img.max(), img.shape,img.type())
#     tweaked_image = img - final_noise_tensor
#     print(tweaked_image.min(), tweaked_image.max(), tweaked_image.shape,tweaked_image.type())

#     tweaked = torch.where(tweaked_image < 0., 0., tweaked_image)
#     print(tweaked.min(), tweaked.max(), tweaked.shape,tweaked.type())

#     tweaked_tensor = transform(tweaked)
#     print(tweaked_tensor.min(), tweaked_tensor.max(), tweaked_tensor.shape)

#     imshow(tweaked,norm_mean, norm_std)
#     return tweaked
#     imshow(images[0])