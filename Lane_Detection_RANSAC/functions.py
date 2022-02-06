import numpy as np

# Convolution
def convolution(img, kernel):
    img_H, img_W = img.shape
    kernel_H, kernel_W = kernel.shape
    output = np.zeros((img_H, img_W))
    pad_width_H = kernel_H // 2
    pad_width_W = kernel_W // 2
    pad = ((pad_width_H, pad_width_H), (pad_width_W, pad_width_W))
    padded_img = np.pad(img, pad, mode='edge')
    kernel = np.flip(kernel)
    for i in range(img_H):
        for j in range(img_W):
            output[i,j] = np.sum(np.multiply(padded_img[i:i+kernel_H,j:j+kernel_W], kernel))
    return output

# Smoothing
def smoothing(img, kernel_size, sigma):
    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    k = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian_kernel[i,j] = 1 / (2*np.pi*np.power(sigma,2)) * np.exp((-(i-k)**2-(j-k)**2) / (2*np.power(sigma,2)))
    soomth_img = convolution(img, gaussian_kernel)
    return soomth_img

# Gradient
def gradient(img, kernel_size, sigma):
    # Smoothing
    smoothed_img = smoothing(img, kernel_size, sigma)
    
    # Gradient
    theta = np.zeros(img.shape)
    kernel_x = np.array([[0.5, 0.0, -0.5]])
    gradient_x = convolution(img, kernel_x)
    kernel_y = np.array([[0.5], [0.0], [-0.5]])
    gradient_y = convolution(img, kernel_y)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    theta = (np.rad2deg(np.arctan2(gradient_y, gradient_x)) + 180) % 360
    return gradient, theta
    
# Non-Maximum Suppression
def NMS(img, kernel_size, sigma):
    # Smoothing, Gradient
    grad, theta = gradient(img, kernel_size, sigma)
    
    # Non-Maximum Suppression
    H, W = grad.shape
    output = np.zeros((H, W))
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)
    G_pad = np.pad(grad, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(1,H+1):
        for j in range(1,W+1):
            if theta[i-1,j-1] == 45 or theta[i-1,j-1] == 225:
                if G_pad[i,j] >= G_pad[i-1,j-1] and G_pad[i,j] >= G_pad[i+1,j+1]:
                    output[i-1,j-1] = G_pad[i,j]
            if theta[i-1,j-1] == 90 or theta[i-1,j-1] == 270:
                if G_pad[i,j] >= G_pad[i-1,j] and G_pad[i,j] >= G_pad[i+1,j]:
                    output[i-1,j-1] = G_pad[i,j]
            if theta[i-1,j-1] == 135 or theta[i-1,j-1] == 315:
                if G_pad[i,j] >= G_pad[i+1,j-1] and G_pad[i,j] >= G_pad[i-1,j+1]:
                    output[i-1,j-1] = G_pad[i,j]
            if theta[i-1,j-1] == 180 or theta[i-1,j-1] == 0:
                if G_pad[i,j] >= G_pad[i,j-1] and G_pad[i,j] >= G_pad[i,j+1]:
                    output[i-1,j-1] = G_pad[i,j]
    return output
    
# Extract Region of Interest
def ROI(edges):
    H, W = edges.shape
    mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if i > (H / W) * j and i > - (H / W) * j + H:
                mask[i, j] = 1
    roi_img = edges * mask
    return roi_img

# RANSAC
def RANSAC(img):
    