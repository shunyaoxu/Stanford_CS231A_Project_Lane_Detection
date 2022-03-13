import numpy as np
import math
import random

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

# Double Threshold
def double_threshold(img, kernel_size, sigma, high, low):
    # Smoothing, Gradient, and NMS
    NMS_image = NMS(img, kernel_size, sigma)
    
    # Double Threshold
    strong_edges = np.zeros(NMS_image.shape, dtype=np.bool)
    weak_edges = np.zeros(NMS_image.shape, dtype=np.bool)
    strong_edges = NMS_image > high
    weak_edges = (NMS_image > low) & (NMS_image <= high)
    return strong_edges, weak_edges

# Extract Useful Weak Edges (we might or might not use this)
def get_neighbors(y, x, H, W):
    neighbors = []
    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))
    return neighbors

def link_edges(img, kernel_size, sigma, high, low):
    # Smoothing, Gradient, NMS, and double threshold
    strong_edges, weak_edges = double_threshold(img, kernel_size, sigma, high, low)
    
    # Extract Useful Weak Edges
    H, W = strong_edges.shape
    indicesStrEdg = np.stack(np.nonzero(strong_edges)).T # get coordinates of all strong edge pixels in shape (n,2)
    while len(indicesStrEdg) != 0:
        y, x = indicesStrEdg[0]
        indicesStrEdg = np.delete(indicesStrEdg, 0, axis=0)
        neighbors = get_neighbors(y, x, H, W)
        for m, n in neighbors:
            if weak_edges[m,n]:
                if not strong_edges[m,n] == True:
                    strong_edges[m,n] = True
                    indicesStrEdg = np.vstack((indicesStrEdg, (m,n)))
    return strong_edges

def canny_edge_detector(img, kernel_size, sigma, high, low):
    return link_edges(img, kernel_size, sigma, high, low)

# Extract Region of Interest
def ROI_Triangle(edges):
    H, W = edges.shape
    mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if i > (H / W) * j and i > - (H / W) * j + H:
                mask[i, j] = 1
    roi_img = edges * mask
    return roi_img

def ROI_Half(edges):
    H, W = edges.shape
    mask = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            if i > H // 2:
                mask[i, j] = 1
    roi_img = edges * mask
    return roi_img

# Hough Transform
def hough(img, numLinesToFind = 2, onlyLeftAndRight = True):
    """ 
    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize Hough space
    hough_space = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)

    # Transform each point (y, x) in image to sin-like curve in Hough Space
    # Find rho corresponding to values in thetas and increment the Hough Space
    ys, xs = np.nonzero(img)
    for y,x in zip(ys,xs):
        for i in range(num_thetas):
            rho = x * cos_t[i] + y * sin_t[i]
            rho = round(rho)
            j = np.abs(rhos-rho).argmin()
            hough_space[j,i] += 1
    
    if onlyLeftAndRight:
        left  = False
        right = False
        # Find the peak points in Hough Space
        while not left or not right:
            # Find peak point
            idx   = np.argmax(hough_space)
            r_idx = idx // hough_space.shape[1] # Calculate rho-coordinate
            t_idx = idx % hough_space.shape[1]  # Calculate theta-coordinate
            hough_space[r_idx, t_idx] = 0       # Zero out the max value in hough space
    
            # Transform a point in Hough space to a line in xy-space.
            rho   = rhos[r_idx]
            theta = thetas[t_idx]
            a = - (np.cos(theta) / np.sin(theta)) # slope of the line
            b = (rho / np.sin(theta)) # y-intersect of the line

            # Break if both right and left lanes are detected
            if left and right: # Both Lanes Detected
                break
            if a < -0.2: # Left lane detected
                if left:
                    continue
                left    = True
                a_left  = a
                b_left  = b
                y1_left = img.shape[0]                 # Point 1 y
                x1_left = (y1_left - b_left) / a_left  # Point 1 x
                y2_left = img.shape[0] * 0.55          # Point 2 y
                x2_left = (y2_left - b_left) / a_left  # Point 2 x
            elif a > 0.2: # Right Lane detected
                if right:
                    continue
                right    = True
                a_right  = a
                b_right  = b
                y1_right = img.shape[0] * 0.55              # Point 1 y
                x1_right = (y1_right - b_right) / a_right  # Point 1 x
                y2_right = img.shape[0]                    # Point 2 y
                x2_right = (y2_right - b_right) / a_right  # Point 2 x
        x_VP = (b_right - b_left) / (a_left - a_right)
        y_VP = x_VP * a_left + b_left
        VP   = [x_VP, y_VP, 1.]
        leftLane  = [[int(round(x1_left)), int(round(y1_left))], [int(round(x2_left)), int(round(y2_left))]]
        rightLane = [[int(round(x1_right)), int(round(y1_right))], [int(round(x2_right)), int(round(y2_right))]]
        return VP, leftLane, rightLane
    else:
        lines = []
        left     = False
        right    = False
        numLeft  = 0
        numRight = 0
        # Find the peak points in Hough Space
        while len(lines) < numLinesToFind or not left or not right:
            # Find peak point
            idx = np.argmax(hough_space)
            r_idx = idx // hough_space.shape[1] # Calculate rho-coordinate
            t_idx = idx % hough_space.shape[1]  # Calculate theta-coordinate
            hough_space[r_idx, t_idx] = 0       # Zero out the max value in hough space
    
            # Transform a point in Hough space to a line in xy-space.
            rho = rhos[r_idx]
            theta = thetas[t_idx]
            if np.sin(theta) == 0:
                continue
            a = - (np.cos(theta) / np.sin(theta)) # slope of the line
            b = (rho / np.sin(theta)) # y-intersect of the line
            
            if a < -0.25:
                numLeft += 1
                y1 = img.shape[0]        # Point 1 y
                x1 = (y1 - b) / a        # Point 1 x
                y2 = 0                   # Point 2 y
                x2 = (y2 - b) / a        # Point 2 x
                lines.append([x1, y1, x2, y2])
                if left:
                    continue
                else:
                    left = True
                    a_left  = a
                    b_left  = b
                    y1_left = y1
                    x1_left = x1
                    y2_left = img.shape[0] * 0.45
                    x2_left = (y2_left - b_left) / a_left
            elif a > 0.25:
                numRight += 1
                y1 = img.shape[0]        # Point 1 y
                x1 = (y1 - b) / a        # Point 1 x
                y2 = 0                   # Point 2 y
                x2 = (y2 - b) / a        # Point 2 x
                lines.append([x1, y1, x2, y2])
                if right:
                    continue
                else:
                    right = True
                    a_right  = a
                    b_right  = b
                    y1_right = y1
                    x1_right = x1
                    y2_right = img.shape[0] * 0.45
                    x2_right = (y2_right - b_right) / a_right
                    
        lines = np.array(lines).reshape(-1,1,4)
        leftLane  = [[int(round(x1_left)), int(round(y1_left))], [int(round(x2_left)), int(round(y2_left))]]
        rightLane = [[int(round(x1_right)), int(round(y1_right))], [int(round(x2_right)), int(round(y2_right))]]
        x_VP   = (b_right - b_left) / (a_left - a_right)
        y_VP   = x_VP * a_left + b_left
        VP_est = [x_VP, y_VP, 1.]
        return VP_est, leftLane, rightLane, lines, numLeft, numRight
    

# helper function for performing Ransac common intersection
#convert a line in the form of (x1,y1) ( x2, y2) to ax+by+c=0
def convert_homogenous_line(line):

    line=np.array(line)
    dx=line[2]-line[0]
    dy=line[3]-line[1]
    newline=np.zeros(3)
    newline[0]=-dy
    newline[1]= dx
    newline/=np.linalg.norm(newline)
    newline[2]= - (line[0]*newline[0]+line[1]*newline[1])

    return newline
# line point distance line in the form ax+by+c=0, point in the form (x,y) ot (x,y,1)


def line_dist(line, point):
    return abs(line[0]*point[0]+line[1]*point[1]+line[2])



def RANSAC_VP(lines,height_lower=None, height_upper=None,width_lower=None, width_upper=None):

    lines=np.array(lines)
    if lines.shape[0]<2:
        return np.zeros([2])
    p=0.99
    e_guess=0.9
    s=2
    n=math.log(1-p)/math.log(1-(1-e_guess)**s)
    #n=min([math.ceil(n),np.shape(lines)[0]])
    e_max=0
    point_max= np.zeros([2])
    n=math.ceil(n)
    #print('dbg0')
    #print(n)
    for i in range(n):
        line1 = lines[random.randint(0,np.shape(lines)[0]-1),0,:]
        line2 = lines[random.randint(0,np.shape(lines)[0]-1),0,:]
        line1=convert_homogenous_line(line1)
        line2 = convert_homogenous_line(line2)
        curr_intersection= np.cross(line1, line2)
        if(curr_intersection[-1]<10**(-10)):
            continue
        curr_intersection/=curr_intersection[-1]
        inlier=0
        #print('curr_intersection')
        #print(curr_intersection)
        for line in lines:
            line=convert_homogenous_line(line[0])
            #print('dbg2')
            #if(line.all()==line1.all() or line.all()==line2.all()):
            #    continue
            if(abs(line_dist(line, curr_intersection))< 10):
                inlier+=1
                #print('dbg3')

        curr_e=inlier/np.shape(lines)[0]

        if(curr_e>e_max and inlier>1 ):
            if height_lower is not None:
                if curr_intersection[1]<height_lower:
                    continue
            if height_upper is not None:
                if curr_intersection[1] > height_upper:
                    continue
            if width_lower is not None:
                if curr_intersection[0]<width_lower:
                    continue
            if width_upper is not None:
                if curr_intersection[0] > width_upper:
                    continue
            #print('dbg4')
            e_max=curr_e
            point_max=curr_intersection.copy()
            #print('dbg')
            #print(e_max)

    #n_new= math.log(1-p)/math.log(1-(1-e_max)**s)
    #if(n_new > n):

    return point_max

def Angle_btw_vect(vec1, vec2):
    vec1/=np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    return np.arccos(np.dot(vec1,vec2))