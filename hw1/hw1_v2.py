# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import os
from random import randint
import cv2
import sys

DATADIR='exposures/'

def readImg():
	tmp = []
	for i in sorted(os.listdir(DATADIR)):
		tmp.append(io.imread(DATADIR + i))
	return np.array(tmp)

def sample(w,h,N=50):
	s = []
	for i in range(N):
		s.append([randint(0,w-1),randint(0,h-1)])
	return np.array(s)

def getSamplePoint(x):
	# random get sample point * 50
	# S = sample(x.shape[1],x.shape[2])
	S = [ [i,j] for i in range(0,x.shape[1],40) for j in range(0,x.shape[2],40) ]
	sp = []
	for img in x:
		tmp = []
		for p in S:
			tmp.append(img[p[0]][p[1]])
		sp.append(tmp) # shape = [pic number , sample point,ch ]

	return np.array(sp).transpose((1,0,2)) # shape = [ sample point, pic number,ch ]




def buildLinearSystem(sp,B,lam,ch,w):
	A = np.zeros(shape=(sp.shape[0] * sp.shape[1] + 1 + 254, 256 + sp.shape[0] ) )
	b = np.zeros(shape=(sp.shape[0] * sp.shape[1] + 1 + 254,1))

	it = 0

	# first term of objective function
	for i in range(sp.shape[0]):
		for j in range(sp.shape[1]):
			A[it][sp[i][j][ch]]   = w[sp[i][j][ch]]
			A[256 + i] = - w[sp[i][j][ch]]
			b[it][0] = w[sp[i][j][ch]] * B[j]
			it += 1

	# g(127) = 0
	A[it][127] = 1
	b[it][0] = 0
	it += 1

	# second term of objective function
	for i in range(1,255):
		A[it][i-1] = lam* w[i]
		A[it][i] = lam* w[i] * (-2)
		A[it][i+1] = lam* w[i]
		b[it][0] = 0
		it += 1


	print(it)
	assert it == sp.shape[0] * sp.shape[1] + 1 + 254
	print(A.shape,b.shape)
	return A,b

def solver(A,b):
	print(b.shape)
	U, s, V = np.linalg.svd(A, full_matrices=False)
	print(U.shape, V.shape, s.shape)
	s_plus = np.diag(1/s)
	s_plus.resize(V.T.shape[1],U.T.shape[0])
	# (306,306) (306,306) (306,905) (905, 1 )
	x = np.dot(np.dot(np.dot(V.T, s_plus), U.T), b)
	print(x.shape)
	return x

def recon(imgpool,B,x,w):
	from tqdm import tqdm
	print(imgpool.shape)
	hdr = np.zeros(shape=(imgpool.shape[1],imgpool.shape[2],imgpool.shape[3]))
	for i in tqdm(range(imgpool.shape[1])):
		for j in range(imgpool.shape[2]):
			for ch in range(imgpool.shape[3]):
				bot = 0
				top = 0
				for k in range(imgpool.shape[0]):
					bot += w[imgpool[k][i][j][ch]]
					top += w[imgpool[k][i][j][ch]] * (x[ch][imgpool[k][i][j][ch]] - B[k])
				hdr[i][j][ch] = np.exp(top / bot) #### 
	return hdr

def localTM(Lm,alpha,op=True):
	if op == False:
		return Lm
	else:
		print(Lm.shape)
		Ls = np.zeros(shape=Lm.shape)
		from scipy.ndimage.filters import gaussian_filter
		blurred = []
		for s in np.arange(0,2,0.1):
			blurred.append(gaussian_filter(a, sigma=s) )
		################## need to add something ##############3

		return Lm

def ToneMapping(hdr, alpha=0.5, delta=1e-6, Lwhite=0 ):
	# Y' = 0.299 R + 0.587 G + 0.114 B 
	
	Lw = hdr[:,:,0] * 0.299 + hdr[:,:,1] * 0.587 + hdr[:,:,2] * 0.114
	LwBar = np.exp(np.mean(np.log(delta + Lw)))
	Lm = alpha / LwBar * Lw
	Ls = localTM(Lm,alpha,True)
	Ld = Lm * (1 + Lm * (Lwhite ** 2) )/ (1 + Ls) 

	ldr = np.zeros(shape=hdr.shape)
	for i in range(3):
		ldr[:,:,i] = Ld / Lw * hdr[:,:,i]
	ldr = np.clip( ldr * 255 , 0 , 255).astype('uint8')
	return ldr

def aligment(imgpool):
	images = imgpool[:,:,::-1]
	alignMTB = cv2.createAlignMTB()
	alignMTB.process(images, images)
	imgpool = images[:,:,::-1]
	return imgpool

if __name__ == '__main__':

	print(len(sys.argv))
	if len(sys.argv) == 1 or sys.argv[1] != 'pre':
		# read all the image
		imgpool = readImg()

		# aligment
		imgpool = aligment(imgpool)


		# shape [imgNum,pic size(2d) ,channel]

		# getSamplePoint
		sp = getSamplePoint(imgpool)

		# generate B(log delta-t array)
		# !!!!!!!!!!!!!!!!!!!!!!!!need to type artificially!!!!!!!!!!!!!!!!!!!!!!!!!!
		B = np.log([32 * (0.5**x) for x in range(16)])

		#  generate weighted parameter
		w = [ z - 0 if z < 0.5*(0 + 255) else 255 - z for z in range(256)]

		x = []
		for ch in range(3):
			# build the linear system and solve it
			A,b = buildLinearSystem(sp,B,0.3,ch,w)
			# solve the linear system
			x.append(solver(A,b))
		x = np.array(x) # shape = [ch , x_result(306) , 1]
		# [g(0)...g(255) | ln(E0) ... ln(E49)]

		# we can use ln(Ei) directly, or need to implement pp.77 


		# reconstruct the (ir-)radiance map
		hdr = recon(imgpool,B,x,w)

		# hdr = hdr_debvec[:, :, ::-1]  # if go to opencv
		print(hdr.max(),hdr.min())
	
		# save hdr data
		np.save('hdr_raw.npy',hdr)
	else:
		hdr = np.load('hdr_raw.npy')
	# tone mapping
	ldr = ToneMapping(hdr)
	cv2.imwrite("ldr.jpg", ldr[:,:,::-1])