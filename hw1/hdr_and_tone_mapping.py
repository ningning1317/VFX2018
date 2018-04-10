# Implement Paul Debeve’s HDR and Photographic tone mapping
# Usage:
# python3 hdr_and_tone_mapping.py -i <img series num> -d <data directory>

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import os
import argparse
import random
from random import randint,choice
import cv2
import numpy as np

IMG_NUM=0
DATA_DIR=''
RESULT_DIR=''
DRAW_RC=True

def readInfo():
    images, shutters, max_shift = [], [], 0
    with open('%s/info.txt'%DATA_DIR, 'r') as f:
        for n, line in enumerate(f):
            if '%d_'%IMG_NUM in line:
                line = line.rstrip().split(',')
                images.append(line[0])
                shutters.append(1/int(line[1]))
                shift_x, shift_y = abs(int(line[2])), abs(int(line[3]))
                max_shift = shift_x if shift_x > max_shift else\
                            shift_y if shift_y > max_shift else\
                            max_shift
    return images, np.array(shutters, dtype=np.float32), max_shift

def readImg(images):
    tmp = []
    for f in images:
        tmp.append(cv2.imread('%s/%s'%(DATA_DIR, f)))
    return np.array(tmp)

def sample(h, w, margin, N=50):
    s = []
    random.seed(1234)
    for i in range(N):
        s.append([randint(margin, h-1-margin), randint(margin, w-1-margin)])
    return np.array(s)

def sampleGAll(x, margin):

    # shape [imgNum,pic size(2d) ,channel]
    # choose the middle pic and sample all g(.)
    random.seed(1234)
    midPic = x[ x.shape[0] // 2][margin:-margin, margin:-margin]
    candidate = []
    for i in range(256):
        tmp = list(zip(*np.where( midPic == i)))
        if tmp == []:
            continue
        else:
            py, px = choice(tmp)[:2]
            candidate.append( (py+margin, px+margin) )
    return candidate

def getSamplePoint(x, margin=0):
    random.seed(1234)
    # random get sample point * 50
    S = sample(x.shape[1], x.shape[2], margin)
#     S = [ [i,j] for i in range(margin,x.shape[1]-margin,100) for j in range(margin,x.shape[2]-margin,100) ]
#     S = sampleGAll(x)
    sp = []
    for img in x:
        tmp = []
        for py,px in S:
            tmp.append(img[py][px])
        sp.append(tmp) # shape = [pic number , sample point,ch ]

    return np.array(sp).transpose((1,0,2)) # shape = [ sample point, pic number,ch ]

def getSampleGray(x, margin=0):
    S = sample(x.shape[1], x.shape[2], margin)
    sp = []
    for img in x:
        tmp = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # sample on gray
        for py, px in S:
            tmp.append(gray[py][px])
        sp.append(tmp) # shape = [pic number , sample point]
    sp = np.asarray(sp)
    sp = sp[:,:,np.newaxis]
    return sp.transpose((1,0,2))

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
    A[it][127] = 10
    b[it][0] = 0
    it += 1

    # second term of objective function
    for i in range(1,255):
        A[it][i-1] = lam* w[i]
        A[it][i] = lam* w[i] * (-2)
        A[it][i+1] = lam* w[i]
        b[it][0] = 0
        it += 1

    assert it == sp.shape[0] * sp.shape[1] + 1 + 254

    return A,b

def solver(A,b):
    U, s, V = np.linalg.svd(A, full_matrices=False)

    s_plus = np.diag(1/s)
    s_plus.resize(V.T.shape[1],U.T.shape[0])
    x = np.dot(np.dot(np.dot(V.T, s_plus), U.T), b)
    return x

def recon(imgpool,B,x,w):
    from tqdm import tqdm

    hdr = np.zeros(shape=(imgpool.shape[1],imgpool.shape[2],imgpool.shape[3]))
    for i in tqdm(range(imgpool.shape[1])):
        for j in range(imgpool.shape[2]):
            for ch in range(imgpool.shape[3]):
                bot = 0
                top = 0
                for k in range(imgpool.shape[0]):
                    bot += w[imgpool[k][i][j][ch]]
                    top += w[imgpool[k][i][j][ch]] * (x[ch][imgpool[k][i][j][ch]] - B[k])
                if bot == 0: # handle the divide by zero exp.
                    hdr[i][j][ch] = np.exp(x[ch][imgpool[k//2][i][j][ch]] - B[k//2])
                else:
                    hdr[i][j][ch] = np.exp(top / bot)

    return hdr

def localTM(Lm,alpha,op=True):
    if op == False:
        return Lm
    else:
        Ls = np.zeros(shape=Lm.shape)
        from scipy.ndimage.filters import gaussian_filter
        blurred = []
        for s in np.arange(0,2,0.1):
            blurred.append(gaussian_filter(a, sigma=s) )
        ################## need to add something ##############3

        return Lm

def ToneMapping(hdr, alpha=0.5, delta=1e-6, Lwhite=0.5 ):
    # Y' = 0.299 R + 0.587 G + 0.114 B
    Lw = hdr[:,:,0] * 0.299 + hdr[:,:,1] * 0.587 + hdr[:,:,2] * 0.114
    LwBar = np.exp(np.mean(np.log(delta + Lw)))
    Lm = alpha / LwBar * Lw
    Ls = localTM(Lm,alpha,False)
    Lwhite *= Lm.max()
    Ld = Lm * (1 + Lm / (Lwhite ** 2) )/ (1 + Ls)

    # to handle denominator=0, since Ld=0 when Lw=0, we can set Lw=whatever
    Lw[Lw == 0] = 1
    ldr = np.zeros(shape=hdr.shape)
    for i in range(3):
        ldr[:,:,i] = Ld / Lw * hdr[:,:,i]

    ldr = np.clip( ldr * 255, 0 , 255).astype('uint8')

    return ldr

def drawRC(x):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(256), x[0][:256], 'b.', np.arange(256), x[1][:256], 'g.', np.arange(256), x[2][:256], 'r.')
    plt.title('Response Curve')
    plt.xlabel('g(.)')
    plt.ylabel('log exposure ( ln(E * delta-t ) ')
    plt.savefig('%s/RC_%d.png'%(RESULT_DIR,IMG_NUM))
    return

def doneByOpenCV():
    print('doing HDR and TM by openCV')
    images, shutters, max_shift = readInfo()
    images = readImg(images)
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, shutters)
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, shutters, responseDebevec)
    print('HDR max:%.2f, min:%.2f'%(hdrDebevec.max(), hdrDebevec.min()))
    cv2.imwrite('%s/hdr_%d_OpenCV.hdr'%(RESULT_DIR,IMG_NUM), hdrDebevec)
    tonemapDrago = cv2.createTonemapDrago(1, 0.8) # hand set params
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 255 * ldrDrago * 1.5 # hand set params
    print('LDR max:%.2f, min:%.2f'%(ldrDrago.max(), ldrDrago.min()))
    cv2.imwrite('%s/ldr_%d_OpenCV.jpg'%(RESULT_DIR,IMG_NUM), ldrDrago)

def main(args):
    global DATA_DIR, IMG_NUM, DRAW_RC, RESULT_DIR
    DATA_DIR = '%s/aligned'%args.data_dir
    RESULT_DIR = '%s/result'%args.data_dir
    IMG_NUM = args.img_num
    DRAW_RC = args.draw_rc

    # the image is aligned(by aligment_MTB.py)
    # images: [img file names], shutters: np.array([delta-t]), max_shift: int>=0
    images, shutters, max_shift = readInfo()
    if len(images) == 0:
        print('ERROR: no images')
        return

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    imgpool = readImg(images) # shape [imgNum, pic size(2d) ,channel]

    # getSamplePoint
    sp = getSamplePoint(imgpool, margin=max_shift)

    # generate B: [log delta-t]
    B = np.log(shutters)

    #  generate weighted parameter
    w = [ z - 0 if z < 0.5*(0 + 255) else 255 - z for z in range(256)]

    x = []
    for ch in range(3):
        # build the linear system and solve it
        A,b = buildLinearSystem(sp,B,100,ch,w)
        # solve the linear system
        x.append(solver(A,b))
    x = np.array(x) # shape = [ch , x_result(306) , 1]
    # [g(0)...g(255) | ln(E0) ... ln(E49)]

    # draw the resopnse curve
    if DRAW_RC == True:
        drawRC(x.copy())

    # reconstruct the (ir-)radiance map
    hdr = recon(imgpool,B,x,w)
    print('HDR max:%.2f, min:%.2f'%(hdr.max(), hdr.min()))

    # save hdr data
    hdr = hdr.astype(np.float32) # to imwrite() correctly
    hdr = hdr[:,:,::-1] # convert to RGB
    cv2.imwrite('%s/hdr_%d.hdr'%(RESULT_DIR, IMG_NUM), hdr)

    # tone mapping
    ldr = ToneMapping(hdr)
    print('LDR max:%.2f, min:%.2f'%(ldr.max(), ldr.min()))
    cv2.imwrite('%s/ldr_%d.jpg'%(RESULT_DIR, IMG_NUM), ldr[:,:,::-1])

    if args.done_by_openCV:
        doneByOpenCV()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_num', type=int,
                        default='1', dest='img_num',
                        help='Img series number')
    parser.add_argument('-d', '--data_dir', type=str,
                        default='jpg/aligned', dest='data_dir',
                        help='Data directory')
    parser.add_argument('--draw_rc', type=bool,
                        default=True, dest='draw_rc',
                        help='Whether to draw resopnse curve')
    parser.add_argument('--done_by_openCV', type=bool,
                        default=True, dest='done_by_openCV',
                        help='Whether to call openCV doing HDR & TM')
    main(parser.parse_args())
