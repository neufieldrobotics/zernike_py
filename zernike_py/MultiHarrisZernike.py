#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:02:43 2019

@author: vik748
"""
import numpy as np
np.set_printoptions(precision=5,suppress=True)
import cv2
import time
from packaging import version
#from matlab_imresize.imresize import imresize
#from scipy.ndimage import convolve

OPENCV_NEWER_THAN_4_5_2 = version.parse(cv2.__version__) > version.parse('4.5.2')

# Inheriting from opencv class causes segfaults
#class MultiHarrisZernike (cv2.Feature2D):
class MultiHarrisZernike:
    '''
    MultiHarrisZernike feature detector which uses multi-level harris corners
    along with Zernike parameters on 2 different radii discs as the feature detector
    A class as a child of cv2.Feature2D

    Parameters
    ----------
    Nfeats : int, optional
        Number of features per image. Default is 600
    seci : int, optional
        Number of vertical sectors (tiles) for keypoint thresholding. Default
        is 2
    secj : int, optional
        Number of horizontal sectors (tiles) for keypoint thresholding. Default
        is 3
    levels : int, optional
        Number of levels in image pyramid. Default is 6
    ratio : float, optional
        Scaling ratio between levels, default is 0.75
    sigi : float, optional
        Integration scale, default is 1.0
    sigd : float, optional
        Derivation scale, default is 2.75
    nmax : int, optional
        Zernike order
    lmax_nd : int, optional
        Feature neighborhood size for local maximum filter, default is 3
    like_matlab : bool, optional
        Flag to replicate Oscar's Matlab version (slr) when true. Default is
        false

    ----------
    Example usage:
        img = cv2.imread('test.png',1)
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        a = MultiHarrisZernike()
        kp, des = a.detectAndCompute(gr)

        outImage	 = cv2.drawKeypoints(gr, kp, gr,color=[255,255,0],
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        fig, ax= plt.subplots(dpi=200)
        plt.title('Multiscale Harris with Zernike Angles')
        plt.axis("off")
        plt.imshow(outImage)
        plt.show()

    '''
    def __init__(self,  Nfeats= 600, seci = 2, secj = 3, levels = 6, ratio = 0.75,
                 sigi = 2.75, sigd = 1.0, nmax = 8, like_matlab=False, lmax_nd = 3, harris_threshold = None):

        if like_matlab:
            from matlab_imresize.imresize import imresize as ml_imresize
            from scipy.ndimage import convolve as sk_convolve
            self.ml_imresize = ml_imresize
            self.sk_convolve = sk_convolve

        self.Nfeats  = Nfeats        # number of features per image
        self.seci    = seci          # number of vertical sectors
        self.secj    = secj          # number of horizontal sectors
        self.levels  = levels        # pyramid levels
        self.ratio   = ratio         # scaling between levels
        self.sigi    = sigi          # integration scale 1.4.^[0:7];%1.2.^[0:10]
        self.sigd    = sigd          # derivation scale
        self.nmax    = nmax          # zernike order
        self.exact   = like_matlab   # Flag to replicate Oscar's Matlab version (slower)
        self.lmax_nd = lmax_nd       # Feature neighborhood size for local maximum filter
        self.zrad    = np.ceil(self.sigi*8).astype(int) # radius for zernike disk
        self.brad    = np.ceil(0.5*self.zrad).astype(int)    # radius for secondary zernike disk
        self.non_max_kernel = np.ones((self.lmax_nd,self.lmax_nd), np.uint8)
        self.harris_threshold = harris_threshold # Minimum harris threshold for a keypoint

        if self.exact:
            self.Gi     = MultiHarrisZernike.fspecial_gauss(11,self.sigi)
            self.pyrlpf = MultiHarrisZernike.fspecial_gauss(int(np.ceil(7*self.sigd)),self.sigd)

        self.ZstrucZ, self.ZstrucZ_rav, self.ZstrucNdesc = MultiHarrisZernike.zernike_generate(self.nmax, self.zrad)
        self.BstrucZ, self.BstrucZ_rav, self.BstrucNdesc = MultiHarrisZernike.zernike_generate(self.nmax, self.brad)

    @staticmethod
    def fspecial_gauss(size, sigma):
        """
        Function to mimic the 'fspecial' gaussian MATLAB function
        """
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    @staticmethod
    def xy_gradients(img):
        '''
        Return x and y gradients of an image. Similar to np.gradient
        '''
        kernelx = 1/2*np.array([[-1,0,1]])
        kernely = 1/2*np.array([[-1],[0],[1]])
        fx = cv2.filter2D(img,cv2.CV_32F,kernelx)
        fy = cv2.filter2D(img,cv2.CV_32F,kernely)
        return fy, fx

    @staticmethod
    def zernike_generate(nmax,radius,verbose=False):
        '''
        generate the zernike filter to caluculate zernike coefficient n,m
        '''
        desc = 0
        Zfilt=[[0 for x in range(nmax+1)] for y in range(nmax+1)]
        Zfilt_rav=[[0 for x in range(nmax+1)] for y in range(nmax+1)]

        for n in range(nmax+1):
            for m in range(n%2,n+1,2):
                desc = desc+1
                if verbose:
                    print([radius, n, m, desc])
                Zfilt[n][m] = MultiHarrisZernike.zerfilt(n,m,radius)
                Zfilt_rav[n][m] = Zfilt[n][m].ravel()
        return Zfilt,Zfilt_rav,desc

    @staticmethod
    def zerfilt(n,m,r):
        '''
        n and m integers that specify coeficient n > 0, m < n, m-n even
        r radius in pixels on image
        '''
        xdim = 2*r+1
        ydim = 2*r+1
        Z = np.zeros((xdim,ydim),dtype=np.complex64)

        for y in range(-r,r+1):
            for x in range (-r,r+1):
                theta = np.arctan2(y,x)
                rho = ((x**2+y**2)**0.5)/r
                if rho <= 1:
                    Z[y+r,x+r] = (n+1)/np.pi*np.exp(-m*theta*1j)*MultiHarrisZernike.zerrad(n,m,rho)
        return Z

    @staticmethod
    def zerrad(n,m,rho):
        R = 0.0
        for s in range(0,int((n-m)/2)+1):
            R = R + ((-1)**s*np.math.factorial(n-s)) / \
                     (np.math.factorial(s)* \
                      np.math.factorial(int((n+m)/2-s))* \
                      np.math.factorial(int((n-m)/2-s)) \
                     )* \
                     rho**(n-2*s)
        return R

    @staticmethod
    def plot_zernike(Z):
        '''
        Plot the generated zernike polynomials
        '''
        from matplotlib import pyplot as plt

        nm=len(Z)
        f, axes = plt.subplots(nm, nm, sharey=True)
        f.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)


        w, h = Z[0][0].shape
        for n in range(nm):
            for m in range(nm):
                if isinstance(Z[n][m], int) and Z[n][m] == 0:
                    pass
                else:
                    axes[n,m].imshow(np.real(Z[n][m]),cmap='gray')
                axes[n,m].axis('off')
                
    def getNfeats(self):
        return self.Nfeats

    def getseci(self):
        return self.seci

    def getsecj(self):
        return self.secj

    def getlevels(self):
        return self.levels

    def getratio(self):
        return self.ratio

    def getsigi(self):
        return self.sigi

    def getsigd(self):
        return self.sigd

    def getnmax(self):
        return self.nmax

    def getlmax_nd(self):
        return self.lmax_nd

    def getharris_threshold(self):
        if self.harris_threshold is None:
            return 0
        else:
            return self.harris_threshold

    def generate_pyramid(self, img, mask=None):
        '''
        Generate image pyramid, based on settings in the MultiHarrisZernike object
        '''
        sigd_list = [self.sigd]
        sigi_list = [self.sigi]
        images = [np.float32(img)]

        if mask is not None:
            masks = [mask]
        else:
            masks = None

        # convolve matches matlab version better, filter is 3 times faster
        if self.exact:
            lpimages = [self.sk_convolve(images[0],self.pyrlpf,mode='constant')]
        else:
            #lpimages = [gaussian_filter(images[0],sigma=self.sigd,mode='constant',truncate=3.0)]
            lpimages = [cv2.GaussianBlur(images[0], ksize=(7,7), sigmaX=self.sigd, sigmaY=self.sigd, borderType = cv2.BORDER_CONSTANT)]

        for k in range(1,self.levels):
            if self.exact:
            		# CV2 version of imresize is faster but doesn't have antialiasing
            		# so results in fewer matches
                images += [self.ml_imresize(images[-1], self.ratio, method='bilinear').astype('float32')]

                # convolve matches matlb version better, filter is 3 times faster

                lpimages += [self.sk_convolve(images[-1],self.pyrlpf,mode='constant')]
            else:
                images += [cv2.resize(images[-1], (0,0), fx=self.ratio,
                           fy=self.ratio, interpolation=cv2.INTER_AREA)]
                #lpimages += [gaussian_filter(images[-1],sigma=self.sigd,mode='constant',truncate=3.0)]
                # kzise = 11,11 matches best with exact version, lower size more efficient
                lpimages += [cv2.GaussianBlur(images[-1], ksize=(5,5), sigmaX=self.sigd, sigmaY=self.sigd, borderType = cv2.BORDER_CONSTANT)]

            if mask is not None:
                masks += [cv2.resize(masks[-1], images[-1].shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)]

            sigd_list += [sigd_list[-1]/self.ratio] #equivalent sigdec at max res
            sigi_list += [sigi_list[-1]/self.ratio]
        return {'images':images, 'lpimages':lpimages, 'sigd':sigd_list,
                'sigi':sigi_list, 'masks':masks}
        
    def generate_Pyramid_image_dims(self, gr_shape):
        x,y = gr_shape
        P_dim = []
        for s in range(self.levels):
            #print(x, y)
            P_dim.append([x,y])
            x = round(x * self.ratio)
            y = round(y * self.ratio)            
        return np.array(P_dim)

    def eigen_image_p(self,lpf,scale, compute_eigenvals=False):
        '''
        ef2,nL = eigen_image_p(lpf,scale)
        set up in pyramid scheme with detection scaled smoothed images
        ef2 is the interest point eigen image
        lpf smoothed by the detection scale gaussian
        Gi = fspecial('gaussian',ceil(7*sigi),sigi);
        '''
        [fy,fx] = MultiHarrisZernike.xy_gradients(lpf)

        [fxy,fxx] = MultiHarrisZernike.xy_gradients(fx)
        [fyy,fyx] = MultiHarrisZernike.xy_gradients(fy)
        nL = scale**(-2)*np.abs(fxx+fyy)

        if self.exact:
            Mfxx = self.sk_convolve(np.square(fx),self.Gi,mode='constant')
            Mfxy = self.sk_convolve(fx*fy,self.Gi,mode='constant')
            Mfyy = self.sk_convolve(np.square(fy),self.Gi,mode='constant')
        else:
            # Significantly Faster than convolve
            # kzise = 11,11 matches best with exact version, lower size more efficient
            Mfxx = cv2.GaussianBlur(np.square(fx), ksize=(5,5), sigmaX=self.sigi,
                                    sigmaY=self.sigi, borderType = cv2.BORDER_CONSTANT)
            Mfxy = cv2.GaussianBlur(fx * fy, ksize=(5,5), sigmaX=self.sigi,
                                    sigmaY=self.sigi, borderType = cv2.BORDER_CONSTANT)
            Mfyy = cv2.GaussianBlur(np.square(fy), ksize=(5,5), sigmaX=self.sigi,
                                    sigmaY=self.sigi, borderType = cv2.BORDER_CONSTANT)

        Tr = Mfxx+Mfyy
        Det = Mfxx*Mfyy-np.square(Mfxy)
        with np.errstate(invalid='ignore'):
            sqrterm = np.sqrt(np.square(Tr)-4*Det)

        ef2 = scale**(-2)*0.5*(Tr - sqrterm)

        if compute_eigenvals:           
            M = np.zeros((lpf.shape[0], lpf.shape[1], 2,2))
            M[:,:,0,0] = Mfxx
            M[:,:,0,1] = Mfxy
            M[:,:,1,0] = Mfxy
            M[:,:,1,1] = Mfyy
    
            eig_vals = np.linalg.eigvals(M)
            return np.nan_to_num(ef2),nL, eig_vals
        else:
            return np.nan_to_num(ef2),nL
    

    def feat_extract_p2 (self, ImgPyramid):
        '''
        Extract multiscaled features from a Pyramid of images
        '''
        scales = self.levels
        ratio = self.ratio
        border = self.zrad
        lpimages = ImgPyramid['lpimages']
        masks = ImgPyramid['masks']
        [rows,cols] = lpimages[0].shape

        eig = [None] * scales
        nL = [None] * scales
        border_mask = [None] * scales
        threshold_mask = [None] * scales
        regmask=[None] * scales
        ivec= [None] * scales
        jvec= [None] * scales

        for k in range(scales):
            [eig[k], nL[k]] = self.eigen_image_p(lpimages[k],ratio**(k))
            # extract regional max and block out borders (edge effect)

            # generate mask for border
            border_mask[k] = np.zeros_like(eig[k],dtype=bool)
            border_mask[k][border:-border,border:-border]=True

            regmask[k] = cv2.dilate(eig[k], self.non_max_kernel, iterations=1) <= eig[k]

            regmask[k] = np.logical_and(regmask[k], border_mask[k])

            if self.harris_threshold is not None:
                threshold_mask[k] = eig[k] >= self.harris_threshold
                regmask[k] = np.logical_and(regmask[k], threshold_mask[k])

            if masks is not None:
                regmask[k] = np.logical_and(masks[k],regmask[k])

            #print("K: ",k," - ",np.sum(regmask[k]))
            #[ivec[k], jvec[k]] = np.nonzero(regmask[k]) #coordinates of 1s in regmask
            # Just to match matlab version, can be reverted to optimise
            [jvec[k], ivec[k]] = np.nonzero(regmask[k].T)

        # INITIALIZE feature positions and scales at highest level
        # at highest resolution coordinates of features:
        Fivec = ivec[0]
        #print("len of Fivec:",len(ivec[0]))
        Fjvec = jvec[0]
        Fsvec = np.zeros_like(Fivec) #initial scale
        Fevec = eig[0][ivec[0],jvec[0]] #access the elements of eig at locations given by ivec,jvec

        #i,j position of feature at the characteristic scale
        Fsivec = np.copy(Fivec)
        Fsjvec = np.copy(Fjvec)

        nLvec = nL[0][ivec[0],jvec[0]]
        pivec = np.copy(Fivec)
        pjvec = np.copy(Fjvec)
        pind = np.array(list(range(len(Fivec))))
        k = 1

        while  (k < scales) & (len(pivec) > 0):
            mx = (np.floor(cols*ratio)-1)/(cols-1)  #scale conversion to next level
            my = (np.floor(rows*ratio)-1)/(rows-1)

            [rows,cols]  = eig[k].shape #dimensions of next level
            pendreg = np.zeros_like(pivec)
            # match matlab output
            sivec = np.round(pivec*my+np.finfo(np.float32).eps).astype(int) #next scale ivec
            sjvec = np.round(pjvec*mx+np.finfo(np.float32).eps).astype(int) #next scale jvec

            csivec = np.copy(sivec)
            csjvec = np.copy(sjvec)

            for u in [-1, 0, 1]:  #account for motion of feature points between scales
                sojvec = sjvec+u #next scale jvec
                for v in [-1, 0, 1]:
                    soivec = sivec+v #next scale ivec
                    uvpend = regmask[k][soivec,sojvec] == 1
                    pendreg = np.logical_or(pendreg,uvpend)
                    csivec[uvpend] = soivec[uvpend]
                    csjvec[uvpend] = sojvec[uvpend]

            pend = np.logical_and(pendreg, nL[k][csivec,csjvec] >= nLvec)
            pind = pind[pend]

            Fsvec[pind] = k #scale is k or larger
            Fevec[pind] = eig[k][csivec[pend],csjvec[pend]] #eigen value is given at
                                                 #level k or larger
            Fsivec[pind] = csivec[pend]
            Fsjvec[pind] = csjvec[pend]

            pivec = csivec[pend]
            pjvec = csjvec[pend]
            nLvec = nL[k][csivec[pend],csjvec[pend]]
            #print(np.sum(Fsvec==k))
            k = k+1
        F = {'ivec':Fivec, 'jvec':Fjvec, 'svec':Fsvec,
             'evec':Fevec, 'sivec':Fsivec, 'sjvec':Fsjvec}
        return F

    def feat_thresh_sec(self,F,rows,cols):

        Nsec = self.seci*self.secj
        Nfsec = np.ceil(self.Nfeats/Nsec).astype(int)

        seclimi = np.linspace(0,rows-1,self.seci+1)
        seclimj = np.linspace(0,cols-1,self.secj+1)

        Fivec = F['ivec']
        Fjvec = F['jvec']
        Fevec = F['evec']
        Fsvec = F['svec']
        Fsivec = F['sivec']
        Fsjvec = F['sjvec']
        select = np.array([],dtype=int) #zeros(size(F.ivec))
        selind = np.array(list(range(len(Fivec))))
        for i_ll,i_ul in zip(seclimi[:-1],seclimi[1:]):
            selecti = np.logical_and( Fivec >= i_ll , Fivec < i_ul)
            for j_ll,j_ul in zip(seclimj[:-1],seclimj[1:]):
                selectj = np.logical_and(Fjvec >= j_ll, Fjvec < j_ul)
                selectsec = np.logical_and(selecti, selectj)
                evec = Fevec[selectsec]
                selindsec = selind[selectsec]
                N,bin_centers = np.histogram(evec,50)
                X = bin_centers[:-1] + np.diff(bin_centers)/2

                C = np.cumsum(N[::-1])

                bins = X[::-1]

                if C[-1]<Nfsec:
                    thresh = bins[-1]
                    selecte=np.ones_like(evec,dtype=bool)

                else:
                    k = 0
                    while C[k] < Nfsec and k < 50-1:
                        k = k+1

                    thresh = bins[k]
                    selecte = evec > thresh
                    while np.sum(selecte) > Nfsec:
                        thresh = thresh*1.2
                        selecte = evec > thresh

                    while np.sum(selecte) < Nfsec and thresh > 1e-9:
                        thresh = thresh*0.9
                        selecte = evec > thresh

                select = np.append(select, selindsec[selecte])

        Fout = {'ivec':Fivec[select], 'jvec':Fjvec[select], 'evec':Fevec[select],
                'sivec':Fsivec[select], 'sjvec':Fsjvec[select], 'svec':Fsvec[select]}
        Fout['Nfeats']=len(Fout['ivec'])
        #Fout['thresh'] = thresh
        return Fout

    def z_jet_p2(self,ImgPyramid,F):
        '''
        Local jet of order three of interest points i,j
        '''
        feats = len(F['ivec'])
        Fsvec = F['svec']
        Fsivec = F['sivec']
        Fsjvec = F['sjvec']
        images = ImgPyramid['images']

        JAcoeff=[[0 for x in range(self.nmax+1)] for y in range(self.nmax+1)]
        JBcoeff=[[0 for x in range(self.nmax+1)] for y in range(self.nmax+1)]

        #initialize
        for n in range(self.nmax+1):
            for m in range(n%2,n+1,2):
                JAcoeff[n][m] = np.zeros(feats,dtype=np.complex64)
                JBcoeff[n][m] = np.zeros(feats,dtype=np.complex64)

        for k in range(feats): #(feats+1):
            sk = Fsvec[k] #scale of feature
            i_s = Fsivec[k]
            j_s = Fsjvec[k]
            # window size
            # [size(P(sk).im) is-zrad is+zrad js-zrad js+zrad]
            W = images[sk][i_s-self.zrad:i_s+self.zrad+1,
                           j_s-self.zrad:j_s+self.zrad+1]
            Wh = W-np.mean(W)
            W = Wh/(np.sum(Wh**2)**0.5)
            W_rav = W.ravel()

            Wb = images[sk][i_s-self.brad:i_s+self.brad+1,
                            j_s-self.brad:j_s+self.brad+1]
            Wbh = Wb-np.mean(Wb)
            Wb = Wbh/((np.sum(Wbh**2))**0.5)
            Wb_rav = Wb.ravel()

            for n in range(self.nmax+1):
                for m in range(n%2,n+1,2):
                    #JAcoeff[n][m][k] = np.sum(W*self.ZstrucZ[n][m])
                    #JBcoeff[n][m][k] = np.sum(Wb*self.BstrucZ[n][m])
                    JAcoeff[n][m][k] = W_rav.dot(self.ZstrucZ_rav[n][m])
                    JBcoeff[n][m][k] = Wb_rav.dot(self.BstrucZ_rav[n][m])

        return JAcoeff, JBcoeff

    def corner_angle(self,ImgPyramid,F):
        '''
        Determine feature angle
        '''
        feats = len(F['ivec'])
        Fsvec = F['svec']
        Fsivec = F['sivec']
        Fsjvec = F['sjvec']
        images = ImgPyramid['images']

        #initialize
        JAcoeff = np.zeros(feats,dtype=np.complex64)

        for k in range(feats): #(feats+1):
            sk = Fsvec[k] #scale of feature
            i_s = Fsivec[k]
            j_s = Fsjvec[k]
            # window size
            # [size(P(sk).im) is-zrad is+zrad js-zrad js+zrad]
            W = images[sk][i_s-self.zrad:i_s+self.zrad+1,
                           j_s-self.zrad:j_s+self.zrad+1]
            Wh = W-np.mean(W,dtype='uint')#np.mean(W)
            W = np.divide(Wh, np.sum(Wh**2)**0.5, dtype=np.float32)
            W_rav = W.ravel()

            JAcoeff[k] = W_rav.dot(self.ZstrucZ_rav[1][1])
            alpha = np.angle(JAcoeff)

        return alpha

    def zinvariants4(self, JA, JB):
        '''
        oriented invariants
        invariance to affine changes in intensity
        '''
        rows, = JA[0][0].shape
        Va = np.zeros((rows, self.ZstrucNdesc),dtype=np.float32)
        Aa = np.zeros((rows, self.ZstrucNdesc),dtype=np.float32)
        Vb = np.zeros((rows, self.ZstrucNdesc),dtype=np.float32)
        Ab = np.zeros((rows, self.ZstrucNdesc),dtype=np.float32)
        #1 through 7 are oriented gradients, relative to maximum direction
        k = 0
        for n in range(self.nmax+1):
            for m in range(n%2,n+1,2):
                Va[:,k] = np.abs(JA[n][m])
                Vb[:,k] = np.abs(JB[n][m])
                Aa[:,k] = np.angle(JA[n][m])
                Ab[:,k] = np.angle(JB[n][m])
                k = k+1
        V = np.hstack((Va, Vb))
        A = np.hstack((Aa, Ab))
        alpha = np.angle(JA[1][1])
        return V,alpha,A
    
    def FtDict2KeypointList(self, Ft, alpha):
        '''
        Convert MultiHarris Ft dict to OpenCV keypoint list
        '''
        keypoints = []
        for x,y,ang,res,sc in zip(Ft['jvec'], Ft['ivec'], alpha, Ft['evec'], Ft['svec']):
            pt_x = float(x)
            pt_y = float(y)
            angle = float(np.rad2deg(ang))
            response = float(res)
            octave = int(sc)
            size = float(self.zrad*(octave+1)*2)
            keypoints.append(cv2.KeyPoint(pt_x, pt_y, size, _angle=angle,
                                          _response=response, _octave=octave))
        return keypoints
        
    def keypointList2FtDict(self, keypoints, gr_shape):
        '''
        Convert an OpenCV keypoint list to MultiHarris style Ft dict
        '''
        ivec = []
        jvec = []
        evec = []
        svec = []
        sivec = []
        sjvec = []

        for kp in keypoints:
            jvec.append(kp.pt[0])
            ivec.append(kp.pt[1])
            evec.append(kp.response)
            svec.append(kp.octave)
            sjvec.append(int(np.round_(kp.pt[0]*self.ratio**kp.octave)))
            sivec.append(int(np.round_(kp.pt[1]*self.ratio**kp.octave)))

        ivec_arr = np.array(ivec,dtype=int)
        jvec_arr = np.array(jvec,dtype=int)
        svec_arr = np.array(svec)
        sivec_arr = np.array(sivec,dtype=int)
        sjvec_arr = np.array(sjvec,dtype=int)
        evec_arr = np.array(evec,dtype=np.float32)
        
        P_dims = self.generate_Pyramid_image_dims(gr_shape)
        img_dims_arr = P_dims[svec_arr]
        
	#handle situations where the feature falls too close to the edge
	#after rounding for scale
        for index, (i, j, (dim_i, dim_j)) in enumerate(zip(sivec_arr, sjvec_arr, img_dims_arr)):
            #print(index,i,j,dim_i,dim_j)
            if dim_i - i <= self.zrad:
                #print(index,i,j,dim_i,dim_j)
                #print("Less: ",dim_i, i)
                sivec_arr[index] = dim_i - self.zrad - 1
            elif i <= self.zrad:        
                #print(index,i,j,dim_i,dim_j)
                #print("Less: ",dim_i, i)
                sivec_arr[index] = self.zrad + 1
            
            if dim_j - j <= self.zrad:
                #print(index,i,j,dim_i,dim_j)
                #print("Less: ",dim_j, j, dim_j - zernike.zrad - 1)
                sjvec_arr[index] = dim_j - self.zrad - 1
            elif j <= self.zrad:        
                #print(index,i,j,dim_i,dim_j)
                #print("Less: ",dim_j, j)
                sjvec_arr[index] = self.zrad + 1            
                           

        Ft = {'ivec':ivec_arr, 'jvec':jvec_arr, 'svec':svec_arr,
              'sivec':sivec_arr, 'sjvec': sjvec_arr, 'evec':evec_arr,
              'Nfeats':len(keypoints)}
        return Ft

    def getDefaultName(self):
        return "MultiHarrisZernike"

    def defaultNorm(self):
        return cv2.NORM_L2
    
    def detectAndCompute(self, gr_img, mask=None, timing=False, computeEigVals=False, Ft=None):
        '''
        cv2.Feature2D style detectAndCompute.  Takes a grayscale image and
        optionally a mask and returns OpenCV keypoints and descriptors

        Parameters
        ----------
        gr_img : 2D-array (image)
            The input grayscale image
        mask : 2D-array, optional
            Image mask with 1s where keypoints are permissible
        timing : bool, optional
            Display timing in various parts of algorithm

        ----------
        Example usage:
            kp, des = a.detectAndCompute(gr, mask=m1)

        '''
        if len(gr_img.shape)!=2:
            raise ValueError("Input image is not a 2D array, possibile non-grayscale")
        if timing: st=time.time()
        P = self.generate_pyramid(gr_img,mask=mask)
        if timing: print("Generate pyramid - {:0.4f}".format(time.time()-st)); st=time.time()
        F = self.feat_extract_p2(P)
        if timing: print("Extract features - {:0.4f}".format(time.time()-st)); st=time.time()
        if Ft is None:
            Ft = self.feat_thresh_sec(F,*gr_img.shape)
        if timing: print("Feature Threshold - {:0.4f}".format(time.time()-st)); st=time.time()
        JA,JB = self.z_jet_p2(P,Ft)
        if timing: print("Feature jets - {:0.4f}".format(time.time()-st)); st=time.time()
        V,alpha,A = self.zinvariants4(JA, JB)
        if timing: print("Feature invariants - {:0.4f}".format(time.time()-st)); st=time.time()
           
        kp = self.FtDict2KeypointList(Ft, alpha)
              
        if computeEigVals:
            scales = self.levels
            ratio = self.ratio
            lpimages = P['lpimages']
            #masks = ImgPyramid['masks']
            #[rows,cols] = lpimages[0].shape
    
            eig = [None] * scales
            nL = [None] * scales
            eigVals = [None] * scales
    
            for k in range(scales):
                [eig[k], nL[k], eigVals[k]] = self.eigen_image_p(lpimages[k],ratio**(k), compute_eigenvals=True)
    
            kp_eigVals = np.zeros((Ft['Nfeats'],2))
            # i at specified scale, j at specified scale and s scale         
            if OPENCV_NEWER_THAN_4_5_2:
                kp = [cv2.KeyPoint(x=float(x),y=float(y),size=float(self.zrad*(sc+1)*2),angle=float(ang),response=float(res),octave=int(sc))
                      for x,y,ang,res,sc in zip(Ft['jvec'], Ft['ivec'], np.rad2deg(alpha), 
                                                Ft['evec'],Ft['svec'])]
            else:
                kp = [cv2.KeyPoint(float(x),float(y),float(self.zrad*(sc+1)*2),_angle=float(ang),_response=float(res),_octave=int(sc))
                      for x,y,ang,res,sc in zip(Ft['jvec'], Ft['ivec'], np.rad2deg(alpha), 
                                                Ft['evec'],Ft['svec'])]

            return kp, V, kp_eigVals, Ft
        
        else:
            return kp, V, #Ft, F, Ft, JA, JB, alpha, A

    def detect(self, gr_img, mask=None, timing=False):
        '''
        cv2.Feature2D style detectAndCompute.  Takes a grayscale image and
        optionally a mask and returns OpenCV keypoints and descriptors

        Parameters
        ----------
        gr_img : 2D-array (image)
            The input grayscale image
        mask : 2D-array, optional
            Image mask with 1s where keypoints are permissible
        timing : bool, optional
            Display timing in various parts of algorithm

        ----------
        Example usage:
            kp, des = a.detectAndCompute(gr, mask=m1)

        '''
        if len(gr_img.shape)!=2:
            raise ValueError("Input image is not a 2D array, possibile non-grayscale")
        if timing: st=time.time()
        P = self.generate_pyramid(gr_img,mask=mask)
        if timing: print("Generate pyramid - {:0.4f}".format(time.time()-st)); st=time.time()
        F = self.feat_extract_p2(P)
        if timing: print("Extract features - {:0.4f}".format(time.time()-st)); st=time.time()
        Ft = self.feat_thresh_sec(F,*gr_img.shape)
        if timing: print("Feature Threshold - {:0.4f}".format(time.time()-st)); st=time.time()
        alpha = self.corner_angle(P, Ft)
        if timing: print("Angle computation - {:0.4f}".format(time.time()-st)); st=time.time()
        if OPENCV_NEWER_THAN_4_5_2:
            kp = [cv2.KeyPoint(x=x,y=y,size=self.zrad*(sc+1)*2,angle=ang,response=res,octave=sc)
                  for x,y,ang,res,sc in zip(Ft['jvec'], Ft['ivec'], np.rad2deg(alpha),
                                            Ft['evec'],Ft['svec'])]
        else:
            kp = [cv2.KeyPoint(x,y,self.zrad*(sc+1)*2,_angle=ang,_response=res,_octave=sc)
                  for x,y,ang,res,sc in zip(Ft['jvec'], Ft['ivec'], np.rad2deg(alpha),
                                            Ft['evec'],Ft['svec'])]
        
        if timing: print("Keypoint export - {:0.4f}".format(time.time()-st)); st=time.time()

        return kp

    def compute(self, gr_img, keypoints, timing=False, mask=None, Ft=None):
        '''
        cv2.Feature2D style compute.  Takes a grayscale image and keypoints
        and returns OpenCV keypoints and descriptors

        Parameters
        ----------
        gr_img : 2D-array (image)
            The input grayscale image
        keypoints : list of OpenCV keypoint objects
        timing : bool, optional
            Display timing in various parts of algorithm

        ----------
        Example usage:
            kp, des = a.compute(gr, kp, mask=m1)

        '''
        Ft = self.keypointList2FtDict(keypoints, gr_img.shape)
        Ft, V, Ft = self.computeFromFt(gr_img, Ft, timing=time, mask=mask, computeEigVals=False)  

        return keypoints, V
    
    def computeFromFt(self, gr_img, Ft, timing=False, mask=None, computeEigVals=False):
        '''
        cv2.Feature2D style compute.  Takes a grayscale image and keypoints
        and returns OpenCV keypoints and descriptors

        Parameters
        ----------
        gr_img : 2D-array (image)
            The input grayscale image
        keypoints : list of OpenCV keypoint objects
        timing : bool, optional
            Display timing in various parts of algorithm

        ----------
        Example usage:
            kp, des = a.compute(gr, kp, mask=m1)

        '''
        if len(gr_img.shape)!=2:
            raise ValueError("Input image is not a 2D array, possibile non-grayscale")
        if timing: st=time.time()
        P = self.generate_pyramid(gr_img,mask=mask)
        if timing: print("Generate pyramid - {:0.4f}".format(time.time()-st)); st=time.time()

        JA,JB = self.z_jet_p2(P,Ft)
        if timing: print("Feature jets - {:0.4f}".format(time.time()-st)); st=time.time()
        V,alpha,A = self.zinvariants4(JA, JB)
        if timing: print("Feature invariants - {:0.4f}".format(time.time()-st)); st=time.time()
        keypoints = self.FtDict2KeypointList(Ft, alpha)        
    
        if computeEigVals:
            scales = self.levels
            ratio = self.ratio
            lpimages = P['lpimages']
    
            eig = [None] * scales
            nL = [None] * scales
            eigVals = [None] * scales
    
            for k in range(scales):
                [eig[k], nL[k], eigVals[k]] = self.eigen_image_p(lpimages[k],ratio**(k), compute_eigenvals=True)
    
            kp_eigVals = np.zeros((Ft['Nfeats'],2))
            # i at specified scale, j at specified scale and s scale         
            for i, (si, sj, s) in enumerate(zip(Ft['sivec'], Ft['sjvec'], Ft['svec'])):
                kp_eigVals[i,:] = eigVals[s][si,sj]

            return keypoints, V, Ft, kp_eigVals
        else:
            return keypoints, V, Ft #, F, Ft, JA, JB, alpha, A

    def computeHarrisEigenVals(self, gr_img, mask= None, timing=False):
        '''
        THIS FUNCTIONS NEEDS TO BE WRITTEN TO COMPUTE EIGENVALUES GIVEN A LIST OF KEYPOINTS
        Compute Harris Eigen Values: Takes a grayscale image, keypoints
        optionally a mask and returns OpenCV keypoints and descriptors

        Parameters
        ----------
        gr_img : 2D-array (image)
            The input grayscale image
        keypoints : list of OpenCV keypoint objects
        mask : 2D-array, optional
            Image mask with 1s where keypoints are permissible
        timing : bool, optional
            Display timing in various parts of algorithm

        ----------
        Example usage:
            kp, des = a.detectAndCompute(gr, mask=m1)

        
        #Ft = keypointList2FtDict(keypoints)
        
        if len(gr_img.shape)!=2:
            raise ValueError("Input image is not a 2D array, possibile non-grayscale")
        P = self.generate_pyramid(gr_img,mask=mask)
        F = self.feat_extract_p2(P)
        Ft = self.feat_thresh_sec(F,*gr_img.shape)        
                
        scales = self.levels
        ratio = self.ratio
        lpimages = P['lpimages']
        #masks = ImgPyramid['masks']
        #[rows,cols] = lpimages[0].shape

        eig = [None] * scales
        nL = [None] * scales
        eigVals = [None] * scales

        for k in range(scales):
            [eig[k], nL[k], eigVals[k]] = self.eigen_image_p(lpimages[k],ratio**(k), compute_eigenvals=True)

        kp_eigVals = np.zeros((Ft['Nfeats'],2))
        # i at specified scale, j at specified scale and s scale         
        for i, (si, sj, s) in enumerate(zip(Ft['sivec'], Ft['sjvec'], Ft['svec'])):
            kp_eigVals[i,:] = eigVals[s][si,sj]
        '''
        return None