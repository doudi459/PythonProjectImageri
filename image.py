import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from skimage import feature
from scipy import ndimage
from math import cos,sin,radians,sqrt
from scipy.stats import itemfreq
from scipy import interpolate
import h5py
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor,wait
from scipy.optimize import linear_sum_assignment

class Image:
	"""
		classe définissant une image caractérisée par
		- son chemin d'accés
		- image originale 
		- vecteur descriptif
	"""
	def __init__(self,path):
		""" constructeur de notre classe qui prend en entrée 
			le path de l'image 	"""
		self._path = path
		self._image = cv2.imread(path)
		self._initialiser()

	# lissage
	def _gaussianBlur(self,k):
		# filtre gaussien
		self._image = cv2.GaussianBlur(self._imgOriginale,(k,k),0)

	def _medianBlur(self,k):
		# filre medien
		self._image = cv2.medianBlur(self._imgOriginale,k) 

	def _getContourExterne(self):
		
		# suppression du bruit avec le filtre Gaussien
		imgGray = cv2.cvtColor(self._image,cv2.COLOR_BGR2GRAY) 
		blur =cv2.GaussianBlur(imgGray,(5,5),0)
		
		# binarisation 
		_,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		
		# extraction du contour externe 
		_,contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnt = max(contours,key=len)
		
		return cnt

	def _getContourInterne(self,sigma):
		
		""" 1ere étape extraction des contour (interne 
				+ externe) avec canny """
		
		#	extraction des trois matrices de couleur
		blue,green,red = cv2.split(self._image)
		blur =cv2.GaussianBlur(red,(5,5),0)
		
		# extraction des countours
		edges = feature.canny(blur, sigma=sigma)
		
		# image du contour externe
		X = np.uint8(255)
		det = X*edges

		""" 2eme étape extraction du contour externe avec 
			la méthode précédente  """
		
		cnt = self._getContourExterne()
		M = np.zeros(self._image.shape[:2],dtype=np.uint8)
		cv2.drawContours(M,[cnt],-1,(255,255,255),2)

		""" 3eme étape	calcul  de la difference entre les deux image """

		# application de la dilataion 
		kernel = np.ones((7,7),np.uint8)
		det = cv2.dilate(det,kernel,iterations = 2)
		M = cv2.dilate(M,kernel,iterations = 2)
		
		#calcul de la différence
		R = det - M

		#extraction du contour interne 
		R = cv2.erode(R,kernel,iterations = 2)
		R = cv2.GaussianBlur(R,(5,5),0)
		_,contourInt,_ = cv2.findContours(R,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		if  len(contourInt) != 0  :
			maximum = max(contourInt,key=len)
			if (len(maximum) >= 500):
				return  maximum
		return []


	def _getPointsCourbure(self,contour,k):

		#approximation polygonale
		epsilon = k*cv2.arcLength(contour,True)
		approx = cv2.approxPolyDP(contour,epsilon,True)

		cnt = approx
		hull = cv2.convexHull(cnt,returnPoints = False)
		
		defects = cv2.convexityDefects(cnt,hull)
		concaves = []
		convexes = []
		points = []

		# on teste si l'image possede des points de courbure
		if (defects is not None):
			#extraction des points de courbures
			for i in range(defects.shape[0]):
				s,e,f,d = defects[i,0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				concaves.append(far)
				convexes.append(start)
				convexes.append(end)
		convexes=list(set(convexes))
		
		for i in approx[:,0]:
			points.append(tuple(i))

		return convexes,concaves,[approx],points

	def _distanceRadiale(self):
		# contour externe 
		contour = self._getContourExterne()

		# ellipse
		ellipse = cv2.fitEllipse(contour)
		p1,p2=ellipse[0]
		#centre de gravité (centre de l'ellipse)
		centre=(int(p1),int(p2))

		# detection du point de contour
		for i in contour[:,0]:
			x = i[0]
			if ( (x == int(p1)) or (abs(x-int(p1) )<=50 ) ):
				x = int(p1)
				#calcul de la distance radiale
				distR= sqrt( (x-centre[0])**2 + (i[1]-centre[1])**2 )
				break

		# normalisation de la distance
		area = cv2.contourArea(contour)
		height,width,_ = self._image.shape
		pr = area/(height*width)
		
		distR = distR * pr

		return distR

	def _distanceRadiale1(self):
		# contour externe 
		contour = self._getContourExterne()

		# ellipse
		ellipse = cv2.fitEllipse(contour)
		p1,p2=ellipse[0]
		#centre de gravité (centre de l'ellipse)
		centre=(int(p1),int(p2))

		# detection du point de contour
		for i in contour[:,0]:
			x = i[0]
			if ( (x == int(p1)) or (abs(x-int(p1) )<=50 ) ):
				x = int(p1)
				#calcul de la distance radiale
				distR= sqrt( (x-centre[0])**2 + (i[1]-centre[1])**2 )
				break

		# normalisation de la distance
		leftmost = tuple(contour[contour[:,:,0].argmin()][0])
		rightmost = tuple(contour[contour[:,:,0].argmax()][0])
		l = (leftmost[0],centre[1])
		r =  (rightmost[0],centre[1])

	
		pr = sqrt( (l[0]-r[0])**2 + (l[1]-r[1])**2 )
		distR = distR / pr

		return distR

	def _extractForm(self):
		""" 
		cette fonction extrait le forme contenue dans l'image
		      en utilisant un masque contruit à partir du contour externe 
		"""

		# extraction du contour externe
		imgGray = cv2.cvtColor(self._image,cv2.COLOR_BGR2GRAY)
		blur =cv2.GaussianBlur(imgGray,(5,5),0)
		_,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		_,contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		
		# points du contour externe
		pts = max(contours,key=len)

		# mask
		mask =  np.zeros(self._image.shape[:2],np.uint8)
		cv2.drawContours(mask,[pts],-1,(255,255,255),-1,cv2.LINE_AA)

		self._image = cv2.bitwise_and(self._image,self._image, mask = mask)
		self._image = cv2.GaussianBlur(self._image,(5,5),0)

	def _calculRotationAngle(self):

		contourEx = self._getContourExterne()

		#approximation
		epsilon = 0.01*cv2.arcLength(contourEx,True)
		approx = cv2.approxPolyDP(contourEx,epsilon,True)	

		# clacul de l'angle de rotation 
		angle = 0
		if (len (approx) > 4):
			ellipse = cv2.fitEllipse(approx)
			if (ellipse[2] != 0):
				angle = ellipse[2]-90

		return angle

	def _rotateImage(self,angle):

		self._image = ndimage.rotate(self._image,angle,reshape=True)
		self._imgOriginale = ndimage.rotate(self._imgOriginale,angle,reshape=True)

	def _initialiser(self):
		"""
		#extraction de la forme + rotation

		self._extractForm()
		self._imgOriginale = self._image.copy()
		self._rotateImage(self._calculRotationAngle()) 
		"""
		
		# faire une copie de l'imge 
		self._imgOriginale = self._image.copy()

		self.contourImages(1)
		self.approxImages(0.005)
		self.calculNbArrete(1)


	def contourImages(self,sigma):

		# extraction des contours 
		M1 = np.zeros(self._image.shape[:2],dtype=np.uint8)
		M2 = np.zeros(self._image.shape[:2],dtype=np.uint8)

		contourEx = self._getContourExterne()		
		contourIn = self._getContourInterne(sigma)
		
		# construction des images des contours 
		cv2.drawContours(M1,[contourEx],-1,(255,255,255),3)
		if len(contourIn) != 0  :
			cv2.drawContours(M2,[contourIn],-1,(255,255,255),3)
		
		self._contourIn = cv2.GaussianBlur(M2,(5,5),0)
		self._contourEx = cv2.GaussianBlur(M1,(5,5),0)

	def approxImages(self,k):
		self._approxImage = self._image.copy()
		self._pointsCimage = self._image.copy()

		_,_,approx,points = self._getPointsCourbure(self._getContourExterne(),k)
		self._nbPointsC = len(points)

		cv2.drawContours(self._approxImage,approx,-1,(255,0,0),5)

		for i in points:
   			cv2.circle(self._pointsCimage,i,14,[255,0,0],-1)

	def calculNbArrete(self,sigma):
		cnt = self._getContourInterne(sigma)
		if len(cnt) != 0:
			convexes,_,_,_ = self._getPointsCourbure(cnt,0.005)
			self._nbArrete = len(convexes)
		else:
			self._nbArrete = 0


	def _save(self,path):
		cv2.imwrite(path,self._image)

	# ***************************************************** #	


	def _histogrammecouleur(self):
		img = cv2.cvtColor(self._image, cv2.COLOR_BGR2HLS)
		h = np.zeros((3,256),dtype=np.uint8)
		s,t,k = img.shape
		
		for c in range(k):
			for j in range(s):               #histograme compte le nombre de pixelle couleur dans chacune des matrice couleur
				for i in range(t):
					valeur = image[j,i,c]
					if (valeur != 0):
						h[c][valeur] += 1
		return h

	def _extract_lbp_feateur(self,img,points):
	
		
		pts=np.array(points)
		
		
		for i in pts:
			if i[0]< 0:
				i[0] = 0
			if i[1] < 0:
				i[1] = 0

		
		
		rect = cv2.boundingRect(pts)
		x,y,w,h = rect
		croped = img[y:y+h, x:x+w].copy()
		pts = pts - pts.min(axis=0)

		mask = np.zeros(croped.shape[:2], np.uint8)
		cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

		## (3) do bit-op
		dst = cv2.bitwise_and(croped, croped, mask=mask)

		## (4) add the white background
		bg = np.ones_like(croped, np.uint8)*255
		cv2.bitwise_not(bg,bg, mask=mask)
		dst2 = bg+ dst
		k=0
		
		ligne,col = dst.shape
		
		
		n_bins = dst.max() + 1
		#hist_lbp, _ = np.histogram(dst, normed=True, bins=n_bins, range=(2, n_bins))
		
		x = itemfreq(dst.ravel())
		hist_lbp = x[:,1]/ sum(x[:,1])
		
		
		return hist_lbp

	def get_pixel(self,img, center, x, y):
	    new_value = 0
	    try:
	        if img[x][y] >= center:
	            new_value = 1
	    except:
	        pass
	    return new_value

	def _cerculelement(self, nbtranche,nbcercul):

		"""
	trouve le rayon de l'image est la decoupe en cercle puit on tranche a l'interier de chaque cercle 
	nbtranche : nombre de tranche a decouper 
	nbcercul : nb cercle 
	retourne une liste des coeficient des tranche 

		"""
		img = self.lbpImg.copy()
		cnt = self._getContourExterne()

		#(x,y),_ = cv2.minEnclosingCircle(cnt)

		elipse =cv2.fitEllipse(cnt)	
		(x,y) = elipse[0]
		centre = (int(x),int(y))

		hull = cv2.convexHull(cnt)

		distMax = math.sqrt(math.pow(x - hull[0,0,0],2) + math.pow(y - hull[0,0,1],2))

		X=0

		Y=0

		for i in range(len(hull)):
			dist = math.sqrt(math.pow(x - hull[i,0,0],2) + math.pow(y - hull[i,0,1],2))
			if dist > distMax:
				distMax = dist
				X = hull[i,0,0]
				Y = hull[i,0,1]
		rad = int(distMax/nbcercul)
		distMax = int(distMax)
		rad1 = rad
		a= int(x + rad)
		b= int(y)
		cv2.line(img,(a,b),centre,[255,255,255] ,2)
		rayon = 360 / nbtranche
		boole = True
		i=0
		featurs = []
		while (rad1 <= distMax):

			rk = rayon
			kb = 0

			while(rk <= 360):
				points = []
				points = [[int(x),int(y)]]
				if boole :
					points.append([int(a),int(b)])
					boole = False
				else:
					points.append([int(d),int(v)])
				
				
				jk = rayon/6 + (i*kb)

				while (jk <= rk):
					d,v=self.rotation(a,b,x,y,jk)
					points.append([int(d),int(v)])
					jk+=rayon/6
				
				d,v=self.rotation(a,b,x,y,rk)
				
				point =(int(d),int(v))

				points.append([int(d),int(v)])
				
				
				featurs.append(self._extract_lbp_feateur(img,points))
				
				cv2.line(img,point,centre,[255,255,255] ,2)
				
				
				kb = rk
				rk+=rayon
				i=1

			cv2.circle(img, centre, rad1,[255,255,255] ,2)
			boole = True
			rad1+=rad
			if rad1 <= distMax :
				a= int(x + rad1)
				cv2.line(img,(a,b),centre,[255,255,255] ,2)

		return img,featurs

#****************************************************************************************************
	
	def rotation(self,x,y,a,b,angle):
		X = x - a
		Y = y - b
		xBis = a + X * cos(radians(angle)) - Y * sin(radians(angle))
		yBis = b + X * sin(radians(angle)) + Y * cos(radians(angle))
		return xBis,yBis



	def change_img_background(self):
		img = self._image

		hls = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

		lower = np.array([0,30,0],dtype='uint8')

		upper = np.array([70,255,255],dtype='uint8')

		mask = cv2.inRange(hls, lower, upper)

		res = cv2.bitwise_and(img,img, mask=mask)
		res = cv2.GaussianBlur(res,(5,5),0)

		self._image = res


	def _histcolor(self,img):
		image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		color = ('r','g','b')
		feature = []
		n_bins = image.max() + 1
		for i,col in enumerate(color):
			hist_coul = cv2.calcHist([image], [i],None, [255],[2,255])
			cv2.normalize(hist_coul,hist_coul)
			feature.append(hist_coul)
		return feature
	

	def _fd_histogram(self):
	    
	    # converssion HLS 
	    image = cv2.cvtColor(self._image, cv2.COLOR_BGR2HLS)
	    # calcul d'histogramme
	    hist  = cv2.calcHist([image], [0, 1,2], None, [255,255,255], [5, 254,5,254,5,254])


	    cv2.normalize(hist, hist)
	    #histed = hist[0:2,:,:]

	    return hist.mean(axis=0).flatten()
	def _lbp_calculated_pixel(self, angle, rayon):
	    """
	   Nous nous somme basé sure se model est nous avons essayé de tirer parti de ses avantage  est changer ses inconvenant on nous basons 
	   sur les travaux de (Ojala et al)  nous avant ajouter 2 paramètre :
		1-	Le rayon du cercle r, qui permet de rendre compte des différentes échelles
		2-	L’ongle de rotation donner on degré, permet de déterminer un voisinage de symétrie circulaire 

	     64 | 128 |   1
	    ----------------
	     32 |   0 |   2
	    ----------------
	     16 |   8 |   4    

	    """
	    self.change_img_background()
	    img = cv2.cvtColor(self._image,cv2.COLOR_BGR2GRAY)

	    height, width = img.shape

	    
	    img_lbp = np.zeros((height, width), np.uint8)
	    
	    
	    for i in range(0, height):
	    	a=i+rayon
	    	for j in range(0, width):
				
		
	    		center = img[i][j]

	    		if center == 0  :
	    			img_lbp[i][j] = 0
	    			continue

	    		
	    		b=j
	    		rk = angle
	    		val_ar = []
	    		val_ar.append(self.get_pixel(img, center, a, b))
	    		while(rk < 360):
	    			d,v=self.rotation(a, b, i, j,rk)
	    			d = int(d)
	    			v = int(v)
	    			val_ar.append(self.get_pixel(img, center, d,v))
	    			rk+=angle
	    		val = 0
	    		
	    		for x in range(len(val_ar)):
	    			val += val_ar[x] * pow(2,x)

	    		#print(val)	
	    		img_lbp[i][j]=val
	    self.lbpImg = np.copy(img_lbp)
	    self.lbpImg=cv2.GaussianBlur(self.lbpImg,(5,5),0)

	    return img_lbp 

    
	def _show_output(self,output_list):
		output_list_len = len(output_list)
		figure = plt.figure()
		for i in range(output_list_len):
		    current_dict = output_list[i]
		    current_img = current_dict["img"]
		    current_xlabel = current_dict["xlabel"]
		    current_ylabel = current_dict["ylabel"]
		    current_xtick = current_dict["xtick"]
		    current_ytick = current_dict["ytick"]
		    current_title = current_dict["title"]
		    current_type = current_dict["type"]
		    current_plot = figure.add_subplot(1, output_list_len, i+1)
		    if current_type == "gray":
		     	current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
		     	current_plot.set_title(current_title)
		     	current_plot.set_xticks(current_xtick)
		     	current_plot.set_yticks(current_ytick)
		     	current_plot.set_xlabel(current_xlabel)
		     	current_plot.set_ylabel(current_ylabel)
		    elif current_type == "histogram":
		     	current_plot.plot(current_img)
		     	current_plot.set_xlabel(current_xlabel)
		     	current_plot.set_ylabel(current_ylabel)
		     	current_plot.set_title(current_title)


		plt.show()
#****************************************Courbe****************************************
	
	def _B_spline_evaluation(self,pts):
		ctr=np.array(pts)
		x=ctr[:,0]
		y=ctr[:,1]

		l=len(x)  

		t=np.linspace(0,1,l-2,endpoint=True)
		t=np.append([0,0,0],t)
		t=np.append(t,[1,1,1])

		tck=[t,[x,y],3]
		u3=np.linspace(0,1,num=100,endpoint=True)
		out = interpolate.splev(u3,tck)
		"""
		plt.plot(x,y,'k--',label='polygone de controle',marker='o',markerfacecolor='red')
		#plt.plot(x,y,'ro',label='Control points only')
		plt.plot(out[0],out[1],'b',linewidth=2.0,label='Courbe B-spline ')
		plt.legend(loc='best')
		plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
		plt.title('B-spline evaluation courbe')
		plt.show()
		"""
		return out
	

	def _B_spline_interpolite(self,pts,n):
		ctr=np.array(pts)
		x=ctr[:,0]
		y=ctr[:,1]

		tck,u = interpolate.splprep([x,y],k=3,s=0)
		u=np.linspace(0,1,n,endpoint=True)
		out = interpolate.splev(u,tck)
		"""
		plt.figure()
		plt.plot(x, y, 'ro', out[0], out[1], 'b')
		plt.legend(['Points', 'Interpolated B-spline', 'True'],loc='best')
		plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
		plt.title('B-Spline interpolation')
		plt.show()
		"""
		return out

#****************************************Couleur**************************************
	
	def _cerculelementcouleur(self, nbtranche,nbcercul):

		"""
	trouve le rayon de l'image est la decoupe en cercle puit on tranche a l'interier de chaque cercle 
	nbtranche : nombre de tranche a decouper 
	nbcercul : nb cercle 
	retourne une liste des coeficient des tranche 

		"""
		img = self._image.copy()
		cnt = self._getContourExterne()

		#(x,y),_ = cv2.minEnclosingCircle(cnt)

		elipse =cv2.fitEllipse(cnt)	
		(x,y) = elipse[0]
		centre = (int(x),int(y))

		hull = cv2.convexHull(cnt)

		distMax = math.sqrt(math.pow(x - hull[0,0,0],2) + math.pow(y - hull[0,0,1],2))

		X=0

		Y=0

		for i in range(len(hull)):
			dist = math.sqrt(math.pow(x - hull[i,0,0],2) + math.pow(y - hull[i,0,1],2))
			if dist > distMax:
				distMax = dist
				X = hull[i,0,0]
				Y = hull[i,0,1]
		rad = int(distMax/nbcercul)
		distMax = int(distMax)
		rad1 = rad
		a= int(x + rad)
		b= int(y)
		cv2.line(img,(a,b),centre,[255,255,255] ,2)
		rayon = 360 / nbtranche
		boole = True
		i=0
		featurs = []
		while (rad1 <= distMax):

			rk = rayon
			kb = 0

			while(rk <= 360):
				
				points = [[int(x),int(y)]]
				if boole :
					points.append([int(a),int(b)])
					boole = False
				else:
					points.append([int(d),int(v)])
				
				
				jk = rayon/6 + (i*kb)

				while (jk <= rk):
					d,v=self.rotation(a,b,x,y,jk)
					points.append([int(d),int(v)])
					jk+=rayon/6
				
				d,v=self.rotation(a,b,x,y,rk)
				
				point =(int(d),int(v))

				points.append([int(d),int(v)])
				
				
				featurs.append(self._extract_Couleur_feateur(img,points))
				
				cv2.line(img,point,centre,[255,255,255] ,2)
				
							
				kb = rk
				rk+=rayon
				i=1

			cv2.circle(img, centre, rad1,[255,255,255] ,2)
			boole = True
			rad1+=rad
			if rad1 <= distMax :
				a= int(x + rad1)
				cv2.line(img,(a,b),centre,[255,255,255] ,2)

		return img,featurs

#******************************************

	def	_extract_Couleur_feateur(self,img,points):
		pts=np.array(points)
		
		
		for i in range(np.size(pts[:,0])):
			if pts[i,0] < 0:
				pts [i,0] = 0
			if pts[i,1] < 0:
				pts[i,1] =0

		
		rect = cv2.boundingRect(pts)
		x,y,w,h = rect
		croped = img[y:y+h, x:x+w].copy()
		pts = pts - pts.min(axis=0)

		mask = np.zeros(croped.shape[:2], np.uint8)
		cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

		## (3) do bit-op

		dst = cv2.bitwise_and(croped, croped, mask=mask)

		## (4) add the white background
		bg = np.ones_like(croped, np.uint8)*255
		cv2.bitwise_not(bg,bg, mask=mask)
		dst2 = bg+ dst
		k=0
			
		output_list = []
		
		
		#hist_lbp, _ = np.histogram(dst, normed=True, bins=n_bins, range=(2, n_bins))
		
		hist_couleur=self._histcolor(dst)
		hist_couleur=np.array(hist_couleur)

		"""
		plt.subplot(121),plt.imshow(dst),plt.xticks([]),plt.yticks([]),plt.title("Couleur")
		plt.subplot(122),plt.plot(hist_couleur[0],color= 'r'),plt.plot(hist_couleur[1],color= 'g'),plt.plot(hist_couleur[2],color= 'b'),plt.title("Couleur Histogramme"),plt.xlim([0,256])
		plt.show()
		self._show_output(output_list)
		"""
		return hist_couleur
	
	
	def _distance_Eclide (self,out):
		distancepoint = []
		for i  in range(len(out[0]) -1 ):
			dist = math.sqrt(math.pow(out[0][i] - out[0][i+1],2) + math.pow(out[1][i] - out[1][i+1],2))
			distancepoint.append(dist)
		distancepoint=np.array(distancepoint)
		return np.sum(distancepoint)
	
	
	def _distance_Max (self,out):
		cnt = self._getContourExterne()

		#(x,y),_ = cv2.minEnclosingCircle(cnt)

		elipse =cv2.fitEllipse(cnt)	
		(x,y) = elipse[0]
		centre = (int(x),int(y))

		hull = cv2.convexHull(cnt)

		distMax = math.sqrt(math.pow(x - hull[0,0,0],2) + math.pow(y - hull[0,0,1],2))

		X=0

		Y=0

		for i in range(len(hull)):
			dist = math.sqrt(math.pow(x - hull[i,0,0],2) + math.pow(y - hull[i,0,1],2))
			if dist > distMax:
				distMax = dist
				X = hull[i,0,0]
				Y = hull[i,0,1]
		return distMax

	
	def _chabyshev_distancepoint(cls,p, q):

	    return max(abs(p[0]-q[0]),abs(p[1]-q[1]))
	_chabyshev_distancepoint = classmethod(_chabyshev_distancepoint)
	

	def _MatDistance(cls,point,point2):
		CoMatrice =[]
		assert len(point) == len(point2) 
		for p in point:
			Ligne = []
			for q in point2:
				Ligne.append(cls._chabyshev_distancepoint(p, q))
			CoMatrice.append(Ligne)
		return CoMatrice
	_MatDistance = classmethod(_MatDistance)
	

	def _TextureCompare(cls,lbpfatear1):
		mon_fichier = h5py.File('./New.h5', 'r')

		featurs =[]  
		fg = OrderedDict() 
		list_elem=[key for key in mon_fichier['/']["Texture"].keys()]
		list_elem=sorted(list_elem)
		for z in list_elem:
		    print(z)
		    elem =[key for key in mon_fichier['/']["Texture"][z].keys()]
		    elem  = sorted(elem)
		    for x in elem:
		        print(x)
		        val = cls.TextureComp(lbpfatear1, z, x)
		       	ig = OrderedDict()
		       	
		       	ig[x] = [val,z]
		       	fg.update(ig)
		clean_dict = OrderedDict(filter(lambda tup: not math.isnan(tup[1][0]), fg.items()))
		d=sorted(clean_dict.items(),key=lambda f: f[1][0])
		mon_fichier.close()
		return d[:100]
	_TextureCompare = classmethod(_TextureCompare)
#**********************************Couleur Compare**************************************************
	def _CouleurCompare(cls,featurecoul):
		mon_fichier = h5py.File('./New.h5', 'r')
		color = ('r','g','b')
		fg = OrderedDict()  
		list_elem=[key for key in mon_fichier['/']["Couler"].keys()]
		list_elem=sorted(list_elem)
		featurs =[]
		for z in list_elem:
		    print(z)
		    elem =[key for key in mon_fichier['/']["Couler"][z].keys()]
		    elem  = sorted(elem)
		    for x in elem:
		        val = cls.CouleurComp(featurecoul, z, x)
		        ig = OrderedDict()
		        
		        ig[x] = [val,z]
		        fg.update(ig)


		clean_dict = OrderedDict(filter(lambda tup: not math.isnan(tup[1][0]), fg.items()))
		d=sorted(clean_dict.items(),key=lambda f: f[1][0])
		mon_fichier.close()
		return d[:100]
	_CouleurCompare = classmethod(_CouleurCompare)
#******************************************************Courbe Compare***********************************************
	def _CourbeCompare(cls,out,d2,out2):
		mon_fichier = h5py.File('./New.h5', 'r')
		point3=[]
		if out2 !=[]:
			point3=[[g,l] for g,l in zip(out2[0],out2[1])]
		point1=[[g,y] for g,y in zip(out[0],out[1])]

		fg = OrderedDict() 
		list_elem=[key for key in mon_fichier['/']["CourbeEXT6"].keys()]
		list_elem=sorted(list_elem)
		for z in list_elem:
		    print(z)
		    elem =[key for key in mon_fichier['/']["CourbeEXT6"][z].keys()]
		    elem  = sorted(elem)
		    for x in elem:
		        print(x)
		          
		        #k+=coff_contour
		        val = cls.ComparForm(point1, point3, d2, z, x)
		        ig = OrderedDict()
		        ig[x] = [val,z]
		        fg.update(ig)
		clean_dict = OrderedDict(filter(lambda tup: not math.isnan(tup[1][0]), fg.items()))
		d=sorted(clean_dict.items(),key=lambda f: f[1][0])
		mon_fichier.close()
		
		return d[:100]
	_CourbeCompare = classmethod(_CourbeCompare)
#****************************************************************Hebride******************************
			
	def TextureComp(cls,lbpfatear1,z,x):
		featurs1 =[]	
		mon_fichier = h5py.File('./New.h5', 'r')
		for t in range(30):
			s=mon_fichier["/Texture/"+ z +"/" + x +"/" + str(t)]
			y=min(len(lbpfatear1[t]),len(s[:]))
			score = cv2.compareHist(np.array(lbpfatear1[t][:y],dtype=np.float32), np.array(s[:y],dtype=np.float32),cv2.HISTCMP_CHISQR)
			featurs1.append(round(score,3))
		rgt = np.array(featurs1)
		valtx = rgt.mean(axis=0)
		mon_fichier.close()
		return valtx
	TextureComp = classmethod(TextureComp)
	
	def CouleurComp(cls,featurecoul,z,x):
		featurs =[]
		color =('r','g','b')
		mon_fichier = h5py.File('./New.h5', 'r')
		t=0
		for fatear in featurecoul:
			result =0
			score =1
			for v,i in enumerate(color):
				s=mon_fichier["/Couler/"+ z +"/" + x +"/" + str(t) + "/" + i]
				score = cv2.compareHist(np.array(fatear[v],dtype=np.float32), np.array(s[:],dtype=np.float32),cv2.HISTCMP_CHISQR)
				result += round(score,3)
			t+=1
			featurs.append(result/3)
	    
		rg = np.array(featurs)
		valcl = rg.mean(axis=0)
		return valcl
	CouleurComp = classmethod(CouleurComp)
	
	def ComparForm(cls,point1,point3,d2,z,x):
		mon_fichier = h5py.File('./New.h5', 'r')

		aut=mon_fichier["/CourbeEXT6/"+ z +"/" + x +"/" + "X"]
		aut1=mon_fichier["/CourbeEXT6/"+ z +"/" + x +"/" + "Y"]
		dist=mon_fichier["/CourbeEXT6/"+ z +"/" + x +"/" + "Dist"]
		coff_contour1 = 10000
		point2=[[g,l] for g,l in zip(aut[:],aut1[:])]
		if(([key for key in mon_fichier['/']["CourbeINT8"][z][x].keys()]!=[]) and point3 != [] ):
			aut2=mon_fichier["/CourbeINT8/"+ z +"/" + x +"/" + "X"]
			aut3=mon_fichier["/CourbeINT8/"+ z +"/" + x +"/" + "Y"]
			point4=[[g,l] for g,l in zip(aut2[:],aut3[:])]
			mat1=cls._MatDistance(point3,point4)
			cost1 = np.array(mat1)
			lign1,col1=linear_sum_assignment(cost1)
			coff_contour1=cost1[lign1,col1].sum()
		mat=cls._MatDistance(point1,point2)
		cost = np.array(mat)
		lign,col=linear_sum_assignment(cost)
		coff_contour=cost[lign,col].sum()
		d1=dist.value
		distance = abs(d2 - d1)
		mon_fichier.close()
		return coff_contour + distance*5000 + coff_contour1/7
	ComparForm = classmethod(ComparForm)

#******************************************************************************
	
	def _CompareHebride(cls,point1,point3,d2,featurecoul,lbpfatear1,lens):
		mon_fichier = h5py.File('./New.h5', 'r')
		fg = OrderedDict()  
		list_elem=[key for key in mon_fichier['/']["Couler"].keys()]
		list_elem=sorted(list_elem)
		for z in list_elem:
		    print(z)
		    elem =[key for key in mon_fichier['/']["Couler"][z].keys()]
		    elem  = sorted(elem)
		    for x in elem:
		    	with ThreadPoolExecutor(max_workers=3) as execution:
		    		Work =execution.submit(cls.TextureComp,lbpfatear1, z,x)
		    		Work1 =execution.submit(cls.CouleurComp,featurecoul, z,x)
		    		Work2 =execution.submit(cls.ComparForm,point1,point3,d2,z,x)
		    	while not ((Work.done()) and (Work1.done()) and (Work2.done())):
				    continue
		    	valtx=Work.result()
		    	valcl=Work1.result()
		    	form =Work2.result()
		    	val = ((form/10000) + (valtx *10))*0.8+ (valcl/300)*0.2
		    	ig = OrderedDict()
		    	ig[x] = [val,z]
		    	fg.update(ig)


		clean_dict = OrderedDict(filter(lambda tup: not math.isnan(tup[1][0]), fg.items()))
		d=sorted(clean_dict.items(),key=lambda f: f[1][0])
		mon_fichier.close()
		return d[:lens]
	_CompareHebride = classmethod(_CompareHebride)
	