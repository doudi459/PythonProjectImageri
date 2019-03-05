import traceback,sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import numpy as np
from matplotlib.figure import Figure
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from MainWindow import *
from image import *
import cv2
import copy
import time
import os
from collections import OrderedDict

PROGRESS_BAR = """ 
QProgressBar {
border : 2px solid gray;
border-radius : 5px;
text-align : center ;
}

QProgressBar::chunk{
	background-color : #4CAF50;
	width : 10px;
	border-radius : 4px;
}
"""

GROUP_BOX = """
QGroupBox{
border : 2px solid gray;
border-radius : 5px;
background-color:#f2f2f2;
}
"""

GROUP_BOX2 ="""
QGroupBox{
border : 2px solid gray;
border-radius : 5px;
}
"""

MAIN_TAB ="""
QTabWidget{
font-size:18px;
}
"""
class MainWindow(QMainWindow,Ui_MainWindow):
	def __init__(self,parent=None):
		QMainWindow.__init__(self)
		self.setupUi(parent)
		
		self.threadpool = QThreadPool()
		self.approxValue = 0.005
		self.sigmaXvalue = 1
		self.modele= False
		self.listRecherche = lambda: None

		self.widgetPlotImage = MplWidget(self.groupBoxAffichage)
		self.widgetPlotImage.setObjectName("widgetPlotImage")
		self.layoutPlotImage.addWidget(self.widgetPlotImage)
		
		self.widgetPlotContour = MplWidget(self.groupBoxAffichage)
		self.widgetPlotContour.setObjectName("widgetPlotContour")
		self.layoutPlotContour.addWidget(self.widgetPlotContour)

		# courbes b-spline
		self.widgetPlotCourbe = MplWidget(self.forme)
		self.widgetPlotCourbe.setObjectName("widgetPlotCourbe")
		self.layoutBspline.addWidget(self.widgetPlotCourbe)

		self.widgetPlotCourbe2 = MplWidget(self.forme)
		self.widgetPlotCourbe2.setObjectName("widgetPlotCourbe2")
		self.layoutBspline2.addWidget(self.widgetPlotCourbe2)


		# histogramme LBP
		self.widgetPlothistLBP = MplWidget(self.forme)
		self.widgetPlothistLBP.setObjectName("widgetPlothistLBP")
		self.layoutHistLBP.addWidget(self.widgetPlothistLBP)
		
		# histogramme couleur
		self.widgetPlothistHLS = MplWidget(self.forme)
		self.widgetPlothistHLS.setObjectName("widgetPlothistHLS")
		self.layoutPlothistHLS.addWidget(self.widgetPlothistHLS)


		#save_btn clicked event
		self.save_btn.clicked.connect(self.save_image)
		#image_btn clicked event	
		self.image_btn.clicked.connect(self.import_image)

		#pushB clicked events
		self.pushBrotateLeft.clicked.connect(lambda:self.pushBclick(self.pushBrotateLeft))
		self.pushBrotateRight.clicked.connect(lambda:self.pushBclick(self.pushBrotateRight)) 
		self.pushBrotate.clicked.connect(lambda:self.pushBclick(self.pushBrotate))
		self.pushBimageFS.clicked.connect(lambda:self.pushBclick(self.pushBimageFS))
		self.pushBcontourFS.clicked.connect(lambda:self.pushBclick(self.pushBcontourFS))
		self.pushBvider.clicked.connect(lambda:self.pushBclick(self.pushBvider))

		#radioBs clicked events
		self.radioBcontourEx.clicked.connect(lambda:self.radioBClick(self.radioBcontourEx))
		self.radioBcontourIn.clicked.connect(lambda:self.radioBClick(self.radioBcontourIn))
		self.radioBimgSource.clicked.connect(lambda:self.radioBClick(self.radioBimgSource))
		self.checkBshowApprox.clicked.connect(lambda:self.radioBClick(self.checkBshowApprox))
		self.checkBshowPointsC.clicked.connect(lambda:self.radioBClick(self.checkBshowPointsC))

		self.radioBtexture.clicked.connect(lambda:self.radioBClick(self.radioBtexture))
		self.radioBcouleur.clicked.connect(lambda:self.radioBClick(self.radioBcouleur))
		self.radioBforme.clicked.connect(lambda:self.radioBClick(self.radioBforme))
		self.radioBtous.clicked.connect(lambda:self.radioBClick(self.radioBtous))

		#horizontalSliderG valueChanged event
		self.horizontalSliderG.valueChanged.connect(lambda:self.horizontalSliderFlouValueChanged(self.horizontalSliderG))
		#horizontalSliderM valueChanged event
		self.horizontalSliderM.valueChanged.connect(lambda:secheckBClicklf.horizontalSliderFlouValueChanged(self.horizontalSliderM))


		# spinBsigmaX value changed event
		self.spinBsigmaX.valueChanged.connect(lambda:self.spinBvalueChanged(self.spinBsigmaX))
		#spinBapprox value changed event
		self.spinBapprox.valueChanged.connect(lambda:self.spinBvalueChanged(self.spinBapprox))

		#checkBoxfrome clicked event
		self.checkBcolor.clicked.connect(lambda:self.checkBClick(self.checkBcolor))
		self.checkBtexture.clicked.connect(lambda:self.checkBClick(self.checkBtexture))

		self.pushBrechechre.clicked.connect(self.pushBrechechreClicked)
		self.mainTab.currentChanged.connect(self.mainTab_tabChanged)



		#Style sheet 
		self.progressBar.setStyleSheet(PROGRESS_BAR)
		self.groupBox_3.setStyleSheet(GROUP_BOX)
		self.groupBox_4.setStyleSheet(GROUP_BOX)
		self.groupBspline.setStyleSheet(GROUP_BOX2)
		self.groupBLBP.setStyleSheet(GROUP_BOX2)
		self.groupBcouleur.setStyleSheet(GROUP_BOX2)
		self.groupBox.setStyleSheet(GROUP_BOX2)

		self.mainTab.setStyleSheet(MAIN_TAB)
	
	def pushBrechechreClicked(self):
		try:
			self.viderLayout()
			self.lcdNumber.display(0)
			self.labelProgression.setText("Recherche en cours. Veuillez patientez !!")

			worker = Worker(self.recherche)
			worker.signals.finished.connect(self.afficher_Recherche)
			# lancement du thread
			self.threadpool.start(worker)
			
		except Exception as e:
			print(e)

	def recherche(self):
		try:	
			if(self.radioBtexture.isChecked() == True):
				self.listRecherche = Image._TextureCompare(self.histTexture)
			elif(self.radioBcouleur.isChecked() == True):
				self.listRecherche = Image._CouleurCompare(self.featureCoul)
			elif(self.radioBforme.isChecked() == True):
				self.listRecherche = Image._CourbeCompare(self.outEx,self.distanceR,self.outIn)
			elif(self.radioBtous.isChecked() == True):
				point1 =[[x,y] for x,y in zip(self.outEx[0],self.outEx[1])]
				point3 = []
				if(self.outIn != []):
					point3 = [[x,y] for x,y in zip(self.outIn[0],self.outIn[1])]

				self.listRecherche = Image._CompareHebride(point1,point3,self.distanceR,self.featureCoul,self.histTexture,120)
			
		except Exception as e:
			print(e)

	def afficher_Recherche(self):
		try:
			if(self.listRecherche is not None):
				x = self.listRecherche
				taux = 0
				classe, nb = self.imageClasse()
				
				for e,i in enumerate(x):
					
					if(classe == i[1][1]):
						taux +=1
					path = "Leaves/"+i[0]
					m = self.createLabel(path)
					if(e%7 == 0):
						self.layoutImages1.addWidget(m)
					elif(e%7 == 1):
						self.layoutImages2.addWidget(m)
					elif(e%7 == 2):
						self.layoutImages3.addWidget(m)
					elif(e%7 == 3):
						self.layoutImages4.addWidget(m)
					elif(e%7 == 4):
						self.layoutImages5.addWidget(m)
					elif(e%7 == 5):
						self.layoutImages6.addWidget(m)
					elif(e%7 == 6):
						self.layoutImages7.addWidget(m)

				print("Classe",classe)
				print("nombre d'image de la classe",nb)
				taux = taux/nb
				print("taux",taux)

				self.lcdNumber.display(taux*100)
			
			self.labelProgression.setText("")
		except Exception as e:
			print(e)
	
	def imageClasse(self):
		try:
			path = self.lineEditPath.text()
			l = path.split("/")
			myImageName = l[-1]
			print("My image Name",myImageName)

			classNames = os.listdir("LeavesClasses/")
			for file in classNames:
				imageNames = os.listdir("LeavesClasses/"+file+"/")
				for name in imageNames:
					if(name == myImageName):
						return file,len(imageNames)
		except Exception as e:
			print(e)
	

	def createLabel(self,path):
		img =  QLabel(self.reconnaissance)
		pixmap = QPixmap(path)
		img.resize(105,90)
		pixmap = pixmap.scaled(img.width(),img.height(),Qt.KeepAspectRatio)
		img.setPixmap(pixmap)

		return img

	def viderLayout(self):
		while self.layoutImages1.count():
			child = self.layoutImages1.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
		
		while self.layoutImages2.count():
			child = self.layoutImages2.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
		
		while self.layoutImages3.count():
			child = self.layoutImages3.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
		
		while self.layoutImages4.count():
			child = self.layoutImages4.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
		
		while self.layoutImages5.count():
			child = self.layoutImages5.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
		
		while self.layoutImages6.count():
			child = self.layoutImages6.takeAt(0)
			if child.widget():
				child.widget().deleteLater()
		
		while self.layoutImages7.count():
			child = self.layoutImages7.takeAt(0)
			if child.widget():
				child.widget().deleteLater()


	# sauvegarde de l'image
	def save_image(self):
		file = QFileDialog.getSaveFileName(self,"savegarde d'image" ,'c:/' , "Image files (*.jpg )")
		if file[0] != "":
			self.imgSource._save(file[0])

	# importation de l'image
	def import_image(self):
		try :
			file  = QFileDialog.getOpenFileName(self,"ouverture d'image",'c:/',"Image files (*.jpg)")
			if file[0]!="":

				worker = Worker(self.load_Image,file[0])
				# lancement du thread
				self.threadpool.start(worker)
				
				# progression
				self.progression(5,"Chargement en cours")
				self.resetElements()
				self.enableButtons()
				self.radioBcontourEx.setChecked(True)
				self.radioBimgSource.setChecked(True)
		except Exception as e:
			print(e)

	def progression(self,duree,label):
		self.labelOpr.setText(label)
		u = 100/duree
		for i in range(0,duree):
			time.sleep(1)
			self.progressBar.setValue(i*u + u)
				
	
	def load_Image(self,path):
		try:	
			# construction de l'objet "imgSource" de la classe Image
			self.imgSource = Image(path)
			self.modele = False
			self.show_image(self.widgetPlotImage,self.imgSource._image)
			self.show_image(self.widgetPlotContour,self.imgSource._contourEx)
			self.nbPointsC.setText(str(self.imgSource._nbPointsC))
			self.spinBArretes.setValue(self.imgSource._nbArrete)
			self.lineEditPath.setText(path)
			cv2.imwrite("color.jpg",self.imgSource._image)

			self.clearAxis()
		except Exception as e:
			print(e)
	
	def clearAxis(self):
		try:
			self.widgetPlothistLBP.canvas.ax.cla()
			self.widgetPlothistLBP.canvas.ax.plot([])
			self.widgetPlothistLBP.canvas.draw()

			self.widgetPlothistHLS.canvas.ax.cla()
			self.widgetPlothistHLS.canvas.ax.plot([])
			self.widgetPlothistHLS.canvas.draw()

			self.widgetPlotCourbe2.canvas.ax.cla()
			self.widgetPlotCourbe2.canvas.ax.plot([])
			self.widgetPlotCourbe2.canvas.draw()

			self.widgetPlotCourbe.canvas.ax.cla()
			self.widgetPlotCourbe.canvas.ax.plot([])
			self.widgetPlotCourbe.canvas.draw()
			print("Axis cleard")
		except Exception as e:
			print(e)
	
	def show_image(self,obj,img):
		try:
			obj.canvas.ax.cla()
			obj.canvas.ax.imshow(img,'gray')
			obj.canvas.ax.set_xticks([])
			obj.canvas.ax.set_yticks([])
			obj.canvas.draw()
		
		except Exception as e:
			print(e)
	
	def show_bspline(self,obj,plot,title):
		try:
			obj.canvas.ax.cla()
			obj.canvas.ax.plot(plot[0],plot[1],color='b',linewidth=2.0)
			obj.canvas.ax.set_title(title)
			obj.canvas.draw()
		
		except Exception as e:
			print(e)
	
	def show_histLBP(self):
		try:
			self.widgetPlothistLBP.canvas.ax.cla()
			self.widgetPlothistLBP.canvas.ax.plot(self.hist)
			self.widgetPlothistLBP.canvas.ax.set_title("Histogramme LBP")
			self.widgetPlothistLBP.canvas.ax.set_xlabel("Bins")
			self.widgetPlothistLBP.canvas.ax.set_ylabel("number of pixels")
			self.widgetPlothistLBP.canvas.draw()
		except Exception as e:
			print(e)

	def show_histCouleur(self):
		try:
			R=[]
			V=[]
			B=[]

			for x in self.featureCoul:
				R.append(x[0])
				V.append(x[1])
				B.append(x[2])

			self.widgetPlothistHLS.canvas.ax.cla()
			self.widgetPlothistHLS.canvas.ax.plot(np.array(R).flatten(),color='r')
			self.widgetPlothistHLS.canvas.ax.plot(np.array(V).flatten(),color='g')
			self.widgetPlothistHLS.canvas.ax.plot(np.array(B).flatten(),color='b')
			self.widgetPlothistHLS.canvas.ax.set_xlim([0,256])
			self.widgetPlothistHLS.canvas.ax.set_title("Histogramme couleur")
			self.widgetPlothistHLS.canvas.ax.set_xlabel("value")
			self.widgetPlothistHLS.canvas.ax.set_ylabel("number of pixels")
			self.widgetPlothistHLS.canvas.ax.legend(['rouge','vert','bleu'],loc='upper right')
			self.widgetPlothistHLS.canvas.draw()
		
		except Exception as e:
			print(e)
	

	def pushBclick(self,obj):
		try:
			if(obj.objectName() == "pushBrotateLeft"):
				worker = Worker(self.rotateImage,90)
				workerP = Worker(self.progression,6,"Rotation en cours")
				worker.signals.finished.connect(self.display)
				self.threadpool.start(worker)
				self.threadpool.start(workerP)
				cv2.imwrite("color.jpg",self.imgSource._image)
				

			elif (obj.objectName() ==  "pushBrotateRight"):
				worker = Worker(self.rotateImage,-90)
				workerP = Worker(self.progression,6,"Rotation en cours")
				worker.signals.finished.connect(self.display)
				self.threadpool.start(worker)
				self.threadpool.start(workerP)
				cv2.imwrite("color.jpg",self.imgSource._image)
				
			elif (obj.objectName() == "pushBrotate"):
				val = self.spinBangle.value()
				if (val != 0):
					worker = Worker(self.rotateImage,val)
					workerP = Worker(self.progression,6,"Rotation en cours")
					worker.signals.finished.connect(self.display)
					self.threadpool.start(worker)
					self.threadpool.start(workerP)
					cv2.imwrite("color.jpg",self.imgSource._image)	
			
			elif (obj.objectName() == "pushBimageFS"):
				i = -1
				if (self.radioBimgSource.isChecked() == True):
					plt.subplot(111),plt.imshow(self.imgSource._image,'gray'),plt.show()
				elif (self.checkBshowApprox.isChecked() == True):
					plt.subplot(111),plt.imshow(self.imgSource._approxImage,'gray'),plt.show()
				else:
					plt.subplot(111),plt.imshow(self.imgSource._pointsCimage,'gray'),plt.show()
			
			elif (obj.objectName() == "pushBcontourFS"):
				i = -1
				if (self.radioBcontourEx.isChecked() == True ):
					plt.subplot(111),plt.imshow(self.imgSource._contourEx,'gray'),plt.show()	
				else:
					plt.subplot(111),plt.imshow(self.imgSource._contourIn,'gray'),plt.show()

			elif(obj.objectName() == "pushBvider"):
					self.viderLayout()
					self.lcdNumber.display(0)
		
		except Exception as e:
			print(e)
	
	def rotateImage(self,angle):
		self.imgSource._rotateImage(angle)

	def resetElements(self):
		self.horizontalSliderG.setValue(0)
		self.horizontalSliderM.setValue(0)

		self.spinBangle.setValue(0)
		self.spinBsigmaX.setValue(1)

		self.checkBtexture.setEnabled(False)
		self.checkBcolor.setEnabled(False)

		self.labelOpr.setText('')
		self.lcdNumber.display(0)
		self.viderLayout()
		self.labelTexture.setPixmap(QPixmap())
		self.labelCouleur.setPixmap(QPixmap())
		self.labelimg.setPixmap(QPixmap())
		

	def enableButtons(self):
		self.horizontalSliderG.setEnabled(True)
		self.horizontalSliderM.setEnabled(True)

		self.radioBimgSource.setEnabled(True)
		self.radioBcontourEx.setEnabled(True)
		self.radioBcontourIn.setEnabled(True)

		self.checkBshowApprox.setEnabled(True)
		self.checkBshowPointsC.setEnabled(True)

		self.save_btn.setEnabled(True)
		self.pushBimageFS.setEnabled(True)
		self.pushBcontourFS.setEnabled(True)

		self.spinBangle.setEnabled(True)
		self.pushBrotateLeft.setEnabled(True)
		self.pushBrotateRight.setEnabled(True)
		self.pushBrotate.setEnabled(True)

		self.spinBArretes.setEnabled(True)
		self.spinBsigmaX.setEnabled(True)
		self.spinBapprox.setEnabled(True)	

	def radioBClick(self,obj):
		try:
			if (obj.objectName() == "radioBcontourEx"):
				self.show_image(self.widgetPlotContour,self.imgSource._contourEx)			
			elif (obj.objectName() == "radioBcontourIn"):
				self.show_image(self.widgetPlotContour,self.imgSource._contourIn)
			elif (obj.objectName() == "radioBimgSource"):
				self.show_image(self.widgetPlotImage,self.imgSource._image)
			elif (obj.objectName() == "checkBshowApprox"):
				self.show_image(self.widgetPlotImage,self.imgSource._approxImage)		
			elif (obj.objectName() == "checkBshowPointsC"):
				self.show_image(self.widgetPlotImage,self.imgSource._pointsCimage)
			elif(obj.objectName() == "radioBtexture" or obj.objectName() == "radioBcouleur" or obj.objectName() == "radioBforme" or obj.objectName() == "radioBtous"):
				self.pushBrechechre.setEnabled(True)
		
		except Exception as e:
			print(e)
	
	def checkBClick(self,obj):
		try:
			if(obj.objectName() == "checkBtexture"):
				if(obj.isChecked() == True):

					pixmap = QPixmap("lbpCercle.jpg")
					pixmap = pixmap.scaled(self.labelTexture.width(),self.labelTexture.height(),Qt.KeepAspectRatio)
					self.labelTexture.setPixmap(pixmap)
				else:
					pixmap = QPixmap("lbp.jpg")
					pixmap = pixmap.scaled(self.labelTexture.width(),self.labelTexture.height(),Qt.KeepAspectRatio)
					self.labelTexture.setPixmap(pixmap)

			elif(obj.objectName() == "checkBcolor"):
				if(obj.isChecked() == True):
					pixmap = QPixmap("colorCercle.jpg")
					pixmap = pixmap.scaled(self.labelCouleur.width(),self.labelCouleur.height(),Qt.KeepAspectRatio)
					self.labelCouleur.setPixmap(pixmap)

				else:
					pixmap = QPixmap("color.jpg")
					pixmap = pixmap.scaled(self.labelCouleur.width(),self.labelCouleur.height(),Qt.KeepAspectRatio)
					self.labelCouleur.setPixmap(pixmap)
		
		except Exception as e:
			print(e)
	def horizontalSliderFlouValueChanged(self,obj):
		
		val = obj.value()
		if( val == 0):
			pass
		elif (val == 1):
			flou = 3
		elif (val == 2):
			flou = 5
		elif (val == 3):
			flou =7 
		
		if (obj.objectName() == "horizontalSliderG" and val!=0):
			self.imgSource._gaussianBlur(flou)
			worker = Worker(self.display)
			self.threadpool.start(worker)
			self.imgSource.approxImages(self.approxValue)
			self.horizontalSliderM.setValue(0)
			
					
		elif (obj.objectName() == "horizontalSliderM" and val!=0):
			self.imgSource._medianBlur(flou)
			worker = Worker(self.display)
			self.threadpool.start(worker)
			self.imgSource.approxImages(self.approxValue)		
			self.horizontalSliderG.setValue(0)
	

	def spinBvalueChanged(self,obj):
	
		if (obj.objectName() == "spinBsigmaX"):
			self.sigmaXvalue = obj.value()
			worker = Worker(self.fonction,1)
			worker.signals.finished.connect(lambda:self.fonction(3))
			self.threadpool.start(worker)			
			

		elif (obj.objectName() == "spinBapprox"):
			self.approxValue = float("0.00"+str(obj.value()))
			worker = Worker(self.fonction,0)
			worker.signals.finished.connect(lambda:self.fonction(2))
			self.threadpool.start(worker)
			self.checkBshowApprox.setChecked(True)

	def fonction(self,i):
		if (i == 0 ):
			self.imgSource.approxImages(self.approxValue)
			self.nbPointsC.setText(str(self.imgSource._nbPointsC))
		elif (i == 1):
			self.imgSource.contourImages(self.sigmaXvalue)
			self.imgSource.calculNbArrete(self.sigmaXvalue)
			self.spinBArretes.setValue(self.imgSource._nbArrete)
		elif (i == 2):
			self.show_image(self.widgetPlotImage,self.imgSource._approxImage)
		elif (i == 3):
			self.show_image(self.widgetPlotContour,self.imgSource._contourIn)
			self.radioBcontourIn.setChecked(True)

	def display(self):
		try:
			self.radioBimgSource.setChecked(True)
			self.show_image(self.widgetPlotImage,self.imgSource._image)
			
			self.imgSource.contourImages(self.sigmaXvalue)
			
			# affichage
			if (self.radioBcontourEx.isChecked() == True):
				self.show_image(self.widgetPlotContour,self.imgSource._contourEx)
			else:
				self.show_image(self.widgetPlotContour,self.imgSource._contourIn)	
			
			self.imgSource.approxImages(self.approxValue)
			self.labelOpr.setText('')			
		except Exception as e:
			print(e)
	def mainTab_tabChanged(self):
		try:
			if((self.mainTab.currentIndex() == 1) and (self.lineEditPath.text() != '') and (not self.modele ) ):
				
				self.widgetPlothistLBP.canvas.ax.clear()

				self.distanceR = self.imgSource._distanceRadiale1()
				self.distR.display(self.distanceR)
				# Textures et Couleur
				worker = Worker(self.textures_couleur,1)
				worker.signals.finished.connect(lambda:self.afficher_hist(1))
				worker2 = Worker(self.textures_couleur,2)
				worker2.signals.finished.connect(lambda:self.afficher_hist(2))
		
				#courbe bspline contour externe 
				cnt = self.imgSource._getContourExterne()
				_,_,_,points = self.imgSource._getPointsCourbure(cnt,0.001)
				
				# B-spline externe
				self.outEx=self.imgSource._B_spline_interpolite(points,50)
				# normalisation de la courbe
				cv2.normalize(self.outEx[0],self.outEx[0],0,255,cv2.NORM_MINMAX)
				cv2.normalize(self.outEx[1],self.outEx[1],0,255,cv2.NORM_MINMAX)
				
				#affichage de la courbe du contour externe
				out = self.imgSource._B_spline_interpolite(points,500)
				
				#normalisation de la courbe
				cv2.normalize(out[0],out[0],0,255,cv2.NORM_MINMAX)
				cv2.normalize(out[1],out[1],0,255,cv2.NORM_MINMAX)
				titre = "Courbe B-spline externe"
				self.show_bspline(self.widgetPlotCourbe,out,titre)

				#B-spline interne
				cnt = self.imgSource._getContourInterne(1)
				self.outIn = []
				if len(cnt) != 0:
					_,_,_,points = self.imgSource._getPointsCourbure(cnt[:,0],0.00001)
					# B-spline externe
					self.outIn=self.imgSource._B_spline_interpolite(points,50)
					# normalisation de la courbe
					cv2.normalize(self.outIn[0],self.outIn[0],0,255,cv2.NORM_MINMAX)
					cv2.normalize(self.outIn[1],self.outIn[1],0,255,cv2.NORM_MINMAX)


					out=self.imgSource._B_spline_interpolite(points,500)
					# normalisation de la courbe
					cv2.normalize(out[0],out[0],0,255,cv2.NORM_MINMAX)
					cv2.normalize(out[1],out[1],0,255,cv2.NORM_MINMAX)

					titre = "Courbe B-spline interne"
					self.show_bspline(self.widgetPlotCourbe2,out,titre)
				else :
					self.widgetPlotCourbe2.canvas.ax.cla()
				
				self.labelModTexture.setText("Modélisation en cours. Veuillez patientez !!")
				self.labelModCouleur.setText("Modélisation en cours. Veuillez patientez !!")
				# lancement des threads
				self.threadpool.start(worker)
				self.threadpool.start(worker2)
				self.modele = True

			elif(self.mainTab.currentIndex() == 2 and self.lineEditPath.text() != '' ):
				pixmap = QPixmap("color.jpg")
				pixmap = pixmap.scaled(self.labelimg.width(),self.labelimg.height(),Qt.KeepAspectRatio)
				self.labelimg.setPixmap(pixmap)
				self.radioBtexture.setEnabled(True)
				self.radioBcouleur.setEnabled(True)
				self.radioBforme.setEnabled(True)
				self.radioBtous.setEnabled(True)

		except Exception as e:
			print(e)	

	def textures_couleur(self,i):
		try :
			if( i==1 ):
				lbp = self.imgSource._lbp_calculated_pixel(30,4)
				lbpCercle,self.histTexture = self.imgSource._cerculelement(10,3)
				
				lbp = cv2.GaussianBlur(lbp,(5,5),0)
				lbpCercle = cv2.GaussianBlur(lbpCercle,(5,5),0)
				cv2.imwrite("lbp.jpg",lbp)
				cv2.imwrite("lbpCercle.jpg",lbpCercle)
				
				self.hist = np.array(self.histTexture).flatten()
				l = []
				for i in self.hist:
					l = np.concatenate((l, i))
				
				self.hist = np.array(l)

			else :
				colorCercle,self.featureCoul =  self.imgSource._cerculelementcouleur(30, 1)
				colorCercle = cv2.GaussianBlur(colorCercle,(5,5),0)
				cv2.imwrite("colorCercle.jpg",colorCercle)
		except Exception as e:
			print(e)
	
	def afficher_hist(self,i):
		try:
			if( i==1 ):
				pixmap = QPixmap("lbp.jpg")
				pixmap = pixmap.scaled(self.labelTexture.width(),self.labelTexture.height(),Qt.KeepAspectRatio)
				self.labelTexture.setPixmap(pixmap)
				self.labelModTexture.setText("")
				self.show_histLBP()
				self.checkBtexture.setEnabled(True)
				

			else :
				pixmap = QPixmap("color.jpg")
				pixmap = pixmap.scaled(self.labelCouleur.width(),self.labelCouleur.height(),Qt.KeepAspectRatio)
				self.labelCouleur.setPixmap(pixmap)
				self.labelModCouleur.setText("")
				self.show_histCouleur()
				self.checkBcolor.setEnabled(True)
		except Exception as e:
			print(e)

class MplCanvas(Canvas):
	def __init__(self):
		self.fig = Figure()
		self.ax = self.fig.add_subplot(111)
		Canvas.__init__(self,self.fig)
		Canvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
		Canvas.updateGeometry(self)


class MplWidget(QWidget):
	def __init__(self,parent=None):
		QWidget.__init__(self,parent)
		self.canvas = MplCanvas()
		self.vbl = QVBoxLayout()
		self.vbl.addWidget(self.canvas)
		self.setLayout(self.vbl)

class WorkerSignals(QObject):

	finished = pyqtSignal()
	error = pyqtSignal(tuple)
	result = pyqtSignal(object)

class Worker(QRunnable):
	def __init__(self,fn,*args,**kwargs):
		super(Worker,self).__init__()
		
		self.fn = fn
		self.args = args
		self.kwargs = kwargs
		self.signals = WorkerSignals()		

	@pyqtSlot()
	def run(self):

		try:
			result = self.fn(*self.args,**self.kwargs)
		except:
			traceback.print_exc()
			exctype, value = sys.exc_info()[:2]
			self.signals.error.emit((exctype, value, traceback.format_exc()))
		else:
			self.signals.result.emit(result)
		finally:
			self.signals.finished.emit()


if __name__=="__main__":
	app = QApplication(sys.argv)
	window = QMainWindow()
	c = MainWindow(window)
	window.showMaximized()
	sys.exit(app.exec_())
