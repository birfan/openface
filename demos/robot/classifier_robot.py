#!/usr/bin/env python2
"""
/*******************************************************************************
*   Copyright (c) 2016-present, Bahar Irfan. All rights reserved.              *
*                                                                              *
*   This file is adapted from the classifier_webcam.py by Brandon Amos &       *
*   Vijayenthiran, (copyright 2015-2016 Carnegie Mellon University).           *
*   It runs a classifier from an image obtained from the robot, and saves the  *
*   the recognition image as the concatenation of the captured image and the   *
*   closest person (in the trained model) image with the confidence score.     *
*                                                                              *
*   Please cite the following if using this work:                              *
*                                                                              *
*     B. Irfan, "OpenFace for NAO and Pepper Robots", University of Plymouth,  * 
*     UK. https://github.com/birfan/openface. 2016.                            *
*                                                                              *
*     B. Amos, B. Ludwiczuk, M. Satyanarayanan, "OpenFace: A general-purpose   *
*     face recognition library with mobile applications," CMU-CS-16-118,       *
*     CMU School of Computer Science, Tech. Rep., 2016.                        *
*                                                                              *
*   This work is distributed in the hope that it will be useful, but WITHOUT   *
*   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      *
*   FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License      *
*   for more details. You should have received a copy of the GNU General       *
*   Public License along with ComparisonService.  If not, see                  *
*   <http://www.gnu.org/licenses/>. OpenFace is licensed under Apache License, *
*   Version 2.0 and copyrighted by Carnegie Mellon University.                 *
*******************************************************************************/
"""

__version__ = '0.0.1'

__copyright__ = 'Copyright (c) 2016-present, Bahar Irfan. All rights reserved.'
__author__ = 'Bahar Irfan'
__email__ = 'bahar.irfan@plymouth.ac.uk'


import time

start = time.time()

import cv2
import os
import pickle
import Image

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface

import re
import shutil

import sys
                 
class Classifier(object):  
    def __init__(self, args):
        self.args = args
           
        self.align = openface.AlignDlib(self.args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(
            self.args.networkModel,
            imgDim=self.args.imgDim,
            cuda=self.args.cuda)
        
        self.noFaceDetected = False
    
    def getRep(self, bgrImg):
        """Aligns the faces, gets the largest bounding box and passes through forward neural network."""
        if self.args.verbose:
            start = time.time()

        if bgrImg is None:
            raise Exception("Unable to load image/frame")

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.args.verbose:
            print("  + Original size: {}".format(rgbImg.shape))
            print("Loading the image took {} seconds.".format(time.time() - start))
            start = time.time()

        # Get the largest face bounding box
        bb1 = self.align.getLargestFaceBoundingBox(rgbImg) #Bounding box
        bb = [bb1]
        # Get all bounding boxes
        # bb = self.align.getAllFaceBoundingBoxes(rgbImg)

        if len(bb) == 0 or (bb1 is None):
            # raise Exception("Unable to find a face: {}".format(imgPath))
            return None
        if self.args.verbose:
            print("Face detection took {} seconds.".format(time.time() - start))
            start = time.time()

        alignedFaces = []
        for box in bb:
            alignedFaces.append(
                self.align.align(
                    self.args.imgDim,
                    rgbImg,
                    box,
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

        if alignedFaces is None:
            raise Exception("Unable to align the frame")
        if self.args.verbose:
            print("Alignment took {} seconds.".format(time.time() - start))
            start = time.time()

        reps = []
        for alignedFace in alignedFaces:
            reps.append(self.net.forward(alignedFace))

        if self.args.verbose:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
            
        return reps


    def infer(self, img):
        """Predicts the closest face(s) (from the trained model) to the face(s) in the captured image, and returns names and confidence scores."""
        with open(self.args.classifierModel, 'r') as f:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)  # le - label and clf - classifer
            else:
                (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

        reps = self.getRep(img)
        persons = []
        confidences = []
        if reps is None:
            print("No Face detected")
            return (None, None)
        
        # for all the faces in the image
        for rep in reps:
            try:
                rep = rep.reshape(1, -1)
            except:
                print("No Face detected")
                return (None, None)

            if self.args.verbose:
                start = time.time()

            predictions = clf.predict_proba(rep).ravel()
            # print(predictions)

            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            persons.append(person)
            confidences.append(predictions[maxI])

            # prints the second prediction
            # max2 = np.self.argsort(predictions)[-3:][::-1][1]
            # print(str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))

            if self.args.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
            # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
            if isinstance(clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                # print("  + Distance from the mean: {}".format(dist))
        return (persons, confidences)

    def convertImage(self, capturedImg):
        """Converts a naoqi (captured) image into an opencv MAT format"""
        imageWidth = capturedImg[0]
        imageHeight = capturedImg[1]
        array = capturedImg[6]
        image_string = str(bytearray(array))
        im = Image.frombytes("RGB", (imageWidth, imageHeight), image_string) # captured image to PIL
        opencvImage = np.array(im) #PIL to opencv image
        opencvImage = opencvImage[:, :, ::-1].copy() # Convert RGB to BGR
        return opencvImage
        
    def compare(self, capturedImg, imageDirectory, databaseFolder, saveImgFolder, serverDir):
        """Compares an image with the trained models and saves recognition image with confidence"""
        self.noFaceDetected = False
        if capturedImg == None:
            print("cannot capture.")
            return (None, None, None)
        elif capturedImg[6] == None:
            print("no image data string.")
            return (None, None, None)
        else:
            frame = self.convertImage(capturedImg)

        # for all people in the image, get the most similar users from database/trained model and the confidence values
        persons, confidences = self.infer(frame)
    
        if persons is None:
            self.noFaceDetected = True
            return (None, None, None)

        # assume only one person in frame (image)
        closestPerson = str(persons[0])
        confidencesP = float(confidences[0])*100

        # if confidence is below threshold (0.5), recognise person as unknown (use this for open set/open world recognition)
#         for i, c in enumerate(confidences):
#             if c <= self.args.threshold:  # 0.5 is kept as threshold for known face.
#                 persons[i] = "_unknown"

        # save recognition image with confidence
        comparisonSaved = self.saveRecogImg(frame, imageDirectory, databaseFolder, saveImgFolder, serverDir, closestPerson, confidencesP)

        if not comparisonSaved:
            return (None, None, None)

        #put space between the name and surname if there isn't one
        closestPersonName = re.sub(r"(\w)([A-Z])", r"\1 \2", closestPerson)
        
        return (comparison, closestPersonName, confidencesP)

    def saveRecogImg(self, frame, imageDirectory, databaseFolder, saveImgFolder, serverDir, closestPerson, confidencesP):
        """Saves captured image and closest person image concetanated with confidence"""
        curDir = os.getcwd()
        closestPersonImDir = os.path.join(curDir, imageDirectory, databaseFolder, closestPerson.replace(" ", "") + ".jpg")
        closestPersonImg = cv2.imread(closestPersonImDir)   

        # concatenate images of the person in the image and the most similar user image and add confidence
        confidencesPstr = "%.1f" % confidencesP
        confidencesPstr = confidencesPstr + "%"
        recogImageDir = os.path.join(curDir, imageDirectory, saveImageFolder)
        serverImagePath = os.path.join(serverDir, "recog.png")
    
        # put two images side by side
        comparison = np.concatenate((frame, closestPersonImg), axis=1) 
        
        cv2.rectangle(comparison,(540, 190),(750, 270),(255,255,255),-1)
        # Add confidence value on the frame
        cv2.putText(comparison, "{}".format(confidencesPstr),
            (550, 250), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 0), 3)
        
        # save image to server
        comparisonSaved = cv2.imwrite(serverImagePath, comparison)
        
        # get number of images in the recognition image folder
        num_imgs = len([name for name in os.listdir(recogImageDir) if os.path.isfile(os.path.join(recogImageDir, name))])
        imagePath = os.path.join(recogImageDir, "recog"+str(num_imgs+1)+".png")

        # save image to the recognition image folder e.g., "recog5.png" for the 5th recognition
        shutil.copy2(serverImagePath, imagePath)        
        
        return comparisonSaved
            

