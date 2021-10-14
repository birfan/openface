#!/usr/local/bin/python
"""
/*******************************************************************************
*   Copyright (c) 2016-present, Bahar Irfan. All rights reserved.              *
*                                                                              *
*   This code is the local version for comparison_remote_robot.py to run the   *
*   code locally on NAO or Pepper robot, as a service. BUT IT WAS NOT TESTED.  *
*   We recommend using comparison_remote_robot instead, unless you can compile *
*   all dependencies for NAOqi or you have a robot with an Ubuntu computer     *
*   (e.g., Adapted Pepper robot in MuMMER project). Please submit an issue on  *
*   GitHub if you test this file.                                              *
*                                                                              *
*   This file allows NAO or Pepper robot to classify a person based on an      *
*   OpenFace trained model. The robot captures an image when it is touched on  *
*   the head, and runs classify_robot.py to get the closest person in the      *
*   trained model (default is the celebrity trained model by OpenFace but you  *
*   can change it) and saves the recognition image. It says the closest person *
*   and its confidence score. If the robot is Pepper, the saved recognition    *
*   image is displayed on the tablet. The classification stops when the right  *
*   bumper is pressed. See README-ROBOT.md file for instructions on how to run *
*   this service on the robot.                                                 *
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
*   Public License along with this work.  If not, see                          *
*   <http://www.gnu.org/licenses/>. OpenFace is licensed under Apache License, *
*   Version 2.0 and copyrighted by Carnegie Mellon University.                 *
*******************************************************************************/
"""

__version__ = '0.0.1'

__copyright__ = 'Copyright (c) 2016-present, Bahar Irfan. All rights reserved.'
__author__ = 'Bahar Irfan'
__email__ = 'bahar.irfan@plymouth.ac.uk'


import qi

import stk.runner
import stk.events
import stk.services
import stk.logging

import SimpleHTTPServer
import SocketServer

import socket
import fcntl
import struct

import functools

from classifier_robot import Classifier

import os

class argus: pass

class OpenFaceRecognition(object):
    "NAOqi service."
    APP_ID = "com.aldebaran.OpenFaceRecognition"
    def __init__(self, qiapp):
        # generic activity boilerplate
        self.qiapp = qiapp
        
        self.events = stk.events.EventHelper(qiapp.session) 
        self.s = stk.services.ServiceCache(qiapp.session) 
        self.logger = stk.logging.get_logger(qiapp.session, self.APP_ID)

        self.setArgs()

        self.clas = Classifier(self.args)

        # subscribe to camera
        resolution = 2    # VGA
        colorSpace = 11   # RGB
        # subscribe is deprecated, use subscribeCamera instead
        # self.s.ALVideoDevice.setActiveCamera(self.args.captureDevice)
        # self.videoClient = self.s.ALVideoDevice.subscribe("python_client", resolution, colorSpace, 5)
        self.videoClient = self.s.ALVideoDevice.subscribeCamera("python_client", self.args.captureDevice, resolution, colorSpace, 5)

        # connect to Pepper tablet
        self.tabletConnected = False
        if self.args.robotName == "pepper":
            try:
                # Ensure that the tablet wifi is enabled
                self.s.ALTabletService.enableWifi()
                self.s.ALTabletService.showWebview()
                # self.s.ALTabletService.hideImage()
                self.tabletConnected = True
            except:
                print("Cannot connect to ALTabletService")

        # Head touched event initialisation
        self.touchHeadEvent = "FrontTactilTouched"
        self.touchHead = None
        self.idHead = -1
        self.touchHead = self.s.ALMemory.subscriber(self.touchHeadEvent)
        self.idHead = self.touchHead.signal.connect(functools.partial(self.onTouchedHead, self.touchHeadEvent))

        # Right bumper event initialisation
        self.touchBumperEvent = "RightBumperPressed"
        self.touchBumper = None
        self.idBumper = -1
        self.touchBumper = self.s.ALMemory.subscriber(self.touchBumperEvent)
        self.idBumper = self.touchBumper.signal.connect(functools.partial(self.onTouchedBumper, self.touchBumperEvent))

        self.s.ALTextToSpeech.say("Touch the top of my head to compare yourself to a celebrity.")

    @qi.bind(returnType=qi.Void, paramsType=[])
    def stop(self):
        "Stop the service."
        
        self.logger.info("OpenFaceRecognition stopped by user request.")
        self.unsubscribeBumper()
        self.unsubscribeHead()
        self.unsubscribeServices()
        self.qiapp.stop()

    @qi.nobind
    def on_stop(self):
        "Cleanup (add yours if needed)"
        self.stop()
        self.logger.info("OpenFaceRecognition finished.") 
        
    @qi.bind(returnType=qi.Void, paramsType=[])
    def startAutonomousLife(self):
        "Stop autonomous life"
        self.s.ALAutonomousLife.setState("solitary")
    
    @qi.bind(returnType=qi.Void, paramsType=[])
    def stopAutonomousLife(self):
        "Stop autonomous life"
        self.s.ALAutonomousLife.setState("disabled")
        self.s.ALMotion.wakeUp()

    @qi.bind(returnType=qi.Void, paramsType=[])
    def unsubscribeServices(self):
        """Unsubscribe video and tablet."""
        self.s.ALVideoDevice.unsubscribe(self.videoClient)
        if self.args.robotName == "pepper" and self.tabletConnected:  
            # Hide the web view
            self.s.ALTabletService.hideImage()
#             self.s.ALTabletService.resetToDefaultValue()
        self.app.stop()

    @qi.bind(returnType=qi.Void, paramsType=[])           
    def unsubscribeHead(self):
        """Unsubscribe the head sensor."""
        if(self.touchHead is not None and self.idHead != -1):
            self.touchHead.signal.disconnect(self.idHead)
        self.touchHead = None
        self.idHead = -1

    @qi.bind(returnType=qi.Void, paramsType=[])        
    def unsubscribeBumper(self):
        """Unsubscribe the bumper."""
        if(self.touchBumper is not None and self.idBumper != -1):
            self.touchBumper.signal.disconnect(self.idBumper)
        self.touchBumper = None
        self.idBumper = -1    

    @qi.bind(returnType=qi.Void, paramsType=[])
    def onTouchedHead(self, strVarName, value):
        """This will be called each time a touchBumper is detected."""
        # Unsubscribe to the event when talking,
        # to avoid repetitions
         
        self.touchHead.signal.disconnect(self.idHead)
        
        self.s.ALTextToSpeech.say("Taking a picture in \\rspd=50\\ three, two, one. \\rspd=100\\ \\emph=1\\ Smile!")
        
        # capture image
        robotImage = self.s.ALVideoDevice.getImageRemote(self.videoClient)

        # classify image, save the recognition image with confidence, and get the closest person and confidence score
        comparison, closestPerson, confidencesP = self.clas.compare(robotImage, self.args.imageDirectory, self.args.databaseFolder, self.args.saveImgFolder, self.args.serverDir)

        image_name = "http://%s/apps/openface-recognition/recog.png" % self.s.ALTabletService.robotIp()

        if self.clas.noFaceDetected:
            self.s.ALTextToSpeech.say("I couldn't see your face.") 
        elif comparison is None:
            self.stop()
            return
        self.s.ALTextToSpeech.say("I am " + str(confidencesP) + " percent confident that you are " + closestPerson)

        if self.args.robotName == "pepper" and self.tabletConnected:
            # Show the resulting image on the tablet
            self.s.ALTabletService.cleanWebview()
            self.s.ALTabletService.showImageNoCache(image_name)
        
        self.s.ALTextToSpeech.say("\\wait=5\\ Touch the top of my head to try again or touch my right bumper to exit!")
               
        # Reconnect again to the event
        self.idHead = self.touchHead.signal.connect(functools.partial(self.onTouchedHead, self.touchHeadEvent))

    @qi.bind(returnType=qi.Void, paramsType=[])          
    def onTouchedBumper(self, strVarName, value):
        """This will be called each time a touch is detected."""      
        self.stop()

    @qi.bind(returnType=qi.Void, paramsType=[])          
    def setArgs(self, robotName="pepper", captureDevice=0,
                dlibFacePredictor="68FACE", classifierModel="CELEB", networkModel="CELEB",
                imgDim=96, width=320, height=240, threshold=0.5, cuda=False, verbose=False,
                imageDirectory="demos/robot/images/", databaseFolder="celeb_images", saveImageFolder="recog_images", 
                serverDir="LOCAL"):
        """Set options."""
        fileDir = os.path.dirname(os.path.realpath(__file__))
        modelDir = os.path.join(fileDir, '../..', 'models')
        dlibselfModelDir = os.path.join(modelDir, 'dlib')
        openfaceselfModelDir = os.path.join(modelDir, 'openface')

        argus.robotName = robotName
        argus.captureDevice = captureDevice
        if dlibFacePredictor == "68FACE":
            argus.dlibFacePredictor = os.path.join(dlibselfModelDir, "shape_predictor_68_face_landmarks.dat")
        else:
            argus.dlibFacePredictor = dlibFacePredictor
        if classifierModel == "CELEB":
            argus.classifierModel = os.path.join(openfaceselfModelDir, "celeb-classifier.nn4.small2.v1.pkl")
        else:
            argus.classifierModel = classifierModel
        if networkModel == "CELEB":
            argus.networkModel = os.path.join(openfaceselfModelDir, "nn4.small2.v1.t7")
        else:
            argus.networkModel = networkModel
        argus.imgDim = imgDim
        argus.width = width
        argus.height = height
        argus.threshold = threshold
        argus.cuda = cuda
        argus.verbose = verbose
        
        argus.imageDirectory = imageDirectory
        argus.databaseFolder = databaseFolder
        argus.saveImgFolder = saveImageFolder
        if serverDir == "LOCAL":
            argus.serverDir = "/home/nao/.local/share/PackageManager/apps/openface-recognition/html/"
        else:
            argus.serverDir = serverDir

        self.args=argus
        
####################
# Setup and Run
####################

if __name__ == "__main__":
    stk.runner.run_service(OpenFaceRecognition)


