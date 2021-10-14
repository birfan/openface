#!/usr/bin/env python2
"""
/*******************************************************************************
*   Copyright (c) 2016-present, Bahar Irfan. All rights reserved.              *
*                                                                              *
*   This file allows NAO or Pepper robot to classify a person based on an      *
*   OpenFace trained model. The robot captures an image when it is touched on  *
*   the head, and runs classify_robot.py to get the closest person in the      *
*   trained model (default is the celebrity trained model by OpenFace but you  *
*   can change it) and saves the recognition image. It says the closest person *
*   and its confidence score. If the robot is Pepper, the saved recognition    *
*   image is displayed on the tablet. The classification stops when the right  *
*   bumper is pressed. This code was tested on NAOqi 2.1 and 2.4. The file     *
*   should be run remotely from the computer with internet access to the robot.*
*   See README-ROBOT.md file for instructions.                                 *
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

import argparse

import SimpleHTTPServer
import SocketServer

import socket
import fcntl
import struct

import functools

from classifier_robot import Classifier

import os

@qi.multiThreaded()
class OpenFaceRecognition(object):
    def __init__(self, app, args):
        self.args = args
        self.app = app
        self.app.start()
        self.session = self.app.session
        
        self.clas = Classifier(self.args)

        self.imageDirectory = self.args.imageDirectory
        self.databaseFolder = self.args.databaseFolder
        self.saveImgFolder = self.args.saveImageFolder

        if self.args.serverDir.startswith("/home"):
            #if it is an absolute path
            self.serverDir = self.args.serverDir
        else:
            #if relative path to server
            self.serverDir = os.path.join(os.path.expanduser("~"), self.args.serverDir)

        self.life_service = self.session.service("ALAutonomousLife")

        # subscribe to camera
        self.video_service = self.session.service("ALVideoDevice")
        resolution = 2    # VGA
        colorSpace = 11   # RGB
        # subscribe is deprecated, use subscribeCamera instead
        # self.video_service.setActiveCamera(self.args.captureDevice)
        # self.videoClient = self.video_service.subscribe("python_client", resolution, colorSpace, 5)
        self.videoClient = self.video_service.subscribeCamera("python_client", self.args.captureDevice, resolution, colorSpace, 5)
        
        # connect to Pepper tablet
        self.tabletConnected = False
        if self.args.robotName == "pepper":
            try:
                self.tablet_service = self.session.service("ALTabletService")
                # Ensure that the tablet wifi is enabled
                self.tablet_service.enableWifi()
                self.tablet_service.showWebview()
                # self.tablet_service.hideImage()
                self.tabletConnected = True
                # Get IP of the tablet
                computerIp = str(getIpAddressComputer(self.args.comm))
                self.computerIp = "http://" + computerIp + "/openface/recog.png" #use a web server (e.g., XAMPP) on local computer and add an 'openface' folder inside the 'WebServer' folder. We need to access the local file from the tablet. You can also use client/server system, but this is simpler.
            except:
                print("Cannot connect to ALTabletService")
        
        # Head touched event initialisation
        self.memory_service = self.session.service("ALMemory")
        self.touchHeadEvent = "FrontTactilTouched"
        self.touchHead = None
        self.idHead = -1
        self.touchHead = self.memory_service.subscriber(self.touchHeadEvent)
        self.idHead = self.touchHead.signal.connect(functools.partial(self.onTouchedHead, self.touchHeadEvent))
        
        # Right bumper event initialisation
        self.touchBumperEvent = "RightBumperPressed"
        self.touchBumper = None
        self.idBumper = -1
        self.touchBumper = self.memory_service.subscriber(self.touchBumperEvent)
        self.idBumper = self.touchBumper.signal.connect(functools.partial(self.onTouchedBumper, self.touchBumperEvent))

        self.speech_service = self.session.service("ALTextToSpeech")
        self.speech_service.say("Touch the top of my head to compare yourself to a celebrity.")

    def startAutonomousLife(self):
        """Start autonomous life."""
        if self.life_service.getState() == "disabled":
            self.life_service.setState("solitary")

    def stopAutonomousLife(self):
        """Stop autonomous life."""
        if not (self.life_service.getState() == "disabled"):
            self.life_service.setState("disabled")
            self.motion_service.wakeUp()

    def unsubscribeServices(self):
        """Unsubscribe video and tablet."""
        self.video_service.unsubscribe(self.videoClient)
        if self.args.robotName == "pepper" and self.tabletConnected:  
            # Hide the web view
            self.tablet_service.hideImage()
#             self.tablet_service.resetToDefaultValue()
        self.app.stop()
            
    def unsubscribeHead(self):
        """Unsubscribe the head sensor."""
        if(self.touchHead is not None and self.idHead != -1):
            self.touchHead.signal.disconnect(self.idHead)
        self.touchHead = None
        self.idHead = -1
        
    def unsubscribeBumper(self):
        """Unsubscribe the bumper."""
        if(self.touchBumper is not None and self.idBumper != -1):
            self.touchBumper.signal.disconnect(self.idBumper)
        self.touchBumper = None
        self.idBumper = -1    
     
    def onTouchedHead(self, strVarName, value):
        """This will be called each time a touchBumper is detected."""
        # Unsubscribe to the event when talking,
        # to avoid repetitions
         
        self.touchHead.signal.disconnect(self.idHead)
        
        self.speech_service.say("Taking a picture in \\rspd=50\\ three, two, one. \\rspd=100\\ \\emph=1\\ Smile!")
        
        # capture image
        robotImage = self.video_service.getImageRemote(self.videoClient)

        # classify image, save the recognition image with confidence, and get the closest person and confidence score
        comparison, closestPerson, confidencesP = self.clas.compare(robotImage, self.imageDirectory, self.databaseFolder, self.saveImgFolder, self.serverDir)

        if self.clas.noFaceDetected:
            self.speech_service.say("I couldn't see your face.") 
        elif comparison is None:
            self.unsubscribeBumper()
            self.unsubscribeHead()
            self.unsubscribeServices()
            return
        self.speech_service.say("I am " + str(confidencesP) + " percent confident that you are " + closestPerson)

        if self.args.robotName == "pepper" and self.tabletConnected:
            # Show the resulting image on the tablet
            self.tablet_service.cleanWebview()
            self.tablet_service.showImageNoCache(self.computerIp)
        
        self.speech_service.say("\\wait=5\\ Touch the top of my head to try again or touch my right bumper to exit!")
               
        # Reconnect again to the event
        self.idHead = self.touchHead.signal.connect(functools.partial(self.onTouchedHead, self.touchHeadEvent))
             
    def onTouchedBumper(self, strVarName, value):
        """This will be called each time a touch is detected."""      
        self.unsubscribeBumper()
        self.unsubscribeHead()
        self.unsubscribeServices()
        
    def getIpAddressComputer(self, ifname):
        """Get IP address of the computer to access the saved recognition image."""      
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', ifname[:15])
            )[20:24])

def parserArgs():
    """Parse options from the command line."""      
    fileDir = os.path.dirname(os.path.realpath(__file__))
    modelDir = os.path.join(fileDir, '../..', 'models')
    dlibselfModelDir = os.path.join(modelDir, 'dlib')
    openfaceselfModelDir = os.path.join(modelDir, 'openface')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--robotIp', type=str, default='127.0.0.1',
                    help='Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.')
    parser.add_argument('--robotPort', type=int, default=9559,
                        help='Naoqi port number')
    parser.add_argument('--robotName', type=str, default='pepper',
                        help='Robot name: nao or pepper. Default is pepper.')
    parser.add_argument('--robotComm', type=str, default='ens33',
                        help='Internet port: ens33 or eth0 or wlan0. Check using ifconfig.')
    parser.add_argument(
            '--captureDevice',
            type=int,
            default=0,
            help='Capture device. 0 for top camera and 1 for bottom camera of Nao/Pepper. Default is 0.')

    parser.add_argument(
            '--dlibFacePredictor',
            type=str,
            default=os.path.join(
                dlibselfModelDir,
                'shape_predictor_68_face_landmarks.dat'),
            help='Path to dlib face predictor.')
    parser.add_argument(
            '--classifierModel',
            type=str,
            default=os.path.join(
                openfaceselfModelDir,
                'celeb-classifier.nn4.small2.v1.pkl'),
            help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.') #celebrity trained model. You can change this to another trained model
    parser.add_argument(
            '--networkModel',
            type=str,
            default=os.path.join(
                openfaceselfModelDir,
                'nn4.small2.v1.t7'),
            help='Path to Torch network model.') #celebrity trained model. You can change this to another trained model.
    parser.add_argument('--imgDim', type=int,
                        help='Default image dimension.', default=96)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--imageDirectory', type=str, default='demos/robot/images/',
                        help='Relative directory (within openface folder) of the images folder. Default is demos/robot/images/.')
    parser.add_argument('--databaseFolder', type=str, default='celeb_images',
                        help='Name of the database folder (folder where the comparison images are). Default is celeb_images.')
    parser.add_argument('--saveImageFolder', type=str, default='recog_images',
                        help='Name of the folder to save the recognition images. Default is recog_images.')
    parser.add_argument('--serverDir', type=str, default='WebServer/openface/',
                        help='Relative directory (from /home/USER/) for the server (XAMPP) to save the recognition images to display on Pepper's tablet. Default is WebServer/openface/. You can also use an absolute path.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parserArgs()
    
    try:
        # Initialize qi framework.
        connection_url = "tcp://" + args.robotIp + ":" + str(args.robotPort)
        app = qi.Application(["OpenFaceRecognition", "--qi-url=" + connection_url])
    except RuntimeError:
        print("Can't connect to Naoqi at ip \"" + args.robotIp + "\" on port " + str(args.robotPort) +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
         
    open_face_recog = OpenFaceRecognition(app, args)
    app.run()
 
