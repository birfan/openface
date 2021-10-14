# OpenFace for NAO and Pepper Robots

This project extends OpenFace (Brandon et al., 2016) for use on NAO and Pepper robots (SoftBank Robotics Europe, France). OpenFace is a free and open source face recognition software with deep neural networks (`https://github.com/cmusatyalab/openface`). This software was tested on NAOqi 2.1 and 2.4 on NAO and Pepper robots.

## Install OpenFace on Local Computer:

See [OpenFace website](https://cmusatyalab.github.io/openface/setup/) for installation instructions. Build from source code (either using Docker, conda or manually).

If using a Pepper robot, install a web server (e.g., XAMPP) to display the resulting recognition image (captured image and closest person image side-by-side with the confidence score) from the Pepper's tablet. Name the folder WebServer and put an "openface" folder inside.

## Run OpenFace for NAO and Pepper Robots Remotely:

The files for the NAO and Pepper robots is in "demos/robot/" folder. 

* Turn on the robot. Connect to the robot (through ethernet or WiFi), and get ROBOT_IP by clicking on the chest button.

* In Docker or local computer (see [OpenFace instructions to run code](https://cmusatyalab.github.io/openface/setup/)), run in the root ("openface") folder:

```
    $ ./demos/robot/comparison_remote_robot.py --robotIp ROBOT_IP --robotPort ROBOT_PORT --robotName ROBOT_NAME --robotComm COMMUNICATION_PORT 
```

ROBOT_NAME is "pepper" by default, use "nao" for NAO. ROBOT_PORT is by default 9559. COMMUNICATION_PORT to the robot is "ens33" by default, other options are "eth0" or "wlan0". Find it using:

```
    $ ifconfig
```

Other options are:

 * --captureDevice 0 for top camera and 1 for bottom camera of Nao/Pepper. Default is 0.
 * --imageDirectory Default is "demos/robot/images/" (relative directory of images)
 * --databaseFolder Default is "celeb_images" if comparing with celebrities (name of the folder in imageDirectory). If using another trained model with different images, change this directory.
 * --saveImageFolder Default is "recog_images" (name of the folder in imageDirectory)
 * --serverDir Default is "WebServer/openface/". This is the relative directory (from /home/USER/) for the server (XAMPP) to save the recognition images to display on Pepper's tablet. You can also use an absolute path.
 * --dlibFacePredictor Path to dlib's face predictor. Default is path to "shape_predictor_68_face_landmarks.dat".
 * --classifierModel The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel. Default is path to "celeb-classifier.nn4.small2.v1.pkl" (celebrity classifier). You can change this to another trained model.
 * --networkModel Path to Torch network model. Default is path to "nn4.small2.v1.t7" celebrity trained model. You can change this to another trained model.
 * --imgDim Default image dimension is 96.
 * --width Image width. Default is 320.
 * --height Image width. Default is 240.
 * --threshold Threshold for unknown user recognition (if confidence score is below the threshold, the user is declared unknown). This is not used for the robots. Default is 0.5.
 * --width Image width. Default is 320.
 * --cuda Cuda is used, if called.
 * --verbose Log the times, if called.

* Touch the head of the robot to start the recognition. This code will run until the right bumper is touched (or the code is stopped from the terminal, but it is best to touch the bumper instead to ensure all channels are unsubscribed).

## Run OpenFace for NAO and Pepper Robots Locally:

If you are using a Pepper robot with an Ubuntu computer or if you need the code to be run on-board the robot, use *comparison_local_robot.py*. This script was NOT tested, please submit a Github issue if you do. This project does NOT contain the compiled NAOqi libraries of OpenFace, OpenCV or any other dependencies. See B. Irfan and S. Lemaignan (2016), "Chilitags for NAO Robot", `https://github.com/birfan/chilitags` to see an example on how to compile code for NAOqi.

```
    $ scp -r openface nao@ROBOT_IP:/home/nao/
```

The *openface-recognition.pml* (in demos/robot/) is the main project file that you open through Choregraphe. This contains the robot files. You can see the files in the 'Project content' tab. To install on the robot, upload the files to your local robot using the 'Package and install current project to the robot' option in the 'Robot Applications' tab. You do not need to run it - it will automatically start, although you may need to restart NAOqi if it does not work straight away.

* On the robot (ssh to the robot):

```
    $ cd openface
    $ cd ./demos/robot/comparison_local_robot.py
```

* Set options (from above) by calling `setArgs` function (e.g., using `qicli call` in terminal or with a script). *robotIp*, *robotPort* and *robotComm* are not used for this script. 

* Touch the head of the robot to start the recognition. This code will run until the right bumper is touched.

## License

OpenFace for NAO and Pepper Robots is released under GNU General Public License v3.0 (GPL3). Cite the following if using this work:

 * B. Irfan, "OpenFace for NAO and Pepper Robots", University of Plymouth, UK. `https://github.com/birfan/openface`. 2016.

 * B. Amos, B. Ludwiczuk, M. Satyanarayanan, "OpenFace: A general-purpose face recognition library with mobile applications," CMU-CS-16-118, CMU School of Computer Science, Tech. Rep., 2016.

```
	@misc{openfaceRobot,
		title = {OpenFace for NAO and Pepper Robots},
		author={Irfan, Bahar},
		publisher={University of Plymouth, UK},
		url={https://github.com/birfan/chilitags},
		year={2016}
	}

    @techreport{amos2016openface,
      title={OpenFace: A general-purpose face recognition library with mobile applications},
      author={Amos, Brandon and Bartosz Ludwiczuk and Satyanarayanan, Mahadev},
      year={2016},
      institution={CMU-CS-16-118, CMU School of Computer Science},
    }
```

OpenFace is released under Apache License, Version 2.0 and copyrighted by Carnegie Mellon University. See README file for information.

## Contact

If you need further information about using OpenFace with the NAO or Pepper robot, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk (the most recent contact information is available at [personal website](https://www.baharirfan.com)).
