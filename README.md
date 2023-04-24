## ir_racecar
This repository is for a lane detection system on a MIT-RACEcar style robot. Multiple approaches will be explored, one will be using infrared lights.


## pwm_Gen.ino:  
The code above sets the PWM frequency and duty cycle on an Arduino board. The code sets the frequency and duty cycle using the setPwmFrequency and setPwmDutyCycle functions, respectively. The PWM frequency and duty cycle can be changed by sending an input over the serial port using the Arduino's Serial.parseInt() function. The input is a 32-bit number where the first two bits indicate the type of input (frequency or duty cycle) and the remaining 30 bits contain the value of the input.

In the setup function, the code sets the PWM and interrupt pin as output, sets the initial PWM frequency, and enables the interrupt for the interrupt pin. In the loop function, the code checks if the PWM is active and sets the PWM value to the value of pwmDuty if it is. The code also checks for serial input and updates the PWM frequency or duty cycle accordingly.

## toSerialApp.py  
### Introduction  
This script allows the user to control the PWM frequency and duty cycle on an Arduino board using a simple GUI. The user can input the PWM frequency and duty cycle values in the GUI and then click the submit button to send the values over the serial port to the Arduino board. The Arduino board will then update the PWM frequency and duty cycle accordingly.  

### Requirements  
Python 3  
The serial and tkinter modules  
### Usage  
To run the script, use the following command:  
```
python toSerialApp.py  
```
The script will open the GUI, which will look like this:  

![image](https://user-images.githubusercontent.com/82340152/206192625-6f232108-07b7-4e09-b40d-583ecb57c22b.png)

The user can input the PWM frequency and duty cycle values in the input fields and then click the submit button to send the values to the Arduino board. The Arduino board will then update the PWM frequency and duty cycle accordingly.  

### Configuration  
The script uses the serial module to communicate with the Arduino board over the serial port. The serial port and baud rate may need to be adjusted depending on the specific configuration of the Arduino board. The default values are '/dev/ttyACM0' for the serial port and 9600 for the baud rate. These values can be modified in the following line of the script:  

```
ser = serial.Serial('/dev/ttyACM0', 9600)
```
Additionally, the code that handles the serial input on the Arduino side needs to be modified to handle the encoded values as described in the previous answer.

As an alternative Version toSerial.py is provided, to grant a easier use in cli enviroments.


# imageDiff.py

This script uses the cv2 module (part of the OpenCV computer vision library) to compare two images and generate a series of images showing the differences between the two.
Dependencies

    OpenCV
    NumPy

### Usage

To use the script, you need to provide the paths to the two images you want to compare as arguments to the cv2.imread() function. The script will then generate a series of images showing the differences between the two images with increasing thresholds for the difference values. The resulting images will be saved to the specified output directory with filenames that include the threshold value used for each image.


# showCam.py

This script contains functions to display a video feed from a webcam, show the difference between two frames, and transform the video feed.
## Dependencies

    cv2
    numpy
    asyncio

## Functions
### showCam(cam)

This function takes in a video capture object cam and displays the video feed in a window named "test". The function also converts the frame to grayscale and improves the contrast using histogram equalization. The processed frame is then returned.
### showDiff(frame, oldFrame)

This function takes in two frames, frame and oldFrame, and displays the difference between the two frames in a window named "diff". If an error is encountered, the original frame is returned.
### showTransDiff(frame, oldFrame)

This function is similar to showDiff, but it also transforms the frame before displaying the difference. If an error is encountered, the original frame is returned.
This function is currently disabled
### showTrans(frame,oldDst)

This function takes in a frame frame and an old frame oldDst, and performs a perspective transform on frame to correct for lens distortion. The transformed frame is then displayed in a window named "transformed", and the difference between the transformed frame and oldDst is displayed in a window named "diff". The transformed frame is returned.
### main()

This is the main function of the script, which initializes a video capture object and displays the video feed using the functions defined above. The loop continues until the user hits the 'esc' key.

# Using multiple Cameras
One approach uses https://github.com/linrl3/Image-Stitching-OpenCV, this however is now where near real time capable.

# CAD
The CAD files are meant to improve testing, allowing for easy maipulation for parameters and providing a interface between the used IC and an arduino.
The CAD software used is KiCAD.

## analyzer_pcb_without_ic
Since the LMH1980 is in the very small 8-SOIC package, we decided to extend the pins using thin wire and not mill the appropriate size, since it would most likely end in a desaster, given the limits of the available equipment. While this chip works very good, the complexity of this circuit deter you from using it.  

## lm1881
This PCB also provides an interface with its IC, this chip how ever is big enough for a mill with a .3mm milling attachement to manufacture. Another plus point for this IC is the simplicity of the circuit.  

Documentations:
LMH1980 : https://www.ti.com/product/LMH1980
LM1881  : https://www.ti.com/product/LM1881
IR-LEDs : https://www.digikey.de/de/products/detail/american-bright-optoelectronics-corporation/BWIR-35C2O48/9678149
Filter  : https://www.ebay.de/itm/223839832714
Current Source: https://www.led-stuebchen.de/de/3x-bausatz-led-konstantstromquelle-700ma
Camera: https://www.dronecosmo.com/products/foxeer-falkor-1200tvl-1-8mm-fpv-camera-limited-edition-white

# TODOs:  

✅ Upload CAD files for the PCBs   
✅ Upload CAD files for the Camera and LED fixture  
✅ Add Documentation for the used ICs  
✅ Upload the openCV test scripts  
✅ Link documentation for used ICs 
✅ refactor stitching files to accept args
✅ refactor stitching to switch types on launch

~  Declutter openCV test scripts  

❌ Link documentation for the used Filter  
❌ Link documentation for used camera  
❌ Fix .launch file (image_proc broken)
❌ add image stitching to .launch file
❌ refactor for general code hygiene
❌ delete broken / duplicate files
❌ maybe add folders for further repo hygiene
❌ comment code more, add explainations why something is bad

❓  Use openCV output and odometrie to generate a ROS map  

