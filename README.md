## ir_racecar
This repository is for a lane detection system using infrared lights, multiple approaches will be explored


## pwm_Gen_frame_sync:  
The code above sets the PWM frequency and duty cycle on an Arduino board. The code sets the frequency and duty cycle using the setPwmFrequency and setPwmDutyCycle functions, respectively. The PWM frequency and duty cycle can be changed by sending an input over the serial port using the Arduino's Serial.parseInt() function. The input is a 32-bit number where the first two bits indicate the type of input (frequency or duty cycle) and the remaining 30 bits contain the value of the input.

In the setup function, the code sets the PWM and interrupt pin as output, sets the initial PWM frequency, and enables the interrupt for the interrupt pin. In the loop function, the code checks if the PWM is active and sets the PWM value to the value of pwmDuty if it is. The code also checks for serial input and updates the PWM frequency or duty cycle accordingly.

## toSerial.py  
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

# TODOs:  

❌ Upload CAD files for the PCBs   
❌ Upload CAD files for the Camera and LED fixture  
❌ Write Documentation for the used ICs  
❌ Upload the openCV test scripts  
❌ Declutter openCV test scripts  
❌ Link documentation for the used Filter  
❌ Link documentation for used ICs  
❌ Link documentation for used camera  
❌ Use openCV output and odometrie to generate a ROS map  

<sup> Parts of this Repository were generatet using ChatGPT</sup>
