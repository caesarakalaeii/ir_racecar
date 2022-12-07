# ir_racecar
This repository is for a lane detection system using infrared lights, multiple approaches will be explored


pwm_Gen_frame_sync:
The code above sets the PWM frequency and duty cycle on an Arduino board. The code sets the frequency and duty cycle using the setPwmFrequency and setPwmDutyCycle functions, respectively. The PWM frequency and duty cycle can be changed by sending an input over the serial port using the Arduino's Serial.parseInt() function. The input is a 32-bit number where the first two bits indicate the type of input (frequency or duty cycle) and the remaining 30 bits contain the value of the input.

In the setup function, the code sets the PWM and interrupt pin as output, sets the initial PWM frequency, and enables the interrupt for the interrupt pin. In the loop function, the code checks if the PWM is active and sets the PWM value to 128 if it is. The code also checks for serial input and updates the PWM frequency or duty cycle accordingly.
