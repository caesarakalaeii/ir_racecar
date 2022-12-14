# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:17:01 2022

@author: Caesar
"""

import tkinter as tk
import serial

# Set the serial port and baud rate
ser = serial.Serial('/dev/ttyACM0', 9600)

def set_pwm():
  # Get the PWM frequency and duty cycle values from the input fields
  pwm_frequency = int(frequency_field.get())
  pwm_duty_cycle = int(duty_cycle_field.get())

  # Encode the frequency and duty cycle values as a 32-bit integer
  # where the first two bits indicate the type of input (frequency or duty cycle)
  # and the remaining 30 bits contain the value of the input
  pwm_frequency_encoded = (1 << 30) | (pwm_frequency & 0x3fffffff)
  pwm_duty_cycle_encoded = (2 << 30) | (pwm_duty_cycle & 0x3fffffff)

  # Send the encoded frequency and duty cycle values over the serial port
  ser.write(pwm_frequency_encoded)
  ser.write(pwm_duty_cycle_encoded)

# Create the main window
root = tk.Tk()
root.title('PWM Control')

# Create the input fields for the PWM frequency and duty cycle
frequency_label = tk.Label(root, text='Frequency (Hz)')
frequency_field = tk.Entry(root)
duty_cycle_label = tk.Label(root, text='Duty Cycle (%)')
duty_cycle_field = tk.Entry(root)

# Create the submit button
submit_button = tk.Button(root, text='Submit', command=set_pwm)

# Place the input fields and submit button in the window
frequency_label.grid(row=0, column=0)
frequency_field.grid(row=0, column=1)
duty_cycle_label.grid(row=1, column=0)
duty_cycle_field.grid(row=1, column=1)
submit_button.grid(row=2, columnspan=2)

# Run the main event loop
root.mainloop()

# Close the serial port
ser.close()
