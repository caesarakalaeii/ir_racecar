import asyncio

import serial
import time




ser = serial.Serial()
class config():
    frequency = 100
    duty_cycle = 10
    baudrate = 9200 
    port = '/dev/ttyUSB0'




    

def write_to_serial(mode, value):
    data= (mode <<30) + value
    data_byte = str(data).encode()
    print("Value: {} ".format(value) + "Binary: {0:b}".format(value))
    print("Mode: {} ".format(mode) + "Binary: {0:b}".format(mode))
    print("Int data: {} ".format(data)+"Binary data:{0:b}".format(data))
    ser.write(data_byte)

async def serialPlot():
        print(ser.read_all)


async def async_main():
    stop = False
    while (not stop):
        stringArray = input().split("=")
        mode = 0
        value = -1
        
        if(stringArray[0] == "c"):
            stop = True
            print("Stopping...")
        elif (stringArray[0] == "f"):
            try:
                
                value = int(stringArray[1])
            except:
                print("Typecast failed, no or non numerical value was given")
                continue
            if(value>0):
                frequency = value
                mode = 1
                write_to_serial(mode,value)
                stringArray[0]=0
                print("Frequency set to {}Hz".format(value))
            else:
                    print("Frequency may not be 0 or negative. {} was given.".format(value))
                         
        elif (stringArray[0] == "d"):
            try:
                value = int(stringArray[1])
            except:
                print("Typecast failed, no or non numerical value was given")
                continue
            if(value>0 and value <= 100):
                duty_cycle = value
                mode = 2
                write_to_serial(mode,value)
                stringArray[0]=0
                print("Duty cycle set to {}%".format(value))
            else:
                print("Duty cycle only allowes values between 0 and 100. {} was given".format(value))
        elif(stringArray[0]=='s'):
            try:
                stop = int(stringArray[1])
                start = int(stringArray[2])
                step = int(stringArray[3])

            except:
                print("Typecast failed, no or non numerical value was given")
                continue

            for i in range(start,stop,step):
                write_to_serial(1,i)
                print("Sweeping: frequency is: {}Hz".format(i))
                time.sleep(3)
                for j in range(0, 100, 10):
                    write_to_serial(2,j)
                    print("Sweeping: Duty cycle is: {}".format(j))
                    time.sleep(3)
        elif(stringArray[0] =='a'):
            write_to_serial(3,0)


            
        else:
            print("not a known command")

    print("Stopped")

if __name__ == '__main__':
    print("Serial initialized")
    ser.baudrate = config.baudrate
    print("baudrate set to {}".format(ser.baudrate))
    ser.port = config.port
    print("Serial port is set to {}".format(ser.port))
    frequency = config.frequency
    print("Frequency is set to {}Hz".format(frequency) )
    duty_cycle = config.duty_cycle
    print("Duty cycle is set to {}%".format(duty_cycle) )
    print("Use the following input:\nf=int to change the frequency\nd=int (0-100) to change the duty_cycle\nc to stop")
    print("s=target=start=step, sweep frequencies, start defines starting frequency, target defines target frequency, stpe defines step size")
    ser.open()
    print("Opening port")
    if(ser.isOpen()):
        asyncio.run(async_main())
    else:
        print("Could not open port")



    
