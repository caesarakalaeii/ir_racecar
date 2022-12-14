// Set the PWM frequency (in Hz)
long input;
const int TYPE_FREQ = 1;
const int TYPE_DUTY = 2;
long clockSpeed = 16000000; // Set to whatever clockspeed the arduino is running at
long pwmFrequency = 400000; // Works best to avoid stripes on Camera frame
int pwmDuty = 128;
int value;

// Set the interrupt pin
const int interruptPin = 2;

// Set the PWM pin
const int pwmPin = 3;

// Set a variable indicating whether the PWM is active or not
bool pwmActive = true;

void setPwmFrequency(long frequency) {
  // Calculate the prescaler value based on the input frequency
  int prescaler = clockSpeed / frequency;

  // Set the prescaler value for timer 2 by manipulating the timer registers
  TCCR2B = (TCCR2B & 0b11111000) | (prescaler & 0x07);
}

void setPwmDutyCycle(long dutyCycle){
  pwmDuty = (int) 255*(dutyCycle/100);
  
}

// this function is called when the interrupt is triggered
void handleInterrupt() {
  // Change the state of the PWM
  pwmActive = !pwmActive;
}

void setup() {
  // Set the PWM and interrupt pin as output
  pinMode(pwmPin, OUTPUT);
  pinMode(interruptPin, INPUT_PULLUP);

  // Set the PWM frequency
  setPwmFrequency(pwmFrequency);

  // Enable the interrupt for the interrupt pin
  attachInterrupt(digitalPinToInterrupt(interruptPin), handleInterrupt, FALLING);
}

void loop() {
  if (pwmActive) {
    // Set the PWM value to 128, which corresponds to a duty cycle of 50%.
    analogWrite(pwmPin, pwmDuty);
  }
  if (Serial.available() > 0) {
    if (Serial.available()) { //for Serial inputs
    input=Serial.parseInt();
    Serial.print("input read:");
    Serial.println(input);
    int type = (int) (input >> 30); //bit shift to access type integer
    input = input << 2;
    value = input >> 2;
    if(type == TYPE_FREQ){
      setPwmFrequency(value);
    }
    if(type == TYPE_FREQ){
      setPwmDutyCycle(value);
    }
    
  } 
  }
  
}
