long input;
const int TYPE_FREQ = 1;
const int TYPE_DUTY = -2;
const int TYPE_DEV = -1;
const int pwmPin = 6; // PWM output pin
const int triggerPin = 3; // Input pin for triggering
long pwmFrequency = 13371337; // Works best to avoid stripes on Camera frame
int pwmDuty = 128;
int value;
bool pwmActive = true;

int lastTriggerState = LOW; // Variable to store the previous state of the trigger pin
bool pwmEnabled = false; // Flag to track PWM enable/disable status

void setup() {
  pinMode(pwmPin, OUTPUT); // Set the PWM pin as an output
  pinMode(triggerPin, INPUT); // Set the trigger pin as an input
}

void loop() {
  if(pwmActive){
  int triggerState = digitalRead(triggerPin); // Read the current state of the trigger pin

  // Check for rising edge on the trigger pin
  if (triggerState == HIGH && lastTriggerState == LOW) {
    pwmEnabled = !pwmEnabled; // Toggle the PWM enable/disable status

    if (pwmEnabled) {
      // Enable PWM on pin 6 with a duty cycle of 50%
      analogWrite(pwmPin, pwmDuty);
      Serial.println("PWM Enabled");
    } else {
      // Disable PWM on pin 6
      analogWrite(pwmPin, 0);
      Serial.println("PWM Disabled");
    }
  }

  lastTriggerState = triggerState; // Update the previous trigger state

}
}


void serialEvent(){
    if (Serial.available()) { //for Serial inputs
    input=Serial.parseInt();
    int input_int = input;
    Serial.print("input read:");
    Serial.println(input, BIN);
    int type = input >> 30;
    input = input << 2;
    value = (int)input >> 2;
    Serial.print("value is:");
    Serial.println(value, DEC);
    Serial.print("type is:");
    Serial.println(type, DEC);
    if(type == TYPE_FREQ){
      pwmFrequency = value;
      Serial.print("frequency set");
    }
    else if(type == TYPE_DUTY){
      pwmDuty = value;
      Serial.print("cycle set");
    }
    else if(type == TYPE_DEV){
      pwmActive = !pwmActive;
      Serial.print("pwmActive changed");
    }
    
  } 
}
