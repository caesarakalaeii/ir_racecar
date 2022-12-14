
#define OUTPUT_PIN 6
#define INTERRUPT_PIN 3
long input;
int TYPE_FREQ = 1;
int TYPE_DUTY = -2;
int TYPE_DEV = -1;
bool dev_mode;
int pin_control_freq_A= 6;
int pin_control_freq_B= 7;
int pin_control_duty_A= 8;
int pin_control_duty_B= 9;
double frequency; //freq in Hz
double duty_cycle; // Cycle in %
double value;
// Set a variable indicating whether the PWM is active or not
bool pwmActive = true;
int count = 0;

void handleInterrupt() {
  // Change the state of the PWM
  if(count == 1){
    pwmActive = !pwmActive;
    count = 0;
  }
  else{
    count = count +1;
  }
}

void setup() {
  pinMode(pin_control_freq_A, INPUT);
  pinMode(pin_control_freq_B, INPUT);
  pinMode(pin_control_duty_A, INPUT);
  pinMode(pin_control_duty_B, INPUT);
  pinMode(OUTPUT_PIN, OUTPUT);
  pinMode(INTERRUPT_PIN, INPUT);

  // Enable the interrupt for the interrupt pin
  attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), handleInterrupt, CHANGE); 
  
  frequency = 400000;
  duty_cycle = 50;
  Serial.begin(9600);
  input = 0;
  pwmActive = true;
  dev_mode = false;
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
      frequency = value;
      Serial.print("frequency set");
    }
    else if(type == TYPE_DUTY){
      duty_cycle = value;
      Serial.print("cycle set");
    }
    else if(type == TYPE_DEV){
      pwmActive = !pwmActive;
      Serial.print("pwmActive changed");
    }
    
  } 
}


void loop() {

  // Calculate the period and the amount of time the output is on for (HIGH) and 
  // off for (LOW).
  if(pwmActive){
    double period = 1000000 / frequency;
    double offFor = period - (period * (duty_cycle/100));
    double onFor = period - offFor;
  
  
    if( period > 16383 ) {
      // If the period is greater than 16383 then use the millisecond timer delay,
      // otherwise use the microsecond timer delay. Currently, the largest value that
      // will produce an accurate delay for the microsecond timer is 16383.
  
      digitalWrite(OUTPUT_PIN, HIGH);
      delay((long)onFor/1000);
      
      digitalWrite(OUTPUT_PIN, LOW);
      delay((long)offFor/1000);
    } else {
      digitalWrite(OUTPUT_PIN, HIGH);
      delayMicroseconds((long)onFor);
      
      digitalWrite(OUTPUT_PIN, LOW);
      delayMicroseconds((long)offFor);
    }
  }
}
