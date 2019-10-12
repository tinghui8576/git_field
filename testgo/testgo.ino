const int rightf = 6;
const int rightback = 5;
const int leftf = 9;
const int leftback = 10;
int x = 0;
int p = 0;
char rec[10];
void setup() {
  Serial.begin(9600);
  pinMode(rightf, OUTPUT);
  pinMode(rightback, OUTPUT);
  pinMode(leftf, OUTPUT);
  pinMode(leftback, OUTPUT);
}
void loop() {
   if (Serial.available()) {
    Serial.readBytes(rec, 10);
    x = atoi(rec);
    Serial.print(x);
  }

  if (x ==0 ){
    stooop();
  }
  else if (x == 1){
    back();
  }
  else if(x == 2){
    forward();
  }
  else if(x == 3){
    rrig();
  }
  else if (x > 0) {
    right(x);
  }
  else if (x < 0) {
    int y = -x;
    left(y);
  }
  else forward();


}
void forward() {
  analogWrite(rightf, 130);
  analogWrite(rightback, 0);
  analogWrite(leftf, 100);
  analogWrite(leftback, 0);
  delay(1000);
  x = 0;
}
void back() {
  analogWrite(rightf, 0);
  analogWrite(rightback, 100);
  analogWrite(leftf, 0);
  analogWrite(leftback, 100);
  delay(1000);
  x = 0;
}
void right(int a) {
  analogWrite(rightf, 0);
  analogWrite(rightback, a); //0.3
  analogWrite(leftf, a); //a
  analogWrite(leftback, 0);
  delay(1000);
  x = 0;
}
void left(int a) {
  analogWrite(rightf, a); //a
  analogWrite(rightback, 0);
  analogWrite(leftf, 0);
  analogWrite(leftback,  a); //0.3
  delay(1000);
  x = 0;
}
void stooop() {
  analogWrite(rightf, 0); //a
  analogWrite(rightback, 0);
  analogWrite(leftf, 0);
  analogWrite(leftback,  0); //0.3
}
void rrig() {
  analogWrite(rightf, 170); //a
  analogWrite(rightback, 0);
  analogWrite(leftf, 0);
  analogWrite(leftback,  0); //0.3
  delay(1000);
  x = 0;
}
