const int rightf = 6;
const int rightback = 5;
const int leftf = 9;
const int leftback = 10;
int x = 0;
int p = 0;
char rec[10];
unsigned long currenttime;
unsigned long previoustime;
void setup() {
  Serial.begin(9600);
  pinMode(rightf, OUTPUT);
  pinMode(rightback, OUTPUT);
  pinMode(leftf, OUTPUT);
  pinMode(leftback, OUTPUT);
}
void loop() {
  currenttime = millis();

  if (Serial.available()) {
    Serial.readBytes(rec, 10);
    Serial.setTimeout(50);
    x = atoi(rec);
    previoustime = millis();
  }
  else p = x;

  if (currenttime - previoustime >= 350) {
    if (x == p) {
      x = 0;
    }
  }

  if (x > 0) {
    left(x);
  }
  else if (x < 0) {
    int y = -x;
    right(y);
  }
  else forward();


}
void forward() {
  analogWrite(rightf, 150);
  analogWrite(rightback, 0);
  analogWrite(leftf, 150);
  analogWrite(leftback, 0);
}
void back() {
  analogWrite(rightf, 0);
  analogWrite(rightback, 100);
  analogWrite(leftf, 0);
  analogWrite(leftback, 100);
}
void right(int a) {
  analogWrite(rightf, 0);
  analogWrite(rightback, a  ); //0.3
  analogWrite(leftf,15 + a ); //a
  analogWrite(leftback, 0);
}
void left(int a) {
  analogWrite(rightf, 65); //a
  analogWrite(rightback, 0);
  analogWrite(leftf, 0);
  analogWrite(leftback, 50); //0.3
}
