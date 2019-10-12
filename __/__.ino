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
    x = atoi(rec);
    previoustime = millis();
  }
  else p = x;

  if (currenttime - previoustime >= 1000) {
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
  analogWrite(rightf, 100);
  analogWrite(rightback, 0);
  analogWrite(leftf, 100);
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
  analogWrite(rightback, a); //0.3
  analogWrite(leftf, a); //a
  analogWrite(leftback, 0);
}
void left(int a) {
  analogWrite(rightf, a); //a
  analogWrite(rightback, 0);
  analogWrite(leftf, 0);
  analogWrite(leftback, a); //0.3
}
