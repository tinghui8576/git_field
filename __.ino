const int r1 = 5;
const int r2 = 6;
const int l1 = 9;
const int l2 = 10;
int x = 0;
int p = 0;
char rec[10];
unsigned long currenttime;
unsigned long previoustime;
void setup() {
  Serial.begin(9600);
  pinMode(r1, OUTPUT);
  pinMode(r2, OUTPUT);
  pinMode(l1, OUTPUT);
  pinMode(l2, OUTPUT);
}
void loop() {
  currenttime = millis();

  if (Serial.available()) {
    Serial.readBytes(rec, 10);
    x = atoi(rec);
    previoustime = millis();
  }
  else p = x;

  if (currenttime - previoustime >= 1500) {
    if (x == p) {
      x = 0;
    }
  }

  if (x > 0) {
    right(x);
  }
  else if (x < 0) {
    int y = -x;
    left(y);
  }
  else forward();


}
void forward() {
  analogWrite(r1, 150);
  analogWrite(r2, 0);
  analogWrite(l1, 150);
  analogWrite(l2, 0);
}
void right(int a) {
  analogWrite(r1, 0);
  analogWrite(r2, 0.5 * a);
  analogWrite(l1, a);
  analogWrite(l2, 0);
}
void left(int a) {
  analogWrite(r1, a);
  analogWrite(r2, 0);
  analogWrite(l1, 0);
  analogWrite(l2, 0.5 * a);
}
