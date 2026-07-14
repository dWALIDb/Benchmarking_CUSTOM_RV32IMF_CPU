#pragma GCC optimize ("-O1")
#include"ann_weights2.h"
#include"neural_network.h"
float l0[16],l1[16];
#define LED_PIN 2 // Change to 1 or 4 if 2 does not work

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
delay(10000);
  Serial.print("input,output,cycles taken \n");

}

void loop() {

float input=-3.14159564f;
   for (int i = 0; i < 628; i++)
    {

        uint32_t  start = ESP.getCycleCount();
  
        float inp=standardize_input(input,INPUT_MEAN,INPUT_SCALE);
        forward_pass(&inp, 1, W0 ,BB00, l0, 4);
        tanh_activation(l0, 8);
        
        // uart_write("layer2\n\r",9);
        forward_pass(l0, 4, W1, BB11, l1, 8);
        tanh_activation(l1, 16);
        
        // uart_write("layer3\n\r",9);
        forward_pass(l1, 8, W2, BB22, l0, 8);
        tanh_activation(l0, 16);
        
        // uart_write("layer4\n\r",9);
        forward_pass(l0, 8, W3, BB33, l1, 4);
        tanh_activation(l1, 8);
        
        // Layer 4 (final output)
        forward_pass(l1, 4, W4, BB44, l0, 1);
        uint32_t end= ESP.getCycleCount();;
        uint32_t cycles= end-start;
        
      Serial.printf("%.4f , %.4f , %ld \n",input,l0[0],cycles);
      input+=0.01f;
    }



  digitalWrite(LED_PIN, HIGH);
  delay(1000);
  digitalWrite(LED_PIN, LOW);
  delay(1000);
while (1) {}
}