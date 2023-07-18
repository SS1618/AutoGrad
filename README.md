# AutoGrad
This is an auto-differentiation library built from the ground up in C++.
## Features
+ Easily create N-dimensional arrays  
+ Perform basic operations such as addition, dot products, transposition, etc. on these arrays  
+ Tape-Based Autograd: dynamically creates operation trees for auto differentiation  
+ Create basic MLPs
## Installation
The current best way to install Autograd is from the source. Simply clone this repo and differentiate away!
## Quick Example
```
#include "iostream"
#include "tensor.h"

using namespace std;

int main(){
  unsigned long dimA[2] = {3, 2};
  Tensor* A = Tensor::random(dimA, 2); //creates 3x2 array of random floats
    
  unsigned long dimX[1] = {2};
  float valsX[2] = {1.0f, -1.0f};
  Tensor* x = new Tensor(dimX, valsX, 1); //creates vector [1, -1]

  Tensor* y = Tensor::dot(A, x);
  Tensor* z = Tensor::dot(y, y);

  z->backward();

  x->getAdjoint()->print(); //prints dz/dx

  //clean up memory
  delete A;
  delete x;
  delete z;
}
```
## Neural Networks
Here is an example of a neural network implemented with our library.
```
#include "iostream"
#include "tensor.h"
#include "fstream"

using namespace std;


class Model{
private:
    Linear* layer1;
    Linear* layer2;
public:
    Model(int input_sz){
        layer1 = new Linear(input_sz, 128);
        layer2 = new Linear(128, 10);
    }
    Tensor* feedforward(Tensor* input){
        return Tensor::Sigmoid(layer2->feedforward(Tensor::ReLU(layer1->feedforward(input))));
    }
    Tensor* MSE(Tensor* expected, Tensor* output){
        Tensor* neg = new Tensor(-1);
        neg->set_keep(false);
        Tensor* neg_output = Tensor::mult(neg, output);
        Tensor* z = Tensor::add(expected, neg_output);
        return Tensor::dot(z, z);
    }
    void update(float step_sz){
        layer2->update(step_sz);
        layer1->update(step_sz);
    }
    ~Model(){
        delete layer1;
        delete layer2;
    }
};
int main(){
    /*load data*/

    Model* model = new Model(28*28);
    Tensor* output = model->feedforward(pixel_data[i]); //pixel_data stores input images
    Tensor* error = model->MSE(expected[i], output);
    error->backward();
    model->update(0.001);
    cout << "Error: " << error->getTensor()->values[0] << endl;
    
    delete error;
    delete model;
    
    for(int i = 0; i < number_of_images; i++){
        delete pixel_data[i];
    }
    for(int i = 0; i < 10; i++){
        delete labels_tens[i];
    }
    //free(pixel_data);
    free(labels);
}
```
## Coming Soon
+ CPU optimization
+ GPU integration
+ Additional operations such as convolutions and softmax
