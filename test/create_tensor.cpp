#include "iostream"
#include "tensor.h"

using namespace std;

int main(){
    //create scalar tensor
    Tensor* x = new Tensor(3.0f);
    
    //create n-dim tensor
    unsigned long dims[1] = {5};
    float vals[5] = {1.0, 0.3, 0.4, 0.1, 2.0};
    Tensor* y = new Tensor(dims, vals, 1);

    y->print();

    delete x;
    delete y;

}