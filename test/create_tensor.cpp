#include "iostream"
#include "tensor.h"

using namespace std;

int main(){
    //create scalar tensor
    Tensor* x = new Tensor(3.0f);
    
    //create 1d tensor
    unsigned long dims1d[1] = {5};
    float vals1d[5] = {1.0, 0.3, 0.4, 0.1, 2.0};
    Tensor* y = new Tensor(dims1d, vals1d, 1);

    //create 2d tensor
    unsigned long dims2d[2] = {6, 7};
    float vals2d[42];
    for(int i = 0; i < 42; i++){
        vals2d[i] = rand();
    }
    Tensor* z = new Tensor(dims2d, vals2d, 2);

    //create 3d tensor
    unsigned long dims3d[3] = {3, 4, 2};
    float vals3d[24];
    for(int i = 0; i < 24; i++){
        vals3d[i] = rand();
    }
    Tensor* w = new Tensor(dims3d, vals3d, 3);

    //clean memory
    delete x;
    delete y;
    delete z;
    delete w;

}