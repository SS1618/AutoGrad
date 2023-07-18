#include "iostream"
#include "tensor.h"

using namespace std;
void dforward_test();

int main(){
    dforward_test();
}

void dforward_test(){
    Linear* lin = new Linear(3, 5);
    unsigned long dim[1] = {3};
    float vals[3] = {2, 3, 4};
    Tensor* x = new Tensor(dim, vals, 1);
    x->name = "input";
    Tensor* z = lin->feedforward(x);
    //z->backward();
    z->name = "output";
    z->clearTree();
    delete z;
    z = lin->feedforward(x);
    z->name = "output";
    z->backward();
    delete z;
    delete x;
    delete lin;
}