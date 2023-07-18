#include "iostream"
#include "tensor.h"

using namespace std;

int main(){
	unsigned long dimA[2] = {3, 2};
	Tensor* A = Tensor::random(dimA, 2);
    
    unsigned long dimX[1] = {2};
    float valsX[2] = {1.0f, -1.0f};
    Tensor* x = new Tensor(dimX, valsX, 1);

    Tensor* y = Tensor::dot(A, x);
    Tensor* z = Tensor::dot(y, y);

    z->backward();

    x->getAdjoint()->print();

    delete A;
    delete x;
    delete z;
}