
#include "iostream"
#include "tensor.h"
#include <immintrin.h>

using namespace std;

int main(){
    vector<unsigned long> dim_x{3, 2, 2};
    vector<double> vals_x{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor* x = new Tensor(dim_x, vals_x);
    x->getTensor()->print();
}