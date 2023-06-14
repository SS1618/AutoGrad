#include "iostream"
#include "tensor.h"
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(){
    vector<unsigned long> dims_A{500, 500};
    vector<double> vals;
    for(int i = 0; i < dims_A[0] * dims_A[1]; i++){
        vals.push_back(rand() / double(RAND_MAX));
    }
    Tensor* A = new Tensor(dims_A, vals);
    vector<unsigned long> dims_y{dims_A[0], 1};
    vals.clear();
    for(int i = 0; i < dims_A[0]; i++){
        vals.push_back(0.0);
    }
    Tensor* y = new Tensor(dims_y, vals);

    vals.clear();
    for(int i = 0; i < dims_A[0]; i++){
        vals.push_back(rand() / double(RAND_MAX));
    }
    Tensor* x = new Tensor(dims_y, vals);

    vals.clear();
    vals.push_back(-1);
    dims_y.clear();
    Tensor* neg = new Tensor(dims_y, vals);
    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    for(int i = 0; i < 10; i++){
        Tensor* v = Tensor::dot(A, x);
        Tensor* z = Tensor::add(y, Tensor::mult(neg, v));
        Tensor* k = Tensor::dot(Tensor::transpose(z), z);
        k->backward();
        A->update(0.1);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    cout << time_span.count() << endl;
}