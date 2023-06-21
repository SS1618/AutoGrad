#include "iostream"
#include "tensor.h"
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(){
    vector<unsigned long> dims_A{1000, 1000};
    vector<double> vals;
    for(int i = 0; i < dims_A[0] * dims_A[1]; i++){
        vals.push_back(rand() / double(RAND_MAX));
    }
    Tensor* A = new Tensor(dims_A, vals);
    vector<unsigned long> dims_y{dims_A[0]};
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
    double avg_update = 0.0;
    double avg_back = 0.0;
    for(int j = 0; j < 5; j++){
        Tensor* v = Tensor::dot(A, x);
        Tensor* nv = Tensor::mult(neg, v);
        Tensor* z = Tensor::add(y, nv);
        Tensor* k = Tensor::dot(z, z);
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        k->backward();
        delete k;
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        A->update(0.1);
        high_resolution_clock::time_point t3 = high_resolution_clock::now();
        duration<double> update_time = duration_cast<duration<double>>(t3 - t2);
        duration<double> back_time = duration_cast<duration<double>>(t2 - t1);
        cout << "Backward Time: " << back_time.count() << endl;
        cout << "Update Time: " << update_time.count() << endl;
        avg_update += update_time.count();
        avg_back += back_time.count();

    }
    delete A;
    delete x;
    delete y;
    delete neg;
    cout << "Average Update Time: " << avg_update / 5.0 << endl;
    cout << "Average Backward Time: " << avg_back / 5.0 << endl;
}