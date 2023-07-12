#include "iostream"
#include "tensor.h"
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(){
    unsigned long dims[2] = {5, 2};
    float vals[10];
    for(int i = 0; i < 10; i++){
        vals[i] = i;
    }

    unsigned long dims2[2] = {2, 5};
    float vals2[10];
    for(int i = 0; i < 10; i++){
        vals2[i] = i;
    }
    NDimArray* arr = new NDimArray(dims, vals, 2);
    NDimArray* arr2 = new NDimArray(dims2, vals2, 2);


    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    NDimArray* arrp = NDimArray::dot(arr2, arr);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    arr->print();
    cout << endl;
    arr2->print();
    cout << endl;
    arrp->print();
    cout << endl;
    duration<double> dur = duration_cast<duration<double>>(t2 - t1);
    cout << "Time: " << dur.count() << endl;
    delete arrp;
    delete arr2;
    delete arr;
}