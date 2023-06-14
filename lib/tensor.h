#ifndef TENSOR_H
#define TENSOR_H
#include "includes.h"
using namespace std;

enum Operator {ADD, DOT, ND1D_DOT, NDMD_DOT, SCALARMULT, MULT, TRANS};

class NDimArray
{
private:
    void print_helper(unsigned long level, unsigned long index, unsigned long sz);
public:
    vector<double> values;
    vector<unsigned long> dimension;
    NDimArray(){}
    NDimArray(vector<unsigned long>& dims, vector<double>& vals);
    NDimArray(double val);
    static NDimArray* add(NDimArray* x, NDimArray* y);
    void add(NDimArray* x);
    static NDimArray* dot(NDimArray* x, NDimArray* y);
    static NDimArray* mult(NDimArray* x, NDimArray* y);
    static NDimArray* transpose(NDimArray* x);
    void setzero(vector<unsigned long>& dims);
    void setidentity(vector<unsigned long>& dims);
    double get(vector<unsigned long>& index);
    void print();
};

class Tensor
{
private:
    NDimArray* tensor;
    NDimArray* adjoint;
    vector<Tensor*> parents;
    vector<Tensor*> children;
    Operator op;
public:
    Tensor(){
        adjoint = NULL;
        tensor = NULL;
    }
    Tensor(vector<unsigned long>& dims, vector<double>& vals);
    void backward();
    NDimArray* getTensor();
    NDimArray* getAdjoint();
    void update(double step_sz);
    static Tensor* add(Tensor* x, Tensor* y);
    static Tensor* dot(Tensor* x, Tensor* y);
    static Tensor* mult(Tensor* x, Tensor* y);
    static Tensor* transpose(Tensor* x);
    NDimArray* derivative(Tensor* x, Operator op);
    void print();
};


#endif