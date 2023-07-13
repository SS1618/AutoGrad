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
    float* values;
    unsigned long* dimension;
    unsigned long values_size;
    unsigned long dimension_size;
    NDimArray(){}
    NDimArray(unsigned long* dim, float* vals, unsigned long dim_sz);
    NDimArray(float val);
    static NDimArray* add(NDimArray* x, NDimArray* y);
    void add(NDimArray* x);
    static NDimArray* dot(NDimArray* x, NDimArray* y);
    static NDimArray* dot1d1d(NDimArray* x, NDimArray* y);
    static NDimArray* dotNd1d(NDimArray* x, NDimArray* y);
    static NDimArray* dotNdMd(NDimArray* x, NDimArray* y);
    static NDimArray* dotScalar(NDimArray* x, NDimArray* y);
    static NDimArray* mult(NDimArray* x, NDimArray* y);
    static NDimArray* transpose(NDimArray* x);
    void setzero(unsigned long* dims, unsigned long dim_sz);
    void setidentity(unsigned long* dims, unsigned long dim_sz);
    double get(unsigned long* index);
    void print();
    ~NDimArray();
};

class Tensor
{
private:
    NDimArray* tensor;
    NDimArray* adjoint;
    vector<Tensor*> parents;
    vector<Tensor*> children;
    Operator op;
    bool keep;
public:
    Tensor(){
        adjoint = NULL;
        tensor = NULL;
        keep = false;
    }
    Tensor(unsigned long* dim, float* vals, unsigned long dim_sz);
    Tensor(float val);
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
    Operator getOp();
    bool get_keep();
    void set_keep(bool k);
    ~Tensor();
};


#endif