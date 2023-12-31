#ifndef TENSOR_H
#define TENSOR_H
#include "includes.h"
using namespace std;

enum Operator {ADD, DOT, ND1D_DOT, NDMD_DOT, SCALARMULT, MULT, TRANS, RELU, SIGMOID};

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
    static NDimArray* random(unsigned long* dim, unsigned long dim_sz);
    static NDimArray* eye(unsigned long dim);
    static NDimArray* add(NDimArray* x, NDimArray* y);
    void add(NDimArray* x);
    static NDimArray* dot(NDimArray* x, NDimArray* y);
    static NDimArray* dot1d1d(NDimArray* x, NDimArray* y);
    static NDimArray* dotNd1d(NDimArray* x, NDimArray* y);
    static NDimArray* dotNdMd(NDimArray* x, NDimArray* y);
    static NDimArray* dotScalar(NDimArray* x, NDimArray* y);
    static NDimArray* mult(NDimArray* x, NDimArray* y);
    static NDimArray* transpose(NDimArray* x);
    static NDimArray* ReLU(NDimArray* x);
    static NDimArray* Sigmoid(NDimArray* x);
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
    string name;
    Tensor(){
        adjoint = NULL;
        tensor = NULL;
        keep = false;
        name = "None";
    }
    Tensor(unsigned long* dim, float* vals, unsigned long dim_sz);
    Tensor(float val);
    void set_name(string n){
        name = n;
    }
    void clearTree();
    void backward();
    NDimArray* getTensor();
    NDimArray* getAdjoint();
    void update(double step_sz);
    static Tensor* random(unsigned long* dim, unsigned long dim_sz);
    static Tensor* eye(unsigned long dim);
    static Tensor* add(Tensor* x, Tensor* y);
    static Tensor* dot(Tensor* x, Tensor* y);
    static Tensor* mult(Tensor* x, Tensor* y);
    static Tensor* transpose(Tensor* x);
    static Tensor* ReLU(Tensor* x);
    static Tensor* Sigmoid(Tensor* x);
    NDimArray* derivative(Tensor* x, Operator op);
    NDimArray* derivative_add(Tensor* x);
    NDimArray* derivative_dot1d1d(Tensor* x);
    NDimArray* derivative_dotNd1d(Tensor* x);
    NDimArray* derivative_dotNdMd(Tensor* x);
    NDimArray* derivative_relu(Tensor* x);
    NDimArray* derivative_sigmoid(Tensor* x);
    NDimArray* derivative_transpose(Tensor* x);
    NDimArray* derivative_mult(Tensor* x);
    NDimArray* derivative_scalarmult(Tensor* x);
    void print();
    Operator getOp();
    bool get_keep();
    void set_keep(bool k);
    ~Tensor();
};

class NN
{
    public:
        virtual Tensor* feedforward(Tensor* x) = 0;
};
class Linear : public NN
{
private:
    Tensor* weight;
    Tensor* bias;
public:
    Linear(){
        weight = NULL;
        bias = NULL;
    }
    Linear(unsigned long in_features, unsigned long out_features);
    Tensor* feedforward(Tensor* x);
    void update(float step_sz);
    ~Linear();
};

#endif