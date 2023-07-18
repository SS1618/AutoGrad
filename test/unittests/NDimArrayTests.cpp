#include "iostream"
#include "tensor.h"

using namespace std;

bool assert_equality(NDimArray* x, NDimArray* y);
void add_test();
void dot_test();
void mult_test();
void transpose_test();
void relu_test();
void sigmoid_test();
void setidentity_test();

int main(){
    add_test();
    dot_test();
    mult_test();
    transpose_test();
    relu_test();
    sigmoid_test();
    setidentity_test();
}

void add_test(){
    int pass_count = 0;
    int test_count = 4;
    //scalar add
    NDimArray* x = new NDimArray(3);
    NDimArray* y = new NDimArray(5);
    NDimArray* r = NDimArray::add(x, y);
    NDimArray* exp = new NDimArray(8);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    //1d add
    unsigned long dims1d[1] = {4};
    float valsx[4] = {3, 4, 5, 3};
    float valsy[4] = {6, 7, 1, 2};
    float valsexp[4] = {9, 11, 6, 5};
    x = new NDimArray(dims1d, valsx, 1);
    y = new NDimArray(dims1d, valsy, 1);
    r = NDimArray::add(x, y);
    exp = new NDimArray(dims1d, valsexp, 1);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    //2d add
    unsigned long dims2d[2] = {2, 3};
    float valsx2[6] = {3, 4, 2, 1, 4, 3};
    float valsy2[6] = {2, 7, 9, -1, 0, -10};
    float valsexp2[6] = {5, 11, 11, 0, 4, -7};
    x = new NDimArray(dims2d, valsx2, 1);
    y = new NDimArray(dims2d, valsy2, 1);
    r = NDimArray::add(x, y);
    exp = new NDimArray(dims2d, valsexp2, 1);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    //3d add
    unsigned long dims3d[3] = {2, 3, 2};
    float valsx3[12] = {4, 5, -2, -10, 10, 11, 3, 4, 2, 1, 4, 3};
    float valsy3[12] = {3, -2, -1, -9, -7, 3, 2, 7, 9, -1, 0, -10};
    float valsexp3[12] = {7, 3, -3, -19, 3, 14, 5, 11, 11, 0, 4, -7};
    x = new NDimArray(dims3d, valsx3, 1);
    y = new NDimArray(dims3d, valsy3, 1);
    r = NDimArray::add(x, y);
    exp = new NDimArray(dims3d, valsexp3, 1);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    
    cout << "Testing NDimArray Add: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
    
}
void dot_test(){
    int pass_count = 0;
    int test_count = 4;
    //scalar dot
    NDimArray* x = new NDimArray(3);
    NDimArray* y = new NDimArray(5);
    NDimArray* r = NDimArray::dotScalar(x, y);
    NDimArray* exp = new NDimArray(15);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    //1d1d dot
    unsigned long dims1d[1] = {4};
    float valsx[4] = {3, 4, 5, 3};
    float valsy[4] = {6, 7, 1, 2};
    float valsexp = 18+28+5+6;
    x = new NDimArray(dims1d, valsx, 1);
    y = new NDimArray(dims1d, valsy, 1);
    r = NDimArray::dot1d1d(x, y);
    exp = new NDimArray(valsexp);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    //Nd1d dot
    unsigned long dimsnd[2] = {3, 4};
    unsigned long dimsexp[1] = {3};
    float valsx2[12] = {1, 2, 3, 4, 2, 3, 5, 4, -1, -3, -2, 0};
    float valsy1[4] = {1, 2, 3, 2};
    float valsexp1[3] = {1+4+9+8, 2+6+15+8, -1-6-6};
    x = new NDimArray(dimsnd, valsx2, 2);
    y = new NDimArray(dims1d, valsy1, 1);
    r = NDimArray::dotNd1d(x, y);
    exp = new NDimArray(dimsexp, valsexp1, 1);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    //NdMd dot
    unsigned long dimsmd[2] = {4, 2};
    unsigned long dimsexp2[2] = {3, 2};
    float valsy2[8] = {2, 1, -1, 3, -2, -4, 2, 5};
    float valsexp2[6] = {2-2-6+8, 1+6-12+20, 4-3-10+8, 2+9-20+20, -2+3+4, -1-9+8};
    x = new NDimArray(dimsnd, valsx2, 2);
    y = new NDimArray(dimsmd, valsy2, 2);
    r = NDimArray::dotNdMd(x, y);
    exp = new NDimArray(dimsexp2, valsexp2, 2);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;

    cout << "Testing NDimArray Dot: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}
void mult_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dims1d[1] = {4};
    float valsy[4] = {6, 7, 1, 2};
    float valsexp[4] = {18, 21, 3, 6};
    NDimArray* x = new NDimArray(3);
    NDimArray* y = new NDimArray(dims1d, valsy, 1);
    NDimArray* r = NDimArray::mult(x, y);
    NDimArray* exp = new NDimArray(dims1d, valsexp, 1);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete exp;
    cout << "Testing NDimArray Mult: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}
void transpose_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dimsnd[2] = {3, 4};
    float valsnd[12] = {1, 2, 3, 4, 2, 3, 5, 4, -1, -3, -2, 0};
    unsigned long dimstrans[2] = {4, 3};
    float valstrans[12] = {1, 2, -1, 2, 3, -3, 3, 5, -2, 4, 4, 0};
    NDimArray* x = new NDimArray(dimsnd, valsnd, 2);
    NDimArray* exp = new NDimArray(dimstrans, valstrans, 2);
    NDimArray* r = NDimArray::transpose(x);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete r; delete exp;
    cout << "Testing NDimArray Transpose: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}
void relu_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dims1d[1] = {4};
    float valsx[4] = {0, 4, -4, 3};
    float valsexp[4] = {0, 4, 0, 3};
    NDimArray* x = new NDimArray(dims1d, valsx, 1);
    NDimArray* exp = new NDimArray(dims1d, valsexp, 1);
    NDimArray* r = NDimArray::ReLU(x);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete r; delete exp;
    
    cout << "Testing NDimArray ReLU: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}
void sigmoid_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dims1d[1] = {4};
    float valsx[4] = {0, 4, -4, 3};
    float valsexp[4] = {0.5, 1.0/(1.0 + exp(-4.0)), 1.0/(1.0 + exp(4.0)), 1.0/(1.0 + exp(-3.0))};
    NDimArray* x = new NDimArray(dims1d, valsx, 1);
    NDimArray* exp = new NDimArray(dims1d, valsexp, 1);
    NDimArray* r = NDimArray::Sigmoid(x);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete x; delete r; delete exp;
    
    cout << "Testing NDimArray Sigmoid: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}

void setidentity_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dims[2] = {3, 3};
    float vals[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    NDimArray* r = new NDimArray();
    r->setidentity(dims, 2);
    NDimArray* exp = new NDimArray(dims, vals, 2);
    if(assert_equality(exp, r)){
        pass_count += 1;
    }
    delete r; delete exp;
    
    cout << "Testing NDimArray Set Identity: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}

bool assert_equality(NDimArray* x, NDimArray* y){
    if(x->values_size != y->values_size){
        return false;
    }
    if(x->dimension_size != y->dimension_size){
        return false;
    }
    for(int i = 0; i < x->dimension_size; i++){
        if(x->dimension[i] != y->dimension[i]){
            return false;
        }
    }
    for(int i = 0; i < x->values_size; i++){
        if(x->values[i] != y->values[i]){
             return false;
        }
    }
    return true;
}