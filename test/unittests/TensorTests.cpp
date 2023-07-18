#include "iostream"
#include "tensor.h"

using namespace std;

bool assert_equality(NDimArray* x, NDimArray* y);
void dadd_test();
void ddot1d1d_test();
void ddotnd1d_test();
void ddotndmnd_test();

int main(){
    dadd_test();
    ddot1d1d_test();
    ddotnd1d_test();
    ddotndmnd_test();
}



void dadd_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dims[2] = {3, 2};
    unsigned long jacdim[2] = {6, 6};
    float valsx[6] = {-3, 4, 2, 0, 4, -4};
    float valsy[6] = {-1, 2, 5, 11, 1, 0};
    Tensor* x = new Tensor(dims, valsx, 2);
    Tensor* y = new Tensor(dims, valsy, 2);
    Tensor* r = Tensor::add(x, y);
    NDimArray* drdx = r->derivative(x, ADD);
    NDimArray* drdy = r->derivative(y, ADD);
    NDimArray* exp = new NDimArray();
    exp->setidentity(jacdim, 2);
    if(assert_equality(drdx, exp) && assert_equality(drdy, exp)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete drdx; delete drdy; delete exp;
    cout << "Testing Tensor Derivative Add: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}

void ddot1d1d_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dims[1] = {4};
    unsigned long dimsd[2] = {1, 4};
    float valsx[4] = {1, 4, 5, 6};
    float valsy[4] = {2, 3, 7, 2};
    Tensor* x = new Tensor(dims, valsx, 1);
    Tensor* y = new Tensor(dims, valsy, 1);
    Tensor* r = Tensor::dot(x, y);
    NDimArray* drdx = r->derivative(x, DOT);
    NDimArray* drdy = r->derivative(y, DOT);
    NDimArray* drdx_exp = new NDimArray(dimsd, valsy, 2);
    NDimArray* drdy_exp = new NDimArray(dimsd, valsx, 2);
    if(assert_equality(drdx, drdx_exp) && assert_equality(drdy, drdy_exp)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete drdx; delete drdy; delete drdx_exp; delete drdy_exp;
    cout << "Testing Tensor Derivative Dot 1d 1d: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;

}

void ddotnd1d_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dimx[2] = {3, 3};
    unsigned long dimy[1] = {3};
    unsigned long dimdrdx[2] = {3, 9};
    float valsx[9] = {1, 4, -1, 2, 0, 2, 1, -5, 7};
    float valsy[3] = {1, 3, 4};
    float valsdrdx_exp[27] = {1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 4};
    Tensor* x = new Tensor(dimx, valsx, 2);
    Tensor* y = new Tensor(dimy, valsy, 1);
    Tensor* r = Tensor::dot(x, y);
    NDimArray* drdx = r->derivative(x, ND1D_DOT);
    NDimArray* drdy = r->derivative(y, ND1D_DOT);
    NDimArray* drdx_exp = new NDimArray(dimdrdx, valsdrdx_exp, 2);
    NDimArray* drdy_exp = new NDimArray(dimx, valsx, 2);
    if(assert_equality(drdx, drdx_exp) && assert_equality(drdy, drdy_exp)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete drdx; delete drdy; delete drdx_exp; delete drdy_exp;
    cout << "Testing Tensor Derivative Dot Nd 1d: " << endl;
    cout << "Tests Passed: " << pass_count << " / " << test_count << endl;
}

void ddotndmnd_test(){
    int pass_count = 0;
    int test_count = 1;
    unsigned long dimx[2] = {3, 2};
    unsigned long dimy[2] = {2, 2};
    unsigned long dimdrdx[2] = {6, 6};
    unsigned long dimdrdy[2] = {6, 4};
    float valsx[6] = {1, 4, -2, 0, 0, -5};
    float valsy[4] = {0, 1, 2, 3};
    float valsdrdx[36] = {0, 2, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 3};
    float valsdrdy[24] = {1, 0, 4, 0, 0, 1, 0, 4, -2, 0, 0, 0, 0, -2, 0, 0, 0, 0, -5, 0, 0, 0, 0, -5};
    Tensor* x = new Tensor(dimx, valsx, 2);
    Tensor* y = new Tensor(dimy, valsy, 2);
    Tensor* r = Tensor::dot(x, y);
    NDimArray* drdx = r->derivative(x, NDMD_DOT);
    NDimArray* drdy = r->derivative(y, NDMD_DOT);
    NDimArray* drdx_exp = new NDimArray(dimdrdx, valsdrdx, 2);
    NDimArray* drdy_exp = new NDimArray(dimdrdy, valsdrdy, 2);
    if(assert_equality(drdx, drdx_exp) && assert_equality(drdy, drdy_exp)){
        pass_count += 1;
    }
    delete x; delete y; delete r; delete drdx; delete drdy; delete drdx_exp; delete drdy_exp;
    cout << "Testing Tensor Derivative Dot Nd Md: " << endl;
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