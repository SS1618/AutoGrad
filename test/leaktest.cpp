#include "iostream"
#include "tensor.h"

using namespace std;

int main(){
    vector<unsigned long> dim_x{3, 3};
    vector<double> vals_x{1, 2, 3, 2, 2, 3, 1, 2, 3};

    vector<unsigned long> dim_y{3, 3};
    vector<double> vals_y{4, 5, 6, 1, 2, 3, 1, 2, 3};

    vector<unsigned long> dim_w{3};
    vector<double> vals_w{0, 2, 1};

    Tensor* x = new Tensor(dim_x, vals_x);
    Tensor* y = new Tensor(dim_y, vals_y);
    Tensor* w = new Tensor(dim_w, vals_w);
    Tensor* z = Tensor::add(Tensor::dot(x, w), w);
    //z->set_keep(true);
    z->backward();
    delete x;
    delete y;
    delete w;
    delete z;
}