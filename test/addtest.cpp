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

    vector<unsigned long> index{0};

    cout << z->getTensor()->get(index) << endl;
    z->backward();

    vector<unsigned long>adj_index{1,4};
    cout << x->getAdjoint()->get(adj_index) << endl;

    vector<unsigned long> dim_k;
    vector<double> vals_k{-1};

    Tensor* k = new Tensor(dim_k, vals_k);
    Tensor* p = Tensor::mult(k, x);
    p->backward();

    vector<unsigned long> p_index{2, 6};
    //cout << x->getAdjoint()->get(p_index) << endl;

    Tensor* t = Tensor::transpose(p);
    //cout << t->getTensor()->get(p_index) << endl;
    t->backward();
    cout << p->getAdjoint()->get(p_index) << endl;
}