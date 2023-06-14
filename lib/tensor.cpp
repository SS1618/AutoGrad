#include "tensor.h"

Tensor::Tensor(vector<unsigned long>& dims, vector<double>& vals){
    tensor = new NDimArray(dims, vals);
    adjoint = NULL;
}

NDimArray* Tensor::getTensor(){
    return tensor;
}
NDimArray* Tensor::getAdjoint(){
    return adjoint;
}
void Tensor::update(double step_sz){
    for(int i = 0; i < adjoint->values.size(); i+=tensor->values.size()){
        #pragma omp parallel for
        for(int j = 0; j < tensor->values.size(); j++){
            tensor->values[j] -= step_sz * adjoint->values[i + j];
        }
    }
}
void Tensor::backward(){
    unsigned var_count = 1;
    for(int i = 0; i < tensor->dimension.size(); i++){
        var_count *= tensor->dimension[i];
    }
    vector<unsigned long> dims{var_count, var_count};
    adjoint = new NDimArray();
    adjoint->setidentity(dims);

    vector<Tensor*> visited_nodes;
    stack<Tensor*> nodes;
    for(int i = 0; i < parents.size(); i++){
        nodes.push(parents[i]);
    }
    while(!nodes.empty()){
        Tensor* n = nodes.top(); //get top node

        if(n->adjoint != NULL){ //skip if adjoint already computed
            nodes.pop();
            continue;
        }

        //check if adjoints for children have been computed
        bool comp_adj = true;
        for(int i = 0; i < n->children.size(); i++){
            if(n->children[i]->adjoint == NULL){
                comp_adj = false;
                break;
            }
        }

        if(comp_adj){
            //compute adjoint
            dims[1] = n->getTensor()->values.size();
            n->adjoint = new NDimArray();
            n->adjoint->setzero(dims);
            for(int i = 0; i < n->children.size(); i++){
                n->adjoint->add(NDimArray::dot(n->children[i]->adjoint, n->children[i]->derivative(n, n->children[i]->op)));
            }
            visited_nodes.push_back(nodes.top());
            nodes.pop();

            //add parents
            for(int i = 0; i < n->parents.size(); i++){
                nodes.push(n->parents[i]);
            }
        }
        else{
            //add children
            for(int i = 0; i < n->children.size(); i++){
                if(n->children[i]->adjoint == NULL){
                    nodes.push(n->children[i]);
                }
            }
        }

        #pragma omp parallel for
        for(int i = 0; i < visited_nodes.size(); i++){
            visited_nodes[i]->children.clear();
        }
    }
}

NDimArray* Tensor::derivative(Tensor* x, Operator op){
    NDimArray* jac = new NDimArray();
    if(op == ADD){
        vector<unsigned long> dims{getTensor()->values.size(), x->getTensor()->values.size()};
        jac->setidentity(dims);
    }
    else if(op == DOT){
        vector<unsigned long> dims{1, x->getTensor()->values.size()};
        vector<double> vals;
        Tensor* other = parents[0];
        if(parents[0] == x){
            other = parents[1];
        }
        for(int i = 0; i < other->getTensor()->values.size(); i++){
            vals.push_back(other->getTensor()->values[i]);
        }
        jac = new NDimArray(dims, vals);
    }
    else if(op == ND1D_DOT){
        Tensor* other = parents[0];
        if(parents[0] == x){
            other = parents[1];
        }
        if(x->getTensor()->dimension.size() == 1){
            vector<unsigned long> dims{getTensor()->values.size(), x->getTensor()->values.size()};
            jac = new NDimArray(dims, other->getTensor()->values);
        }
        else{
            vector<unsigned long> dims{getTensor()->values.size(), x->getTensor()->values.size()};
            jac->setzero(dims);

            #pragma omp parallel for
            for(int i = 0; i < jac->values.size(); i+= jac->dimension[1]){
                for(int j = 0; j < other->getTensor()->values.size(); j++){
                    jac->values[i + ((i / jac->dimension[1]) * other->getTensor()->values.size()) + j] = other->getTensor()->values[j];
                }
            }
        }
    }
    else if(op == NDMD_DOT){
        if(parents[0] == x){
            Tensor* other = parents[1];
            vector<unsigned long> dims{getTensor()->values.size(), x->getTensor()->values.size()};
            jac->setzero(dims);

            #pragma omp parallel for
            for(int i = 0; i < jac->values.size(); i+= jac->dimension[1]){
                for(int j = 0; j < x->getTensor()->dimension.back(); j++){
                    int jac_row = i / jac->dimension[1];
                    jac->values[i + ((jac_row / other->getTensor()->dimension.back()) * x->getTensor()->dimension.back()) + j] = other->getTensor()->values[(other->getTensor()->dimension.back() * j) + (jac_row % other->getTensor()->dimension.back())];
                }
            }
        }
        else{
            Tensor* other = parents[0];
            vector<unsigned long> dims{getTensor()->values.size(), x->getTensor()->values.size()};
            jac->setzero(dims);

            #pragma omp parallel for
            for(int i = 0; i < jac->values.size(); i+= jac->dimension[1]){
                for(int j = 0; j < x->getTensor()->dimension.back(); j++){
                    int jac_row = i / jac->dimension[1];
                    jac->values[(j * x->getTensor()->dimension.back()) + (jac_row % x->getTensor()->dimension.back()) + i] = other->getTensor()->values[((jac_row / x->getTensor()->dimension.back()) * other->getTensor()->dimension.back()) + j];
                }
            }
        }
    }
    else if (op == SCALARMULT){
        Tensor* other = parents[0];
        if(parents[0] == x){
            other = parents[1];
        }
        vector<unsigned long> dims{1, 1};
        vector<double> vals{other->getTensor()->values[0]};
        jac = new NDimArray(dims, vals);
    }
    else if(op == MULT){
        if(x->getTensor()->dimension.size() == 0){
            vector<unsigned long> dims{getTensor()->values.size(), 1};
            jac = new NDimArray(dims, parents[1]->getTensor()->values);
        }
        else{
            vector<unsigned long> dims{getTensor()->values.size(), x->getTensor()->values.size()};
            jac->setzero(dims);

            #pragma omp parallel for
            for(int i = 0; i < getTensor()->values.size(); i++){
                jac->values[i + (i * getTensor()->values.size())] = parents[0]->getTensor()->values[0];
            }
        }
    }
    else if(op == TRANS){
        vector<unsigned long> dims{getTensor()->values.size(), x->getTensor()->values.size()};
        jac->setzero(dims);

        #pragma omp parallel for
        for(int i = 0; i < getTensor()->values.size(); i++){
            int r = i / getTensor()->dimension.back();
            int c = i % getTensor()->dimension.back();
            jac->values[(i * x->getTensor()->values.size()) + (c * x->getTensor()->dimension.back()) + r] = 1.0;
        }
    }
    return jac;
}

Tensor* Tensor::add(Tensor* x, Tensor* y){
    Tensor* t = new Tensor();
    x->adjoint = NULL;
    y->adjoint = NULL;
    t->tensor = NDimArray::add(x->getTensor(), y->getTensor());
    t->parents.push_back(x);
    t->parents.push_back(y);
    t->op = ADD;
    x->children.push_back(t);
    y->children.push_back(t);
    return t;
}

Tensor* Tensor::dot(Tensor* x, Tensor* y){
    Tensor* t = new Tensor();
    x->adjoint = NULL;
    y->adjoint = NULL;
    t->tensor = NDimArray::dot(x->getTensor(), y->getTensor());
    t->parents.push_back(x);
    t->parents.push_back(y);
    if(x->getTensor()->dimension.size() == 1 and y->getTensor()->dimension.size() == 1){
        t->op = DOT;
    }
    else if(y->getTensor()->dimension.size() == 1){
        t->op = ND1D_DOT;
    }
    else if(x->getTensor()->dimension.size() == 0 and y->getTensor()->dimension.size() == 0){
        t->op = SCALARMULT;
    }
    else{
        t->op = NDMD_DOT;
    }
    x->children.push_back(t);
    y->children.push_back(t);
    return t;
}

Tensor* Tensor::mult(Tensor* x, Tensor* y){
    Tensor* t = new Tensor();
    x->adjoint = NULL;
    y->adjoint = NULL;
    t->tensor = NDimArray::mult(x->getTensor(), y->getTensor());
    t->parents.push_back(x);
    t->parents.push_back(y);
    t->op = MULT;
    x->children.push_back(t);
    y->children.push_back(t);
    return t;
}

Tensor* Tensor::transpose(Tensor* x){
    Tensor* t = new Tensor();
    x->adjoint = NULL;
    t->tensor = NDimArray::transpose(x->getTensor());
    t->parents.push_back(x);
    t->op = TRANS;
    x->children.push_back(t);
    return t;
}

void Tensor::print(){
    cout << "tensor(";
    tensor->print();
    cout << ")" << endl;
}

NDimArray::NDimArray(vector<unsigned long>& dims, vector<double>& vals){
    for(int i = 0; i < dims.size(); i++){
        dimension.push_back(dims[i]);
    }
    for(int i = 0; i < vals.size(); i++){
        values.push_back(vals[i]);
    }
}

NDimArray::NDimArray(double val){
    values.push_back(val);
}

NDimArray* NDimArray::add(NDimArray* x, NDimArray* y){
    assert(x->dimension.size() == y->dimension.size());

    #pragma omp parallel for
    for(int i = 0; i < x->dimension.size(); i++){
        assert(x->dimension[i] == y->dimension[i]);
    }
    assert(x->values.size() == y->values.size());
    vector<double> vals;
    for(int i = 0; i < x->values.size(); i++){
        vals.push_back(x->values[i] + y->values[i]);
    }
    return new NDimArray(x->dimension, vals);
}

void NDimArray::add(NDimArray* x){
    assert(x->dimension.size() == dimension.size());
    
    #pragma omp parallel for
    for(int i = 0; i < x->dimension.size(); i++){
        assert(x->dimension[i] == dimension[i]);
    }
    assert(x->values.size() == values.size());
    for(int i = 0; i < values.size(); i++){
        values[i] += x->values[i];
    }
}

NDimArray* NDimArray::dot(NDimArray* x, NDimArray* y){
    if(x->dimension.size() == 1 && y->dimension.size() == 1){
        assert(x->dimension[0] == y->dimension[0]);
        vector<unsigned long> dims{1};
        vector<double> vals{0.0};
        for(int i = 0; i < x->values.size(); i++){
            vals[0] += x->values[i] * y->values[i];
        }
        return new NDimArray(dims, vals);
    }
    else if(y->dimension.size() == 1){
        assert(y->dimension[0] == x->dimension.back());
        vector<unsigned long> dims;
        vector<double> vals;
        for(int i = 0; i < x->values.size(); i+=y->dimension[0]){
            double sum = 0.0;
            for(int j = 0; j < y->dimension[0]; j++){
                sum += (x->values[i + j] * y->values[j]);
            }
            vals.push_back(sum);
        }
        for(int i = 0; i < x->dimension.size() - 1; i++){
            dims.push_back(x->dimension[i]);
        }
        return new NDimArray(dims, vals);
    }
    else if(x->dimension.size() == 0 && y->dimension.size() == 0){
        return new NDimArray(x->values[0] * y->values[0]);
    }
    else{
        assert(x->dimension.back() == y->dimension[y->dimension.size() - 2]);
        vector<unsigned long> dims;
        vector<double> vals;
        for(int i = 0; i < x->values.size(); i+=x->dimension.back()){
            for(int j = 0; j < y->dimension.back(); j++){
                double sum = 0.0;
                for(int k = 0; k < x->dimension.back(); k++){
                    sum += x->values[i + k] * y->values[j + (k * y->dimension.back())];
                }
                vals.push_back(sum);
            }
        }
        for(int i = 0; i < x->dimension.size() - 1; i++){
            dims.push_back(x->dimension[i]);
        }
        dims.push_back(y->dimension.back());
        return new NDimArray(dims, vals);
    }
}

NDimArray* NDimArray::mult(NDimArray* x, NDimArray* y){
    assert(x->dimension.size() == 0);
    vector<double> vals;
    for(int i = 0; i < y->values.size(); i++){
        vals.push_back(y->values[i] * x->values[0]);
    }
    return new NDimArray(y->dimension, vals);
}

NDimArray* NDimArray::transpose(NDimArray* x){
    assert(x->dimension.size() >= 2);
    vector<unsigned long> dims;
    for(int i = 0; i < x->dimension.size() - 2; i++){
        dims.push_back(x->dimension[i]);
    }
    dims.push_back(x->dimension.back());
    dims.push_back(x->dimension[x->dimension.size() - 2]);
    vector<double> vals;
    for(int i = 0; i < x->dimension.back(); i++){
        for(int j = 0; j < x->values.size(); j+=x->dimension.back()){
            vals.push_back(x->values[j + i]);
        }
    }
    return new NDimArray(dims, vals);
}

void NDimArray::setzero(vector<unsigned long>& dims){
    dimension.clear();
    unsigned long var_count = 1;
    for(int i = 0; i < dims.size(); i++){
        dimension.push_back(dims[i]);
        var_count *= dims[i];
    }
    values.clear();
    if(dimension.size() == 0){
        values.push_back(0.0);
    }
    else{
        for(int i = 0; i < var_count; i++){
            values.push_back(0.0);
        }
    }
}

void NDimArray::setidentity(vector<unsigned long>& dims){
    dimension.clear();
    values.clear();
    unsigned long var_count = 1;
    for(int i = 0; i < dims.size(); i++){
        dimension.push_back(dims[i]);
        var_count *= dims[i];
    }
    if(dimension.size() == 0){
        values.push_back(1.0);
    }
    else if(dimension.size() == 1){
        for(int i = 0; i < values.size(); i++){
            values.push_back(1.0);
        }
    }
    else{
        for(int i = 0; i < var_count; i++){
            if(i / dimension.back() == i % dimension.back()){
                values.push_back(1.0);
            }
            else{
                values.push_back(0.0);
            }
        }
    }
}

double NDimArray::get(vector<unsigned long>& index){
    unsigned long shift = 0;
    unsigned long mult = 1;
    for(int i = index.size() - 1; i >= 0; i--){
        shift += mult * index[i];
        mult *= dimension[i];
    }
    return values[shift];
}

void NDimArray::print(){
    print_helper(0, 0, values.size());
}

void NDimArray::print_helper(unsigned long level, unsigned long index, unsigned long sz){
    //cout << "func " << level << ", " << index << endl;
    if(dimension.size() == 0){
        cout << "[" << values[0] << "]";
    }
    else if(level == dimension.size() - 1){
        cout << "[";
        for(int i = 0; i < dimension[level]; i++){
            cout << values[index + i];
            if(i != dimension[level] - 1){
                cout << ", ";
            }
        }
        cout << "]";
    }
    else{
        cout << "[";
        unsigned long size = dimension[level];
        for(int i = 0; i < size; i++){
            unsigned long new_index = index + i * (sz / size);
            print_helper(level + 1, new_index, sz / size);
            if(i != size - 1){
                cout << ", ";
            }
        }
        cout << "]";
    }
}