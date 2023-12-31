#include "tensor.h"

Linear::Linear(unsigned long in_features, unsigned long out_features){
    unsigned long dimsW[2] = {out_features, in_features};
    float valsW[in_features * out_features];
    for(int i = 0; i < in_features * out_features; i++){
        valsW[i] = rand() / (1000 * float(RAND_MAX));
    }
    weight = new Tensor(dimsW, valsW, 2);
    weight->set_name("weights");
    unsigned long dimsB[1] = {out_features};
    float valsB[out_features];
    for(int i = 0; i < out_features; i++){
        valsB[i] = rand() / (1000 * float(RAND_MAX));
    }
    bias = new Tensor(dimsB, valsB, 1);
    bias->set_name("bias");
}

Tensor* Linear::feedforward(Tensor* x){
    return Tensor::add(Tensor::dot(weight, x), bias);
}

void Linear::update(float step_sz){
    weight->update(step_sz);
    bias->update(step_sz);
}

Linear::~Linear(){
    delete weight;
    delete bias;
}

Tensor::Tensor(unsigned long* dim, float* vals, unsigned long dim_sz){
    tensor = new NDimArray(dim, vals, dim_sz);
    adjoint = NULL;
    keep = true;
}

Tensor::Tensor(float val){
    tensor = new NDimArray(val);
    adjoint = NULL;
    keep = true;
}

Tensor::~Tensor(){
    delete tensor;
    delete adjoint;
    parents.clear();
    children.clear();
}

NDimArray* Tensor::getTensor(){
    return tensor;
}
NDimArray* Tensor::getAdjoint(){
    return adjoint;
}
Operator Tensor::getOp(){
    return op;
}
void Tensor::clearTree(){
    vector<Tensor*> visited_nodes;
    stack<Tensor*> nodes;
    for(int i = 0; i < parents.size(); i++){
        nodes.push(parents[i]);
    }
    adjoint = new NDimArray();
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
            n->adjoint = new NDimArray();
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
    }
    for(int i = 0; i < visited_nodes.size(); i++){
        if(visited_nodes[i] != NULL){
            visited_nodes[i]->children.clear();
            if(!visited_nodes[i]->get_keep()){
                delete visited_nodes[i];
                visited_nodes[i] = NULL;
            }
        }
    }
}
void Tensor::update(double step_sz){
    for(int i = 0; i < adjoint->values_size; i+=tensor->values_size){
        #pragma omp parallel for
        for(int j = 0; j < tensor->values_size; j++){
            tensor->values[j] -= step_sz * adjoint->values[i + j];
        }
    }
}
void Tensor::backward(){
    
    unsigned var_count = 1;
    for(int i = 0; i < tensor->dimension_size; i++){
        var_count *= tensor->dimension[i];
    }
    
    unsigned long dims[2] = {var_count, var_count};
    adjoint = new NDimArray();
    adjoint->setidentity(dims, 2);

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

            dims[1] = n->getTensor()->values_size;
            //n->adjoint->setzero(dims, 2);
            if(n->children.size() > 0){
                NDimArray* jac = n->children[0]->derivative(n, n->children[0]->op);
                n->adjoint = NDimArray::dot(n->children[0]->adjoint, jac);
                delete jac;
            }
            for(int i = 1; i < n->children.size(); i++){
                NDimArray* jac = n->children[i]->derivative(n, n->children[i]->op);
                NDimArray* prod = NDimArray::dot(n->children[i]->adjoint, jac);
                n->adjoint->add(prod);
                delete prod;
                delete jac;
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
    }
    for(int i = 0; i < visited_nodes.size(); i++){
        if(visited_nodes[i] != NULL){
            visited_nodes[i]->children.clear();
            if(!visited_nodes[i]->get_keep()){
                delete visited_nodes[i];
                visited_nodes[i] = NULL;
            }
        }
    }
}

NDimArray* Tensor::derivative(Tensor* x, Operator op){
    if(op == ADD){
        return derivative_add(x);
    }
    else if(op == DOT){
        return derivative_dot1d1d(x);
    }
    else if(op == ND1D_DOT){
        return derivative_dotNd1d(x);
    }
    else if(op == NDMD_DOT){
        return derivative_dotNdMd(x);
    }
    else if (op == SCALARMULT){
        return derivative_scalarmult(x);
    }
    else if(op == MULT){
        return derivative_mult(x);
    }
    else if(op == TRANS){
        return derivative_transpose(x);
    }
    else if(op == RELU){
        return derivative_relu(x);
    }
    else if(op == SIGMOID){
        return derivative_sigmoid(x);
    }
    return NULL;
}

NDimArray* Tensor::derivative_add(Tensor* x){
    unsigned long dims[2] = {getTensor()->values_size, x->getTensor()->values_size};
    NDimArray* jac = new NDimArray();
    jac->setidentity(dims, 2);
    return jac;
}

NDimArray* Tensor::derivative_dot1d1d(Tensor* x){
    unsigned long dims[2] = {1, x->getTensor()->values_size};
    Tensor* other = parents[0];
    if(parents[0] == x){
        other = parents[1];
    }
    float vals[other->getTensor()->values_size];
    for(int i = 0; i < other->getTensor()->values_size; i++){
        vals[i] = other->getTensor()->values[i];
    }
    return new NDimArray(dims, vals, 2);
}

NDimArray* Tensor::derivative_dotNd1d(Tensor* x){
    Tensor* other = parents[0];
    if(parents[0] == x){
        other = parents[1];
    }
    if(x->getTensor()->dimension_size == 1){
        unsigned long dims[2] = {getTensor()->values_size, x->getTensor()->values_size};
        return new NDimArray(dims, other->getTensor()->values, 2);
    }
    else{
        unsigned long dims[2] = {getTensor()->values_size, x->getTensor()->values_size};
        NDimArray* jac = new NDimArray();
        jac->setzero(dims, 2);
        int incr = jac->dimension[1];
        //#pragma omp parallel for
        for(int i = 0; i < jac->values_size; i+= incr){
            for(int j = 0; j < other->getTensor()->values_size; j++){
                jac->values[i + ((i / jac->dimension[1]) * other->getTensor()->values_size) + j] = other->getTensor()->values[j];
            }
        }
        return jac;
    }
    return NULL;
}

NDimArray* Tensor::derivative_dotNdMd(Tensor* x){
    NDimArray* jac = new NDimArray();
    if(parents[0] == x){
        Tensor* other = parents[1];
        unsigned long dims[2] = {getTensor()->values_size, x->getTensor()->values_size};
        jac->setzero(dims, 2);
        for(int i = 0; i < jac->values_size; i+= jac->dimension[1]){
            for(int j = 0; j < x->getTensor()->dimension[x->getTensor()->dimension_size - 1]; j++){
                int jac_row = i / jac->dimension[1];
                jac->values[i + ((jac_row / other->getTensor()->dimension[other->getTensor()->dimension_size - 1]) * x->getTensor()->dimension[x->getTensor()->dimension_size - 1]) + j] = other->getTensor()->values[(other->getTensor()->dimension[other->getTensor()->dimension_size - 1] * j) + (jac_row % other->getTensor()->dimension[other->getTensor()->dimension_size - 1])];
            }
        }
    }
    else{
        Tensor* other = parents[0];
        unsigned long dims[2] = {getTensor()->values_size, x->getTensor()->values_size};
        jac->setzero(dims, 2);
        for(int i = 0; i < jac->values_size; i+= jac->dimension[1]){
            for(int j = 0; j < x->getTensor()->dimension[x->getTensor()->dimension_size - 1]; j++){
                int jac_row = i / jac->dimension[1];
                jac->values[(j * x->getTensor()->dimension[x->getTensor()->dimension_size - 1]) + (jac_row % x->getTensor()->dimension[x->getTensor()->dimension_size - 1]) + i] = other->getTensor()->values[((jac_row / x->getTensor()->dimension[x->getTensor()->dimension_size - 1]) * other->getTensor()->dimension[other->getTensor()->dimension_size - 1]) + j];
            }
        }
    }
    return jac;
}

NDimArray* Tensor::derivative_scalarmult(Tensor* x){
    Tensor* other = parents[0];
    if(parents[0] == x){
        other = parents[1];
    }
    unsigned long dims[2] = {1, 1};
    float vals[1] = {other->getTensor()->values[0]};
    return new NDimArray(dims, vals, 2);
}

NDimArray* Tensor::derivative_mult(Tensor* x){
    if(x->getTensor()->dimension_size == 0){
        unsigned long dims[2] = {getTensor()->values_size, 1};
        return new NDimArray(dims, parents[1]->getTensor()->values, 2);
    }
    else{
        unsigned long dims[2] = {getTensor()->values_size, x->getTensor()->values_size};
        NDimArray* jac = new NDimArray();
        jac->setzero(dims, 2);

        for(int i = 0; i < getTensor()->values_size; i++){
            jac->values[i + (i * getTensor()->values_size)] = parents[0]->getTensor()->values[0];
        }
        return jac;
    }
    return NULL;
}

NDimArray* Tensor::derivative_transpose(Tensor* x){
    unsigned long dims[2] = {getTensor()->values_size, x->getTensor()->values_size};
    NDimArray* jac = new NDimArray();
    jac->setzero(dims, 2);

    for(int i = 0; i < getTensor()->values_size; i++){
        int r = i / getTensor()->dimension[getTensor()->dimension_size - 1];
        int c = i % getTensor()->dimension[getTensor()->dimension_size - 1];
        jac->values[(i * x->getTensor()->values_size) + (c * x->getTensor()->dimension[x->getTensor()->dimension_size - 1]) + r] = 1.0;
    }
    return jac;
}

NDimArray* Tensor::derivative_relu(Tensor* x){
    assert(getTensor()->values_size == x->getTensor()->values_size);
    unsigned long dims[2] = {x->getTensor()->values_size, x->getTensor()->values_size};
    float vals[x->getTensor()->values_size*x->getTensor()->values_size];
    for(int i = 0; i < getTensor()->values_size*getTensor()->values_size; i+=getTensor()->values_size){
        for(int j = 0; j < getTensor()->values_size; j++){
            if(j == i / getTensor()->values_size && x->getTensor()->values[j] > 0){
                vals[i + j] = 1;
            }
            else{
                vals[i + j] = 0;
            }
        }
    }
    return new NDimArray(dims, vals, 2);
}

NDimArray* Tensor::derivative_sigmoid(Tensor* x){
    assert(getTensor()->values_size == x->getTensor()->values_size);
    unsigned long dims[2] = {x->getTensor()->values_size, x->getTensor()->values_size};
    float vals[x->getTensor()->values_size*x->getTensor()->values_size];
    for(int i = 0; i < getTensor()->values_size*getTensor()->values_size; i+=getTensor()->values_size){
        for(int j = 0; j < getTensor()->values_size; j++){
            if(j == i / getTensor()->values_size){
                vals[i + j] = getTensor()->values[j] * (1.0f - getTensor()->values[j]);
            }
            else{
                vals[i + j] = 0;
            }
        }
    }
    return new NDimArray(dims, vals, 2);
}

Tensor* Tensor::random(unsigned long* dim, unsigned long dim_sz){
    Tensor* t = new Tensor();
    t->tensor = NDimArray::random(dim, dim_sz);
    t->adjoint = NULL;
    t->keep = true;
    return t;
}

Tensor* Tensor::eye(unsigned long dim){
    Tensor* t = new Tensor();
    t->tensor = NDimArray::eye(dim);
    t->adjoint = NULL;
    t->keep = true;
    return t;
}

Tensor* Tensor::add(Tensor* x, Tensor* y){
    Tensor* t = new Tensor();
    delete x->adjoint;
    delete y->adjoint;
    x->adjoint = NULL;
    y->adjoint = NULL;
    t->tensor = NDimArray::add(x->getTensor(), y->getTensor());
    t->parents.push_back(x);
    t->parents.push_back(y);
    t->op = ADD;
    t->name = "add";
    x->children.push_back(t);
    y->children.push_back(t);
    return t;
}

Tensor* Tensor::dot(Tensor* x, Tensor* y){
    Tensor* t = new Tensor();
    delete x->adjoint;
    delete y->adjoint;
    x->adjoint = NULL;
    y->adjoint = NULL;
    t->name = "dot";

    //t->tensor = NDimArray::dot(x->getTensor(), y->getTensor());
    t->parents.push_back(x);
    t->parents.push_back(y);
    if(x->getTensor()->dimension_size == 1 and y->getTensor()->dimension_size == 1){
        t->op = DOT;
        t->tensor = NDimArray::dot1d1d(x->getTensor(), y->getTensor());
    }
    else if(y->getTensor()->dimension_size == 1){
        t->op = ND1D_DOT;
        t->tensor = NDimArray::dotNd1d(x->getTensor(), y->getTensor());
    }
    else if(x->getTensor()->dimension_size == 0 and y->getTensor()->dimension_size == 0){
        t->op = SCALARMULT;
        t->tensor = NDimArray::dotScalar(x->getTensor(), y->getTensor());
    }
    else{
        t->op = NDMD_DOT;
        t->tensor = NDimArray::dotNdMd(x->getTensor(), y->getTensor());
    }
    x->children.push_back(t);
    y->children.push_back(t);
    return t;
}

Tensor* Tensor::mult(Tensor* x, Tensor* y){
    Tensor* t = new Tensor();
    delete x->adjoint;
    delete y->adjoint;
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
    delete x->adjoint;
    x->adjoint = NULL;
    t->tensor = NDimArray::transpose(x->getTensor());
    t->parents.push_back(x);
    t->op = TRANS;
    x->children.push_back(t);
    return t;
}

Tensor* Tensor::ReLU(Tensor* x){
    Tensor* t = new Tensor();
    delete x->adjoint;
    x->adjoint = NULL;
    t->tensor = NDimArray::ReLU(x->getTensor());
    t->parents.push_back(x);
    t->op = RELU;
    t->name = "RELU";
    x->children.push_back(t);
    return t;
}
Tensor* Tensor::Sigmoid(Tensor* x){
    Tensor* t = new Tensor();
    delete x->adjoint;
    x->adjoint = NULL;
    t->tensor = NDimArray::Sigmoid(x->getTensor());
    t->parents.push_back(x);
    t->op = SIGMOID;
    x->children.push_back(t);
    return t;
}

void Tensor::print(){
    cout << "tensor(";
    tensor->print();
    cout << ")" << endl;
}

bool Tensor::get_keep(){return keep;}
void Tensor::set_keep(bool k){keep = k;}

NDimArray::NDimArray(unsigned long* dim, float* vals, unsigned long dim_sz){
    dimension = new unsigned long[dim_sz];
    unsigned long vals_sz = 1;
    for(int i = 0; i < dim_sz; i++){
        dimension[i] = dim[i];
        vals_sz *= dim[i];
    }
    values = new float[vals_sz];
    for(int i = 0; i < vals_sz; i++){
        values[i] = vals[i];
    }
    dimension_size = dim_sz;
    values_size = vals_sz;
}

NDimArray::NDimArray(float val){
    values = new float[1];
    values[0] = val;
    values_size = 1;
}

NDimArray::~NDimArray(){
    delete[] values;
    delete[] dimension;
}

NDimArray* NDimArray::random(unsigned long* dim, unsigned long dim_sz){
    NDimArray* n = new NDimArray();
    n->dimension = new unsigned long[dim_sz];
    unsigned long vals_sz = 1;
    for(int i = 0; i < dim_sz; i++){
        n->dimension[i] = dim[i];
        vals_sz *= dim[i];
    }
    n->values = new float[vals_sz];
    for(int i = 0; i < vals_sz; i++){
        n->values[i] = rand() / float(RAND_MAX);
    }
    n->dimension_size = dim_sz;
    n->values_size = vals_sz;
    return n;
}
NDimArray* NDimArray::eye(unsigned long dim){
    unsigned long dims[2] = {dim, dim};
    NDimArray* n = new NDimArray();
    n->setidentity(dims, 2);
    return n;
}

NDimArray* NDimArray::add(NDimArray* x, NDimArray* y){
    assert(x->dimension_size == y->dimension_size);

    for(int i = 0; i < x->dimension_size; i++){
        assert(x->dimension[i] == y->dimension[i]);
    }
    assert(x->values_size == y->values_size);
    float vals[x->values_size];
    for(int i = 0; i < x->values_size; i++){
        vals[i] = x->values[i] + y->values[i];
    }
    return new NDimArray(x->dimension, vals, x->dimension_size);
}

void NDimArray::add(NDimArray* x){
    assert(x->dimension_size == dimension_size);
    
    for(int i = 0; i < x->dimension_size; i++){
        assert(x->dimension[i] == dimension[i]);
    }
    assert(x->values_size == values_size);

    for(int i = 0; i < values_size; i++){
        values[i] += x->values[i];
    }
}

NDimArray* NDimArray::dot(NDimArray* x, NDimArray* y){
    if(x->dimension_size == 1 && y->dimension_size == 1){
        return dot1d1d(x, y);
    }
    else if(y->dimension_size == 1){
        return dotNd1d(x, y);
    }
    else if(x->dimension_size == 0 && y->dimension_size == 0){
        return dotScalar(x, y);
    }
    else{
        return dotNdMd(x, y);
    }
}

NDimArray* NDimArray::dot1d1d(NDimArray* x, NDimArray* y){
    assert(x->dimension[0] == y->dimension[0]);
    float val = 0;
    for(int i = 0; i < x->values_size; i++){
        val += x->values[i] * y->values[i];
    }
    return new NDimArray(val);
}
NDimArray* NDimArray::dotNd1d(NDimArray* x, NDimArray* y){
    assert(y->dimension[0] == x->dimension[x->dimension_size - 1]);
    unsigned long dims[x->dimension_size - 1];
    float vals[x->values_size / y->dimension[0]];
    for(int i = 0; i < x->values_size / y->dimension[0]; i++){
        vals[i] = 0.0f;
    }
    for(int i = 0; i < x->values_size; i+=y->dimension[0]){
        __m256 sum_vec = _mm256_setzero_ps();
        for(int j = 0; j < (y->dimension[0] / 8) * 8; j+=8){
            __m256 a_vec = _mm256_loadu_ps(x->values + i + j);
            __m256 b_vec = _mm256_loadu_ps(y->values + j);
            sum_vec = _mm256_add_ps(_mm256_mul_ps(a_vec, b_vec), sum_vec);
            //sum += (x->values[i + j] * y->values[j]);
        }
        float store_prods[8];
        _mm256_storeu_ps(store_prods, sum_vec);
        float sum = store_prods[0] + store_prods[1] + store_prods[2] + store_prods[3]+ store_prods[4]+ store_prods[5]+ store_prods[6]+ store_prods[7];
        for(int j = (y->dimension[0] / 8) * 8; j < y->dimension[0]; j++){
            sum += (x->values[i + j] * y->values[j]);
        }
        //vals.push_back(sum);
        vals[i / y->dimension[0]] = sum;
    }
    for(int i = 0; i < x->dimension_size - 1; i++){
        dims[i] = x->dimension[i];
    }
    return new NDimArray(dims, vals, x->dimension_size - 1);
}
NDimArray* NDimArray::dotNdMd(NDimArray* x, NDimArray* y){
    assert(x->dimension[x->dimension_size - 1] == y->dimension[y->dimension_size - 2]);
    unsigned long dims[x->dimension_size];
    float vals[(x->values_size / x->dimension[x->dimension_size - 1]) * y->dimension[y->dimension_size - 1]];
    for(int i = 0; i < x->values_size; i+=x->dimension[x->dimension_size - 1]){
        for(int j = 0; j < y->dimension[y->dimension_size - 1]; j++){
            double sum = 0.0;
            for(int k = 0; k < x->dimension[x->dimension_size - 1]; k++){
                sum += x->values[i + k] * y->values[j + (k * y->dimension[y->dimension_size - 1])];
            }
            vals[((i / x->dimension[x->dimension_size - 1]) * y->dimension[y->dimension_size - 1]) + j] = sum;
        }
    }
    for(int i = 0; i < x->dimension_size - 1; i++){
        dims[i] = x->dimension[i];
    }
    dims[x->dimension_size - 1] = y->dimension[y->dimension_size - 1];
    return new NDimArray(dims, vals, x->dimension_size);
}
NDimArray* NDimArray::dotScalar(NDimArray* x, NDimArray* y){
    return new NDimArray(x->values[0] * y->values[0]);
}

NDimArray* NDimArray::mult(NDimArray* x, NDimArray* y){
    assert(x->dimension_size == 0);
    float vals[y->values_size];
    for(int i = 0; i < y->values_size; i++){
        vals[i] = y->values[i] * x->values[0];
    }
    return new NDimArray(y->dimension, vals, y->dimension_size);
}

NDimArray* NDimArray::transpose(NDimArray* x){
    assert(x->dimension_size >= 2);
    unsigned long dims[x->dimension_size];
    for(int i = 0; i < x->dimension_size - 2; i++){
        dims[i] = x->dimension[i];
    }
    dims[x->dimension_size - 2] =  x->dimension[x->dimension_size - 1];
    dims[x->dimension_size - 1] = x->dimension[x->dimension_size - 2];
    float vals[x->dimension[x->dimension_size - 1] * (x->values_size / x->dimension[x->dimension_size - 1])];
    for(int i = 0; i < x->dimension[x->dimension_size - 1]; i++){
        for(int j = 0; j < x->values_size; j+=x->dimension[x->dimension_size - 1]){
            vals[(i * (x->values_size / x->dimension[x->dimension_size - 1])) + (j / x->dimension[x->dimension_size - 1])] = x->values[i + j];
        }
    }
    return new NDimArray(dims, vals, x->dimension_size);
}

NDimArray* NDimArray::ReLU(NDimArray* x){
    float vals[x->values_size];
    for(int i = 0; i < x->values_size; i++){
        if(x->values[i] < 0){
            vals[i] = 0;
        }
        else{
            vals[i] = x->values[i];
        }
    }
    return new NDimArray(x->dimension, vals, x->dimension_size);
}
NDimArray* NDimArray::Sigmoid(NDimArray* x){
    float vals[x->values_size];
    for(int i = 0; i < x->values_size; i++){
        vals[i] = 1.0/(1.0 + exp(-x->values[i]));
    }
    return new NDimArray(x->dimension, vals, x->dimension_size);
}

void NDimArray::setzero(unsigned long* dims, unsigned long dim_sz){
    unsigned long var_count = 1;
    dimension = new unsigned long[dim_sz];
    for(int i = 0; i < dim_sz; i++){
        dimension[i] = dims[i];
        var_count *= dims[i];
    }
    values = new float[var_count];
    if(dim_sz == 0){
        values[0] = 0.0f;
    }
    else{
        for(int i = 0; i < var_count; i++){
            values[i] = 0.0f;
        }
    }
    dimension_size = dim_sz;
    values_size = var_count;
}

void NDimArray::setidentity(unsigned long* dims, unsigned long dim_sz){
    dimension = new unsigned long[dim_sz];
    unsigned long var_count = 1;
    for(int i = 0; i < dim_sz; i++){
        dimension[i] = dims[i];
        var_count *= dims[i];
    }
    values = new float[var_count];
    if(dim_sz == 0){
        values[0] = 1.0f;
    }
    else if(dim_sz == 1){
        for(int i = 0; i < dimension[0]; i++){
            values[i] = 1.0f;
        }
    }
    else{
        assert(var_count / dimension[dim_sz - 1] == dimension[dim_sz - 1]);
        for(int i = 0; i < var_count; i++){
            if(i / dimension[dim_sz - 1] == i % dimension[dim_sz - 1]){
                values[i] = 1.0f;
            }
            else{
                values[i] = 0.0f;
            }
        }
    }
    dimension_size = dim_sz;
    values_size = var_count;
}

double NDimArray::get(unsigned long* index){
    unsigned long shift = 0;
    unsigned long mult = 1;
    for(int i = dimension_size - 1; i >= 0; i--){
        shift += mult * index[i];
        mult *= dimension[i];
    }
    return values[shift];
}

void NDimArray::print(){
    print_helper(0, 0, values_size);
}

void NDimArray::print_helper(unsigned long level, unsigned long index, unsigned long sz){
    //cout << "func " << level << ", " << index << endl;
    if(dimension_size == 0){
        cout << "[" << values[0] << "]";
    }
    else if(level == dimension_size - 1){
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

