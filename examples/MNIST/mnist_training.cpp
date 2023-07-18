#include "iostream"
#include "tensor.h"
#include "fstream"

using namespace std;

int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

class Model{
private:
    Linear* layer1;
    Linear* layer2;
public:
    Model(int input_sz){
        layer1 = new Linear(input_sz, 128);
        layer2 = new Linear(128, 10);
    }
    Tensor* feedforward(Tensor* input){
        return Tensor::Sigmoid(layer2->feedforward(Tensor::ReLU(layer1->feedforward(input))));
    }
    Tensor* MSE(Tensor* expected, Tensor* output){
        Tensor* neg = new Tensor(-1);
        neg->set_keep(false);
        Tensor* neg_output = Tensor::mult(neg, output);
        Tensor* z = Tensor::add(expected, neg_output);
        return Tensor::dot(z, z);
    }
    void update(float step_sz){
        layer2->update(step_sz);
        layer1->update(step_sz);
    }
    ~Model(){
        delete layer1;
        delete layer2;
    }
};
int main(){
    ifstream fin("test/mnist/train-images.idx3-ubyte", std::ios::binary);

    int magic_number=0;
    int number_of_images=0;
    int n_rows=0;
    int n_cols=0;
    if(fin.is_open()){
        fin.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        fin.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        fin.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        fin.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
    }
    //int** pixel_data = (int**)malloc(number_of_images*sizeof(int*));
    Tensor* pixel_data[number_of_images];
    unsigned long img_dim[1] = {28*28};
    
    if(fin.is_open()){
        for(int i=0;i<number_of_images;++i)
        {
            //pixel_data[i] = (int*)malloc(n_rows*n_cols*sizeof(int));
            float data[n_rows * n_cols];
            for(int j = 0; j < n_rows * n_cols; j++){
                unsigned char temp=0;
                fin.read((char*)&temp,sizeof(temp));
                //pixel_data[i][j] = (int)temp;
                data[j] = ((int) temp) / 255.0f;
            }
            pixel_data[i] = new Tensor(img_dim, data, 1);
        }
    }
    fin.close();
    fin.open("test/mnist/train-labels.idx1-ubyte", std::ios::binary);

    Tensor* labels_tens[10];
    int* labels = (int*)malloc(number_of_images*sizeof(int));
    unsigned long label_dim[1] = {10};
    float label_vals[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for(int i = 0; i < 10; i++){
        label_vals[i] = 1.0f;
        labels_tens[i] = new Tensor(label_dim, label_vals, 1);
        label_vals[i] = 0;
    }
    if(fin.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        fin.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        fin.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        for(int i = 0; i < number_of_images; i++){
            unsigned char temp = 0;
            fin.read((char*)&temp, sizeof(temp));
            labels[i] = (int) temp;
        }
    }
    fin.close();

    Model* model = new Model(28*28);
    float avg_error = 0;
    for(int i = 10; i < 10000; i++){
        Tensor* output = model->feedforward(pixel_data[i]);
        Tensor* error = model->MSE(labels_tens[labels[i]], output);
        error->backward();
        model->update(0.001);
        avg_error += error->getTensor()->values[0];
        delete error;
        if(i > 0 && i % 100 == 0){
            cout << "Train Error: " << avg_error / 100.0 << endl;
            avg_error = 0;
            for(int j = 0; j < 10; j++){
                output = model->feedforward(pixel_data[j]);
                error = model->MSE(labels_tens[labels[j]], output);
                //error->backward();
                avg_error += error->getTensor()->values[0];
                error->clearTree();
                delete error;
            }
            cout << "Test Error: " << avg_error / 10.0 << endl;
            avg_error = 0;            
        }
    }

    delete model;
    
    for(int i = 0; i < number_of_images; i++){
        delete pixel_data[i];
    }
    for(int i = 0; i < 10; i++){
        delete labels_tens[i];
    }
    //free(pixel_data);
    free(labels);
}