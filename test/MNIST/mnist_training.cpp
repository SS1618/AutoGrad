#include "iostream"
#include "tensor.h"
#include "fstream"

using namespace std;
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
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
    int** pixel_data = (int**)malloc(number_of_images*sizeof(int*));
    int* labels = (int*)malloc(number_of_images*sizeof(int));
    if(fin.is_open()){
        for(int i=0;i<number_of_images;++i)
        {
            pixel_data[i] = (int*)malloc(n_rows*n_cols*sizeof(int));
            for(int j = 0; j < n_rows * n_cols; j++){
                unsigned char temp=0;
                fin.read((char*)&temp,sizeof(temp));
                pixel_data[i][j] = (int)temp;
            }
        }
    }
    fin.close();
    fin.open("test/mnist/train-labels.idx1-ubyte", std::ios::binary);
    if(fin.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        fin.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        fin.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        cout << number_of_images << endl;
        for(int i = 0; i < number_of_images; i++){
            unsigned char temp = 0;
            fin.read((char*)&temp, sizeof(temp));
            labels[i] = (int) temp;
        }
    }
    fin.close();
    //cout << labels[0] << endl;

    for(int i = 0; i < number_of_images; i++){
        free(pixel_data[i]);
    }
    free(pixel_data);
    free(labels);
}