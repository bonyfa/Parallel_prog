#include <iostream>
#include <cstdlib>
//#include <string> 
#include <fcntl.h>   // для open()
#include <unistd.h>  // для close(), read(), write()
#include <vector>

using namespace std;

#include <fstream>
#include <iomanip>

bool make_matrix(size_t size, const char* path) {
    ofstream file(path);
    if (!file.is_open()) return false;

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            file << rand() % 100 << " ";
        }
        file << "\n";
    }
    
    return true;
}

int main(){
    srand(1000);

    make_matrix(100, "./matrix_a.txt");
    make_matrix(100, "./matrix_b.txt");
    
    return 0;
}
