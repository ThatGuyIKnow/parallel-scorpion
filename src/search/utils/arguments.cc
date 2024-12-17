#include "arguments.h"

#include <string.h>

using namespace std;

namespace utils {
    char** copy_argv(int argc, const char** argv) {
        // Allocate memory for the array of pointers, +1 for a nullptr terminator if needed
        char** new_argv = new char*[argc + 1];
        
        for(int i = 0; i < argc; ++i){
            // Allocate memory for each string (+1 for the null terminator)
            new_argv[i] = new char[strlen(argv[i]) + 1];
            
            // Copy the string content
            strcpy(new_argv[i], argv[i]);
        }
        
        // Optional: Null-terminate the array
        new_argv[argc] = nullptr;
        
        return new_argv;
    }

    void free_argv(int argc, char** argv){
        for(int i = 0; i < argc; ++i){
            delete[] argv[i];  // Free each string
        }
        delete[] argv;  // Free the array of pointers
    }
}