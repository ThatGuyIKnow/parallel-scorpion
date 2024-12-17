#include <string.h>

namespace utils {
    char** copy_argv(int argc, const char** argv);
    void free_argv(int argc, char** argv);
}