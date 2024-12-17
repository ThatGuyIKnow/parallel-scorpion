// mpi_shared_istream.cc
#include "mpi_share_stream.h"
#include <iostream>
#include <sstream>
#include <vector>

namespace utils {

/**
 * @brief Initializes MPI if it has not been initialized yet.
 */
bool initialize_mpi(int& argc, char**& argv, bool& was_initialized) {
    int flag = 0;
    MPI_Initialized(&flag);
    was_initialized = false;
    if (!flag) {
        int ret = MPI_Init(&argc, &argv);
        if (ret != MPI_SUCCESS) {
            std::cerr << "Error initializing MPI." << std::endl;
            return false;
        }
        was_initialized = true;
    }
    return true;
}

/**
 * @brief Finalizes MPI if it was initialized by initialize_mpi.
 */
void finalize_mpi(bool was_initialized) {
    if (was_initialized) {
        MPI_Finalize();
    }
}

/**
 * @brief Broadcasts a string from the root process to all other MPI processes.
 */
std::string mpi_broadcast_string(const std::string& data, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Broadcast the length of the string first
    int length = 0;
    if (rank == root) {
        length = static_cast<int>(data.size());
    }
    MPI_Bcast(&length, 1, MPI_INT, root, comm);

    // Allocate buffer for the string
    std::string buffer;
    if (rank != root) {
        buffer.resize(length);
    }

    // Broadcast the string data
    MPI_Bcast(rank == root ? const_cast<char*>(data.data()) : &buffer[0], length, MPI_CHAR, root, comm);

    // Return the received string
    return (rank == root) ? data : buffer;
}

/**
 * @brief Reads input on the root process, broadcasts it to all processes, and converts it into an std::istringstream.
 */
std::istringstream broadcast_input_to_all(std::istream& input, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::string shared_data;

    if (rank == root) {
        // Read the entire input into a string
        std::ostringstream oss;
        oss << input.rdbuf();
        shared_data = oss.str();
    }

    // Broadcast the string to all processes
    shared_data = mpi_broadcast_string(shared_data, root, comm);

    // Convert the shared string into an std::istringstream
    std::istringstream iss(shared_data);

    return iss;
}

// /**
//  * @brief Example main function demonstrating the usage of mpi_shared_istream functions.
//  *
//  * To compile:
//  *     mpic++ -o mpi_shared_istream mpi_shared_istream.cc
//  *
//  * To run with 4 processes:
//  *     mpirun -np 4 ./mpi_shared_istream < input.txt
//  */
// int main(int argc, char** argv) {
//     bool was_initialized = false;

//     // Initialize MPI
//     if (!initialize_mpi(argc, argv, was_initialized)) {
//         return EXIT_FAILURE;
//     }

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // Broadcast input from rank 0 to all processes
//     std::istringstream shared_istream = broadcast_input_to_all(std::cin, 0, MPI_COMM_WORLD);

//     // Example processing: Each process reads from the shared_istream and prepares output
//     std::ostringstream local_output;
//     std::string line;
//     while (std::getline(shared_istream, line)) {
//         local_output << "Rank " << rank << " read line: " << line << "\n";
//     }

//     // Gather all outputs to rank 0 for organized printing
//     std::string output = local_output.str();
//     int output_length = static_cast<int>(output.size());

//     // Gather lengths of each process's output
//     std::vector<int> recv_lengths(size, 0);
//     MPI_Gather(&output_length, 1, MPI_INT, recv_lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Prepare for gathering all outputs
//     std::vector<char> recv_buffer;
//     std::vector<int> displs(size, 0);
//     if (rank == 0) {
//         int total_length = 0;
//         for (int i = 0; i < size; ++i) {
//             displs[i] = total_length;
//             total_length += recv_lengths[i];
//         }
//         recv_buffer.resize(total_length);
//     }

//     // Gather all outputs
//     MPI_Gatherv(output.c_str(), output_length, MPI_CHAR,
//                 rank == 0 ? recv_buffer.data() : nullptr,
//                 rank == 0 ? recv_lengths.data() : nullptr,
//                 rank == 0 ? displs.data() : nullptr,
//                 MPI_CHAR, 0, MPI_COMM_WORLD);

//     // Rank 0 prints all outputs in order
//     if (rank == 0) {
//         std::cout << "=== All Processes Outputs ===\n";
//         for (int i = 0; i < size; ++i) {
//             if (recv_lengths[i] > 0) {
//                 std::string proc_output(recv_buffer.data() + displs[i], recv_lengths[i]);
//                 std::cout << proc_output;
//             }
//         }
//     }

//     // Finalize MPI if it was initialized by this function
//     finalize_mpi(was_initialized);

//     return EXIT_SUCCESS;
// }
}