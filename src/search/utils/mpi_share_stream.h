// mpi_shared_istream.h
#ifndef MPI_SHARED_ISTREAM_H
#define MPI_SHARED_ISTREAM_H

#include <string>
#include <sstream>

#include <mpi.h>

namespace utils { 

    /**
     * @brief Initializes MPI if it has not been initialized yet.
     *
     * @param argc The argument count from the command line.
     * @param argv The argument vector from the command line.
     * @param was_initialized A boolean reference that will be set to true if MPI was initialized by this function.
     * @return true if MPI initialization was successful or already initialized.
     * @return false if MPI initialization failed.
     */
    bool initialize_mpi(int& argc, char**& argv, bool& was_initialized);

    /**
     * @brief Finalizes MPI if it was initialized by initialize_mpi.
     *
     * @param was_initialized A boolean indicating whether MPI was initialized by initialize_mpi.
     */
    void finalize_mpi(bool was_initialized);

    /**
     * @brief Broadcasts a string from the root process to all other MPI processes.
     *
     * @param data The string to broadcast (only used by the root process).
     * @param root The rank of the root process (default is 0).
     * @param comm The MPI communicator (default is MPI_COMM_WORLD).
     * @return The broadcasted string on all processes.
     */
    std::string mpi_broadcast_string(const std::string& data, int root = 0, MPI_Comm comm = MPI_COMM_WORLD);

    /**
     * @brief Reads input on the root process, broadcasts it to all processes, and converts it into an std::istringstream.
     *
     * @param input The input stream to read from (only used by the root process).
     * @param root The rank of the root process (default is 0).
     * @param comm The MPI communicator (default is MPI_COMM_WORLD).
     * @return std::istringstream An input string stream containing the shared data.
     */
    std::istringstream broadcast_input_to_all(std::istream& input, int root = 0, MPI_Comm comm = MPI_COMM_WORLD);

}

#endif // MPI_SHARED_ISTREAM_H