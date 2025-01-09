#include "logging.h"

#include "system.h"
#include "timer.h"

#include "../plugins/plugin.h"

#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

static unsigned int MPI_LOG_MESSAGE = UINT16_MAX;

namespace utils {
/*
  NOTE: When adding more options to Log, make sure to adapt the if block in
  get_log_from_options below to test for *all* default values used for
  global_log here. Also add the options to dump_options().
*/

static shared_ptr<Log> global_log = make_shared<Log>(Verbosity::NORMAL);

LogProxy g_log(global_log);

void add_log_options_to_feature(plugins::Feature &feature) {
    feature.add_option<Verbosity>(
        "verbosity",
        "Option to specify the verbosity level.",
        "normal");
}

tuple<Verbosity> get_log_arguments_from_options(
    const plugins::Options &opts) {
    return make_tuple<Verbosity>(opts.get<Verbosity>("verbosity"));
}

LogProxy get_log_for_verbosity(const Verbosity &verbosity) {
    if (verbosity == Verbosity::NORMAL) {
        return LogProxy(global_log);
    }
    return LogProxy(make_shared<Log>(verbosity));
}

LogProxy get_silent_log() {
    return utils::get_log_for_verbosity(utils::Verbosity::SILENT);
}

ContextError::ContextError(const string &msg)
    : Exception(msg) {
}

const string Context::INDENT = "  ";

Context::Context(const Context &context)
    : initial_stack_size(context.block_stack.size()),
      block_stack(context.block_stack) {
}

Context::~Context() {
    if (block_stack.size() > initial_stack_size) {
        cerr << str() << endl;
        ABORT("A context was destructed with an non-empty stack.");
    }
}

string Context::decorate_block_name(const string &block_name) const {
    return block_name;
}

void Context::enter_block(const string &block_name) {
    block_stack.push_back(block_name);
}

void Context::leave_block(const string &block_name) {
    if (block_stack.empty() || block_stack.back() != block_name) {
        cerr << str() << endl;
        ABORT("Tried to pop a block '" + block_name +
              "' from an empty stack or the block to remove "
              "is not on the top of the stack.");
    }
    block_stack.pop_back();
}

string Context::str() const {
    ostringstream message;
    message << "Traceback:" << endl;
    if (block_stack.empty()) {
        message << INDENT << "Empty";
    } else {
        message << INDENT
                << utils::join(block_stack, "\n" + INDENT + "-> ");
    }
    return message.str();
}

void Context::error(const string &message) const {
    throw ContextError(str() + "\n\n" + message);
}

void Context::warn(const string &message) const {
    utils::g_log << str() << endl << endl << message;
}

TraceBlock::TraceBlock(Context &context, const string &block_name)
    : context(context),
      block_name(context.decorate_block_name(block_name)) {
    context.enter_block(this->block_name);
}

TraceBlock::~TraceBlock() {
    context.leave_block(block_name);
}

MemoryContext _memory_context;

string MemoryContext::decorate_block_name(const string &msg) const {
    ostringstream decorated_msg;
    decorated_msg << "[TRACE] "
                  << setw(TIME_FIELD_WIDTH) << g_timer << " "
                  << setw(MEM_FIELD_WIDTH) << get_peak_memory_in_kb() << " KB";
    for (size_t i = 0; i < block_stack.size(); ++i)
        decorated_msg << INDENT;
    decorated_msg << ' ' << msg << endl;
    return decorated_msg.str();
}

void trace_memory(const string &msg) {
    g_log << _memory_context.decorate_block_name(msg);
}

static plugins::TypedEnumPlugin<Verbosity> _enum_plugin({
        {"silent", "only the most basic output"},
        {"normal", "relevant information to monitor progress"},
        {"verbose", "full output"},
        {"debug", "like verbose with additional debug output"}
    });

void Log::add_prefix(std::ostream &os) const {
    os << "[t=";
    streamsize previous_precision = cout.precision(TIMER_PRECISION);
    ios_base::fmtflags previous_flags = os.flags();
    os.setf(ios_base::fixed, ios_base::floatfield);
    os << g_timer;
    os.flags(previous_flags);
    cout.precision(previous_precision);
    os << ", "
           << get_peak_memory_in_kb() << " KB] ";
}

void Log::send_synchronized_message() {
        std::string message = sync_stream.str();

        // Allocate a buffer for the message to use with MPI_Bsend
        std::vector<char> buffer(MPI_BSEND_OVERHEAD + message.size());
        MPI_Buffer_attach(buffer.data(), buffer.size());

        // Perform the buffered send
        MPI_Bsend(message.data(), message.size(), MPI_CHAR, 0, 
                    MPI_LOG_MESSAGE, MPI_COMM_WORLD);

        // Detach the buffer after sending
        void* detachBuffer;
        int detachSize;
        MPI_Buffer_detach(&detachBuffer, &detachSize);

        sync_stream.str(""); // Clear the stream
        sync_stream.clear();
    }

void Log::empty() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank != 0) {
            return; // Only rank 0 should process remaining messages
        }

        int flag = 0;
        MPI_Status status;

        while (true) {
            // Check if there are any incoming messages
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_LOG_MESSAGE, MPI_COMM_WORLD, &flag, &status);
            if (!flag) {
                break; // Exit the loop when no more messages are pending
            }

            // Determine the size of the incoming message
            int messageSize;
            MPI_Get_count(&status, MPI_CHAR, &messageSize);

            // Receive the message
            std::vector<char> recvBuffer(messageSize);
            MPI_Recv(recvBuffer.data(), messageSize, MPI_CHAR, status.MPI_SOURCE, 
                     MPI_LOG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Print the received message
            std::string message(recvBuffer.begin(), recvBuffer.end());
            stream << "[Rank " << status.MPI_SOURCE << "] " << message << std::endl;
        }
    }
}
