#include "parallel_eager_search.h"

#include "../evaluation_context.h"
#include "../evaluator.h"
#include "../open_list_factory.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../plugins/options.h"
#include "../task_utils/successor_generator.h"
#include "../utils/logging.h"
#include "../state_id.h"

#include "../task_utils/task_properties.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>

using namespace std;

void mpi_arg_min_func(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype)
{   
    if (*len != 2) {
        throw std::invalid_argument("Buffer length must be 2 in MPI_ARG_MIN");
    }
    if (*datatype != MPI_INT) {
        throw std::invalid_argument("MPI_ARG_MIN only supports MPI_INT datatype");
    }
    int* input = (int*)inputBuffer;
    int* output = (int*)outputBuffer;
 
    if (input[1] < output[1]) {
        output[0] = input[0];
        output[1] = input[1];
    }
}
namespace parallel_eager_search {
ParallelEagerSearch::ParallelEagerSearch(
    const shared_ptr<OpenListFactory> &open, bool reopen_closed,
    const shared_ptr<Evaluator> &f_eval,
    const vector<shared_ptr<Evaluator>> &preferred,
    const std::shared_ptr<distribution_hash::MapBasedHash> &hash,
    const shared_ptr<PruningMethod> &pruning,
    const shared_ptr<Evaluator> &lazy_evaluator, OperatorCost cost_type,
    int bound, double max_time, const string &description,
    utils::Verbosity verbosity
) : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      state_byte_size(state_registry.get_state_size_in_bytes()),
      node_byte_size(state_byte_size + sizeof(int) * 7),
      open_list(open->create_state_open_list()),
      f_evaluator(f_eval),
      preferred_operator_evaluators(preferred),
      lazy_evaluator(lazy_evaluator),
      pruning_method(pruning), 
      distribution_hash(hash) {
    if (lazy_evaluator && !lazy_evaluator->does_cache_estimates()) {
        cerr << "lazy_evaluator must cache its estimates" << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void ParallelEagerSearch::initialize() {
    log << "Conducting best first search"
        << (reopen_closed_nodes ? " with" : " without")
        << " reopening closed nodes, (real) bound = " << bound
        << endl;
    assert(open_list);

    set<Evaluator *> evals;
    open_list->get_path_dependent_evaluators(evals);
    /*
      Collect path-dependent evaluators that are used for preferred operators
      (in case they are not also used in the open list).
    */
    for (const shared_ptr<Evaluator> &evaluator : preferred_operator_evaluators) {
        evaluator->get_path_dependent_evaluators(evals);
    }

    /*
      Collect path-dependent evaluators that are used in the f_evaluator.
      They are usually also used in the open list and will hence already be
      included, but we want to be sure.
    */
    if (f_evaluator) {
        f_evaluator->get_path_dependent_evaluators(evals);
    }

    /*
      Collect path-dependent evaluators that are used in the lazy_evaluator
      (in case they are not already included).
    */
    if (lazy_evaluator) {
        lazy_evaluator->get_path_dependent_evaluators(evals);
    }

    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    pruning_method->initialize(task);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    processor_info = {(unsigned int) world_size, (unsigned int) world_rank};
	printf("%d/%d processes\n", world_rank, world_size);

    // Calculate the size of the node MPI package. It's the size of the state + the size of 
    // six additional ints (g, h, operator, distributed_hash, parent_state_id)
	// TODO: not sure we need this or Buffer_attach will do that for us.
	unsigned int buffer_size = (node_byte_size + MPI_BSEND_OVERHEAD) * world_size * 100;


	unsigned int buffer_max = 400000000; // 400 MB TODO: not sure this is enough or too much.

	if (buffer_size > buffer_max) {
		buffer_size = 400000000;
	}

    // All processes starts as idle
    for(unsigned int i = 0; i < processor_info.world_size; i++){
        status_processors.emplace_back(ProcessesStatus::IDLE);
    }

	mpi_buffer = new unsigned char[buffer_size];
	fill(mpi_buffer, mpi_buffer + buffer_size, 0);
	MPI_Buffer_attach((void *) mpi_buffer, buffer_size);

    unsigned int d_hash = distribution_hash->hash(initial_state);
    state_hash_map[initial_state.get_id().value] = d_hash;

    if (processor_info.rank == (d_hash % processor_info.world_size)) {

    /*
      Note: we consider the initial state as reached by a preferred
      operator.
        */
        EvaluationContext eval_context(initial_state, 0, true, &statistics);

        statistics.inc_evaluated_states();

        if (open_list->is_dead_end(eval_context)) {
            log << "Initial state is a dead end." << endl;
        } else {
            if (search_progress.check_progress(eval_context))
                statistics.print_checkpoint_line(0);
            start_f_value_statistics(eval_context);
            SearchNode node = search_space.get_node(initial_state);
            node.open_initial();

            open_list->insert(eval_context, initial_state.get_id());
            processor_info.status = ProcessesStatus::ACTIVE;
            status_processors[processor_info.rank] = ProcessesStatus::ACTIVE;
        }

        print_initial_evaluator_values(eval_context);
    }

    MPI_Op_create(&mpi_arg_min_func, 0, &MPI_ARG_MIN);
    MPI_Barrier(MPI_COMM_WORLD);
    log << "Synced! Processes: " << processor_info.rank << endl;
}

void ParallelEagerSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

unsigned int ParallelEagerSearch::get_assigned_rank(State state) {
    // Extract the buffer value
    unsigned int buffer_value = *state.get_buffer();

    // Apply a uniform hash function
    unsigned int hashed_value = buffer_value;
    hashed_value ^= hashed_value >> 17;
    hashed_value *= 0xed5ad4bb;
    hashed_value ^= hashed_value >> 11;
    hashed_value *= 0xac4c1b51;
    hashed_value ^= hashed_value >> 15;
    hashed_value *= 0x31848bab;
    hashed_value ^= hashed_value >> 14;

    // Compute the assigned rank
    unsigned int assigned_rank = hashed_value % processor_info.world_size;

    return assigned_rank;
}




bool ParallelEagerSearch::detect_if_plan_found() {
    if (plan_found) {
        return true;
    }

    int plan_found_by_other_process;
	MPI_Iprobe(MPI_ANY_SOURCE, MPIMessageType::FOUND_GOAL, MPI_COMM_WORLD, &plan_found_by_other_process,
			MPI_STATUS_IGNORE);
    return plan_found_by_other_process;
}

bool ParallelEagerSearch::check_and_progress_termination(
    ProcessesStatus status, 
    ProcessesStatus next_status, 
    std::vector<ProcessesStatus> statuses) {

    unsigned char message_buffer[2];
    bool all_minimum = std::all_of(statuses.begin(), statuses.end(), 
        [status](ProcessesStatus s) { return s >= status; }); 

    if(all_minimum && awaited_ack == 0) {
        processor_info.status = next_status;
        status_processors[processor_info.rank] = processor_info.status;
        to_byte(parallel::StatusMessage{processor_info.status, processor_info.rank}, message_buffer);

        for(unsigned int i = 1; i < processor_info.world_size; i++) {
            int rank = (processor_info.rank + i) % processor_info.world_size;
            MPI_Bsend(message_buffer, 2, MPI_BYTE, rank, MPIMessageType::STATUS, MPI_COMM_WORLD);
        }
        return true;
    }

    return false;
}

void ParallelEagerSearch::construct_plan(SearchNode goal_node) {
    log << "(Process " << processor_info.rank << ") Starting construct_plan." << endl;

    State current_state = goal_node.get_state();
    int external_state_id = current_state.get_id().value;
    std::vector<OperatorID> path;
    unsigned int assigned_rank = processor_info.rank;

    while (true) {
        log << "(Process " << processor_info.rank << ") Current state ID: " << external_state_id << ", Assigned rank: " << assigned_rank << endl;

        if (processor_info.rank != assigned_rank) {
            if (process_external_node(current_state, external_state_id, assigned_rank, path)) {
                log << "(Process " << processor_info.rank << ") External node processed, ending loop." << endl;
                break;
            }
        } else {
            if (process_internal_node(current_state, assigned_rank, external_state_id, path)) {
                log << "(Process " << processor_info.rank << ") Internal node processed, ending loop." << endl;
                break;
            }
        }
    }

    int term_signal[] = {1, 0};
    for (unsigned int i = 1; i < processor_info.world_size; i++) {
        int rank = (processor_info.rank + i) % processor_info.world_size;
        MPI_Bsend(term_signal, 2, MPI_INT, rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD);
        log << "(Process " << processor_info.rank << ") Sent termination signal to process " << rank << endl;
    }

    std::reverse(path.begin(), path.end());
    set_plan(path);
    log << "(Process " << processor_info.rank << ") Plan constructed and set." << endl;
}

// Sub-function to handle external nodes
bool ParallelEagerSearch::process_external_node(State &current_state, int &external_state_id, unsigned int &assigned_rank, std::vector<OperatorID> &path) {
    log << "(Process " << processor_info.rank << ") Processing external node with state ID: " << external_state_id << endl;

    // Send request for the parent node
    int buffer[2] = {0, external_state_id};
    MPI_Send(buffer, 2, MPI_INT, assigned_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD);
    log << "(Process " << processor_info.rank << ") Sent request to process " << assigned_rank << " for state ID: " << external_state_id << endl;

    // Probe for message size
    MPI_Status status;
    int package_size;
    MPI_Probe(assigned_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &package_size);

    // Receive the message
    std::vector<int> message(package_size);
    MPI_Recv(message.data(), package_size, MPI_INT, assigned_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    log << "(Process " << processor_info.rank << ") Received message from process " << assigned_rank << endl;

    // Parse the received message
    return parse_external_message(message, current_state, external_state_id, assigned_rank, path);
}

// Sub-function to parse received messages for external nodes
bool ParallelEagerSearch::parse_external_message(const std::vector<int> &message, State &current_state, int &external_state_id, unsigned int &assigned_rank, std::vector<OperatorID> &path) {
    log << "(Process " << processor_info.rank << ") Parsing external message." << endl;

    assigned_rank = message[0];
    external_state_id = message[1];
    int segment_length = message[2];

    if (assigned_rank == processor_info.rank) {
        current_state = state_registry.lookup_state(StateID(external_state_id));
        log << "(Process " << processor_info.rank << ") Updated current state from state registry." << endl;
    }

    for (int i = 0; i < segment_length; i++) {
        OperatorID op_id = OperatorID(message[3 + i]);
        if (op_id == OperatorID::no_operator) {
            log << "(Process " << processor_info.rank << ") End of plan segment detected." << endl;
            return true;
        }
        path.push_back(op_id);
    }
    return false;
}

// Sub-function to process internal nodes
bool ParallelEagerSearch::process_internal_node(State &current_state, unsigned int &assigned_rank, int &external_state_id, std::vector<OperatorID> &path) {
    log << "(Process " << processor_info.rank << ") Processing internal node with state ID: " << current_state.get_id().value << endl;

    const SearchNodeInfo &info = search_space.search_node_infos[current_state];
    if (info.creating_operator == OperatorID::no_operator) {
        assert(info.parent_state_id == StateID::no_state);
        log << "(Process " << processor_info.rank << ") Reached root node." << endl;
        return true; // End of plan construction
    }

    path.push_back(info.creating_operator);
    log << "(Process " << processor_info.rank << ") Added operator to path." << endl;

    if (node_messages.contains(info.parent_state_id.value)) {
        log << "(Process " << processor_info.rank << ") Node messages contains ID." << endl;

        assigned_rank = node_messages[info.parent_state_id.value].sender;
        external_state_id = node_messages[info.parent_state_id.value].state_id;
        log << "(Process " << processor_info.rank << ") Parent state is external, assigned to rank " << assigned_rank << endl;
    } else {
        current_state = state_registry.lookup_state(info.parent_state_id);
        log << "(Process " << processor_info.rank << ") Parent state is internal, updated current state." << endl;
    }

    return false;
}

void ParallelEagerSearch::construct_plan_worker(int constructor_rank) {
    log << "(Process " << processor_info.rank << ") Starting construct_plan_worker." << endl;

    unsigned int term_signal;
    int request_state_or_terminate[2];
    std::vector<int> path;
    int assigned_rank = processor_info.rank;

    for (;;) {
        MPI_Recv(&request_state_or_terminate, 2, MPI_INT, constructor_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        log << "(Process " << processor_info.rank << ") Received message from constructor rank " << constructor_rank << endl;

        term_signal = request_state_or_terminate[0];

        if (term_signal != 0) {
            log << "(Process " << processor_info.rank << ") Termination signal received." << endl;
            break;
        }

        StateID state_id = StateID(request_state_or_terminate[1]);
        State current_state = state_registry.lookup_state(state_id);
        int external_state_id = current_state.get_id().value;

        for (;;) {
            const SearchNodeInfo &info = search_space.search_node_infos[current_state];
            if (info.creating_operator == OperatorID::no_operator) {
                assert(info.parent_state_id == StateID::no_state);
                path.push_back(OperatorID::no_operator.get_index());
                log << "(Process " << processor_info.rank << ") Root node reached, ending loop." << endl;
                break;
            }
            path.push_back(info.creating_operator.get_index());

            if (node_messages.contains(current_state.get_id().value)) {
                assigned_rank = node_messages[current_state.get_id().value].sender;
                external_state_id = node_messages[current_state.get_id().value].state_id;
                log << "(Process " << processor_info.rank << ") Transitioned to external node " << external_state_id << " for rank " << assigned_rank << "." << endl;
                break;
            }
            current_state = state_registry.lookup_state(info.parent_state_id);
            log << "(Process " << processor_info.rank << ") Transitioned to internal parent node." << endl;
        }

        int message_size = 3 + path.size();
        std::vector<int> out_buffer(message_size); // Use std::vector for safe buffer management

        out_buffer[0] = assigned_rank;
        out_buffer[1] = external_state_id;
        out_buffer[2] = path.size();
        std::copy(path.begin(), path.end(), out_buffer.begin() + 3); // Copy path into the buffer

        MPI_Send(out_buffer.data(), message_size, MPI_INT, constructor_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD);
        log << "(Process " << processor_info.rank << ") Sent constructed plan segment to constructor rank." << endl;

        path.clear();
    }
}


SearchStatus ParallelEagerSearch::detect_termination() {

    if(detect_if_plan_found()){
        return SearchStatus::SOLVED;
    }

    if((processor_info.status == ProcessesStatus::TERMINATION)) {
        return SearchStatus::FAILED;
    }

    if(processor_info.status == ACTIVE) {
        return SearchStatus::IN_PROGRESS;
    }

    unsigned char message_buffer[2];
    int has_received;

	MPI_Iprobe(MPI_ANY_SOURCE, MPIMessageType::STATUS, MPI_COMM_WORLD, &has_received,
			MPI_STATUS_IGNORE);
    if (has_received) {
        MPI_Recv(&message_buffer, 2, MPI_CHAR, MPI_ANY_SOURCE,
            MPIMessageType::STATUS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        parallel::StatusMessage message;
        to_message(message_buffer, &message);
        
        status_processors[message.rank] = message.status;
    }

    bool all_idle = check_and_progress_termination(
        ProcessesStatus::IDLE, ProcessesStatus::PRE_TERMINATION, status_processors);
    if(all_idle) { return SearchStatus::IN_PROGRESS; }

    bool all_pre_term = check_and_progress_termination(
        ProcessesStatus::PRE_TERMINATION, ProcessesStatus::TERMINATION, status_processors);
    if(all_pre_term) { return SearchStatus::FAILED; }

    return SearchStatus::IN_PROGRESS;
}

void ParallelEagerSearch::to_message(unsigned char* message, parallel::StatusMessage* out) {
    // The message is structured as such:
    //      ...[R][R][R][S][S]
    // S = STATUS INDICATOR (2 bits)
    // R = RANK ID (remainding bits)
    // This function extracts boths and returns them.
    switch (message[0] & 0b11)
    {
        case 0:
            out->status = ProcessesStatus::ACTIVE;
            break;
        case 1:
            out->status = ProcessesStatus::IDLE;
            break;
        case 2:
            out->status = ProcessesStatus::PRE_TERMINATION;
            break;
        case 3:
            out->status = ProcessesStatus::TERMINATION;
        default:
            break;
    }

    out->rank = (message[0] >> 2) | message[1] << (CHAR_BIT-2);
}
void ParallelEagerSearch::to_byte(parallel::StatusMessage message, unsigned char* out) {
    // The message is structured as such:
    //      ...[R][R][R][S][S]
    // S = STATUS INDICATOR (2 bits)
    // R = RANK ID (remainding bits)
    // This function extracts boths and returns them.
    int status;
    switch (message.status)
    {
        case ProcessesStatus::ACTIVE:
            status = 0b00;
            break;
        case ProcessesStatus::IDLE:
            status = 0b01;
            break;
        case ProcessesStatus::PRE_TERMINATION:
            status = 0b10;
            break;
        case ProcessesStatus::TERMINATION:
            status = 0b11;
            break;
        default:
            break;
    }

    int encoded_message = 0 & (message.rank << 2 | status);

    out[0] = encoded_message && UCHAR_MAX;
    out[1] = (encoded_message >> CHAR_BIT) && UCHAR_MAX;
}   
std::vector<unsigned char> ParallelEagerSearch::to_byte_message(SearchNode parent, OperatorProxy op, State state) {
    // Create a vector to hold the serialized data
    std::vector<unsigned char> package(state_byte_size + sizeof(int) * 6);

    // Copy the state buffer into the beginning of the vector
    std::copy((unsigned char*)state.get_buffer(), (unsigned char*)state.get_buffer() + state_byte_size, package.begin());

    // Prepare the metadata as integers
    int g = parent.get_g() + get_adjusted_cost(op);
    int state_id = state.get_id().value;
    std::array<int, 6> info = {
        g,                               // g-value
        0,                               // Placeholder for future heuristic value
        op.get_id(),                     // Operator ID
        std::bit_cast<int>(state_hash_map[state_id]),        // Distributed hash
        parent.get_state().get_id().value,     // Parent state ID
        state_id,            // Current state ID
    };

    // Append the metadata to the vector
    unsigned char* metadata_start = package.data() + state_byte_size;
    std::copy(reinterpret_cast<const unsigned char*>(info.data()),
              reinterpret_cast<const unsigned char*>(info.data()) + sizeof(int) * info.size(),
              metadata_start);

    return package;
}



parallel::NodeMessage ParallelEagerSearch::to_node_message(unsigned char* buffer, int sender, bool* discard) {
    // Interpret the buffer for the state
    state_registry.state_data_pool.push_back(reinterpret_cast<const PackedStateBin*>(buffer));
    StateID id = state_registry.insert_id_or_pop_state();
    State state = state_registry.lookup_state(id);

    state.unpack();

    // Parse the metadata
    const unsigned char* metadata_start = buffer + state_byte_size;
    std::array<int, 6> info;
    std::copy(metadata_start, metadata_start + sizeof(int) * 6,
              reinterpret_cast<unsigned char*>(info.data()));

    int g = info[0];
    int h = info[1];
    int op_id = info[2];
    int distributed_hash = info[3];
    StateID parent_id = StateID(info[4]);
    StateID state_id = StateID(info[5]);

    // Create a SearchNode and validate the path cost
    SearchNode node = search_space.get_node(state);
    state_hash_map[state.get_id().value] = distributed_hash;

    int search_space_g = node.get_g();
    if (((search_space_g != -1) && (g > search_space_g) )|| node.is_dead_end()) {
        *discard = true;
    } else {
        search_space.search_node_infos[state].creating_operator = OperatorID(op_id);
        search_space.search_node_infos[state].g = g;
        search_space.search_node_infos[state].parent_state_id = parent_id;
    }

    parallel::NodeMessage parsed_message = {
        node,
        g,
        h,
        task_proxy.get_operators()[op_id],
        distributed_hash,
        parent_id,
        state_id,
        sender
    };

    return parsed_message;
}


void ParallelEagerSearch::retrieve_nodes_from_queue() {
    MPI_Status mpi_status;
    int has_received = 0;
    unsigned char* buffer = new unsigned char[node_byte_size];


    do {
        MPI_Iprobe(MPI_ANY_SOURCE, MPIMessageType::NODE, MPI_COMM_WORLD, 
                   &has_received, &mpi_status);

        if (has_received) {
            int sender = mpi_status.MPI_SOURCE;
            bool discard = false;

            // Receive the node message
            MPI_Recv(buffer, node_byte_size, MPI_BYTE, sender, MPIMessageType::NODE, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Process the received node message
            parallel::NodeMessage message = to_node_message(buffer, sender, &discard);

            // Validate if the node is a dead end
            if (discard) {
                continue;
            }

            State state = message.node.get_state();

            EvaluationContext eval_context(
                state, message.g, false, &statistics);

            // Add the state to the search space
            SearchNode search_node = search_space.get_node(state);
            if (search_node.is_new()) {
                search_node.open_new_node(message.node, message.creating_operator, message.g);
            }

            // Add to node messages and the open list
            node_messages.insert_or_assign(
                state.get_id().value, 
                parallel::ExternalStateMessage{message.state_id.value, message.sender});

            open_list->insert(eval_context, state.get_id());
        }
    } while (has_received);

    delete[] buffer;
}


void ParallelEagerSearch::retrieve_ack_from_queue() {
    MPI_Status mpi_status;
    int has_received = 0;

    do {
        MPI_Iprobe(MPI_ANY_SOURCE, MPIMessageType::ACK, MPI_COMM_WORLD, 
            &has_received, &mpi_status);

        if(has_received) {
            int sender = mpi_status.MPI_SOURCE;
            MPI_Recv(NULL, 0, NULL, sender, MPIMessageType::ACK, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            awaited_ack = min(awaited_ack - 1, awaited_ack);
        }
    }
    while (has_received);
}

unsigned int ParallelEagerSearch::select_constructor_rank() {
    unsigned int send_buffer[2];
    unsigned int recv_buffer[2];

    send_buffer[0] = processor_info.rank;
    send_buffer[1] = lowest_g;

    MPI_Allreduce(send_buffer, recv_buffer, 2, MPI_INT, 
        MPI_ARG_MIN, MPI_COMM_WORLD);
    
    return recv_buffer[0];
}

SearchStatus ParallelEagerSearch::terminate(SearchStatus status){

    if (status == SearchStatus::SOLVED) { 
        MPI_Barrier(MPI_COMM_WORLD);
        
        unsigned int constructor_rank = select_constructor_rank();
        if (processor_info.rank == constructor_rank){
            log << processor_info.rank << " Started host construction" << endl;
            State goal_state = state_registry.lookup_state(StateID(goal_state_id));
            SearchNode goal_node = search_space.get_node(goal_state);
            construct_plan(goal_node);
        }
        else {
            construct_plan_worker(constructor_rank);
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);
    log << "Terminating process: " << processor_info.rank << endl << std::flush;

    // flush_outgoing_buffer(MPIMessageType::NODE);
    // flush_outgoing_buffer(MPIMessageType::TERMINATE);
    // flush_outgoing_buffer(MPIMessageType::FOUND_GOAL);
    // flush_outgoing_buffer(MPIMessageType::ACK);
    // flush_outgoing_buffer(MPIMessageType::CONSTRUCT_PLAN);
    // flush_outgoing_buffer(MPIMessageType::STATUS);


	int buffer_size;
	MPI_Buffer_detach(&mpi_buffer, &buffer_size);

	delete[] mpi_buffer;

    log << "Finalizing process " << processor_info.rank << endl;

	return status;
}

void ParallelEagerSearch::flush_outgoing_buffer(MPIMessageType tag) {
	int has_received = 1;
    MPI_Status status;
	unsigned char* dummy = new unsigned char[node_byte_size * 10000];
	while (has_received) {
		has_received = 0;
		MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &has_received,
				MPI_STATUS_IGNORE);
		if (has_received) {
			int source = status.MPI_SOURCE;
			int d_size;
			MPI_Get_count(&status, MPI_INT, &d_size); // TODO: = node_byte_size?
			MPI_Recv(dummy, d_size, MPI_INT, source, tag,
					MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
}

bool ParallelEagerSearch::lookup_assigned_rank(SearchNode parent, OperatorProxy op, State succ_state) {
    State parent_state = parent.get_state();
    // unsigned int d_hash = distribution_hash->hash_incremental(
    //     parent_state, 
    //     state_hash_map[parent_state.get_id().value], 
    //     op);
    unsigned int d_hash = distribution_hash->hash(succ_state);
    state_hash_map[succ_state.get_id().value] = d_hash;
    unsigned int assigned_rank = d_hash % processor_info.world_size;

    if (processor_info.rank != assigned_rank) {
        std::vector<unsigned char> buffer = to_byte_message(parent, op, succ_state);

        MPI_Bsend(
            buffer.data(), 
            node_byte_size, 
            MPI_BYTE, 
            assigned_rank, 
            MPIMessageType::NODE, 
            MPI_COMM_WORLD);
            
        ++awaited_ack;
        return true;
    }
    return false;
}
SearchStatus ParallelEagerSearch::step() {
    

    retrieve_ack_from_queue();
    

    retrieve_nodes_from_queue();
    

    SearchStatus status = detect_termination();
    if(status != SearchStatus::IN_PROGRESS) {
        
        return terminate(status);
    }

    optional<SearchNode> node;
    while (true) {
        if (open_list->empty()) {
            
            return IN_PROGRESS;
        }
        StateID id = open_list->remove_min();
        

        State s = state_registry.lookup_state(id);
        

        node.emplace(search_space.get_node(s));

        if (node->is_closed()) {
            
            continue;
        }

        EvaluationContext eval_context(s, node->get_g(), false, &statistics);

        if (lazy_evaluator) {
            if (node->is_dead_end()) {
                log << "(Process " << processor_info.rank << ")" << "NOTE: Node is a dead end. Continuing." << endl;
                continue;
            }

            if (lazy_evaluator->is_estimate_cached(s)) {
                int old_h = lazy_evaluator->get_cached_estimate(s);
                int new_h = eval_context.get_evaluator_value_or_infinity(lazy_evaluator.get());
                if (open_list->is_dead_end(eval_context)) {
                    node->mark_as_dead_end();
                    statistics.inc_dead_ends();
                    log << "(Process " << processor_info.rank << ")" << "NOTE: Node marked as dead end." << endl;
                    continue;
                }
                if (new_h != old_h) {
                    open_list->insert(eval_context, id);
                    log << "(Process " << processor_info.rank << ")" << "NOTE: Updated heuristic value and reinserted into open list." << endl;
                    continue;
                }
            }
        }

        node->close();
        log << "(Process " << processor_info.rank << ")" << "NOTE: Node closed." << endl;
        assert(!node->is_dead_end());
        update_f_value_statistics(eval_context);
        statistics.inc_expanded();
        log << "(Process " << processor_info.rank << ")" << "NOTE: Node expanded." << endl;
        break;
    }

    const State &s = node->get_state();
    log << "(Process " << processor_info.rank << ")" << "NOTE: State gotten." << endl;
    if (task_properties::is_goal_state(task_proxy, s)) {
        log << "(Process " << processor_info.rank << ")" << "NOTE: Goal state found." << endl;
        plan_found = true;
        goal_state_id = s.get_id().value;
        lowest_g = node->get_g();

        for (unsigned int i = 1; i < processor_info.world_size; i++) {
            int rank = (processor_info.rank + i) % processor_info.world_size;
            MPI_Bsend(NULL, 0, MPI_CHAR, rank, MPIMessageType::FOUND_GOAL, MPI_COMM_WORLD);
        }
        return IN_PROGRESS;
    }
    s.unpack();
    
    std::vector<int> vec = s.get_unpacked_values();

    std::vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(s, applicable_ops);
    log << "(Process " << processor_info.rank << ")" << "NOTE: Generated applicable operators." << endl;

    pruning_method->prune_operators(s, applicable_ops);
    log << "(Process " << processor_info.rank << ")" << "NOTE: Pruned operators." << endl;

    EvaluationContext eval_context(s, node->get_g(), false, &statistics, true);
    ordered_set::OrderedSet<OperatorID> preferred_operators;
    for (const shared_ptr<Evaluator> &preferred_operator_evaluator : preferred_operator_evaluators) {
        collect_preferred_operators(eval_context,
                                    preferred_operator_evaluator.get(),
                                    preferred_operators);
    }

    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        if ((node->get_real_g() + op.get_cost()) >= bound)
            continue;

        State succ_state = state_registry.get_successor_state(s, op);
        statistics.inc_generated();
        bool is_preferred = preferred_operators.contains(op_id);

        SearchNode succ_node = search_space.get_node(succ_state);

        for (Evaluator *evaluator : path_dependent_evaluators) {
            evaluator->notify_state_transition(s, op_id, succ_state);
        }

        if (succ_node.is_dead_end()) {
            log << "(Process " << processor_info.rank << ")" << "NOTE: Successor node is a dead end. Continuing." << endl;
            continue;
        }

        if (succ_node.is_new()) {
            int succ_g = node->get_g() + get_adjusted_cost(op);

            EvaluationContext succ_eval_context(
                succ_state, succ_g, is_preferred, &statistics);
            statistics.inc_evaluated_states();

            if (open_list->is_dead_end(succ_eval_context)) {
                succ_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                log << "(Process " << processor_info.rank << ")" << "NOTE: Successor node marked as dead end." << endl;
                continue;
            }
            succ_node.open_new_node(*node, op, get_adjusted_cost(op));

            bool is_external = lookup_assigned_rank(*node, op, succ_node.get_state());
            if (!is_external) {
                open_list->insert(succ_eval_context, succ_state.get_id());
            }
            if (search_progress.check_progress(succ_eval_context)) {
                statistics.print_checkpoint_line(succ_node.get_g());
                reward_progress();
            }
        } else if (succ_node.get_g() > node->get_g() + get_adjusted_cost(op)) {
            if (reopen_closed_nodes) {
                if (succ_node.is_closed()) {
                    statistics.inc_reopened();
                }
                succ_node.reopen_closed_node(*node, op, get_adjusted_cost(op));

                EvaluationContext succ_eval_context(
                    succ_state, succ_node.get_g(), is_preferred, &statistics);

                bool is_external = lookup_assigned_rank(*node, op, succ_node.get_state());
                if (!is_external) {
                    open_list->insert(succ_eval_context, succ_state.get_id());

                    node_messages.erase(succ_state.get_id().value);
                }
            } else {
                succ_node.update_closed_node_parent(*node, op, get_adjusted_cost(op));
            }
        }
    }

    return IN_PROGRESS;
}


void ParallelEagerSearch::reward_progress() {
    // Boost the "preferred operator" open lists somewhat whenever
    // one of the heuristics finds a state with a new best h value.
    open_list->boost_preferred();
}

void ParallelEagerSearch::dump_search_space() const {
    search_space.dump(task_proxy);
}

void ParallelEagerSearch::start_f_value_statistics(EvaluationContext &eval_context) {
    if (f_evaluator) {
        int f_value = eval_context.get_evaluator_value(f_evaluator.get());
        statistics.report_f_value_progress(f_value);
    }
}

/* TODO: HACK! This is very inefficient for simply looking up an h value.
   Also, if h values are not saved it would recompute h for each and every state. */
void ParallelEagerSearch::update_f_value_statistics(EvaluationContext &eval_context) {
    if (f_evaluator) {
        int f_value = eval_context.get_evaluator_value(f_evaluator.get());
        statistics.report_f_value_progress(f_value);
    }
}
}
