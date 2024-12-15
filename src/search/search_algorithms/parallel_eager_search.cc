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
ParallelEagerSearch::ParallelEagerSearch(const plugins::Options &opts)
    : SearchAlgorithm(opts),
      reopen_closed_nodes(opts.get<bool>("reopen_closed")),
      state_byte_size(state_registry.get_state_size_in_bytes()),
      node_byte_size(state_byte_size + sizeof(int) * 7),
      open_list(opts.get<shared_ptr<OpenListFactory>>("open")->
                create_state_open_list()),
      f_evaluator(opts.get<shared_ptr<Evaluator>>("f_eval", nullptr)),
      preferred_operator_evaluators(opts.get_list<shared_ptr<Evaluator>>("preferred")),
      lazy_evaluator(opts.get<shared_ptr<Evaluator>>("lazy_evaluator", nullptr)),
      pruning_method(opts.get<shared_ptr<PruningMethod>>("pruning")) {
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


    MPI_Init(NULL, NULL);

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

    // TO DO: DISTRIBUTION HASH. TO VALIDATE PRA* IMPLEMENTATION, KEEP SIMPLE TO START.
    // srand(42);
	// unsigned int d_hash = rand();

    if (processor_info.rank == get_assigned_rank(state_registry.get_initial_state())) {

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

        for(int i = 1; i < processor_info.world_size; i++) {
            int rank = (processor_info.rank + i) % processor_info.world_size;
            MPI_Bsend(message_buffer, 2, MPI_BYTE, rank, MPIMessageType::STATUS, MPI_COMM_WORLD);
        }
        return true;
    }

    return false;
}


void ParallelEagerSearch::construct_plan(SearchNode goal_node) {
    State current_state = goal_node.get_state();
    int external_state_id = current_state.get_id().value;
    std::vector<OperatorID> path;
    unsigned int assigned_rank = processor_info.rank;

    while (true) {
        if (processor_info.rank != assigned_rank) {
            if(process_external_node(current_state, external_state_id, assigned_rank, path)) {
                break;
            }
        } else {
            if (process_internal_node(current_state, assigned_rank, external_state_id, path)) {
                break;
            }
        }
    }

    int term_signal[] = {1, 0};
    for (int i = 1; i < processor_info.world_size; i++) {
        int rank = (processor_info.rank + i) % processor_info.world_size;
        MPI_Bsend(term_signal, 2, MPI_INT, rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD);
    }
    std::reverse(path.begin(), path.end());
    set_plan(path);
}

// Sub-function to handle external nodes
bool ParallelEagerSearch::process_external_node(State &current_state, int &external_state_id, unsigned int &assigned_rank, std::vector<OperatorID> &path) {
    // Send request for the parent node
    int buffer[2] = {0, external_state_id};
    MPI_Send(buffer, 2, MPI_INT, assigned_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD);

    // Probe for message size
    MPI_Status status;
    int package_size;
    MPI_Probe(assigned_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_INT, &package_size);

    // Receive the message
    std::vector<int> message(package_size);
    MPI_Recv(message.data(), package_size, MPI_INT, assigned_rank, MPIMessageType::CONSTRUCT_PLAN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Parse the received message
    return parse_external_message(message, current_state, external_state_id, assigned_rank, path);
}

// Sub-function to parse received messages for external nodes
bool ParallelEagerSearch::parse_external_message(const std::vector<int> &message, State &current_state, int &external_state_id, unsigned int &assigned_rank, std::vector<OperatorID> &path) {
    assigned_rank = message[0];
    external_state_id = message[1];
    int segment_length = message[2];

    if (assigned_rank == processor_info.rank) {
        current_state = state_registry.lookup_state(StateID(external_state_id));
    }

    for (int i = 0; i < segment_length; i++) {
        OperatorID op_id = OperatorID(message[3 + i]);
        if (op_id == OperatorID::no_operator) {
            return true;
        }
        path.push_back(op_id);
    }
    return false;
}

// Sub-function to process internal nodes
bool ParallelEagerSearch::process_internal_node(State &current_state, unsigned int &assigned_rank, int &external_state_id, std::vector<OperatorID> &path) {
    const SearchNodeInfo &info = search_space.search_node_infos[current_state];
    if (info.creating_operator == OperatorID::no_operator) {
        assert(info.parent_state_id == StateID::no_state);
        return true; // End of plan construction
    }

    path.push_back(info.creating_operator);
    if (node_messages.contains(info.parent_state_id.value)) {
        assigned_rank = node_messages[info.parent_state_id.value].sender;
        external_state_id = node_messages[info.parent_state_id.value].state_id;
    } else {
        current_state = state_registry.lookup_state(info.parent_state_id);
    }

    return false;
}



void ParallelEagerSearch::construct_plan_worker(int constructor_rank) { 
    unsigned int term_signal;
    int request_state_or_terminate[2];
    std::vector<int> path; // Use std::vector for safe dynamic memory management
    int assigned_rank = processor_info.rank;

    for (;;) {
                MPI_Recv(&request_state_or_terminate, 2, MPI_INT, constructor_rank, MPIMessageType::CONSTRUCT_PLAN, 
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        term_signal = request_state_or_terminate[0];
        
        if (term_signal != 0) {
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
                break;
            }
            path.push_back(info.creating_operator.get_index());

            if (node_messages.contains(current_state.get_id().value)) {
                assigned_rank = node_messages[current_state.get_id().value].sender;
                external_state_id = node_messages[current_state.get_id().value].state_id;
                break;
            }
            current_state = state_registry.lookup_state(info.parent_state_id);
        }

        int message_size = 3 + path.size();
        std::vector<int> out_buffer(message_size); // Use std::vector for safe buffer management

        
        out_buffer[0] = assigned_rank;
        out_buffer[1] = external_state_id;
        out_buffer[2] = path.size();
        std::copy(path.begin(), path.end(), out_buffer.begin() + 3); // Copy path into the buffer

        MPI_Send(out_buffer.data(), message_size, MPI_INT, constructor_rank, MPIMessageType::CONSTRUCT_PLAN, 
                 MPI_COMM_WORLD);
        
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
    std::vector<unsigned char> package(state_byte_size + sizeof(int) * 7);

    // Copy the state buffer into the beginning of the vector
    const PackedStateBin* state_buffer = state.get_buffer();
    std::copy(state_buffer, state_buffer + state_byte_size, package.begin());

    // Prepare the metadata as integers
    int g = parent.get_g() + get_adjusted_cost(op);
    std::array<int, 7> info = {
        g,                               // g-value
        0,                               // Placeholder for future heuristic value
        op.get_id(),                     // Operator ID
        0,                               // Distributed hash (placeholder)
        parent.get_state().get_id().value,     // Parent state ID
        state.get_id().value,            // Current state ID
        processor_info.rank              // Processor rank
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
    const PackedStateBin* state_buffer = reinterpret_cast<const PackedStateBin*>(buffer);
    state_registry.state_data_pool.push_back(state_buffer);

    StateID id = state_registry.insert_id_or_pop_state();
    State state = state_registry.lookup_state(id);

    // Parse the metadata
    const unsigned char* metadata_start = buffer + state_byte_size;
    std::array<int, 7> info;
    std::copy(metadata_start, metadata_start + sizeof(int) * 7,
              reinterpret_cast<unsigned char*>(info.data()));

    int g = info[0];
    int h = info[1];
    int op_id = info[2];
    int distributed_hash = info[3];
    StateID parent_id = StateID(info[4]);
    StateID state_id = StateID(info[5]);

    // Create a SearchNode and validate the path cost
    SearchNode node = search_space.get_node(state);

    int search_space_g = search_space.get_node(state).get_g();
    if ((search_space_g != -1) && (g > search_space_g)) {
        *discard = true;
    } else {
        search_space.search_node_infos[state].creating_operator = OperatorID(op_id);
        search_space.search_node_infos[state].g = g;
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
            if (discard || message.node.is_dead_end()) {
                continue;
            }

            State state = message.node.get_state();

            EvaluationContext eval_context(
                state, message.g, false, &statistics);

            // Add the state to the search space
            SearchNode search_node = search_space.get_node(state);
            if (search_node.is_new()) {
                search_node.open(message.node, message.creating_operator, message.g);
            }

            // Add to node messages and the open list
            node_messages.insert_or_assign(
                state.get_id().value, 
                parallel::ExternalStateMessage{message.id.value, message.sender});

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

	MPI_Finalize();
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
    unsigned int assigned_rank = get_assigned_rank(succ_state);
    if (assigned_rank != processor_info.rank) {
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
    // update_incumbent();

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

        if (node->is_closed())
            continue;

        /*
          We can pass calculate_preferred=false here since preferred
          operators are computed when the state is expanded.
        */
        EvaluationContext eval_context(s, node->get_g(), false, &statistics);

        if (lazy_evaluator) {
            /*
              With lazy evaluators (and only with these) we can have dead nodes
              in the open list.
`
              For example, consider a state s that is reached twice before it is expanded.
              The first time we insert it into the open list, we compute a finite
              heuristic value. The second time we insert it, the cached value is reused.

              During first expansion, the heuristic value is recomputed and might become
              infinite, for example because the reevaluation uses a stronger heuristic or
              because the heuristic is path-dependent and we have accumulated more
              information in the meantime. Then upon second expansion we have a dead-end
              node which we must ignore.
            */
            if (node->is_dead_end())
                continue;

            if (lazy_evaluator->is_estimate_cached(s)) {
                int old_h = lazy_evaluator->get_cached_estimate(s);
                int new_h = eval_context.get_evaluator_value_or_infinity(lazy_evaluator.get());
                if (open_list->is_dead_end(eval_context)) {
                    node->mark_as_dead_end();
                    statistics.inc_dead_ends();
                    continue;
                }
                if (new_h != old_h) {
                    open_list->insert(eval_context, id);
                    continue;
                }
            }
        }

        node->close();
        assert(!node->is_dead_end());
        update_f_value_statistics(eval_context);
        statistics.inc_expanded();
        break;
    }

    const State &s = node->get_state();
    if (task_properties::is_goal_state(task_proxy, s)){
        log << "Solution found by process " << processor_info.rank << "!" << endl;
        plan_found = true;
        goal_state_id = s.get_id().value;
        lowest_g = node->get_g();

        
        for(int i = 1; i < processor_info.world_size; i++) {
            int rank = (processor_info.rank + i) % processor_info.world_size;
            MPI_Bsend(NULL, 0, MPI_CHAR, rank, MPIMessageType::FOUND_GOAL, MPI_COMM_WORLD);
        }
        return IN_PROGRESS;
    }
    std::vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(s, applicable_ops);

    /*
      TODO: When preferred operators are in use, a preferred operator will be
      considered by the preferred operator queues even when it is pruned.
    */
    pruning_method->prune_operators(s, applicable_ops);

    // This evaluates the expanded state (again) to get preferred ops
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

        // Previously encountered dead end. Don't re-evaluate.
        if (succ_node.is_dead_end())
            continue;


        if (succ_node.is_new()) {
            // We have not seen this state before.
            // Evaluate and create a new node.

            // Careful: succ_node.get_g() is not available here yet,
            // hence the stupid computation of succ_g.
            // TODO: Make this less fragile.
            int succ_g = node->get_g() + get_adjusted_cost(op);

            EvaluationContext succ_eval_context(
                succ_state, succ_g, is_preferred, &statistics);
            statistics.inc_evaluated_states();

            if (open_list->is_dead_end(succ_eval_context)) {
                succ_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                continue;
            }
            succ_node.open(*node, op, get_adjusted_cost(op));

            bool is_external = lookup_assigned_rank(*node, op, succ_node.get_state());
            if (!is_external) {
                open_list->insert(succ_eval_context, succ_state.get_id());
            }
            if (search_progress.check_progress(succ_eval_context)) {
                statistics.print_checkpoint_line(succ_node.get_g());
                reward_progress();
            }
        } else if (succ_node.get_g() > node->get_g() + get_adjusted_cost(op)) {
            // We found a new cheapest path to an open or closed state.
            if (reopen_closed_nodes) {
                if (succ_node.is_closed()) {
                    /*
                      TODO: It would be nice if we had a way to test
                      that reopening is expected behaviour, i.e., exit
                      with an error when this is something where
                      reopening should not occur (e.g. A* with a
                      consistent heuristic).
                    */
                    statistics.inc_reopened();
                }
                succ_node.reopen(*node, op, get_adjusted_cost(op));

                EvaluationContext succ_eval_context(
                    succ_state, succ_node.get_g(), is_preferred, &statistics);

                /*
                  Note: our old code used to retrieve the h value from
                  the search node here. Our new code recomputes it as
                  necessary, thus avoiding the incredible ugliness of
                  the old "set_evaluator_value" approach, which also
                  did not generalize properly to settings with more
                  than one evaluator.

                  Reopening should not happen all that frequently, so
                  the performance impact of this is hopefully not that
                  large. In the medium term, we want the evaluators to
                  remember evaluator values for states themselves if
                  desired by the user, so that such recomputations
                  will just involve a look-up by the Evaluator object
                  rather than a recomputation of the evaluator value
                  from scratch.
                */
                bool is_external = lookup_assigned_rank(*node, op, succ_node.get_state());
                if (!is_external) {
                    open_list->insert(succ_eval_context, succ_state.get_id());
                }
            } else {
                // If we do not reopen closed nodes, we just update the parent pointers.
                // Note that this could cause an incompatibility between
                // the g-value and the actual path that is traced back.
                succ_node.update_parent(*node, op, get_adjusted_cost(op));
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

void add_options_to_feature(plugins::Feature &feature) {
    SearchAlgorithm::add_pruning_option(feature);
    SearchAlgorithm::add_options_to_feature(feature);
}
}
