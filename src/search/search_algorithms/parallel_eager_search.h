#ifndef SEARCH_ALGORITHMS_PARALLEL_EAGER_SEARCH_H
#define SEARCH_ALGORITHMS_PARALLEL_EAGER_SEARCH_H

#include "../open_list.h"
#include "../search_algorithm.h"
#include "../search_node_info.h"

#include <memory>
#include <vector>

#include <mpi.h>

class Evaluator;
class PruningMethod;

namespace plugins {
class Feature;
}

void mpi_arg_min_func(void* inputBuffer, void* outputBuffer, int* len, MPI_Datatype* datatype);

enum ProcessesStatus {ACTIVE, IDLE, PRE_TERMINATION, TERMINATION};

namespace parallel {

    struct ProcessorInfo {
        unsigned int world_size;
        unsigned int rank;
        ProcessesStatus status = ProcessesStatus::IDLE;
    };

    struct StatusMessage {
        ProcessesStatus status;
        unsigned int rank;
    };


    struct NodeMessage {
        SearchNode node;
        int g;
        int h;
        OperatorProxy creating_operator;
        int distributed_hash;
        StateID parent_id;
        StateID id;
        int sender;
    };


    struct StateSemanticHash {
        int state_size;
        StateSemanticHash(
            int state_size)
            : state_size(state_size) {
        }

        uint64_t operator()(State state) const {
            const PackedStateBin *data = state.get_buffer();
            utils::HashState hash_state;
            for (int i = 0; i < state_size; ++i) {
                hash_state.feed(data[i]);
            }
            return hash_state.get_hash64();
        }
    };

    struct ExternalStateMessage {
        int state_id;
        int sender;
        ExternalStateMessage& operator=(const ExternalStateMessage& other) { 
            state_id = other.state_id;
            sender = other.sender;
            return *this;
        };
    };

}



enum MPIMessageType {NODE, TERMINATE, FOUND_GOAL, ACK, CONSTRUCT_PLAN, STATUS};

namespace parallel_eager_search {
class ParallelEagerSearch : public SearchAlgorithm {

    using NodeMessageMap = phmap::flat_hash_map<int, parallel::ExternalStateMessage>;
    NodeMessageMap node_messages;
    MPI_Op MPI_ARG_MIN;

    const bool reopen_closed_nodes;
    const unsigned int state_byte_size;
    const unsigned int node_byte_size;

    parallel::ProcessorInfo processor_info;
	
    unsigned char* mpi_buffer; // used for MPI_Buffer_attach.
    unsigned int awaited_ack = 0;
    int goal_state_id;
    int lowest_g = INT_MAX;
    bool plan_found = false;
    std::vector<ProcessesStatus> status_processors;

    std::unique_ptr<StateOpenList> open_list;
    std::shared_ptr<Evaluator> f_evaluator;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::vector<std::shared_ptr<Evaluator>> preferred_operator_evaluators;
    std::shared_ptr<Evaluator> lazy_evaluator;

    std::shared_ptr<PruningMethod> pruning_method;

    unsigned int get_assigned_rank(State state);
    bool lookup_assigned_rank(SearchNode parent, OperatorProxy op, State succ_state);
    void send_node_message();
    void insert_incoming_queue();
    StateID messages_to_id(unsigned char* d, unsigned int d_size);
    SearchStatus detect_termination();
    bool detect_if_plan_found();
    void construct_plan(SearchNode goal_node);
    void construct_plan_worker(int constructor_rank);
    unsigned int select_constructor_rank();
    void retrieve_ack_from_queue();
    void retrieve_nodes_from_queue();   
    SearchStatus terminate(SearchStatus status);

    // Handles external node processing
    bool process_external_node(State &current_state, int &external_state_id, unsigned int &assigned_rank, std::vector<OperatorID> &path);

    // Parses the message received for an external node
    bool parse_external_message(const std::vector<int> &message, State &current_state, int &external_state_id, unsigned int &assigned_rank, std::vector<OperatorID> &path);

    // Handles internal node processing
    bool process_internal_node(State &current_state, unsigned int &assigned_rank, int &external_state_id, std::vector<OperatorID> &path);
    void flush_outgoing_buffer(MPIMessageType tag);
    bool is_idle();
    bool check_and_progress_termination(
        ProcessesStatus status, 
        ProcessesStatus next_status, 
        std::vector<ProcessesStatus> statuses);

    void to_message(unsigned char* message, parallel::StatusMessage* out);
    void to_byte(parallel::StatusMessage message, unsigned char* out);
    std::vector<unsigned char> to_byte_message(SearchNode parent, OperatorProxy op, State succ_state);
    parallel::NodeMessage to_node_message(unsigned char* buffer, int sender, bool* discard);

    void start_f_value_statistics(EvaluationContext &eval_context);
    void update_f_value_statistics(EvaluationContext &eval_context);
    void reward_progress();

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit ParallelEagerSearch(const plugins::Options &opts);
    virtual ~ParallelEagerSearch() = default;

    virtual void print_statistics() const override;

    void dump_search_space() const;
private:

};

extern void add_options_to_feature(plugins::Feature &feature);
}

#endif
