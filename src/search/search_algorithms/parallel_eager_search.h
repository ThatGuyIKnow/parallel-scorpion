#ifndef SEARCH_ALGORITHMS_PARALLEL_EAGER_SEARCH_H
#define SEARCH_ALGORITHMS_PARALLEL_EAGER_SEARCH_H

#include "../open_list.h"
#include "../search_algorithm.h"
#include "../search_node_info.h"

#include <memory>
#include <vector>

class Evaluator;
class PruningMethod;

namespace plugins {
class Feature;
}

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
        OperatorProxy op;
        int distributed_hash;
        StateID parent_id;
    };
}



enum MPIMessageType {NODE, TERMINATE, FOUND_PLAN, ACK, CONSTRUCT_PLAN, STATUS};

namespace parallel_eager_search {
class ParallelEagerSearch : public SearchAlgorithm {
    const bool reopen_closed_nodes;
    const unsigned int state_byte_size;
    const unsigned int node_byte_size;

    parallel::ProcessorInfo processor_info;
	
    unsigned char* mpi_buffer; // used for MPI_Buffer_attach.
    unsigned int awaited_ack = 0;
    bool plan_found = false;
    std::vector<ProcessesStatus> status_processors;

    std::unique_ptr<StateOpenList> open_list;
    std::shared_ptr<Evaluator> f_evaluator;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::vector<std::shared_ptr<Evaluator>> preferred_operator_evaluators;
    std::shared_ptr<Evaluator> lazy_evaluator;

    std::shared_ptr<PruningMethod> pruning_method;

    void send_node_message();
    void insert_incoming_queue();
    StateID messages_to_id(unsigned char* d, unsigned int d_size);
    bool detect_termination();
    bool detect_if_plan_found();
    void retrieve_ack_from_queue();
    void retrieve_nodes_from_queue();   
    SearchStatus terminate();
    void flush_outgoing_buffer(MPIMessageType tag);
    bool is_idle();
    bool check_and_progress_termination(
        ProcessesStatus status, 
        ProcessesStatus next_status, 
        std::vector<ProcessesStatus> statuses);

    void ToMessage(unsigned char* message, parallel::StatusMessage* out);
    void ToByte(parallel::StatusMessage message, unsigned char* out);
    void ToByteMessage(unsigned char* out, SearchNode parent,
		OperatorProxy op, parallel::ProcessorInfo originating_process);
    parallel::NodeMessage ToNodeMessage(unsigned char* buffer);

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
