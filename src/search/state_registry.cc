#include "state_registry.h"

#include "plugins/plugin.h"

static plugins::TypedEnumPlugin<StateRegistryType> _enum_plugin({
    {"packed", "state variables are packed into integers which are stored in a segmented vector"},
    {"dtdb_s_packed", "packed + DTDB_S"},
    {"dtdb_h_packed", "packed + DTDB_H"},
});

