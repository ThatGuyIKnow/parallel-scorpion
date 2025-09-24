/*
 * Copyright (C) 2025 Dominik Drexler
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef VALLA_INCLUDE_DTDB_H_HPP_
#define VALLA_INCLUDE_DTDB_H_HPP_

#include "valla/declarations.hpp"
#include "valla/indexed_hash_set.hpp"

#include <iterator>
#include <type_traits>

namespace valla {

/**
 * A minimal unstable indexed hash set for DTDB_H built on top of the existing
 * valla::IndexedHashSet. It stores:
 *  - internal nodes: pairs of indices representing binary tree nodes
 *  - root nodes: pairs (tree_index, size)
 *
 * The interface mirrors what's needed by DTDB_H: insert_internal, insert_root,
 * lookup_internal, lookup_root, and a resize_to_fit no-op.
 */
class DtdbHTable {
public:
    using index_type = Index;
    using value_type = IndexSlot; // pair<Index, Index>

    DtdbHTable() = default;

    // Not copyable or movable to keep internal references stable (mirrors IndexedHashSet policy).
    DtdbHTable(const DtdbHTable&) = delete;
    DtdbHTable& operator=(const DtdbHTable&) = delete;
    DtdbHTable(DtdbHTable&&) = delete;
    DtdbHTable& operator=(DtdbHTable&&) = delete;


    template <class AnySequence>
    void resize_to_fit(const AnySequence&) {}

    // Insert an internal binary node (i1, i2) and return its index.
    Index insert_internal(IndexSlot slot) {
        auto res = m_internal.insert(slot);
        return *res.first;
    }

    // Insert a root node (tree_index, size) and return its index.
    Index insert_root(IndexSlot slot) {
        auto res = m_root.insert(slot);
        return *res.first;
    }

    // Lookup internal node by index.
    IndexSlot lookup_internal(Index i) const { return m_internal[i]; }

    // Lookup root node by index.
    IndexSlot lookup_root(Index i) const { return m_root[i]; }

    size_t internal_size() const { return m_internal.size(); }
    size_t root_size() const { return m_root.size(); }

private:
    IndexedHashSet m_internal;
    IndexedHashSet m_root;
};

// Insert a sequence of indices and return the root index in the root table.
template <class Range>
Index insert_sequence(const Range& sequence, DtdbHTable& table);

// Read back the sequence of indices from the given root index.
template <class OutIterator>
void read_sequence(Index root_index, const DtdbHTable& table, OutIterator out);

} // namespace valla

#include "valla/dtdb_h_impl.hpp"

#endif // VALLA_INCLUDE_DTDB_H_HPP_
