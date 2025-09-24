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

#ifndef VALLA_INCLUDE_DTDB_H_IMPL_HPP_
#define VALLA_INCLUDE_DTDB_H_IMPL_HPP_

#include <bit>
#include <cassert>
#include <iterator>

namespace valla {

namespace detail {
template <class Iterator>
inline Index insert_sequence_recursively(Iterator it, Iterator end, Index size, DtdbHTable& table) {
    if (size == Index{1})
        return *it;
    if (size == Index{2})
        return table.insert_internal(IndexSlot(*it, *(it + 1)));

    const auto mid = std::bit_floor(size - 1);
    const auto mid_it = it + mid;
    const auto i1 = insert_sequence_recursively(it, mid_it, mid, table);
    const auto i2 = insert_sequence_recursively(mid_it, end, size - mid, table);
    return table.insert_internal(IndexSlot(i1, i2));
}
} // namespace detail

template <class Range>
Index insert_sequence(const Range& sequence, DtdbHTable& table) {
    const auto size = static_cast<Index>(std::distance(sequence.begin(), sequence.end()));
    if (size == Index{0})
        return Index{0};

    table.resize_to_fit(sequence);
    const auto tree_index = detail::insert_sequence_recursively(sequence.begin(), sequence.end(), size, table);
    return table.insert_root(IndexSlot(tree_index, size));
}

namespace detail {
template <class OutIterator>
inline void read_sequence_recursively(Index index, Index size, const DtdbHTable& table, OutIterator out) {
    if (size == Index{1}) {
        *out++ = index;
        return;
    }
    if (size == Index{2}) {
        const auto slot = table.lookup_internal(index);
        *out++ = slot.lhs;
        *out++ = slot.rhs;
        return;
    }
    const auto mid = std::bit_floor(size - 1);
    const auto slot = table.lookup_internal(index);
    read_sequence_recursively(slot.lhs, mid, table, out);
    read_sequence_recursively(slot.rhs, size - mid, table, out);
}
} // namespace detail

template <class OutIterator>
void read_sequence(Index root_index, const DtdbHTable& table, OutIterator out) {
    const auto slot = table.lookup_root(root_index);
    if (slot.rhs == Index{0})
        return;
    detail::read_sequence_recursively(slot.lhs, slot.rhs, table, out);
}

} // namespace valla

#endif // VALLA_INCLUDE_DTDB_H_IMPL_HPP_
