#ifndef DATA_NODES
#define DATA_NODES

#include "nodes.hpp"

// Node for reading data from a CSV which has a header, and arbirary number of columns length and
// type Template args are type for data: T, and DataTypes: tuple of types for each column
template <typename T = FloatT, typename DataTypes>
struct CSVDataNode : Input<T>
{
};

#endif