#pragma once

//#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>

#include <cooperative_groups.h>


__constant__ char aminoacid_sequence[150];