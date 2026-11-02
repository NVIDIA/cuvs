/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

#include <cuvs/neighbors/scann.hpp>

#include <rmm/mr/pool_memory_resource.hpp>

#include "common.cuh"

struct scann_args {
  std::string dataset;                 // required: input binary file
  std::string output      = "/tmp";    // output artifact directory
  std::string dtype       = "float";   // float | half | int8 | uint8
  int n_leaves            = 1000;
  int pq_dim              = 2;
  int pq_bits             = 4;
  float partitioning_eta  = 2.0f;
  float soar_lambda       = 1.5f;
  int n_coarse_clusters   = 0;
  int kmeans_n_rows_train = 10000;
  int pq_n_rows_train     = 10000;
  bool bf16               = false;
};

void usage(const char* prog)
{
  std::cerr << "Usage: " << prog
            << " --dataset <file> [--output <dir>] [--dtype float|half|int8|uint8]\n"
               "       [--n_leaves N] [--pq_dim D] [--pq_bits B] [--partitioning_eta E]\n"
               "       [--soar_lambda L] [--n_coarse_clusters C]\n"
               "       [--kmeans_n_rows_train K] [--pq_n_rows_train P] [--bf16 0|1]\n";
  std::cerr << "Input file expected to be a binary file ([uint32 N][uint32 dim][raw data]).\n";
  std::exit(EXIT_FAILURE);
}

scann_args parse_args(int argc, char** argv)
{
  scann_args args;
  bool has_dataset = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--dataset" && i + 1 < argc) {
      args.dataset = argv[++i];
      has_dataset  = true;
    } else if (a == "--output" && i + 1 < argc) {
      args.output = argv[++i];
    } else if (a == "--dtype" && i + 1 < argc) {
      args.dtype = argv[++i];
    } else if (a == "--n_leaves" && i + 1 < argc) {
      args.n_leaves = std::stoi(argv[++i]);
    } else if (a == "--pq_dim" && i + 1 < argc) {
      args.pq_dim = std::stoi(argv[++i]);
    } else if (a == "--pq_bits" && i + 1 < argc) {
      args.pq_bits = std::stoi(argv[++i]);
    } else if (a == "--partitioning_eta" && i + 1 < argc) {
      args.partitioning_eta = std::stof(argv[++i]);
    } else if (a == "--soar_lambda" && i + 1 < argc) {
      args.soar_lambda = std::stof(argv[++i]);
    } else if (a == "--n_coarse_clusters" && i + 1 < argc) {
      args.n_coarse_clusters = std::stoi(argv[++i]);
    } else if (a == "--kmeans_n_rows_train" && i + 1 < argc) {
      args.kmeans_n_rows_train = std::stoi(argv[++i]);
    } else if (a == "--pq_n_rows_train" && i + 1 < argc) {
      args.pq_n_rows_train = std::stoi(argv[++i]);
    } else if (a == "--bf16" && i + 1 < argc) {
      args.bf16 = std::stoi(argv[++i]) != 0;
    } else {
      usage(argv[0]);
    }
  }
  if (!has_dataset) {
    std::cerr << "Error: --dataset is required\n";
    std::exit(EXIT_FAILURE);
  }
  if (!std::filesystem::exists(args.dataset)) {
    std::cerr << "Error: file not found: " << args.dataset << "\n";
    std::exit(EXIT_FAILURE);
  }
  return args;
}

template <typename T>
void scann_build_and_write(raft::device_resources const& dev_resources, const scann_args& args)
{
  using namespace cuvs::neighbors;

  auto dataset = read_bin_dataset<T, int64_t>(dev_resources, args.dataset, INT_MAX);

  scann::index_params index_params;
  index_params.n_leaves            = args.n_leaves;
  index_params.pq_dim              = args.pq_dim;
  index_params.pq_bits             = args.pq_bits;
  index_params.partitioning_eta    = args.partitioning_eta;
  index_params.soar_lambda         = args.soar_lambda;
  index_params.n_coarse_clusters   = args.n_coarse_clusters;
  index_params.kmeans_n_rows_train = args.kmeans_n_rows_train;
  index_params.pq_n_rows_train     = args.pq_n_rows_train;
  index_params.reordering_bf16     = args.bf16;

  std::cout << "Building ScaNN index" << std::endl;

  auto start = std::chrono::system_clock::now();
  auto index =
    scann::build(dev_resources, index_params, raft::make_const_mdspan(dataset.view()));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;

  std::cout << "Time to build index: " << elapsed_seconds.count() << "s\n";

  // Output serialized index artifacts to the output directory
  serialize(dev_resources, args.output, index);
}

int main(int argc, char* argv[])
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use
  // the same pool.
  rmm::mr::pool_memory_resource pool_mr(rmm::mr::get_current_device_resource_ref(),
                                        1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(pool_mr);

  auto args = parse_args(argc, argv);

  if (args.dtype == "float") {
    scann_build_and_write<float>(dev_resources, args);
  } else if (args.dtype == "half" || args.dtype == "fp16") {
    scann_build_and_write<half>(dev_resources, args);
  } else if (args.dtype == "int8") {
    scann_build_and_write<int8_t>(dev_resources, args);
  } else if (args.dtype == "uint8") {
    scann_build_and_write<uint8_t>(dev_resources, args);
  } else {
    usage(argv[0]);
  }
}
