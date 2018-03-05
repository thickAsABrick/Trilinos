// @HEADER
//
// ***********************************************************************
//
//        MueLu: A package for multigrid based preconditioning
//                  Copyright 2012 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact
//                    Jonathan Hu       (jhu@sandia.gov)
//                    Andrey Prokopenko (aprokop@sandia.gov)
//                    Ray Tuminaro      (rstumin@sandia.gov)
//
// ***********************************************************************
//
// @HEADER
#ifndef MUELU_AGGREGATIONPHASE2BALGORITHM_KOKKOS_DEF_HPP
#define MUELU_AGGREGATIONPHASE2BALGORITHM_KOKKOS_DEF_HPP

#ifdef HAVE_MUELU_KOKKOS_REFACTOR

#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <Xpetra_Vector.hpp>

#include "MueLu_AggregationPhase2bAlgorithm_kokkos_decl.hpp"

#include "MueLu_Aggregates_kokkos.hpp"
#include "MueLu_Exceptions.hpp"
#include "MueLu_LWGraph_kokkos.hpp"
#include "MueLu_Monitor.hpp"

namespace MueLu {

  // Try to stick unaggregated nodes into a neighboring aggregate if they are
  // not already too big
  template <class LocalOrdinal, class GlobalOrdinal, class Node>
  void AggregationPhase2bAlgorithm_kokkos<LocalOrdinal, GlobalOrdinal, Node>::
  BuildAggregates(const ParameterList& params, const LWGraph_kokkos& graph,
                  Aggregates_kokkos& aggregates, std::vector<unsigned>& aggStat,
                  LO& numNonAggregatedNodes, Kokkos::View<LO*,
                  typename MueLu::LWGraph_kokkos<LO, GO, Node>::local_graph_type::device_type::
                  memory_space>& colorsDevice, LO& numColors) const {
    Monitor m(*this, "BuildAggregates");

    typedef typename MueLu::LWGraph_kokkos<LO, GO, Node>::local_graph_type graph_t;
    typedef typename graph_t::device_type::memory_space memory_space;
    typedef typename graph_t::device_type::execution_space execution_space;
    typedef typename graph_t::device_type::execution_space::scratch_memory_space scratch_space;
    typedef typename Kokkos::TeamPolicy<execution_space>::member_type member_type;
    typedef Kokkos::View<int*, scratch_space, Kokkos::MemoryTrait<Kokkos::Unmanaged> > shared_int_1d;

    const LO  numRows = graph.GetNodeNumVertices();
    const int myRank  = graph.GetComm()->getRank();

    // Note, lbv 01-05-18: Most of the following should disappear in favor
    //                     of having aggStat as a view instead of having
    //                     aggStat as an std::vector.
    Kokkos::View<unsigned*, memory_space> d_aggStatView("aggStat", numRows);
    typename Kokkos::View<unsigned*, memory_space>::HostMirror h_aggStatView =
      Kokkos::create_mirror_view (d_aggStatView);
    Kokkos::parallel_for("Phase 2b: Initialize aggStatView", numRows,
                         KOKKOS_LAMBDA (const LO i) { h_aggStatView(i) = aggStat[i];});
    Kokkos::deep_copy(d_aggStatView,    h_aggStatView);

    auto vertex2AggIdView = aggregates.GetVertex2AggId()->template getLocalView<memory_space>();
    auto procWinnerView   = aggregates.GetProcWinner()  ->template getLocalView<memory_space>();

    ArrayRCP<LO> vertex2AggId = aggregates.GetVertex2AggId()->getDataNonConst(0);
    ArrayRCP<LO> procWinner   = aggregates.GetProcWinner()  ->getDataNonConst(0);

    LO numLocalAggregates = aggregates.GetNumAggregates();

    const int defaultConnectWeight = 100;
    const int penaltyConnectWeight = 10;

    std::vector<int> aggWeight    (numLocalAggregates, 0);
    std::vector<int> connectWeight(numRows, defaultConnectWeight);
    std::vector<int> aggPenalties (numRows, 0);

    Kokkos::View<int*, memory_space> connectWeightView("connectWeight", numRows);
    Kokkos::View<int*, memory_space> aggPenaltiesView ("aggPenalties",  numRows);

    typename Kokkos::View<int*, memory_space>::HostMirror h_connectWeightView =
      Kokkos::create_mirror_view (connectWeightView);
    Kokkos::parallel_for("Phase 2b: Initialize connectWeightView", numRows,
                         KOKKOS_LAMBDA (const LO i) { h_connectWeightView(i) = defaultConnectWeight;});
    Kokkos::deep_copy(connectWeightView,    h_connectWeightView);

    // We do this cycle twice.
    // I don't know why, but ML does it too
    // taw: by running the aggregation routine more than once there is a chance that also
    // non-aggregated nodes with a node distance of two are added to existing aggregates.
    // Assuming that the aggregate size is 3 in each direction running the algorithm only twice
    // should be sufficient.
    for (int k = 0; k < 2; k++) {
      // total work = numberOfTeams * teamSize
      Kokkos::TeamPolicy<execution_space> outerPolicy<>
      Kokkos::RangePolicy<LO, execution_space> numRowsPolicy(0, numRows);
      Kokkos::parallel_for ("", numRowsPolicy,
                            KOKKOS_LAMBDA (const member_type & teamMember) {

                              // Allocate this locally so that threads do not trash the weigth
                              // when working on the same aggregate.
                              shared_int_1d aggWeightView    ("aggWeight", numLocalAggregates);
                              const unsigned int vertexToAggregate = teamMember.league_rank();

                              if (d_aggStatView(i) != READY)
                                continue;

                              auto neighOfINode = graph.getNeighborVertices(i);

                              for (int j = 0; j < as<int>(neighOfINode.length); j++) {
                                LO neigh = neighOfINode(j);

                                // We don't check (neigh != i), as it is covered by checking (aggStat[neigh] == AGGREGATED)
                                if (graph.isLocalNeighborVertex(neigh) && d_aggStatView(neigh) == AGGREGATED)
                                  aggWeightView(vertex2AggId(neigh, 0)) =
                                    aggWeightView(vertex2AggId(neigh, 0)) + connectWeight(neigh);
                              }

                              int bestScore   = -100000;
                              int bestAggId   = -1;
                              int bestConnect = -1;

                              for (int j = 0; j < as<int>(neighOfINode.length); j++) {
                                LO neigh = neighOfINode(j);

                                if (graph.isLocalNeighborVertex(neigh) && d_aggStatView(neigh) == AGGREGATED) {
                                  int aggId = vertex2AggIdView(neigh, 0);
                                  int score = aggWeightView(aggId) - aggPenaltiesView(aggId);

                                  if (score > bestScore) {
                                    bestAggId   = aggId;
                                    bestScore   = score;
                                    bestConnect = connectWeight(neigh);

                                  } else if (aggId == bestAggId && connectWeight(neigh) > bestConnect) {
                                    bestConnect = connectWeight(neigh);
                                  }

                                  // Reset the weights for the next loop
                                  aggWeightView(aggId) = 0;
                                }
                              }

                              if (bestScore >= 0) {
                                d_aggStatView   (i)    = AGGREGATED;
                                vertex2AggIdView(i, 0) = bestAggId;
                                procWinnerView  (i, 0) = myRank;

                                numNonAggregatedNodes--;

                                // This does not protect bestAggId's aggPenalties from being fetched
                                // by another thread before this update happens, it just guarantees
                                // that the update is performed correctly...
                                Kokkos::atomic_add(&aggPenalties(bestAggId), 1);
                                connectWeightView(i) = bestConnect - penaltyConnectWeight;
                              }
                            });
    }
  }

} // end namespace

#endif // HAVE_MUELU_KOKKOS_REFACTOR
#endif // MUELU_AGGREGATIONPHASE2BALGORITHM_KOKKOS_DEF_HPP
