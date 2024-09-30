/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2022, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   HybridBayesNet.cpp
 * @brief  A Bayes net of Gaussian Conditionals indexed by discrete keys.
 * @author Fan Jiang
 * @author Varun Agrawal
 * @author Shangjie Xue
 * @author Frank Dellaert
 * @date   January 2022
 */

#include <gtsam/discrete/DiscreteBayesNet.h>
#include <gtsam/discrete/DiscreteFactorGraph.h>
#include <gtsam/hybrid/HybridBayesNet.h>
#include <gtsam/hybrid/HybridValues.h>

// In Wrappers we have no access to this so have a default ready
static std::mt19937_64 kRandomNumberGenerator(42);

namespace gtsam {

/* ************************************************************************* */
void HybridBayesNet::print(const std::string &s,
                           const KeyFormatter &formatter) const {
  Base::print(s, formatter);
}

/* ************************************************************************* */
bool HybridBayesNet::equals(const This &bn, double tol) const {
  return Base::equals(bn, tol);
}

/* ************************************************************************* */
DecisionTreeFactor HybridBayesNet::pruneDiscreteConditionals(
    size_t maxNrLeaves) {
  // Get the joint distribution of only the discrete keys
  // The joint discrete probability.
  DiscreteConditional discreteProbs;

  std::vector<size_t> discrete_factor_idxs;
  // Record frontal keys so we can maintain ordering
  Ordering discrete_frontals;

  for (size_t i = 0; i < this->size(); i++) {
    auto conditional = this->at(i);
    if (conditional->isDiscrete()) {
      discreteProbs = discreteProbs * (*conditional->asDiscrete());

      Ordering conditional_keys(conditional->frontals());
      discrete_frontals += conditional_keys;
      discrete_factor_idxs.push_back(i);
    }
  }

  const DecisionTreeFactor prunedDiscreteProbs =
      discreteProbs.prune(maxNrLeaves);

  // Eliminate joint probability back into conditionals
  DiscreteFactorGraph dfg{prunedDiscreteProbs};
  DiscreteBayesNet::shared_ptr dbn = dfg.eliminateSequential(discrete_frontals);

  // Assign pruned discrete conditionals back at the correct indices.
  for (size_t i = 0; i < discrete_factor_idxs.size(); i++) {
    size_t idx = discrete_factor_idxs.at(i);
    this->at(idx) = std::make_shared<HybridConditional>(dbn->at(i));
  }

  return prunedDiscreteProbs;
}

/* ************************************************************************* */
HybridBayesNet HybridBayesNet::prune(size_t maxNrLeaves) const {
  HybridBayesNet copy(*this);
  DecisionTreeFactor prunedDiscreteProbs =
      copy.pruneDiscreteConditionals(maxNrLeaves);

  /* To prune, we visitWith every leaf in the HybridGaussianConditional.
   * For each leaf, using the assignment we can check the discrete decision tree
   * for 0.0 probability, then just set the leaf to a nullptr.
   *
   * We can later check the HybridGaussianConditional for just nullptrs.
   */

  HybridBayesNet prunedBayesNetFragment;

  // Go through all the conditionals in the
  // Bayes Net and prune them as per prunedDiscreteProbs.
  for (auto &&conditional : copy) {
    if (auto gm = conditional->asHybrid()) {
      // Make a copy of the hybrid Gaussian conditional and prune it!
      auto prunedHybridGaussianConditional = gm->prune(prunedDiscreteProbs);

      // Type-erase and add to the pruned Bayes Net fragment.
      prunedBayesNetFragment.push_back(prunedHybridGaussianConditional);

    } else {
      // Add the non-HybridGaussianConditional conditional
      prunedBayesNetFragment.push_back(conditional);
    }
  }

  return prunedBayesNetFragment;
}

/* ************************************************************************* */
GaussianBayesNet HybridBayesNet::choose(
    const DiscreteValues &assignment) const {
  GaussianBayesNet gbn;
  for (auto &&conditional : *this) {
    if (auto gm = conditional->asHybrid()) {
      // If conditional is hybrid, select based on assignment.
      gbn.push_back(gm->choose(assignment));
    } else if (auto gc = conditional->asGaussian()) {
      // If continuous only, add Gaussian conditional.
      gbn.push_back(gc);
    } else if (auto dc = conditional->asDiscrete()) {
      // If conditional is discrete-only, we simply continue.
      continue;
    }
  }

  return gbn;
}

/* ************************************************************************* */
HybridValues HybridBayesNet::optimize() const {
  // Collect all the discrete factors to compute MPE
  DiscreteFactorGraph discrete_fg;

  for (auto &&conditional : *this) {
    if (conditional->isDiscrete()) {
      discrete_fg.push_back(conditional->asDiscrete());
    }
  }

  // Solve for the MPE
  DiscreteValues mpe = discrete_fg.optimize();

  // Given the MPE, compute the optimal continuous values.
  return HybridValues(optimize(mpe), mpe);
}

/* ************************************************************************* */
VectorValues HybridBayesNet::optimize(const DiscreteValues &assignment) const {
  GaussianBayesNet gbn = choose(assignment);

  // Check if there exists a nullptr in the GaussianBayesNet
  // If yes, return an empty VectorValues
  if (std::find(gbn.begin(), gbn.end(), nullptr) != gbn.end()) {
    return VectorValues();
  }
  return gbn.optimize();
}

/* ************************************************************************* */
HybridValues HybridBayesNet::sample(const HybridValues &given,
                                    std::mt19937_64 *rng) const {
  DiscreteBayesNet dbn;
  for (auto &&conditional : *this) {
    if (conditional->isDiscrete()) {
      // If conditional is discrete-only, we add to the discrete Bayes net.
      dbn.push_back(conditional->asDiscrete());
    }
  }
  // Sample a discrete assignment.
  const DiscreteValues assignment = dbn.sample(given.discrete());
  // Select the continuous Bayes net corresponding to the assignment.
  GaussianBayesNet gbn = choose(assignment);
  // Sample from the Gaussian Bayes net.
  VectorValues sample = gbn.sample(given.continuous(), rng);
  return {sample, assignment};
}

/* ************************************************************************* */
HybridValues HybridBayesNet::sample(std::mt19937_64 *rng) const {
  HybridValues given;
  return sample(given, rng);
}

/* ************************************************************************* */
HybridValues HybridBayesNet::sample(const HybridValues &given) const {
  return sample(given, &kRandomNumberGenerator);
}

/* ************************************************************************* */
HybridValues HybridBayesNet::sample() const {
  return sample(&kRandomNumberGenerator);
}

/* ************************************************************************* */
AlgebraicDecisionTree<Key> HybridBayesNet::errorTree(
    const VectorValues &continuousValues) const {
  AlgebraicDecisionTree<Key> result(0.0);

  // Iterate over each conditional.
  for (auto &&conditional : *this) {
    result = result + conditional->errorTree(continuousValues);
  }

  return result;
}

/* ************************************************************************* */
AlgebraicDecisionTree<Key> HybridBayesNet::discretePosterior(
    const VectorValues &continuousValues) const {
  AlgebraicDecisionTree<Key> errors = this->errorTree(continuousValues);
  AlgebraicDecisionTree<Key> p =
      errors.apply([](double error) { return exp(-error); });
  return p / p.sum();
}

/* ************************************************************************* */
double HybridBayesNet::evaluate(const HybridValues &values) const {
  return exp(logProbability(values));
}

/* ************************************************************************* */
HybridGaussianFactorGraph HybridBayesNet::toFactorGraph(
    const VectorValues &measurements) const {
  HybridGaussianFactorGraph fg;

  // For all nodes in the Bayes net, if its frontal variable is in measurements,
  // replace it by a likelihood factor:
  for (auto &&conditional : *this) {
    if (conditional->frontalsIn(measurements)) {
      if (auto gc = conditional->asGaussian()) {
        fg.push_back(gc->likelihood(measurements));
      } else if (auto gm = conditional->asHybrid()) {
        fg.push_back(gm->likelihood(measurements));
      } else {
        throw std::runtime_error("Unknown conditional type");
      }
    } else {
      fg.push_back(conditional);
    }
  }
  return fg;
}

}  // namespace gtsam
