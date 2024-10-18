/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    ISAM2Clique.cpp
 * @brief   Specialized iSAM2 Clique
 * @author  Michael Kaess, Richard Roberts, Frank Dellaert
 */

#include "gtsam/linear/GaussianBayesTree.h"
#include <gtsam/inference/BayesTreeCliqueBase-inst.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/linear/linearAlgorithms-inst.h>
#include <gtsam/nonlinear/ISAM2Clique.h>

#include <stack>
#include <utility>

using namespace std;

namespace gtsam {

class GaussianBayesTree;

// Instantiate base class
template class BayesTreeCliqueBase<ISAM2Clique, GaussianFactorGraph>;

/* ************************************************************************* */
void ISAM2Clique::setEliminationResult(
    const FactorGraphType::EliminationResult& eliminationResult) {
  conditional_ = eliminationResult.first;
  cachedFactor_ = eliminationResult.second;
  // Compute gradient contribution
  gradientContribution_.resize(conditional_->cols() - 1);
  // Rewrite -(R * P')'*d   as   -(d' * R * P')'   for computational speed
  // reasons
  gradientContribution_ << -conditional_->R().transpose() *
                               conditional_->d(),
      -conditional_->S().transpose() * conditional_->d();

  unusedTree_ = nullptr;
  reducedGraph_ = FactorGraphType();
}

/* ************************************************************************* */
bool ISAM2Clique::equals(const This& other, double tol) const {
  return Base::equals(other) &&
         ((!cachedFactor_ && !other.cachedFactor_) ||
          (cachedFactor_ && other.cachedFactor_ &&
           cachedFactor_->equals(*other.cachedFactor_, tol)));
}

/* ************************************************************************* */
void ISAM2Clique::print(const string& s, const KeyFormatter& formatter) const {
  Base::print(s, formatter);
  if (cachedFactor_)
    cachedFactor_->print(s + "Cached: ", formatter);
  else
    cout << s << "Cached empty" << endl;
  if (gradientContribution_.rows() != 0)
    gtsam::print(gradientContribution_, "Gradient contribution: ");
}

/* ************************************************************************* */
bool ISAM2Clique::isDirty(const KeySet& replaced, const KeySet& changed) const {
  // if none of the variables in this clique (frontal and separator!) changed
  // significantly, then by the running intersection property, none of the
  // cliques in the children need to be processed

  // Are any clique variables part of the tree that has been redone?
  bool dirty = replaced.exists(conditional_->frontals().front());
#if !defined(NDEBUG) && defined(GTSAM_EXTRA_CONSISTENCY_CHECKS)
  for (Key frontal : conditional_->frontals()) {
    assert(dirty == replaced.exists(frontal));
  }
#endif

  // If not, then has one of the separator variables changed significantly?
  if (!dirty) {
    for (Key parent : conditional_->parents()) {
      if (changed.exists(parent)) {
        dirty = true;
        break;
      }
    }
  }
  return dirty;
}

/* ************************************************************************* */
/**
 * Back-substitute - special version stores solution pointers in cliques for
 * fast access.
 */
void ISAM2Clique::fastBackSubstitute(VectorValues* delta) const {
#ifdef USE_BROKEN_FAST_BACKSUBSTITUTE
  // TODO(gareth): This code shares a lot of logic w/ linearAlgorithms-inst,
  // potentially refactor

  // Create solution part pointers if necessary and possible - necessary if
  // solnPointers_ is empty, and possible if either we're a root, or we have
  // a parent with valid solnPointers_.
  ISAM2Clique::shared_ptr parent = parent_.lock();
  if (solnPointers_.empty() && (isRoot() || !parent->solnPointers_.empty())) {
    for (Key frontal : conditional_->frontals())
      solnPointers_.emplace(frontal, delta->find(frontal));
    for (Key parentKey : conditional_->parents()) {
      assert(parent->solnPointers_.exists(parentKey));
      solnPointers_.emplace(parentKey, parent->solnPointers_.at(parentKey));
    }
  }

  // See if we can use solution part pointers - we can if they either
  // already existed or were created above.
  if (!solnPointers_.empty()) {
    GaussianConditional& c = *conditional_;
    // Solve matrix
    Vector xS;
    {
      // Count dimensions of vector
      DenseIndex dim = 0;
      FastVector<VectorValues::const_iterator> parentPointers;
      parentPointers.reserve(conditional_->nrParents());
      for (Key parent : conditional_->parents()) {
        parentPointers.push_back(solnPointers_.at(parent));
        dim += parentPointers.back()->second.size();
      }

      // Fill parent vector
      xS.resize(dim);
      DenseIndex vectorPos = 0;
      for (const VectorValues::const_iterator& parentPointer : parentPointers) {
        const Vector& parentVector = parentPointer->second;
        xS.block(vectorPos, 0, parentVector.size(), 1) =
            parentVector.block(0, 0, parentVector.size(), 1);
        vectorPos += parentVector.size();
      }
    }

    // NOTE(gareth): We can no longer write: xS = b - S * xS
    // This is because Eigen (as of 3.3) no longer evaluates S * xS into
    // a temporary, and the operation trashes valus in xS.
    // See: http://eigen.tuxfamily.org/index.php?title=3.3
    const Vector rhs = c.getb() - c.S() * xS;
    const Vector solution = c.R().triangularView<Eigen::Upper>().solve(rhs);

    // Check for indeterminant solution
    if (solution.hasNaN())
      throw IndeterminantLinearSystemException(c.keys().front());

    // Insert solution into a VectorValues
    DenseIndex vectorPosition = 0;
    for (GaussianConditional::const_iterator frontal = c.beginFrontals();
         frontal != c.endFrontals(); ++frontal) {
      solnPointers_.at(*frontal)->second =
          solution.segment(vectorPosition, c.getDim(frontal));
      vectorPosition += c.getDim(frontal);
    }
  } else {
    // Just call plain solve because we couldn't use solution pointers.
    delta->update(conditional_->solve(*delta));
  }
#else
  delta->update(conditional_->solve(*delta));
#endif
}

/* ************************************************************************* */
bool ISAM2Clique::valuesChanged(const KeySet& replaced,
                                const Vector& originalValues,
                                const VectorValues& delta,
                                double threshold) const {
  auto frontals = conditional_->frontals();
  if (replaced.exists(frontals.front())) return true;
  Vector diff = originalValues - delta.vector(frontals);
  return diff.lpNorm<Eigen::Infinity>() >= threshold;
}

/* ************************************************************************* */
/// Set changed flag for each frontal variable
void ISAM2Clique::markFrontalsAsChanged(KeySet* changed) const {
  for (Key frontal : conditional_->frontals()) {
    changed->insert(frontal);
  }
}

/* ************************************************************************* */
void ISAM2Clique::restoreFromOriginals(const Vector& originalValues,
                                       VectorValues* delta) const {
  size_t pos = 0;
  for (Key frontal : conditional_->frontals()) {
    auto v = delta->at(frontal);
    v = originalValues.segment(pos, v.size());
    pos += v.size();
  }
}

/* ************************************************************************* */
// Note: not being used right now in favor of non-recursive version below.
void ISAM2Clique::optimizeWildfire(const KeySet& replaced, double threshold,
                                   KeySet* changed, VectorValues* delta,
                                   size_t* count) const {
  if (isDirty(replaced, *changed)) {
    // Temporary copy of the original values, to check how much they change
    auto originalValues = delta->vector(conditional_->frontals());

    // Back-substitute
    fastBackSubstitute(delta);
    count += conditional_->nrFrontals();

    if (valuesChanged(replaced, originalValues, *delta, threshold)) {
      markFrontalsAsChanged(changed);
    } else {
      restoreFromOriginals(originalValues, delta);
    }

    // Recurse to children
    for (const auto& child : children) {
      child->optimizeWildfire(replaced, threshold, changed, delta, count);
    }
  }
}

size_t optimizeWildfire(const ISAM2Clique::shared_ptr& root, double threshold,
                        const KeySet& keys, VectorValues* delta) {
  KeySet changed;
  size_t count = 0;
  // starting from the root, call optimize on each conditional
  if (root) root->optimizeWildfire(keys, threshold, &changed, delta, &count);
  return count;
}

/* ************************************************************************* */
bool ISAM2Clique::optimizeWildfireNode(const KeySet& replaced, double threshold,
                                       KeySet* changed, VectorValues* delta,
                                       size_t* count) const {
  // TODO(gareth): This code shares a lot of logic w/ linearAlgorithms-inst,
  // potentially refactor
  bool dirty = isDirty(replaced, *changed);
  if (dirty) {
    // Temporary copy of the original values, to check how much they change
    auto originalValues = delta->vector(conditional_->frontals());

    // Back-substitute
    fastBackSubstitute(delta);
    *count += conditional_->nrFrontals();

    if (valuesChanged(replaced, originalValues, *delta, threshold)) {
      markFrontalsAsChanged(changed);
    } else {
      restoreFromOriginals(originalValues, delta);
    }
  }

  return dirty;
}

size_t optimizeWildfireNonRecursive(const ISAM2Clique::shared_ptr& root,
                                    double threshold, const KeySet& keys,
                                    VectorValues* delta) {
  KeySet changed;
  size_t count = 0;

  if (root) {
    std::stack<ISAM2Clique::shared_ptr> travStack;
    travStack.push(root);
    ISAM2Clique::shared_ptr currentNode = root;
    while (!travStack.empty()) {
      currentNode = travStack.top();
      travStack.pop();
      bool dirty = currentNode->optimizeWildfireNode(keys, threshold, &changed,
                                                     delta, &count);
      if (dirty) {
        for (const auto& child : currentNode->children) {
          travStack.push(child);
        }
      }
    }
  }

  return count;
}

/* ************************************************************************* */
void ISAM2Clique::nnz_internal(size_t* result) const {
  size_t dimR = conditional_->rows();
  size_t dimSep = conditional_->S().cols();
  *result += ((dimR + 1) * dimR) / 2 + dimSep * dimR;
  // traverse the children
  for (const auto& child : children) {
    child->nnz_internal(result);
  }
}

/* ************************************************************************* */
size_t ISAM2Clique::calculate_nnz() const {
  size_t result = 0;
  nnz_internal(&result);
  return result;
}

/* ************************************************************************* */
void ISAM2Clique::findAll(const KeySet& markedMask, KeySet* keys) const {
  static const bool debug = false;
  // does the separator contain any of the variables?
  bool found = false;
  for (Key key : conditional_->parents()) {
    if (markedMask.exists(key)) {
      found = true;
      break;
    }
  }
  if (found) {
    // then add this clique
    keys->insert(conditional_->beginFrontals(), conditional_->endFrontals());
    if (debug) print("Key(s) marked in clique ");
    if (debug) cout << "so marking key " << conditional_->front() << endl;
  }
  for (const auto& child : children) {
    child->findAll(markedMask, keys);
  }
}

/* ************************************************************************* */
void ISAM2Clique::addGradientAtZero(VectorValues* g) const {
  // Loop through variables in each clique, adding contributions
  DenseIndex position = 0;
  for (auto it = conditional_->begin(); it != conditional_->end(); ++it) {
    const DenseIndex dim = conditional_->getDim(it);
    const Vector contribution = gradientContribution_.segment(position, dim);
    VectorValues::iterator values_it;
    bool success;
    std::tie(values_it, success) = g->tryInsert(*it, contribution);
    if (!success) values_it->second += contribution;
    position += dim;
  }

  // Recursively add contributions from children
  for (const auto& child : children) {
    child->addGradientAtZero(g);
  }
}

ISAM2Clique::FactorGraphType
ISAM2Clique::separatorMarginal(Eliminate function) const {
  std::lock_guard<std::mutex> marginalLock(cachedSeparatorMarginalMutex_);
  gttic(BayesTreeCliqueBase_separatorMarginal);
  // Check if the Separator marginal was already calculated
  if (!cachedSeparatorMarginal_) {
    gttic(BayesTreeCliqueBase_separatorMarginal_cachemiss);

    // If this is the root, there is no separator
    if (parent_.expired() /*(if we're the root)*/) {
      // we are root, return empty
      FactorGraphType empty;
      cachedSeparatorMarginal_ = empty;
    } else {
      derived_ptr parent(parent_.lock());

      // The variables we want to keepSet are exactly the ones in S
      KeyVector indicesS(this->conditional()->beginParents(),
                         this->conditional()->endParents());

      if (parent->reducedGraph_.size()) {
        FactorGraphType siblingsSeparatorMarginal(parent->reducedGraph_);
        KeyVector indicesSp, indicesSf;
        for (auto key : indicesS) {
          if (siblingsSeparatorMarginal.keys().exists(key)) {
            indicesSp.push_back(key);
          } else {
            indicesSf.push_back(key);
          }
        }

        auto separatorMarginal_Sp =
            siblingsSeparatorMarginal.marginalMultifrontalBayesNet(
                Ordering(indicesSp), function);

        if (parent->unusedTree_) {
          for (auto clique : parent->unusedTree_->nodes()) {
            siblingsSeparatorMarginal.push_back(clique.second->conditional());
            break;
          }
        }

        gttic(BayesTreeCliqueBase_separatorMarginal_incremental);
        boost::shared_ptr<BayesNetType> separatorMarginalInSiblings;
        try {
          separatorMarginalInSiblings =
              siblingsSeparatorMarginal.marginalMultifrontalBayesNet(
                  Ordering(indicesSf), function, boost::none,
                  &parent->unusedTree_);
        } catch (std::exception &e) {
          std::cout << "Error in marginalMultifrontalBayesNet: " << e.what()
                    << std::endl;
          std::cout << "keys to marginalize: ";
          for (auto key : indicesS) {
            std::cout << DefaultKeyFormatter(key) << " ";
          }
          std::cout << std::endl;

          for (auto factor : siblingsSeparatorMarginal) {
            if (auto conditional =
                    boost::dynamic_pointer_cast<ConditionalType>(factor)) {
              conditional->ConditionalType::BaseConditional::print();
            } else {
              factor->print();
            }
          }
          throw;
        }
        gttoc(BayesTreeCliqueBase_separatorMarginal_incremental);

        separatorMarginalInSiblings->push_back(*separatorMarginal_Sp);
        cachedSeparatorMarginal_.reset(*separatorMarginalInSiblings);
        parent->reducedGraph_ = *separatorMarginalInSiblings;

        // for (auto factor : *separatorMarginalInSiblings) {
        //   if (auto conditional =
        //           boost::dynamic_pointer_cast<ConditionalType>(factor)) {
        //     conditional->ConditionalType::BaseConditional::print("p_sib(S):
        //     ");
        //   } else {
        //     factor->print();
        //   }
        // }
      } else {
        // Flatten recursion in timing outline
        gttoc(BayesTreeCliqueBase_separatorMarginal_cachemiss);
        gttoc(BayesTreeCliqueBase_separatorMarginal);

        // Obtain P(S) = \int P(Cp) = \int P(Fp|Sp) P(Sp)
        // initialize P(Cp) with the parent separator marginal
        FactorGraphType p_Cp(parent->separatorMarginal(function)); // P(Sp)

        gttic(BayesTreeCliqueBase_separatorMarginal);
        gttic(BayesTreeCliqueBase_separatorMarginal_cachemiss);

        // std::cout << "p(S_parent): " << std::endl;
        // for (auto factor : p_Cp) {
        //   if (auto conditional =
        //           boost::dynamic_pointer_cast<ConditionalType>(factor)) {
        //     conditional->ConditionalType::BaseConditional::print();
        //   } else {
        //     factor->print();
        //   }
        // }

        // now add the parent conditional
        p_Cp += parent->conditional_; // P(Fp|Sp)
        // parent->conditional_->ConditionalType::BaseConditional::print(
        //     "p(F_parent|S_parent): ");

        // this->conditional()->ConditionalType::BaseConditional::print(
        //     "p(F|S): ");

        // std::cout << "keys in p_Cp" << std::endl;
        // for (auto key : p_Cp.keys()) {
        //   std::cout << DefaultKeyFormatter(key) << " ";
        // }
        // std::cout << std::endl;

        auto separatorMarginal = p_Cp.marginalMultifrontalBayesNet(
            Ordering(indicesS), function, boost::none, &parent->unusedTree_);
        cachedSeparatorMarginal_.reset(*separatorMarginal);
        parent->reducedGraph_ = *separatorMarginal;
      }
    }
  }

  // return the shortcut P(S||B)
  return *cachedSeparatorMarginal_; // return the cached version
}

ISAM2Clique::FactorGraphType ISAM2Clique::marginal2(Eliminate function) const {
  if(this->reducedGraph_.size()) {
    FactorGraphType p_C(this->reducedGraph_);
    if(this->unusedTree_) {
      for (auto clique : this->unusedTree_->nodes()) {
        p_C.push_back(clique.second->conditional());
        break;
      }
    }
    return p_C;
  }

  gttic(BayesTreeCliqueBase_marginal2);
  // initialize with separator marginal P(S)
  FactorGraphType p_C = this->separatorMarginal(function);
  // add the conditional P(F|S)
  p_C += boost::shared_ptr<FactorType>(this->conditional_);
  return p_C;
}

/* ************************************************************************* */
}  // namespace gtsam
