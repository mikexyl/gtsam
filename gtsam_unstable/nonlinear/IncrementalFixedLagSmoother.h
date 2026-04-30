/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    IncrementalFixedLagSmoother.h
 * @brief   An iSAM2-compatible fixed-lag smoother.
 *
 * @author  Michael Kaess, Stephen Williams
 * @date    Oct 14, 2012
 */

// \callgraph
#pragma once

#include <gtsam_unstable/nonlinear/FixedLagSmoother.h>
#include <gtsam/base/debug.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <algorithm>
#include <iostream>
#include <set>
#include <utility>

namespace gtsam {

/**
 * This is a base class for the various HMF2 implementations. The HMF2
 * eliminates the factor graph such that the active states are placed in/near
 * the root. This templated implementation accepts any ISAM2-compatible backend.
 */
template <class ISAM2Like>
class GTSAM_UNSTABLE_EXPORT IncrementalFixedLagSmootherT
    : public FixedLagSmoother {
 public:
  /// Typedef for a shared pointer to an Incremental Fixed-Lag Smoother
  typedef boost::shared_ptr<IncrementalFixedLagSmootherT> shared_ptr;

  /** default constructor */
  IncrementalFixedLagSmootherT(
      double smootherLag = 0.0,
      const ISAM2Params& parameters = DefaultISAM2Params())
      : FixedLagSmoother(smootherLag), isam_(parameters) {}

  /** destructor */
  ~IncrementalFixedLagSmootherT() override {}

  /** Print the factor for debugging and testing (implementing Testable) */
  void print(const std::string& s = "IncrementalFixedLagSmoother:\n",
             const KeyFormatter& keyFormatter = DefaultKeyFormatter) const
      override {
    FixedLagSmoother::print(s, keyFormatter);
  }

  /** Check if two IncrementalFixedLagSmoother Objects are equal */
  bool equals(const FixedLagSmoother& rhs, double tol = 1e-9) const override {
    const IncrementalFixedLagSmootherT* e =
        dynamic_cast<const IncrementalFixedLagSmootherT*>(&rhs);
    return e != nullptr && FixedLagSmoother::equals(*e, tol) &&
           isam_.equals(e->isam_, tol);
  }

  /**
   * Add new factors, updating the solution and re-linearizing as needed.
   * @param newFactors new factors on old and/or new variables
   * @param newTheta new values for new variables only
   * @param timestamps an (optional) map from keys to real time stamps
   * @param factorsToRemove an (optional) list of factors to remove.
   */
  Result update(
      const NonlinearFactorGraph& newFactors = NonlinearFactorGraph(),
      const Values& newTheta = Values(),
      const KeyTimestampMap& timestamps = KeyTimestampMap(),
      const FactorIndices& factorsToRemove = FactorIndices()) override {
    const bool debug = ISDEBUG("IncrementalFixedLagSmoother update");

    if (debug) {
      std::cout << "IncrementalFixedLagSmoother::update() Start" << std::endl;
      PrintSymbolicTree(isam_, "Bayes Tree Before Update:");
      std::cout << "END" << std::endl;
    }

    boost::optional<FastMap<Key, int>> constrainedKeys = boost::none;

    // Update the timestamps associated with the factor keys.
    updateKeyTimestampMap(timestamps);

    const double currentTimestamp = getCurrentTimestamp();

    if (debug) {
      std::cout << "Current Timestamp: " << currentTimestamp << std::endl;
    }

    // Find the set of variables to be marginalized out.
    KeyVector marginalizableKeys =
        findKeysBefore(currentTimestamp - smootherLag_);

    if (debug) {
      std::cout << "Marginalizable Keys: ";
      for (Key key : marginalizableKeys) {
        std::cout << DefaultKeyFormatter(key) << " ";
      }
      std::cout << std::endl;
    }

    // Force iSAM2 to put the marginalizable variables at the beginning.
    createOrderingConstraints(marginalizableKeys, constrainedKeys);

    if (debug) {
      std::cout << "Constrained Keys: ";
      if (constrainedKeys) {
        for (FastMap<Key, int>::const_iterator iter =
                 constrainedKeys->begin();
             iter != constrainedKeys->end();
             ++iter) {
          std::cout << DefaultKeyFormatter(iter->first) << "(" << iter->second
                    << ")  ";
        }
      }
      std::cout << std::endl;
    }

    // Mark additional keys between the marginalized keys and the leaves.
    std::set<Key> additionalKeys;
    for (Key key : marginalizableKeys) {
      ISAM2Clique::shared_ptr clique = isam_[key];
      if (!clique) {
        continue;
      }
      for (const ISAM2Clique::shared_ptr& child : clique->children) {
        recursiveMarkAffectedKeys(key, child, additionalKeys);
      }
    }
    KeyList additionalMarkedKeys(additionalKeys.begin(), additionalKeys.end());

    // Update the iSAM2-compatible backend.
    isamResult_ = isam_.update(newFactors,
                               newTheta,
                               factorsToRemove,
                               constrainedKeys,
                               boost::none,
                               additionalMarkedKeys);

    if (debug) {
      PrintSymbolicTree(isam_,
                        "Bayes Tree After Update, Before Marginalization:");
      std::cout << "END" << std::endl;
    }

    // Marginalize out any needed variables.
    if (!marginalizableKeys.empty()) {
      FastList<Key> leafKeys(marginalizableKeys.begin(),
                             marginalizableKeys.end());
      isam_.marginalizeLeaves(leafKeys);
    }

    eraseKeyTimestampMap(marginalizableKeys);

    if (debug) {
      PrintSymbolicTree(isam_, "Final Bayes Tree:");
      std::cout << "END" << std::endl;
    }

    Result result;
    result.iterations = 1;
    result.linearVariables = 0;
    result.nonlinearVariables = 0;
    result.error = 0;

    if (debug) {
      std::cout << "IncrementalFixedLagSmoother::update() Finish" << std::endl;
    }

    return result;
  }

  /**
   * Compute an estimate from the incomplete linear delta computed during the
   * last update.
   */
  Values calculateEstimate() const override { return isam_.calculateEstimate(); }

  /** Compute an estimate for a single variable. */
  template <class VALUE>
  VALUE calculateEstimate(Key key) const {
    return isam_.template calculateEstimate<VALUE>(key);
  }

  /** return the current set of iSAM2 parameters */
  const ISAM2Params& params() const { return isam_.params(); }

  /** Access the current set of factors */
  const NonlinearFactorGraph& getFactors() const {
    return isam_.getFactorsUnsafe();
  }

  /** Access the current linearization point */
  const Values& getLinearizationPoint() const {
    return isam_.getLinearizationPoint();
  }

  /** Access the current set of deltas to the linearization point */
  const VectorValues& getDelta() const { return isam_.getDelta(); }

  /// Calculate marginal covariance on given variable
  Matrix marginalCovariance(Key key) const {
    return isam_.marginalCovariance(key);
  }

  /// Get results of latest isam2 update
  const ISAM2Result& getISAM2Result() const { return isamResult_; }

 protected:
  template <typename... Args>
  IncrementalFixedLagSmootherT(double smootherLag,
                               std::in_place_t,
                               Args&&... args)
      : FixedLagSmoother(smootherLag),
        isam_(std::forward<Args>(args)...) {}

  /** Create default parameters */
  static ISAM2Params DefaultISAM2Params() {
    ISAM2Params params;
    params.findUnusedFactorSlots = true;
    return params;
  }

  /** An iSAM2-compatible object used to perform inference. */
  ISAM2Like isam_;

  /** Store results of latest isam2 update */
  ISAM2Result isamResult_;

  /** Erase any keys associated with timestamps before the provided time */
  void eraseKeysBefore(double timestamp) {
    TimestampKeyMap::iterator end = timestampKeyMap_.lower_bound(timestamp);
    TimestampKeyMap::iterator iter = timestampKeyMap_.begin();
    while (iter != end) {
      keyTimestampMap_.erase(iter->second);
      timestampKeyMap_.erase(iter++);
    }
  }

  /** Fill in constrained keys so marginalized variables eliminate first. */
  void createOrderingConstraints(
      const KeyVector& marginalizableKeys,
      boost::optional<FastMap<Key, int>>& constrainedKeys) const {
    if (!marginalizableKeys.empty()) {
      constrainedKeys = FastMap<Key, int>();
      for (const TimestampKeyMap::value_type& timestampKey :
           timestampKeyMap_) {
        constrainedKeys->operator[](timestampKey.second) = 1;
      }
      for (Key key : marginalizableKeys) {
        constrainedKeys->operator[](key) = 0;
      }
    }
  }

 private:
  static void recursiveMarkAffectedKeys(
      const Key& key,
      const ISAM2Clique::shared_ptr& clique,
      std::set<Key>& additionalKeys) {
    if (std::find(clique->conditional()->beginParents(),
                  clique->conditional()->endParents(),
                  key) != clique->conditional()->endParents()) {
      for (Key i : clique->conditional()->frontals()) {
        additionalKeys.insert(i);
      }

      for (const ISAM2Clique::shared_ptr& child : clique->children) {
        recursiveMarkAffectedKeys(key, child, additionalKeys);
      }
    }
  }

  /** Private methods for printing debug information */
  static void PrintKeySet(const std::set<Key>& keys,
                          const std::string& label = "Keys:") {
    std::cout << label;
    for (Key key : keys) {
      std::cout << " " << DefaultKeyFormatter(key);
    }
    std::cout << std::endl;
  }

  static void PrintSymbolicFactor(const GaussianFactor::shared_ptr& factor) {
    std::cout << "f(";
    for (Key key : factor->keys()) {
      std::cout << " " << DefaultKeyFormatter(key);
    }
    std::cout << " )" << std::endl;
  }

  static void PrintSymbolicGraph(
      const GaussianFactorGraph& graph,
      const std::string& label = "Factor Graph:") {
    std::cout << label << std::endl;
    for (const GaussianFactor::shared_ptr& factor : graph) {
      PrintSymbolicFactor(factor);
    }
  }

  static void PrintSymbolicTree(const ISAM2Like& isam,
                                const std::string& label = "Bayes Tree:") {
    std::cout << label << std::endl;
    if (!isam.roots().empty()) {
      for (const auto& root : isam.roots()) {
        PrintSymbolicTreeHelper(root);
      }
    } else {
      std::cout << "{Empty Tree}" << std::endl;
    }
  }

  static void PrintSymbolicTreeHelper(
      const ISAM2Clique::shared_ptr& clique,
      const std::string indent = "") {
    std::cout << indent << "P( ";
    for (Key key : clique->conditional()->frontals()) {
      std::cout << DefaultKeyFormatter(key) << " ";
    }
    if (clique->conditional()->nrParents() > 0) {
      std::cout << "| ";
    }
    for (Key key : clique->conditional()->parents()) {
      std::cout << DefaultKeyFormatter(key) << " ";
    }
    std::cout << ")" << std::endl;

    for (const ISAM2Clique::shared_ptr& child : clique->children) {
      PrintSymbolicTreeHelper(child, indent + " ");
    }
  }
};

using IncrementalFixedLagSmoother = IncrementalFixedLagSmootherT<ISAM2>;

}  // namespace gtsam
