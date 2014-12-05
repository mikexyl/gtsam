/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation, 
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  PreintegratedRotation.h
 *  @author Luca Carlone
 *  @author Stephen Williams
 *  @author Richard Roberts
 *  @author Vadim Indelman
 *  @author David Jensen
 *  @author Frank Dellaert
 **/

#pragma once

#include <gtsam/geometry/Rot3.h>

namespace gtsam {

/**
 * PreintegratedRotation is the base class for all PreintegratedMeasurements
 * classes (in AHRSFactor, ImuFactor, and CombinedImuFactor).
 * It includes the definitions of the preintegrated rotation.
 */
class PreintegratedRotation {

  Rot3 deltaRij_;    ///< Preintegrated relative orientation (in frame i)
  double deltaTij_;  ///< Time interval from i to j

  /// Jacobian of preintegrated rotation w.r.t. angular rate bias
  Matrix3 delRdelBiasOmega_;

public:

  /**
   *  Default constructor, initializes the variables in the base class
   */
  PreintegratedRotation() :
    deltaRij_(Rot3()), deltaTij_(0.0),
    delRdelBiasOmega_(Z_3x3) {}

  /// methods to access class variables
  Matrix3 deltaRij() const {return deltaRij_.matrix();} // expensive
  Vector3 thetaRij() const {return Rot3::Logmap(deltaRij_);} // super-expensive
  const double& deltaTij() const{return deltaTij_;}
  const Matrix3& delRdelBiasOmega() const{ return delRdelBiasOmega_;}

  /// Needed for testable
  void print(const std::string& s) const {
    std::cout << s << std::endl;
    deltaRij_.print("  deltaRij ");
  }

  /// Needed for testable
  bool equals(const PreintegratedRotation& expected, double tol) const {
    return deltaRij_.equals(expected.deltaRij_, tol)
    && fabs(deltaTij_ - expected.deltaTij_) < tol
    && equal_with_abs_tol(delRdelBiasOmega_, expected.delRdelBiasOmega_, tol);
  }

  /// Re-initialize PreintegratedMeasurements
  void resetIntegration(){
    deltaRij_ = Rot3();
    deltaTij_ = 0.0;
    delRdelBiasOmega_ = Z_3x3;
  }

  /// Update preintegrated measurements
  void updateIntegratedRotationAndDeltaT(const Rot3& incrR, double deltaT){
    deltaRij_ = deltaRij_ * incrR;
    deltaTij_ += deltaT;
  }

  /**
   *  Update Jacobians to be used during preintegration
   *  TODO: explain arguments
   */
  void update_delRdelBiasOmega(const Matrix3& Jr_theta_incr, const Rot3& incrR,
      double deltaT) {
    const Matrix3 incrRt = incrR.transpose();
    delRdelBiasOmega_ = incrRt * delRdelBiasOmega_ - Jr_theta_incr * deltaT;
  }

  /// Return a bias corrected version of the integrated rotation - expensive
  Rot3 biascorrectedDeltaRij(const Vector3& biasOmegaIncr) const {
    return deltaRij_*Rot3::Expmap(delRdelBiasOmega_ * biasOmegaIncr);
  }

  /// Get so<3> version of bias corrected rotation, with optional Jacobian
  Vector3 biascorrectedThetaRij(const Vector3& biasOmegaIncr,
      boost::optional<Matrix&> H) const {
    // First, we correct deltaRij using the biasOmegaIncr, rotated
    const Rot3 deltaRij_biascorrected = biascorrectedDeltaRij(biasOmegaIncr);
    // This was done via an expmap, now we go *back* to so<3>
    const Vector3 theta_biascorrected = Rot3::Logmap(deltaRij_biascorrected);
    if (H) {
      // We then do a very expensive Jacobian calculation. TODO Right Duy ?
      const Matrix3 Jrinv_theta_bc = //
          Rot3::rightJacobianExpMapSO3inverse(theta_biascorrected);
      const Matrix3 Jr_JbiasOmegaIncr = //
          Rot3::rightJacobianExpMapSO3(delRdelBiasOmega_ * biasOmegaIncr);
      (*H) = Jrinv_theta_bc * Jr_JbiasOmegaIncr * delRdelBiasOmega_;
    }
    return theta_biascorrected;
  }

private:
  /** Serialization function */
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(deltaRij_);
    ar & BOOST_SERIALIZATION_NVP(deltaTij_);
    ar & BOOST_SERIALIZATION_NVP(delRdelBiasOmega_);
  }
};

} /// namespace gtsam
