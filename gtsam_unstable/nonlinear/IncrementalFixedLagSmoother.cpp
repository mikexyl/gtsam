/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    IncrementalFixedLagSmoother.cpp
 * @brief   Explicit instantiation for the default iSAM2 fixed-lag smoother.
 */

#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

namespace gtsam {

template class IncrementalFixedLagSmootherT<ISAM2>;

}  // namespace gtsam
