// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/controllers/bundle_adjustment.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"
#include "colmap/scene/database_cache.h"
#include "colmap/controllers/incremental_pipeline.h"

#include <ceres/ceres.h>

namespace colmap {
namespace {

// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
 public:
  explicit BundleAdjustmentIterationCallback(BaseController* controller)
      : controller_(controller) {}

  virtual ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
    THROW_CHECK_NOTNULL(controller_);
    if (controller_->CheckIfStopped()) {
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    } else {
      return ceres::SOLVER_CONTINUE;
    }
  }

 private:
  BaseController* controller_;
};

}  // namespace

BundleAdjustmentController::BundleAdjustmentController(
    const OptionManager& options,
    std::shared_ptr<Reconstruction> reconstruction)
    : options_(options), reconstruction_(std::move(reconstruction)) {}

void BundleAdjustmentController::Run() {
  THROW_CHECK_NOTNULL(reconstruction_);

  PrintHeading1("Global bundle adjustment");
  Timer run_timer;
  run_timer.Start();

  if (reconstruction_->NumRegImages() < 2) {
    LOG(ERROR) << "Need at least two views.";
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;

  BundleAdjustmentIterationCallback iteration_callback(this);
  ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }
  auto reg_image_ids_it = reconstruction_->RegImageIds().begin();
  ba_config.SetConstantCamPose(*reg_image_ids_it);                // 1st image
  ba_config.SetConstantCamPositions(*(++reg_image_ids_it), {0});  // 2nd image

  // Run bundle adjustment.
  std::unique_ptr<BundleAdjuster> bundle_adjuster = CreateDefaultBundleAdjuster(
      std::move(ba_options), std::move(ba_config), *reconstruction_);
  bundle_adjuster->Solve();

  run_timer.PrintMinutes();
}

void BundleAdjustmentController::RunWithDB() {
  THROW_CHECK_NOTNULL(reconstruction_);

  std::unordered_set<std::string> image_names;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    const auto& image = reconstruction_->Image(image_id);
    image_names.insert(image.Name());
  }

  Database database(*(options_.database_path));
  Timer timer;
  timer.Start();
  const size_t min_num_matches = 0;
  auto database_cache = DatabaseCache::Create(database, min_num_matches, false, image_names);
  timer.PrintMinutes();

  if (database_cache->NumImages() == 0) {
    LOG(WARNING) << "No images with matches found in the database";
    return;
  }

  auto mapper_options = *options_.mapper;

  // If prior positions are to be used and setup from the database, convert
  // geographic coords. to cartesian ones
  if (mapper_options.use_prior_position) {
    database_cache->SetupPosePriors();
  }

  PrintHeading1("Global bundle adjustment");
  Timer run_timer;
  run_timer.Start();

  const std::set<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  if (reg_image_ids.size() < 2) {
    LOG(ERROR) << "Need at least two views.";
    return;
  }

  // Avoid degeneracies in bundle adjustment.
  ObservationManager(*reconstruction_).FilterObservationsWithNegativeDepth();

  BundleAdjustmentOptions custom_ba_options = *options_.bundle_adjustment;
  const size_t kMinNumRegImagesForFastBA = 10;
  if (reg_image_ids.size() < kMinNumRegImagesForFastBA) {
    custom_ba_options.solver_options.function_tolerance /= 10;
    custom_ba_options.solver_options.gradient_tolerance /= 10;
    custom_ba_options.solver_options.parameter_tolerance /= 10;
    custom_ba_options.solver_options.max_num_iterations *= 2;
    custom_ba_options.solver_options.max_linear_solver_iterations = 200;
  }

  BundleAdjustmentIterationCallback iteration_callback(this);
  custom_ba_options.solver_options.callbacks.push_back(&iteration_callback);

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  const bool use_prior_position =
      mapper_options.use_prior_position && reg_image_ids.size() > 2;

  std::unique_ptr<BundleAdjuster> bundle_adjuster;
  if (!use_prior_position) {
    // Fix 7-DOFs of the bundle adjustment problem.
    auto reg_image_ids_it = reg_image_ids.begin();
    ba_config.SetConstantCamPose(*(reg_image_ids_it++));  // 1st image
    ba_config.SetConstantCamPositions(*reg_image_ids_it, {0});  // 2nd image

    bundle_adjuster = CreateDefaultBundleAdjuster(
        std::move(custom_ba_options), std::move(ba_config), *reconstruction_);
  } else {
    PosePriorBundleAdjustmentOptions prior_options;
    prior_options.use_robust_loss_on_prior_position = mapper_options.use_robust_loss_on_prior_position;
    prior_options.prior_position_loss_scale = mapper_options.prior_position_loss_scale;
    bundle_adjuster = CreatePosePriorBundleAdjuster(std::move(custom_ba_options),
                                                    prior_options,
                                                    std::move(ba_config),
                                                    database_cache->PosePriors(),
                                                    *reconstruction_);
  }

  bundle_adjuster->Solve();

  run_timer.PrintMinutes();
}

}  // namespace colmap
