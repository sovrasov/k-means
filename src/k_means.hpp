/*
Copyright (C) 2017 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
*/
#pragma once

#include <vector>

namespace km
{
  using fptype = double;
  using PointNd = std::vector<fptype>;
  enum class InitMethod {RANDOM, PP};

  class KMeans
  {
  private:

    std::vector<int> mCurrentLabels;
    std::vector<int> mClustersSizes;
    std::vector<PointNd> mCurrentClusters;
    std::vector<PointNd> mPreviousClusters;
    const std::vector<PointNd>& mDataPoints;

    int mK;
    int mMaxIters;
    int mDataDim;
    fptype mEps;
    InitMethod mInitMet;

    void InitClusters();
    int getNearestClusterLabel(const PointNd& point) const;
    void UpdateLabels();
    void UpdateClusters();
    bool CheckStopCondition() const;

  public:

    KMeans(const std::vector<PointNd>& inputPoints, int k,
      int maxIters = 50, InitMethod method = InitMethod::RANDOM, fptype eps = 0);

    void run();
    std::vector<PointNd> getClusters() const;
    std::vector<int> getLabels() const;
  };
}
