/*
Copyright (C) 2017 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
*/
#pragma once

#include <vector>

using fptype = double;
using PointNd = std::vector<fptype>;

class KMeans
{
private:

  std::vector<int> mCurrentLabels;
  std::vector<int> mClustersSizes;
  std::vector<PointNd> mCurrentClusters;
  std::vector<fptype> mDistances;
  const std::vector<PointNd>& mDataPoints;
  int mK;
  int mMaxIters;
  int mDataDim;

  void InitClusters();
  int getNearestClusterLabel(const PointNd& point) const;
  void UpdateLabels();
  void UpdateClusters();

public:

  KMeans(const std::vector<PointNd>& inputPoints, int k, int maxIters);

  void run();
  std::vector<PointNd> getClusters() const;
  std::vector<int> getLabels() const;
};
