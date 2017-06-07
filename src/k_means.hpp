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

  void InitClusters();

public:

  KMeans(std::vector<PointNd>& inputPoints, int k, int maxIters);

  void run();
  std::vector<PointNd> getClusters() const;
  std::vector<int> getLabels() const;
};
