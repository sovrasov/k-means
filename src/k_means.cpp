/*
Copyright (C) 2017 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
*/
#include "k_means.hpp"

#include <algorithm>
#include <random>
#include <limits>

using namespace std;
using namespace km;

namespace
{
  fptype euclideanDistanceSQR(const PointNd& p1, const PointNd& p2)
  {
    size_t dim = p1.size();
    const fptype* data1 = p1.data();
    const fptype* data2 = p2.data();
    fptype dist = 0;
    for(size_t i = 0; i < dim; i++)
      dist += (data1[i] - data2[i])*(data1[i] - data2[i]);

    return dist;
  }
  void addVectors(const PointNd& p1, const PointNd& p2, PointNd& dst)
  {
    size_t dim = p1.size();
    const fptype* data1 = p1.data();
    const fptype* data2 = p2.data();
    fptype* dataDst = dst.data();
    for(size_t i = 0; i < dim; i++)
      dataDst[i] = data1[i] + data2[i];
  }
  void scaleVector(PointNd& p1, fptype scale)
  {
    size_t dim = p1.size();
    fptype* data1 = p1.data();
    for(size_t i = 0; i < dim; i++)
      data1[i] *= scale;
  }
}

KMeans::KMeans(const vector<PointNd>& inputPoints, int k,
  int maxIters, InitMethod method, fptype eps) :
  mDataPoints(inputPoints), mK(k), mMaxIters(maxIters), mEps(eps), mInitMet(method)
{
  if(!inputPoints.empty())
    mDataDim = inputPoints[0].size();
}

void KMeans::run()
{
  mCurrentLabels.resize(mDataPoints.size());
  mClustersSizes.resize(mK);

  int itersCounter = 0;

  InitClusters();
  UpdateLabels();

  do
  {
    swap(mCurrentClusters, mPreviousClusters);
    UpdateClusters();
    UpdateLabels();
    itersCounter++;
  }
  while(itersCounter < mMaxIters && !CheckStopCondition());
}

void KMeans::InitClusters()
{
  default_random_engine generator;
  uniform_int_distribution<> dis(0, mDataPoints.size());
  mCurrentClusters.resize(mK);
  mPreviousClusters.resize(mK);

  if(mInitMet == InitMethod::RANDOM)
  {
    for(int i = 0; i < mK; i++)
    {
      mCurrentClusters[i] = mDataPoints[dis(generator)];
      mPreviousClusters[i] = mCurrentClusters[i];
    }
  }
  else
  {
    uniform_real_distribution<fptype> real_dis(0.0, 1.0);
    mCurrentClusters[0] = mPreviousClusters[0] = mDataPoints[dis(generator)];
    vector<fptype> distances(mDataPoints.size());

    for(int k = 1; k < mK; k++)
    {
      fptype distSum = 0;
      for(size_t i = 0; i < mDataPoints.size(); i++)
      {
        fptype minDist = numeric_limits<fptype>::max();
        for(int j = 0; j < k; j++)
          minDist = min(minDist, euclideanDistanceSQR(mDataPoints[i], mCurrentClusters[j]));
        distances[i] = minDist;
        distSum += minDist;
      }
      fptype splitSample = distSum*real_dis(generator);
      distSum = 0;
      size_t i;
      for(i = 0; i < distances.size(); i++)
      {
        distSum += distances[i];
        if(distSum >= splitSample)
          break;
      }
      mCurrentClusters[k] = mPreviousClusters[k] = mDataPoints[i];
    }
  }
}

void KMeans::UpdateClusters()
{
  fill(mClustersSizes.begin(), mClustersSizes.end(), 0);
  for(int k = 0; k < mK; k++)
    fill(mCurrentClusters[k].begin(), mCurrentClusters[k].end(), 0);

  for(size_t i = 0; i < mCurrentLabels.size(); i++)
  {
    int clusterLbl = mCurrentLabels[i];
    mClustersSizes[clusterLbl]++;
    addVectors(mCurrentClusters[clusterLbl], mDataPoints[i], mCurrentClusters[clusterLbl]);
  }

  for(int k = 0; k < mK; k++)
    if(mClustersSizes[k] != 0)
      scaleVector(mCurrentClusters[k], (fptype)1. / mClustersSizes[k]);
}

void KMeans::UpdateLabels()
{
#pragma omp parallel for
  for(size_t i = 0; i < mDataPoints.size(); i++)
    mCurrentLabels[i] = getNearestClusterLabel(mDataPoints[i]);
}

int KMeans::getNearestClusterLabel(const PointNd& point) const
{
  int label;
  fptype minDist = numeric_limits<fptype>::max();
  for(int i = 0; i < mK; i++)
  {
    fptype currentDist = euclideanDistanceSQR(point, mCurrentClusters[i]);
    if(currentDist < minDist)
    {
      minDist = currentDist;
      label = i;
    }
  }

  return label;
}

bool KMeans::CheckStopCondition() const
{
  fptype maxDist = 0;
  for(int k = 0; k < mK; k++)
    maxDist = max(euclideanDistanceSQR(mCurrentClusters[k], mPreviousClusters[k]), maxDist);

  if(maxDist <= mEps)
    return true;
  return false;
}

vector<PointNd> KMeans::getClusters() const
{
  return mCurrentClusters;
}

vector<int> KMeans::getLabels() const
{
  return mCurrentLabels;
}
