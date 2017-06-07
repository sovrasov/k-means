#include "k_means.hpp"

#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

void makeGaussians(vector<PointNd>& points, const vector<PointNd>& centers, int k, int nSamples,
                  int dimension, vector<fptype> sigmas, int seed = 0)
{
  default_random_engine generator;
  vector<normal_distribution<fptype>> distributions;
  for(int i = 0; i < k; i++)
    distributions.push_back(normal_distribution<fptype>((fptype)0, sigmas[i]));

  for(int i = 0; i < k; i++)
    for(int j = 0; j < nSamples; j++)
    {
      PointNd point(2);
      for(int l = 0; l < dimension; l++)
        point[l] = distributions[l](generator) + centers[i][l];
      points.push_back(point);
    }
}

int main(int argc, const char** argv)
{
  const int K = 2;
  const int dimension = 2;

  vector<PointNd> points;
  vector<PointNd> centers(K);
  centers[0] = {0, 0};
  centers[1] = {3, 4.3};

  makeGaussians(points, centers, K, 10000, dimension, { 1., 1.5 });

  KMeans clusterizer(points, K, 1000);
  float start = omp_get_wtime();
  clusterizer.run();
  float finish = omp_get_wtime();

  auto estimatedCenters = clusterizer.getClusters();
  cout << "Time elapsed: " << finish - start << " seconds" << endl;
  cout << "Estimated centers: " << endl;
  for (const auto& center : estimatedCenters)
  {
    for(size_t i = 0; i < center.size(); i++)
      cout << center[i] << " ";
    cout << endl;
  }

  return 0;
}
