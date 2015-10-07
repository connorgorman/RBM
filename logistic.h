#include <fstream>
#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <chrono>
#include <time.h>
#include <sstream>
#include <string>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>


#include <iomanip>
#include <assert.h>
#include <execinfo.h>

double l_sigmoid(double x);
double l_logisticCostFunction(Eigen::MatrixXd x_samples, Eigen::MatrixXd y_samples, Eigen::MatrixXd theta);
Eigen::MatrixXd l_vlogisticGradientFunction(Eigen::MatrixXd x_samples, Eigen::MatrixXd y_samples, Eigen::MatrixXd theta);
void l_runLogisticRegression(Eigen::MatrixXd * inputs, Eigen::MatrixXd * labels, Eigen::MatrixXd * theta, int iterations, double alpha_rate, double reg_const);
double getRandom_01();
