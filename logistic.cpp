#include "logistic.h"

/*** GLOBALS ***/

boost::random::mt19937 gen;

/*** END GLOBALS ***/

double l_sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double l_logisticCostFunction(Eigen::MatrixXd x_samples, Eigen::MatrixXd y_samples, Eigen::MatrixXd theta)
{
	double sum = 0.0;
	int m = x_samples.cols();

	for(int i = 0; i < m; i++)
	{
		Eigen::MatrixXd input = x_samples.col(i);

		unsigned char y = y_samples(i);

		double scalar = (theta.transpose() * input)(0);

		//printf("Scalar %f \n", scalar);

		double h_x = l_sigmoid(  scalar  );

		// printf("Sigmoid: %f \n", h_x);

		double first_log = log( h_x );
		double second_log = log( 1 - h_x );

		sum += ( y * first_log ) + (1 - y) * second_log;
	}

	sum = sum * (-1.0 / m);

	return sum;
}

Eigen::MatrixXd l_vlogisticGradientFunction(Eigen::MatrixXd x_samples, Eigen::MatrixXd y_samples, Eigen::MatrixXd theta)
{
	Eigen::MatrixXd gradient;
	
	int m = x_samples.cols();

	Eigen::MatrixXd sigmoid_matrix;
	sigmoid_matrix.resize(y_samples.rows(), 1);

	Eigen::MatrixXd sub_matrix;
	sub_matrix.resize(y_samples.rows(),1);

	for(int i = 0; i < m; i++)
	{
		double temp = (theta.transpose() * x_samples.col(i))(0);
		sigmoid_matrix(i, 0) = l_sigmoid( temp );
	}

	sub_matrix = sigmoid_matrix - y_samples;

	gradient = ( x_samples * sub_matrix );

	return gradient / m;
}

void l_runLogisticRegression(Eigen::MatrixXd * inputs, Eigen::MatrixXd * labels, Eigen::MatrixXd * theta, int iterations, double alpha_rate, double reg_const)
{

	int div_iter = iterations / 10;
	if(div_iter == 0)
		div_iter = 1;

	for( int i = 0; i < iterations; i++ )
	{

		if( i % div_iter == 0){
			double cost = l_logisticCostFunction(*inputs, *labels, *theta);
			printf("Cost on iteration %d: %f \n", i, cost);
		}

		Eigen::MatrixXd gradient = l_vlogisticGradientFunction(*inputs, *labels, *theta);

		//std::cout << "GRADIENT \n" << gradient << "\n" << std::endl;

		(*theta) = (*theta) - alpha_rate * gradient;
	}
}

double getRandom_01()
{
	double r = ((double) rand() / (RAND_MAX)) + 1;
	return r;
}

