#include <fstream>
#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <chrono>
#include <time.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_01.hpp>


boost::random::mt19937 gen;

const img_width = 28;
const img_height = 28;

const num_input = img_width * img_height;

const num_hidden = 100;

const alpha_rate = 0.01;
const gibbs_iterations = 1;

const batch_size = 20;
const num_batches = 3000;

const data_passes = 1;


Eigen::Matrix<double, num_hidden, num_input> WeightMatrix;
Eigen::Vector<double, num_hidden, 1> HiddenBiasVector;
Eigen::Vector<double, num_input, 1> InputBiasVector;
Eigen::Vector<double, num_hidden, 1> HiddenNodes;
Eigen::Vector<double, num_input, 1> InputNodes;


struct RBM
{

	WeightMatrix weight_matrix;
	HiddenBiasVector b_bias;
	InputBiasVector c_bias;

	HiddenNodes gibbs_hidden;
	InputNodes gibbs_input;

	InputNodes actual_input;

	WeightMatrix weight_gradients;
	HiddenBiasVector b_bias_gradients;
	InputBiasVector c_bias_gradients;

};



double sigmoid(double value)
{
	return 1.0 / ( 1.0 + exp( -value ) );
}

Eigen::MatrixXd * matrix_sigmoid(Eigen::MatrixXd * input, Eigen::MatrixXd * b_bias, Eigen::MatrixXd * weight_matrix )
{

	Eigen::MatrixXd * sigmoid_mat = new Eigen::MatrixXd;
	sigmoid_mat->resize(b_bias->rows(), 1);

	*sigmoid_mat = (*b_bias) + (*weight_matrix) * (*input);

	for(int i = 0; i < b_bias->rows(); i ++)
	{
		(*sigmoid_mat)(i, 0) = sigmoid( (*sigmoid_mat)(i,0));
	}

	return sigmoid_mat;
}

unsigned int calculateInteger(char * buf)
{
	unsigned int integer = 0;

	int count = 0;
	for(int i = 3; i >=0; i--)
	{
		int multiplier = pow(16, count);
		integer += multiplier * (unsigned int) ( (unsigned char) buf[i] );
		count+=2;
	}

	return integer;
}

unsigned char calculatePixel(char * buf)
{

	unsigned char pixel = 0;
	pixel = (unsigned char) buf[0];
	return pixel;
}

std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
}

Eigen::MatrixXd * gibbs_sampling(Eigen::MatrixXd * input, Eigen::MatrixXd * b_bias,  Eigen::MatrixXd * c_bias,  Eigen::MatrixXd * weight_matrix, unsigned int num_hidden, int num_steps)
{

	Eigen::MatrixXd * x_tild = new Eigen::MatrixXd;

	unsigned int num_input = input->rows();

	x_tild->resize(input->rows(), input->cols());

	for(int i = 0; i < input->rows(); i++)
	{
		(*x_tild)(i, 0) = (*input)(i, 0);
	}

	Eigen::MatrixXd * hidden = new Eigen::MatrixXd;
	hidden->resize(num_hidden, 1);

	boost::random::uniform_01<> uniform;

	for(int k = 0; k < num_steps; k++)
	{

		for(unsigned int i = 0; i < num_hidden; i++)
		{
			double scalar = ( (*weight_matrix).row(i) * (*x_tild))(0);
			double prob = sigmoid( (*b_bias)(i) + scalar );
			double draw = uniform(gen);

			if( draw < prob )
			{
				(*hidden)(i, 0) = 1;
			}
			else
			{
				(*hidden)(i, 0) = 0;
			}
		}

		// for(int i = 0; i < 15; i++)
		// {
		// 	printf("Hidden %d: %f \n", i, (*hidden)(i,0));
		// }

		for(unsigned int i = 0; i < num_input; i++)
		{

			double scalar = ( (*hidden).transpose() * (*weight_matrix).col(i) )(0);
			double prob = sigmoid ( (*c_bias)(i) + scalar);

			double draw = uniform(gen);

			if( draw < prob )
			{
				(*x_tild)(i) = 1;
			}
			else
			{
				(*x_tild)(i) = 0;
			}
		}

		// for(int i = 0; i < 15; i++)
		// {
		// 	printf("Observed %d: %f \n", i, (*x_tild)(i,0));
		// }

	}

	return x_tild;
}

struct RBM * loadData(char * filename)
{

	std::ifstream f;

	char * int_buf = new char[4];
	char * pixel_buf = new char;

	const char * filename = "train-images.idx3-ubyte";

	f.open(filename);

	f.read(int_buf, 4);
	unsigned int magic = calculateInteger( int_buf );
	
	f.read(int_buf, 4);
	unsigned int filesize = calculateInteger( int_buf );
	
	f.read(int_buf, 4);
	unsigned int height = calculateInteger( int_buf );
	
	f.read(int_buf, 4);
	unsigned int width = calculateInteger( int_buf );

	printf("Magic: %u, Filesize: %u, width: %u, height: %u \n", magic, filesize, width, height);

	struct RBM rbm = new RBM;

	// std::vector< Eigen::MatrixXd * > input_vectors;

	// for(unsigned int i = 0; i < filesize; i++)
	// {
	// 	Eigen::MatrixXd * input = new Eigen::MatrixXd;
	// 	(*input).resize(num_input, 1);

	// 	for(unsigned int j = 0; j < height; j++)
	// 	{
	// 		for( unsigned int k = 0; k < width; k++ )
	// 		{
	// 			f.read(pixel_buf, 1);
	// 			unsigned char pixel = (unsigned char) calculatePixel(pixel_buf);

	// 			if( pixel > 128 )
	// 				(*input)(j * 28 + k, 0) = 1;
	// 			else
	// 				(*input)(j * 28 + k, 0) = 0;
	// 		}
	// 	}
	// 	input_vectors.push_back(input);
	// }


}

int main(int argc, char * argv[])
{
	boost::random::uniform_01<> uniform;

	

	// for(unsigned int i = 0; i < input_vectors.size(); i++)
	// {
	// 	for(unsigned int j = 0; j < height; j++)
	// 	{
	// 		for(unsigned int k = 0; k < width; k++ )
	// 		{
	// 			Eigen::MatrixXd temp_vector = *( input_vectors[i] );
	// 			unsigned char pixel = temp_vector(j * width + k);

	// 			if( pixel == 1 )
	// 				printf("|");
	// 			else
	// 				printf("-");
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("\n");
	// }

	unsigned int num_hidden = 100;

 	//std::normal_distribution<double> normal_distribution(0.0, 10.0);

 	double lower_bound = -4.0 * sqrt(6.0 / (num_input + num_hidden) );
 	double upper_bound = -lower_bound;

 	printf("lower: %f \n", lower_bound);
 	printf("upper: %f \n", upper_bound);

 	boost::uniform_real<> uniform_distribution(lower_bound, upper_bound);

	Eigen::MatrixXd * weight_matrix = new Eigen::MatrixXd;
	(*weight_matrix).resize(num_hidden, num_input);

	for(unsigned int i = 0; i < num_hidden; i++)
	{
		for(unsigned int j = 0; j < num_input; j++)
		{
			(*weight_matrix)(i,j) = uniform_distribution(gen);
		}
	}

	for(int i = 0; i < 15; i++)
	{

		printf("Row %d: %f \n", i, (*weight_matrix)(0,i) );
	}


	Eigen::MatrixXd * b_bias = new Eigen::MatrixXd;
	(*b_bias).resize(num_hidden, 1);
	
	for(unsigned int i = 0; i < num_hidden; i++)
	{
		(*b_bias)(i,0) = 0;
	}


	Eigen::MatrixXd * c_bias = new Eigen::MatrixXd;
	(*c_bias).resize(num_input, 1);
	
	for(unsigned int i = 0; i < num_input; i++)
	{
		(*c_bias)(i,0) = 0;
	}

	Eigen::MatrixXd * hidden_nodes = new Eigen::MatrixXd;
	(*hidden_nodes).resize(num_hidden, 1);
	// Hidden nodes will be initialized by gibbs sampling

	printf("Sigmoid of .5: %f \n", sigmoid(0));

	//int batch_size = 20;

	//int num_batches = input_vectors.size() / batch_size;

	double alpha = .01;

	int batch_size = 20;
	int num_batches = 3000;

	for(int y = 0; y < 10; y++){

	for(int i = 0; i < num_batches; i++)
	{   

		Eigen::MatrixXd * weight_gradients = new Eigen::MatrixXd;
		weight_gradients->resize( weight_matrix->rows(), weight_matrix->cols());

		for(int j = 0; j < weight_matrix->rows(); j++)
		{
			for(int k = 0; k < weight_matrix->cols(); k++)
			{
				(*weight_gradients)(j,k) = 0;
			}
		}

		Eigen::MatrixXd * b_bias_gradients = new Eigen::MatrixXd;
		b_bias_gradients->resize( b_bias->rows(), b_bias->cols() );

		for(int j = 0; j < b_bias->rows(); j++)
		{
			(*b_bias_gradients)(j, 0) = 0;
		}

		Eigen::MatrixXd * c_bias_gradients = new Eigen::MatrixXd;
		c_bias_gradients->resize( c_bias->rows(), c_bias->cols() );

		for(int j = 0; j < c_bias->rows(); j++)
		{
			(*c_bias_gradients)(j, 0) = 0;
		}

		for(int j = 0; j < batch_size; j++)
		{
			Eigen::MatrixXd * x_tild = gibbs_sampling(input_vectors[ i * batch_size + j], b_bias, c_bias, weight_matrix, num_hidden, 1);

			Eigen::MatrixXd first_weight = (*matrix_sigmoid(input_vectors[i * batch_size + j], b_bias, weight_matrix)) * (*input_vectors[i * batch_size + j]).transpose();
			Eigen::MatrixXd second_weight = ( *matrix_sigmoid(x_tild, b_bias, weight_matrix) ) * (*x_tild).transpose();

			*weight_gradients += first_weight - second_weight;


			*b_bias_gradients += ( *matrix_sigmoid(input_vectors[i * batch_size + j], b_bias, weight_matrix) ) - ( *matrix_sigmoid(x_tild, b_bias, weight_matrix) );

			*c_bias_gradients += (*input_vectors[i * batch_size + j]) - (*x_tild);

		}

		// printf("\n");
		// for(int x = 0; x < 15; x++)
		// {
		// 	printf("weight gradient %d: %f \n", x, (*weight_gradients)(0,x));
		// }
		// printf("\n");

		// for(int x = 0; x < 15; x++)
		// {
		// 	printf("b bias gradient %d: %f \n", x, (*b_bias_gradients)(x,0));
		// }
		// printf("\n");

		// for(int x = 0; x < 15; x++)
		// {
		// 	printf("c bias gradient %d: %f \n", x, (*c_bias_gradients)(x,0));
		// }
		// printf("\n");


		*weight_gradients *= alpha;
		*b_bias_gradients *= alpha;
		*c_bias_gradients *= alpha;


		(*weight_matrix) += (*weight_gradients);
		(*b_bias) += (*b_bias_gradients);
		(*c_bias) += (*c_bias_gradients);

	}
}
		printf("\n");c
		for(int x = 0; x < 15; x++)
		{
			printf("weight  %d: %f \n", x, (*weight_matrix)(0,x));
		}
		printf("\n");

		for(int x = 0; x < 15; x++)
		{
			printf("b bias %d: %f \n", x, (*b_bias)(x,0));
		}
		printf("\n");

		for(int x = 0; x < 15; x++)
		{
			printf("c bias %d: %f \n", x, (*c_bias)(x,0));
		}
		printf("\n");

	//gibbs_sampling(input_vectors[0], b_bias, c_bias, weight_matrix, num_hidden, 1);


	// for(int i = 0; i < num_hidden; i++)
	// {
	// 	(*b_bias)(i,0) = 0;
	// 	for(int j = 0; j < num_input; j++){

	// 		(*weight_matrix)(i,j) = 0;

	// 	}
	// }

	// Eigen::MatrixXd * test = matrix_sigmoid(input_vectors[0], weight_matrix, b_bias);
	// std::cout << *test << std::endl;



	f.close();

	return 0;
	}
