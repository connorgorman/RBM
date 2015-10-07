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

#include "logistic.h"

const int img_width = 28;
const int img_height = 28;
const int num_input = img_width * img_height;
const int num_hidden = 100;
const int data_passes = 1;

boost::random::mt19937 gen1;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> WeightMatrix;
typedef Eigen::Matrix<double, num_hidden, 1> HiddenBiasVector;
typedef Eigen::Matrix<double, num_input, 1> InputBiasVector;
typedef Eigen::Matrix<double, num_hidden, 1> HiddenNodes;
typedef Eigen::Matrix<double, num_input, 1> InputNodes;

/*** Logistic ***/
typedef Eigen::Matrix<double, num_hidden+1, 1> LogisticInputNodes;
typedef Eigen::Matrix<double, num_hidden+1, 1> LogisticParameters;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X_Samples;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Y_Samples;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Sigmoid_Matrix;

/**********************************/

/**** CONTROLLERS ****/
const int epochs = 15;
const int batch_size = 20;
const int num_batches = 3000;
const int gibbs_steps = 15;
const double alpha_rate = 0.1;
const double logisticLearningRate = .1;
const double regularization_const = 1.0;


/* END */

void error(int num)
{
	printf("ERROR %d \n", num);
	exit(-1);
}


double sigmoid(double x) {
	return 1.0/(1.0+exp(-1*x));
}


struct Layer
{

	WeightMatrix weight_matrix;
	HiddenBiasVector b_bias;
	InputBiasVector c_bias;

	HiddenNodes gibbs_hidden;
	InputNodes gibbs_input_tild;

	std::vector<InputNodes> input_vectors;

	WeightMatrix weight_gradients;
	HiddenBiasVector b_bias_gradients;
	InputBiasVector c_bias_gradients;

	HiddenNodes sigmoid_x_tild;
	HiddenNodes sigmoid_x_real;

	LogisticParameters theta;

	void resetGradients()
	{
		for(unsigned int i = 0; i < num_hidden; i++)
		{
			b_bias_gradients(i, 0) = 0;

			for(unsigned int j = 0; j < num_input; j++)
			{
				weight_gradients(i, j) = 0;

				if(i == 0)
				{	
					c_bias_gradients(j, 0) = 0;
				}
			}
		}
	}

	void initializeLayer()
	{

		weight_matrix.resize(num_hidden, num_input);
		weight_gradients.resize(num_hidden, num_input);

		double lower_bound = -4.0 * sqrt(6.0 / (num_input + num_hidden) );
		double upper_bound = -lower_bound;

		printf("lower: %f \n", lower_bound);
		printf("upper: %f \n", upper_bound);

		boost::uniform_real<> uniform_distribution(lower_bound, upper_bound);

		for(unsigned int i = 0; i < num_hidden; i++)
		{
			b_bias(i, 0) = 0.0;
			b_bias_gradients(i, 0) = 0.0;

			for(unsigned int j = 0; j < num_input; j++)
			{
				weight_matrix(i, j) = uniform_distribution(gen1);

				weight_gradients(i, j) = 0.0;

				if(i == 0)
				{	
					c_bias(j,0) = 0.0;
					c_bias_gradients(j, 0) = 0.0;
				}
			}
		}
	}

	void gibbs_sampling(InputNodes * input_ptr, bool first)
	{
		if( first ){
			printf("FIRST SAMPLE!\n");
			for(int i = 0; i < num_input; i++){
				gibbs_input_tild(i,0) = (*input_ptr)(i,0);
			}
		}

		boost::random::uniform_01<> uniform;

		for(int k = 0; k < gibbs_steps; k++)
		{

			for(unsigned int i = 0; i < num_hidden; i++)
			{

				double scalar = ( weight_matrix.row(i) * gibbs_input_tild)(0);

				double prob = sigmoid( b_bias(i) + scalar );
				double draw = uniform(gen1);

				if( draw < prob )
				{
					gibbs_hidden(i, 0) = 1;
				}
				else
				{
					gibbs_hidden(i, 0) = 0;
				}
			}

			for(unsigned int i = 0; i < num_input; i++)
			{

				double scalar = ( gibbs_hidden.transpose() * weight_matrix.col(i) )(0);
				double prob = sigmoid ( c_bias(i) + scalar );

				double draw = uniform(gen1);

				if( draw < prob )
				{
					gibbs_input_tild(i) = 1;
				}
				else
				{
					gibbs_input_tild(i) = 0;
				}
			}
		}
	}

	void updateGradients(InputNodes * real_nodes)
	{
		for(int i = 0; i < num_hidden; i++)
		{
			sigmoid_x_tild(i) = sigmoid( b_bias(i) + (weight_matrix.row(i) * gibbs_input_tild)(0) );
			sigmoid_x_real(i) = sigmoid( b_bias(i) + (weight_matrix.row(i) * (*real_nodes))(0) );
		}

		weight_gradients = weight_gradients + sigmoid_x_real * (*real_nodes).transpose() - sigmoid_x_tild * gibbs_input_tild.transpose();
		b_bias_gradients = b_bias_gradients +  sigmoid_x_real - sigmoid_x_tild;
		c_bias_gradients = c_bias_gradients + (*real_nodes) - gibbs_input_tild;

	}

	void updateParameters()
	{

		weight_gradients = (alpha_rate/batch_size) * weight_gradients;
		b_bias_gradients = (alpha_rate/batch_size) * b_bias_gradients;
		c_bias_gradients = (alpha_rate/batch_size) * c_bias_gradients;
		// weight_gradients = alpha_rate * weight_gradients;
		// b_bias_gradients = alpha_rate * b_bias_gradients;
		// c_bias_gradients = alpha_rate * c_bias_gradients;
		weight_matrix = weight_matrix + weight_gradients;
		b_bias = b_bias + b_bias_gradients;
		c_bias = c_bias + c_bias_gradients;
	}

	void printGradients()
	{

		std::cout << "Printing gradients for W\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: " << i << " " << std::setprecision (15) <<  weight_gradients(0,i) << "\n";
		}

		std::cout << "Printing gradients for b\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: " << i << " " << std::setprecision (15) << b_bias_gradients(i) << "\n";
		}

		std::cout << "Printing gradients for c\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: " << i << " " << std::setprecision (15) << c_bias_gradients(i) << "\n";
		}
	}

	void printParameters()
	{
		std::cout << "Printing first row from weight matrix\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: "<< i << " " << std::setprecision (15) << weight_matrix(0,i) << "\n";
		}

		std::cout << "Printing b bias values\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: "<< i << " " << std::setprecision (15) << b_bias(i,0) << "\n";
		}

		std::cout << "Printing c bias values\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: "<< i << " " << std::setprecision (15) << c_bias(i,0) << "\n";
		}
	}

	void dumpSample(int sampleNum){

		std::ofstream myfile;
		myfile.open ("connor_sample.txt");

		for(int i = 0; i < 784; i++)
		{
			myfile << input_vectors[sampleNum](i) << " ";
		}
		myfile.close();
	}

	void dumpGradients()
	{
		std::ofstream myfile;

		myfile.open ("connor_wgradient.txt");
		for(int i = 0; i < weight_gradients.rows(); i++)
		{
			for(int j = 0; j < weight_gradients.cols(); j++)
			{
				myfile << std::setprecision (15) << weight_gradients(i, j) << " ";
			}
			myfile << "\n";
		}
		myfile.close();

		myfile.open ("connor_bgradient.txt");
		for(int i = 0; i < num_hidden; i++)
		{
			myfile << std::setprecision (15) << b_bias_gradients(i) << " ";
		}
		myfile << "\n";
		myfile.close();

		myfile.open ("connor_cgradient.txt");
		for(int i = 0; i < num_input; i++)
		{
			myfile << std::setprecision (15) << c_bias_gradients(i) << " ";
		}
		myfile << "\n";

		myfile.close();
	}

	void dumpWeightMatrix()
	{

		std::ofstream myfile;
		myfile.open ("weight_dump.txt");
		for(int i = 0; i < weight_matrix.rows(); i++)
		{
			for(int j = 0; j < weight_matrix.cols(); j++)
			{
				myfile << std::setprecision (10) << weight_matrix(i, j) << " ";
			}
			myfile << "\n";
		}
		myfile.close();

		myfile.open("b_bias_dump.txt");
		for(int i = 0; i < b_bias.rows(); i++)
		{
			myfile << std::setprecision (10) << b_bias(i) << " ";
		}
		myfile.close();

		myfile.open("c_bias_dump.txt");
		for(int i = 0; i < c_bias.rows(); i++)
		{
			myfile << std::setprecision (10) << c_bias(i) << " ";
		}
		myfile.close();

	}

	void loadWeightMatrix(const char * file_name)
	{

		std::string line;
		std::ifstream infile(file_name);

		int row = 0;
		while (std::getline(infile, line))  // this does the checking!
		{
			std::istringstream iss(line);
			double c;

			int column = 0;

			while (iss >> c)
			{
				weight_matrix(row,column) = c;
				column++;
			}

			row++;
		}
		infile.close();
	}

	void loadBiases(const char * b_bias_filename, const char * c_bias_filename)
	{

		std::string line;
		std::ifstream infile(b_bias_filename);

		while (std::getline(infile, line))  // this does the checking!
		{
			std::istringstream iss(line);
			double c;

			int row = 0;
			while (iss >> c)
			{
				b_bias(row,0) = c;
				row++;
			}
		}
		infile.close();

		infile.open(c_bias_filename);

		while (std::getline(infile, line))  // this does the checking!
		{
			std::istringstream iss(line);
			double c;

			int row = 0;
			while (iss >> c)
			{
				c_bias(row,0) = c;
				row++;
			}
		}
		infile.close();

	}

	void logisticComputeHidden(X_Samples * x_samples, std::vector<InputNodes> input_vec)
	{

		boost::random::uniform_01<> uniform;

		std::cout << "Input vectors size: " << input_vec.size() << std::endl;

		for(int i = 0; i < input_vec.size(); i++)
		{
			InputNodes input = input_vec[i];

			for(unsigned int j = 0; j < num_hidden; j++)
			{
				double scalar = ( weight_matrix.row(j) * (input)) (0);
				double prob = sigmoid( b_bias(j) + scalar );
				(*x_samples).col(i)(j+1, 0) = prob;
			}

			(*x_samples).col(i)(0,0) = 1;
		}
	}


};

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

struct Layer * load(const char * file_name)
{
	std::ifstream image_file(file_name, std::ifstream::binary);
	int numRows=28;
	int numColumns=28;

	std::cout << "\n" << "MNIST FILE NAME: " << file_name << "\n";

	char *buffer = new char[4];
	image_file.read(buffer,4);
	unsigned long unicodeValue = ((unsigned int) (unsigned char)buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Magic number for MNIST file: " << (int)unicodeValue << std::endl;
	image_file.read(buffer,4);
	int num_images = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of images in MNIST file: " << num_images << "\n";
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of rows in MNIST file: " << unicodeValue << "\n";
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of columns in MNIST file: " << unicodeValue << "\n";

	struct Layer *layer = new Layer;

	for(int i=0; i<num_images; i++) {
		InputNodes temp;
		for(int j=0; j<28; j++) {
			for(int k=0; k<28; k++) {
				image_file.read(buffer,1);//pixel is one byte
				if((int)(unsigned char) buffer[0] > 128) {
					temp(j*28+k) = 1;
				}
				else {
					temp(j*28+k) = 0;
				}
			}
		}
		layer->input_vectors.push_back(temp);

	}

	delete [] buffer;

	std::cout << "\n";

	return layer;
}

void labelLoad(const char * file_name, Y_Samples * labels)
{
	std::cout << "\n";
	std::ifstream image_file(file_name, std::ifstream::binary);
	char *buffer = new char[4];
	image_file.read(buffer,4);
	unsigned long unicodeValue = ((unsigned int) (unsigned char)buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Magic number for MNIST Label file: " << (int)unicodeValue << std::endl;
	image_file.read(buffer,4);
	unsigned long num_items = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of images in MNIST Label file: " << num_items << "\n";
	image_file.read(buffer,4);
	
	// unsigned char * labels = new unsigned char[num_items];

	for(int i = 0; i < num_items; i++)
	{
		image_file.read(buffer, 1);
		(*labels)(i) = (unsigned char) buffer[0];
	}

}

void totalLoad(const char * file_name, Y_Samples * labels)
{

	// labels->resize(10, num_images);
	// inputs->resize(784, num_images);
	std::ifstream ifile(file_name);
	for(int i = 0; i < labels->rows(); i++)
	{
		double val = 0;
		std::string temp;
		ifile >> val;

		std::getline(ifile, temp);

		(*labels)(i) = val;
	}

}

void trainLayers(Layer * layer)
{

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < epochs; i ++)
	{
		std::cout << "Starting Epoch: " << i << "\n";
		std::chrono::high_resolution_clock::time_point time_temp = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::seconds>( time_temp - t1 ).count();
		std::cout << "Time (Seconds): " << duration << std::endl;

		for(int j = 0; j < num_batches; j++ )
		{

			layer->resetGradients();

			for(int k = 0; k < batch_size; k++)
			{
				unsigned int current_index = j * batch_size + k;
				if( i == 0 && current_index == 0){
					layer->gibbs_sampling( &(layer->input_vectors[current_index]), true);
				}
				else
					layer->gibbs_sampling( &(layer->input_vectors[current_index]), false);

				layer->updateGradients( &(layer->input_vectors[current_index]) );
			}

			layer->updateParameters();
		}
	}
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
	std::cout << "DURATION (Seconds): " << duration << std::endl;
}

void logisticDumpParameters( std::vector<LogisticParameters *> logisticParameters )
{

	std::ofstream myfile;
	myfile.open ("logisticParameters.txt");

	for(int i = 0; i < logisticParameters.size(); i++)
	{

		for(int j = 0; j < logisticParameters[i]->rows(); j++)
		{
			myfile << (*(logisticParameters[i]))(j);
			if( j != logisticParameters[i]->rows() - 1 ) 
				myfile << " ";
		}
		myfile << "\n";
	}

	myfile.close();
}

std::vector<LogisticParameters *> loadLogisticParameters(const char * file_name)
{

	std::string line;
	std::ifstream infile(file_name);

	std::vector<LogisticParameters *> logisticParameters;

	while (std::getline(infile, line))  // this does the checking!
	{
		LogisticParameters * parameters = new LogisticParameters;

		std::istringstream iss(line);
		double c;

		int column = 0;
		while (iss >> c)
		{
			(*parameters)(column, 0) = c;
			column++;
		}

		logisticParameters.push_back(parameters);
	}
	infile.close();

	return logisticParameters;
}

void normalizeData( Eigen::MatrixXd * inputs)
{

	for(int i = 1; i < inputs->rows(); i++){
		
		double mean = 0.0;

		for(int j = 0; j< inputs->cols(); j++)
		{
			mean += (*inputs).row(i)(j);
		}

		mean = mean / inputs->cols();

		double std_dev = 0.0;
		for(int j = 0; j< inputs->cols(); j++)
		{
			double minus = ( (*inputs).row(i)(j) - mean);
			std_dev = std_dev + ( minus * minus );
		}

		std_dev = sqrt( std_dev / inputs->cols() );
		if( std_dev == 0)
			std_dev == 1;

		for(int j = 0; j< inputs->cols(); j++)
		{
			(*inputs).row(i)(j) = ( (*inputs).row(i)(j) - mean) / std_dev;
		}

	}
}


void dumpHidden(Eigen::MatrixXd * training_samples, Eigen::MatrixXd * training_labels)
{

	std::ofstream myfile;
	myfile.open ("hidden.txt");

	for(int i = 0; i < 60000; i++){
		for(int j = 1; j < 101; j++)
		{

			myfile << (*training_samples).col(i)(j);

			if( j != 100)
				myfile << " ";
		}
		myfile << "\n";
	}

	myfile.close();

	myfile.open("labels.txt");

	for(int i = 0; i < 60000; i++){
		myfile << (*training_samples)(i);
		if( i != 59999)
			myfile << " ";
	}

	myfile.close();


}

int main(int argc, char * argv[])
{
	Layer * layer = load("train-images-idx3-ubyte");	
	layer->initializeLayer();

	bool load_weights_from_file = false;


	if( !load_weights_from_file ){

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		trainLayers(layer);
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::minutes>( t2 - t1 ).count();
		std::cout << "DURATION: " << duration << std::endl;
		layer->printParameters();
		layer->dumpWeightMatrix();
	}
	else
	{
		layer->loadWeightMatrix("weight_dump.txt");
		layer->loadBiases("b_bias_dump.txt", "c_bias_dump.txt");
		layer->printParameters();
	}

/******** PREDICTION ************/

	bool load_logistic_parameters = true;

	std::vector <Eigen::MatrixXd *> logistic_parameters;

	Y_Samples training_labels;
	training_labels.resize(60000, 1);

	X_Samples training_samples;
	training_samples.resize(101, 60000);

	totalLoad("mnist_train_spaces.csv", &training_labels);

	std::vector< Y_Samples *> all_y_samples;
	for(int i = 0; i < 10; i++)
	 {

	 	Y_Samples * temp_y = new Y_Samples;
	 	(*temp_y).resize(60000, 1);

		for(int j = 0; j < training_samples.cols(); j++)
		{

			if( training_labels(j,0) == i )
				(*temp_y)(j,0) = 1;
			else
				(*temp_y)(j,0) = 0;
		}

		all_y_samples.push_back(temp_y);
	}

 	layer->logisticComputeHidden(&training_samples, layer->input_vectors);
 	normalizeData(&training_samples);

 	dumpHidden(&training_samples, &training_labels);

 	boost::normal_distribution<> nd(0.0, 10);

	for (int i = 0; i < 10; i++ )
	 {
		Eigen::MatrixXd * theta = new Eigen::MatrixXd;
		(*theta).resize(num_hidden +1, 1);
		for(int j = 0; j < num_hidden+1; j++){
			//(*theta)(j) = nd(gen1);
			(*theta)(j) = 0;
		}

		l_runLogisticRegression(&training_samples, all_y_samples[i], theta, 100, 2.0, 0.0);

	 	logistic_parameters.push_back(theta);
	 }

	 for(int j = 0; j < 10; j++)
	{
		double num_true_correct = 0.0;
	    double num_correct = 0.0;
	    for(int i = 0; i < training_samples.cols(); i++)
	    {
	        Eigen::MatrixXd * new_theta = logistic_parameters[j];

	        double h_x = l_sigmoid( ( (*new_theta).transpose() * training_samples.col(i) )(0) );

	        if (h_x >= 0.5 && training_labels(i) == j){
	        	num_true_correct += 1.0;
	            num_correct += 1.0;
	        }
	        else if( h_x < 0.5 && training_labels(i) != j) {
	            num_correct += 1.0;
	        }
	     }

	    printf("num correct for %d is %f \n", j, num_correct);
	    printf("num true correct for %d is %f out of %f\n\n", j, num_true_correct, num_correct);
	}


/**** RUNNING ON TEST SET ****/

	Y_Samples testing_labels;
	testing_labels.resize(10000, 1);

	X_Samples testing_samples;
	testing_samples.resize(101, 10000);

	totalLoad("mnist_test_spaces.csv", &testing_labels);

	Layer * layer1 = load("t10k-images-idx3-ubyte");
	layer->logisticComputeHidden(&testing_samples, layer1->input_vectors);
 	normalizeData(&testing_samples);

 	double num_correct = 0.0;
 	for(int i = 0; i < testing_samples.cols(); i++)
 	{

 		double best_prob = 0.0;
 		double best_idx = 0.0;

 		for(int j = 0; j < logistic_parameters.size(); j++)
 		{
 			 Eigen::MatrixXd * new_theta = logistic_parameters[j];

			 Eigen::MatrixXd column = testing_samples.col(i);
 			 double prob = l_sigmoid( ( new_theta->transpose() * column )(0) );

 			 if( prob > best_prob )
 			 {
 			 	best_prob = prob;
 			 	best_idx = j;
 			 }
 		}

 		if( best_idx == testing_labels(i) )
 			num_correct++;

 	}

 	double accuracy = num_correct / 10000;

 	printf("Accuracy: %f \n", accuracy);

	delete layer;
	//delete test_layer;

	return 0;
}
