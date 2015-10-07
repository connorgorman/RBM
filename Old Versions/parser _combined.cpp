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

#include <iomanip>
#include <assert.h>
#include <execinfo.h>


/******** Connors Portion *********/
const int img_width = 28;
const int img_height = 28;
const int num_input = img_width * img_height;
const int num_hidden = 100;
const double alpha_rate = 0.01;
//const int gibbs_steps = 1;
//const int batch_size = 1;
//const int num_batches = 2;
const int data_passes = 1;

boost::random::mt19937 gen1;
/**********************************/

/******** Barret's Portion *********/
//Number of hidden and visible units for th RBM
const int numHiddenUnits=100;
const int numVisibleUnits=784; //28*28
//Contrastive Divergence Parameters
const double learningRate=0.01;
const int gibbsIter=1;
// //Constants for training the RBM
// const int miniBatchSize = 1;
// const int numBatches = 2;

/**********************************/

/******** Connors Portion *********/
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> WeightMatrix;
typedef Eigen::Matrix<double, num_hidden, 1> HiddenBiasVector;
typedef Eigen::Matrix<double, num_input, 1> InputBiasVector;
typedef Eigen::Matrix<double, num_hidden, 1> HiddenNodes;
typedef Eigen::Matrix<double, num_input, 1> InputNodes;
/**********************************/

/******** Barret's Portion *********/
//Typedefs for the weights
typedef Eigen::Matrix<double,numVisibleUnits,1> VisibleBiasVec;
typedef Eigen::Matrix<double,numHiddenUnits,1> HiddenBiasVec;
typedef Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> ConnectionWeights;// numHiddenUnits x numVisibleUnits

//Typedefs for the inputs for the image
typedef Eigen::Matrix<double,numHiddenUnits,1> HiddenVec;
typedef Eigen::Matrix<double,numVisibleUnits,1> InputVec;
//Training examples are stored in this
std::vector<InputVec> inputs;

//Random number generator
boost::random::mt19937 gen2;
/**********************************/

/**** CONTROLLERS ****/
const int epochs = 5;
const int batch_size = 20;
const int num_batches = 3000;
const int gibbs_steps = 1;
/* END */



/******** Barret's Portion *********/
//Info for MNIST dataset
void loadTrainImageFile(std::string file_name) {
	std::ifstream image_file(file_name, std::ifstream::binary);
	int numImages=60000;
	int numRows=28;
	int numColumns=28;

	char *buffer = new char[4];
	image_file.read(buffer,4);
	unsigned long unicodeValue = ((unsigned int) (unsigned char)buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Magic number for MNIST file: " << (int)unicodeValue << std::endl;
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of images in MNIST file: " << unicodeValue << "\n";
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of rows in MNIST file: " << unicodeValue << "\n";
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of columns in MNIST file: " << unicodeValue << "\n";

	for(int i=0; i<numImages; i++) {
		InputVec temp;
		for(int j=0; j<28; j++) {
			for(int k=0; k<28; k++) {
				image_file.read(buffer,1);//pixel is one byte
				if((int)(unsigned char) buffer[0] > 128) {
					temp(j*28+k) = 1;
					//std::cout << temp(j*28+k) << " ";
				}
				else {
					temp(j*28+k) = 0;
					//std::cout << temp(j*28+k) << " ";
				}
			}
			//std::cout << "\n";
		}
		inputs.push_back(temp);
	}
	delete [] buffer;
}
/**********************************/

void error(int num)
{
	printf("ERROR %d \n", num);
	exit(-1);
}

//Hidden layer vector
struct HiddenLayer {
	HiddenBiasVec biasWeights;

	void initializeWeights() {
		for(int i=0; i<numHiddenUnits; i++) {
			biasWeights(i)=0;
		}
	}

	void dumpWeights() {
		
		std::ofstream myfile;
		
		myfile.open ("barrett_bweight.txt");
		for(int i = 0; i < numHiddenUnits; i++)
		{
			myfile << std::setprecision (15) << biasWeights(i) << " ";
		}
		myfile << "\n";

		myfile.close();

	}
};

//Input layer vector
struct InputLayer {
	VisibleBiasVec biasWeights;

	void initializeWeights() {
		for(int i=0; i<numVisibleUnits; i++) {
			biasWeights(i)=0;
		}
	}

	void dumpWeights() {
		
		std::ofstream myfile;
		myfile.open ("barrett_cweight.txt");
		for(int i = 0; i < numVisibleUnits; i++)
		{
			myfile << std::setprecision (15) << biasWeights(i) << " ";
		}
		myfile << "\n";
	
		myfile.close();
	}

};

double sigmoid(double x) {
	return 1.0/(1.0+exp(-1*x));
}

//Layer between input and output layer, or output and output in stacked RBM's
struct Weights {

	ConnectionWeights weights; // hidden x visible
	ConnectionWeights gradients_W; //Stores new weights when updating parameters using gradient descent, then puts back into weights
	VisibleBiasVec gradients_c; //gradients for c
	HiddenBiasVec gradients_b; //gradients for b

	//Draws from the gibbs sampler
	InputVec gibbsDraw_Input;
	HiddenVec gibbsDraw_Hidden;

	//Sigmoid matrices for gradient updates, notation consistent with source
	HiddenVec h_sample;
	HiddenVec h_input;

	//White for positive, black for negative, grey for close to zero
	void visualizeWeights() {
		std::cout << std::setprecision(2);
		for(int i=0; i<numHiddenUnits; i++) {
			std::cout << "Visualizing hidden node: " << i << "\n";
			for(int j=0; j<28; j++) {
				for(int k=0; k<28; k++) {
					//std::cout << weights(i,j*28+k) << " ";
					if(weights(i,j*28+k) > 7) {
						std::cout << "1 ";
					}
					else if(weights(i,j*28+k) < -7) {
						std::cout << "0 ";
					}
					else {
						std::cout << "- ";
					}
				}
				std::cout << "\n";
			}
			std::cout << "\n\n";
		}
	}

	void dumpWeightMatrix()
	{

		std::ofstream myfile;
		myfile.open ("barrett_wweight.txt");

		for(int i = 0; i < weights.rows(); i++)
		{
			for(int j = 0; j < weights.cols(); j++)
			{
  				(myfile) << std::setprecision (15) << weights(i, j) << " ";
			}
			(myfile) << "\n";
		}
		myfile.close();
	}

	void dumpGradients(){

		std::ofstream gradFile;

		gradFile.open ("barrett_wgradient.txt");
		for(int i = 0; i < gradients_W.rows(); i++)
		{
			for(int j = 0; j < gradients_W.cols(); j++)
			{
  				gradFile << std::setprecision (15) << gradients_W(i, j) << " ";
			}
			gradFile << "\n";
		}
		gradFile.close();

		gradFile.open ("barrett_bgradient.txt");
		for(int i = 0; i < numHiddenUnits; i++)
		{
			gradFile << std::setprecision (15) << gradients_b(i) << " ";
		}
		gradFile << "\n";
		gradFile.close();

		gradFile.open ("barrett_cgradient.txt");
		for(int i = 0; i < numVisibleUnits; i++)
		{
			gradFile << std::setprecision (15) << gradients_c(i) << " ";
		}
		gradFile << "\n";

		gradFile.close();

	}

	void initializeWeights() {
		//Resize Weight Matrix
		weights.resize(numHiddenUnits,numVisibleUnits);
		gradients_W.resize(numHiddenUnits,numVisibleUnits);

		//Initialize the weights to random values
		double lower = -4.0*sqrt(6.0/(numVisibleUnits+numHiddenUnits));
		double upper = 4.0*sqrt(6.0/(numVisibleUnits+numHiddenUnits));
		std::cout << "Lower " << lower << "\n";
		std::cout << "Upper " << upper << "\n";
		boost::uniform_real<> distribution(lower,upper);

		for(int i=0; i<numHiddenUnits; i++) {
			for(int j=0; j<numVisibleUnits; j++) {
				weights(i,j) = distribution(gen2);
			}
		}

		// for(int i =0; i<15;i++)
		// {
		// 	printf("Weight Initialization %d: %f \n", i, weights(0,i));
		// }

	}

	void clearGradients() {
		for(int i=0; i<numHiddenUnits; i++) {
			gradients_b(i) = 0.0;
			for(int j=0; j<numVisibleUnits; j++) {
				gradients_W(i,j) = 0.0;
				if(i==0) {
					gradients_c(j) = 0.0;
				}
			}
		}
	}

	//Dump weights to file
	void dumpWeights() {

	}

	void gibbsSampler(InputLayer *inputLayer,HiddenLayer *hiddenLayer,InputVec *initialSample) {
		//Uniform RV generator
		boost::random::uniform_01<> uniform;

		// for(int i = 0; i < 30; i++){
		// 	printf("Bias Weight %d: %f \n", i, hiddenLayer->biasWeights(i));
		// }

		// for(int i = 0; i < 30; i++){
		// 	printf("Initial Weights %d: %f \n", i, (*initialSample)(i,0));
		// }

		// double sum = 0.0;
		// for(int i = 0; i < 784; i++){
		// 	// printf("Sample Value %d: %f \n", i, (*initialSample)(i,0));
		// 	sum += (*initialSample)(i,0);
		// }

		// printf("\n SUM: %f \n", sum);

		//Sample h given the input
		for(int i=0; i<numHiddenUnits; i++) {
			double threshold = sigmoid(hiddenLayer->biasWeights(i) + (weights.row(i)*(*initialSample))(0) );
			double draw = uniform(gen2);

				// if( i < 30){
				// 	printf("Scalar %d: %f \n", i, (weights.row(i)*(*initialSample))(0));
				// }

				// if( i < 15){
				// 	printf("Threshold %d: %f \n", i, threshold);
				// 	printf("Draw %d: %f \n", i, draw);
				// }


			if(draw < threshold) {
				gibbsDraw_Hidden(i) = 1;
			}
			else {
				gibbsDraw_Hidden(i) = 0;
			}
		}
		//Sample x given the hidden
		for(int i=0; i<numVisibleUnits; i++) {
			double threshold = sigmoid(inputLayer->biasWeights(i) + (gibbsDraw_Hidden.transpose() * weights.col(i)));
			if(uniform(gen2) < threshold) {
				gibbsDraw_Input(i) = 1;
			}
			else {
				gibbsDraw_Input(i) = 0;
			}
		}

		//Now full gibbs iterations
		for(int i = 1; i<gibbsIter; i++) {
			//Sample h given the previous gibb input
			for(int i=0; i<numHiddenUnits; i++) {
				double threshold = sigmoid(hiddenLayer->biasWeights(i) + (weights.row(i)*gibbsDraw_Input)(0) );
				if(uniform(gen2) < threshold) {
					gibbsDraw_Hidden(i) = 1;
				}
				else {
					gibbsDraw_Hidden(i) = 0;
				}
			}
			//Sample x given the hidden
			for(int i=0; i<numVisibleUnits; i++) {
				double threshold = sigmoid(inputLayer->biasWeights(i) + (gibbsDraw_Hidden.transpose() * weights.col(i)));
				if(uniform(gen2) < threshold) {
					gibbsDraw_Input(i) = 1;
				}
				else {
					gibbsDraw_Input(i) = 0;
				}
			}
		}
	}

	//Update the gradient after a round of gibbs sampling
	void updateGradients(HiddenLayer *hiddenLayer, InputVec *initialSample) {

		for(int i=0; i<numHiddenUnits; i++) {
			h_input(i) = sigmoid(hiddenLayer->biasWeights(i) + (weights.row(i)*(*initialSample))(0));
		}

		for(int i=0; i<numHiddenUnits; i++) {
			h_sample(i) = sigmoid(hiddenLayer->biasWeights(i) + (weights.row(i)*gibbsDraw_Input)(0));
		}
		gradients_W = gradients_W + h_input*(initialSample->transpose()) - h_sample*(gibbsDraw_Input.transpose());
		gradients_b = gradients_b + h_input - h_sample;
		gradients_c = gradients_c + (*initialSample) - gibbsDraw_Input;
	}

	//Call after every minibatch
	void updateParameters(InputLayer *inputLayer,HiddenLayer *hiddenLayer) {
		
		gradients_W = learningRate * gradients_W;
		gradients_b = learningRate * gradients_b;
		gradients_c = learningRate * gradients_c;

		weights = weights + gradients_W;
		hiddenLayer->biasWeights = hiddenLayer->biasWeights + gradients_b;
		inputLayer->biasWeights = inputLayer->biasWeights + gradients_c;
	}

	void debug_gradients(InputLayer *inputLayer,HiddenLayer *hiddenLayer) {

		std::cout << "Printing gradients for W\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: " << i << " " << std::setprecision (15) << gradients_W(0,i) << "\n";
		}

		std::cout << "Printing gradients for b\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: " << i << " " << std::setprecision (15) << gradients_b(i) << "\n";
		}

		std::cout << "Printing gradients for c\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: " << i << " " << std::setprecision (15) << gradients_c(i) << "\n";
		}

	}

	void debug(InputLayer *inputLayer,HiddenLayer *hiddenLayer) {
		std::cout << "Printing first row from weight matrix\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: "<< i << " " << std::setprecision (15) << weights(0,i) << "\n";
		}

		std::cout << "Printing b bias values\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: "<< i << " " << std::setprecision (15) << hiddenLayer->biasWeights(i) << "\n";
		}

		std::cout << "Printing c bias values\n";
		for(int i=0; i<15; i++) {
			std::cout << "Index: "<< i << " " << std::setprecision (15) << inputLayer->biasWeights(i) << "\n";
		}

	}
};

struct RBM
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

	void initializeRBM()
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

		// for(int i =0; i<15;i++)
		// {
		// 	printf("Weight Initialization %d: %f \n", i, weight_matrix(0,i));
		// }

	}

	void gibbs_sampling(InputNodes * input_ptr)
	{

		//gibbs_input_tild = *input_ptr;
		for(int i = 0; i < num_input; i++){
			gibbs_input_tild(i,0) = (*input_ptr)(i,0);
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

		weight_gradients = alpha_rate * weight_gradients;
		b_bias_gradients = alpha_rate * b_bias_gradients;
		c_bias_gradients = alpha_rate * c_bias_gradients;

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

	void trainRBM()
	{

		for(int x = 0; x < data_passes; x++){
			printf("Starting New Pass %d \n", x);
			for(int i = 0; i < num_batches; i++)
			{
				resetGradients();


				

				for( int j = 0; j < batch_size; j++)
				{
					InputNodes * currentSet = &(input_vectors[i * batch_size + j] );
					gibbs_sampling( currentSet );
					updateGradients( currentSet );
				}

				updateParameters();

				if( i == num_batches -1)
				{
					dumpWeightMatrix();
				}

				if( i == num_batches -1)
				{
					dumpGradients();
				}
			}
		}
		printParameters();

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
		myfile.open ("connor_wweight.txt");
		for(int i = 0; i < weight_matrix.rows(); i++)
		{
			for(int j = 0; j < weight_matrix.cols(); j++)
			{
				myfile << std::setprecision (15) << weight_matrix(i, j) << " ";
			}
			myfile << "\n";
		}
		myfile.close();

		myfile.open ("connor_bweight.txt");
		for(int i = 0; i < num_hidden; i++)
		{
			myfile << std::setprecision (15) << b_bias(i) << " ";
		}
		myfile << "\n";
		myfile.close();

		myfile.open ("connor_cweight.txt");
		for(int i = 0; i < num_input; i++)
		{
			myfile << std::setprecision (15) << c_bias(i) << " ";
		}
		myfile << "\n";

		myfile.close();

	}

	void dumpBiasWeights(std::ofstream * myfile) {

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


struct RBM * barrettLoad(const char * file_name)
{
	std::ifstream image_file(file_name, std::ifstream::binary);
	int numImages=60000;
	int numRows=28;
	int numColumns=28;

	char *buffer = new char[4];
	image_file.read(buffer,4);
	unsigned long unicodeValue = ((unsigned int) (unsigned char)buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Magic number for MNIST file: " << (int)unicodeValue << std::endl;
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of images in MNIST file: " << unicodeValue << "\n";
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of rows in MNIST file: " << unicodeValue << "\n";
	image_file.read(buffer,4);
	unicodeValue = ((unsigned int) buffer[0])*(16777216)+((unsigned int) (unsigned char)buffer[1])*(65536) + ((unsigned int) (unsigned char)buffer[2])*(256) + ((unsigned int) (unsigned char)buffer[3]);
	std::cout << "Number of columns in MNIST file: " << unicodeValue << "\n";

	struct RBM *rbm = new RBM;

	for(int i=0; i<numImages; i++) {
		InputNodes temp;
		for(int j=0; j<28; j++) {
			for(int k=0; k<28; k++) {
				image_file.read(buffer,1);//pixel is one byte
				if((int)(unsigned char) buffer[0] > 128) {
					temp(j*28+k) = 1;
					//std::cout << temp(j*28+k) << " ";
				}
				else {
					temp(j*28+k) = 0;
					//std::cout << temp(j*28+k) << " ";
				}
			}
			//std::cout << "\n";
		}
		rbm->input_vectors.push_back(temp);

		

		//std::cout << "\n\n";
	}

	// double sum = 0;
	// for(int h = 0; h < 784; h++){
	// 	sum+= inputs[0](h);
	// }
	// printf("\n Input SUM %f \n", sum);

	// for(int i=0; i<inputs.size(); i++)
	// {
	// 	for(int j=0; j<28; j++) {
	// 		for(int k=0; k<28; k++) {
	// 			std::cout << inputs[i](j*28+k) << " ";
	// 		}
	// 		std::cout << "\n";
	// 	}
	// 	std::cout << "\n\n";
	// }
	delete [] buffer;

	return rbm;
}



struct RBM * loadData(const char * filename)
{

	std::ifstream f;

	char * int_buf = new char[4];
	char * pixel_buf = new char;

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

	struct RBM * rbm = new RBM;

	for(unsigned int i = 0; i < filesize; i++)
	{

		InputNodes temp_nodes;

		for(unsigned int j = 0; j < height; j++)
		{
			for( unsigned int k = 0; k < width; k++ )
			{
				f.read(pixel_buf, 1);
				unsigned char pixel = (unsigned char) calculatePixel(pixel_buf);

				if( pixel > 128 )
					temp_nodes(j * 28 + k, 0) = 1;
				else
					temp_nodes(j * 28 + k, 0) = 0;
			}
		}
		rbm->input_vectors.push_back(temp_nodes);
	}

	delete [] int_buf;
	delete pixel_buf;

	f.close();

	return rbm;
}

void validateInputs(RBM * rbm)
{
	if(rbm->input_vectors.size() != inputs.size() ) error(3);

	for(int i = 0; i > inputs.size(); i++)
	{
		if(rbm->input_vectors[i].rows() != inputs[i].rows()) error(4);
		if(rbm->input_vectors[i].cols() != inputs[i].cols()) error(5);

		for(int j = 0; j < inputs[i].rows(); i++)
		{
			if(rbm->input_vectors[i](i, 0) != inputs[i](i, 0)) error(6);
		}
	}
}

void validateWeights( RBM * rbm, InputLayer * inputLayer, Weights * layer01, HiddenLayer * hiddenLayer )
{

	if(rbm->b_bias.rows() != hiddenLayer->biasWeights.rows()) error(7);

	if(rbm->b_bias.cols() != hiddenLayer->biasWeights.cols()) error(8);

	if(rbm->c_bias.rows() !=inputLayer->biasWeights.rows()) error(9);

	if(rbm->c_bias.cols() != inputLayer->biasWeights.cols()) error(10);

	if(rbm->weight_matrix.rows() != layer01->weights.rows()) error(11);

	if(rbm->weight_matrix.cols() != layer01->weights.cols()) error(12);

	for(int i = 0; i < rbm->b_bias.rows(); i++)
	{
		if(rbm->b_bias(i, 0) != hiddenLayer->biasWeights(i,0)) error(13);
	}

	for(int i = 0; i < rbm->c_bias.rows(); i++)
	{
		if(rbm->c_bias(i, 0) != inputLayer->biasWeights(i,0)) error(14);
	}

	for(int i = 0; i< rbm->weight_matrix.rows(); i++)
	{
		for(int j = 0; j < rbm->weight_matrix.cols(); j++)
		{
			if(rbm->weight_matrix(i,j) != layer01->weights(i,j)) error(15);
		}
	}

}

void
print_trace (void)
{
  void *array[10];
  size_t size;
  char **strings;
  size_t i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);

  printf ("Obtained %zd stack frames.\n", size);

  for (i = 0; i < size; i++)
     printf ("%s\n", strings[i]);

  free (strings);
}

void validateGradients( RBM * rbm, Weights * layer01)
{

	if(rbm->b_bias_gradients.rows() != layer01->gradients_b.rows()) error(16);

	if(rbm->b_bias_gradients.cols() != layer01->gradients_b.cols()) error(17);

	if(rbm->c_bias_gradients.rows() != layer01->gradients_c.rows()) error(18);

	if(rbm->c_bias_gradients.cols() != layer01->gradients_c.cols()) error(19);

	if(rbm->weight_gradients.rows() != layer01->gradients_W.rows()) error(20);

	if(rbm->weight_gradients.cols() != layer01->gradients_W.cols()) error(21);

	for(int i = 0; i < rbm->b_bias_gradients.rows(); i++)
	{
		if(rbm->b_bias_gradients(i, 0) != layer01->gradients_b(i,0)){
			printf("Connor's B Bias: %f <-> Barret's B Bias: %f at index %d \n", rbm->b_bias_gradients(i, 0), layer01->gradients_b(i,0), i);
			
			//print_trace();
			break;
			//error(22);
		}
	}

	for(int i = 0; i < rbm->c_bias_gradients.rows(); i++)
	{
		if(rbm->c_bias_gradients(i, 0) != layer01->gradients_c(i,0)) error(1);
	}

	for(int i = 0; i< rbm->weight_gradients.rows(); i++)
	{
		for(int j = 0; j < rbm->weight_gradients.cols(); j++)
		{
			if(rbm->weight_gradients(i,j) != layer01->gradients_W(i,j)) error(2);
		}
	}

}

void validateStates(RBM * rbm, Weights * layer01)
{

	if(rbm->gibbs_input_tild.rows() != layer01->gibbsDraw_Input.rows()) error(23);
	if(rbm->gibbs_input_tild.cols() != layer01->gibbsDraw_Input.cols()) error(24);
	
	if(rbm->gibbs_hidden.rows() != layer01->gibbsDraw_Hidden.rows()) error(25);
	if(rbm->gibbs_hidden.cols() != layer01->gibbsDraw_Hidden.cols() ) error(26);

	for(int i = 0; i < rbm->gibbs_input_tild.rows(); i++)
	{
		//printf("Input: %f %f \n", rbm->gibbs_input_tild(i,0), layer01->gibbsDraw_Input(i,0) );
		if( rbm->gibbs_input_tild(i,0) != layer01->gibbsDraw_Input(i,0)) error(27);
	}

	for(int i = 0; i < rbm->gibbs_hidden.rows(); i++)
	{
		//printf("Hidden: %f %f \n", rbm->gibbs_hidden(i,0), layer01->gibbsDraw_Hidden(i,0) );
		if( rbm->gibbs_hidden(i,0) != layer01->gibbsDraw_Hidden(i,0)) error(28);
	}
}

void trainRBMs(RBM * rbm, Weights& layer01, InputLayer& inputLayer, HiddenLayer& hiddenLayer)
{

	for(int i = 0; i < epochs; i ++)
	{

		std::cout << "Starting Epoch: " << i << "\n";

		for(int j = 0; j < num_batches; j++ )
		{

			rbm->resetGradients();
			layer01.clearGradients();

			//Validation
			validateGradients(rbm, &layer01);

			for(int k = 0; k < batch_size; k++)
			{
				unsigned int current_index = j * batch_size + k;

				layer01.gibbsSampler(&inputLayer, &hiddenLayer, &inputs[current_index]);
				rbm->gibbs_sampling( &(rbm->input_vectors[current_index]));

				validateStates(rbm, &layer01);

				layer01.updateGradients(&hiddenLayer, &inputs[current_index]);
				rbm->updateGradients( &(rbm->input_vectors[current_index]) );

				validateGradients(rbm, &layer01);

			}

			layer01.updateParameters(&inputLayer, &hiddenLayer);
			rbm->updateParameters();

			validateWeights(rbm, &inputLayer, &layer01, &hiddenLayer);

		}
	}
}

int main(int argc, char * argv[])
{
	/* Barret */
	loadTrainImageFile("train-images.idx3-ubyte");
	Weights layer01;
	InputLayer inputLayer;
	HiddenLayer hiddenLayer1;
	
	inputLayer.initializeWeights();
	layer01.initializeWeights();
	hiddenLayer1.initializeWeights();
	/* End */

	/* Connor */
	RBM * rbm = barrettLoad("train-images.idx3-ubyte");	
	rbm->initializeRBM();
	/* End */

	/* Validation */
	validateInputs(rbm);
	validateWeights(rbm, &inputLayer, &layer01, &hiddenLayer1);
	/* End */

	printf("BEGIN TRAINING RBMS \n");

	trainRBMs(rbm, layer01, inputLayer, hiddenLayer1);

	rbm->printParameters();

	// rbm->trainRBM();

	// rbm->dumpSample(num_batches - 1);
	// //rbm->dumpGradients();

	delete rbm;

	return 0;
}
