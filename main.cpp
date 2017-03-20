#include <iostream>
#include <math.h>
#include "LeastSquare.h"

// case 1. sine fitting
template <typename Tx = double, typename Ty = double> 
class SineModel : public LeastSquare::BaseModel<Tx,Ty>
{
        using LeastSquare::BaseModel<Tx,Ty>::NumDomain;
        using LeastSquare::BaseModel<Tx,Ty>::NumParameter;
        using LeastSquare::BaseModel<Tx,Ty>::NumDimension;
        using LeastSquare::BaseModel<Tx,Ty>::xi;
        using LeastSquare::BaseModel<Tx,Ty>::yi;

public:
        explicit SineModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, 4)
        {}

	virtual double modelFunc(const double* parameter, const int Index) const
	{
		return parameter[0]*std::sin(parameter[1]*xi[Index] + parameter[2]) + parameter[3];
	}

/*
        virtual void get_jacobian(const double* parameter, double* jacobian) const
        {
                for(int i=0; i<NumDomain; ++i)
                {
                        jacobian[i*NumParameter + 0] = std::sin(parameter[1]*xi[i] + parameter[2]);
                        jacobian[i*NumParameter + 1] = parameter[0]*std::cos(parameter[1]*xi[i] + parameter[2])*xi[i];
                        jacobian[i*NumParameter + 2] = parameter[0]*std::cos(parameter[1]*xi[i] + parameter[2]);
                        jacobian[i*NumParameter + 3] = 1.;
		}
        }
*/
};


// case 2. multi-linear fitting
template <typename Tx = double, typename Ty = double> 
class MultiLinearModel : public LeastSquare::BaseModel<Tx,Ty>
{
        using LeastSquare::BaseModel<Tx,Ty>::NumDomain;
        using LeastSquare::BaseModel<Tx,Ty>::NumParameter;
        using LeastSquare::BaseModel<Tx,Ty>::NumDimension;
        using LeastSquare::BaseModel<Tx,Ty>::xi;
        using LeastSquare::BaseModel<Tx,Ty>::yi;

public:
        explicit MultiLinearModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, 4, 3)
        {}

	virtual double modelFunc(const double* parameter, const int Index) const
	{
		return parameter[3]*xi[NumDimension*Index+2] + parameter[2]*xi[NumDimension*Index+1] + parameter[1]*xi[NumDimension*Index] + parameter[0];
	}

/*
        virtual void get_jacobian(const double* parameter, double* jacobian) const
        {
                for(int i=0; i<NumDomain; ++i)
                {
                        jacobian[i*NumParameter + 0] = 1.;
                        jacobian[i*NumParameter + 1] = xi[NumDimension*i];
                        jacobian[i*NumParameter + 2] = xi[NumDimension*i+1];
                        jacobian[i*NumParameter + 3] = xi[NumDimension*i+2];
		}
        }
*/
};


// case 3. non-linear fitting
template <typename Tx = double, typename Ty = double> 
class ExponentialModel : public LeastSquare::BaseModel<Tx,Ty>
{
        using LeastSquare::BaseModel<Tx,Ty>::NumDomain;
        using LeastSquare::BaseModel<Tx,Ty>::NumParameter;
        using LeastSquare::BaseModel<Tx,Ty>::NumDimension;
        using LeastSquare::BaseModel<Tx,Ty>::xi;
        using LeastSquare::BaseModel<Tx,Ty>::yi;

public:
        explicit ExponentialModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, 4, 3)
        {}

	virtual double modelFunc(const double* parameter, const int Index) const
	{
		double p_x32 = parameter[3] - xi[Index*NumDimension + 2];
		double p_x21 = parameter[2] - xi[Index*NumDimension + 1];
		double p_x10 = parameter[1] - xi[Index*NumDimension + 0];
		return std::exp(-(std::pow(p_x32,2) + std::pow(p_x21,2) + std::pow(p_x10,2))/std::pow(parameter[0],2));
	}
};


int main(int argc, char* argv[])
{

	typedef LeastSquare::BaseModel<double,double> baseModel;
	const int NumDomain = 20;
	const int NumParameter = 4;

	std::vector<double> x;
	std::vector<double> y;
	baseModel* model;
	double parameter[NumParameter] = {10,4,1,1};

	LeastSquare::loadtxt("input1.dat", 1, x, y);

	model = new SineModel<>(y.size());
	model -> inData(x, y);

	std::cout<<"init cost(sine):"<<model -> cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model, parameter, (int)1e4, 1e-15);
	delete model;

	LeastSquare::loadtxt("input2.dat", 3, x, y);
	model = new MultiLinearModel<>(y.size());
	model -> inData(x, y);

	std::cout<<"init cost(linear):"<<model -> cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model, parameter, (int)1e4, 1e-15);
	delete model;

	LeastSquare::loadtxt("input.dat", 3, x, y);
	model = new ExponentialModel<>(y.size());
	model -> inData(x, y);

	std::cout<<"init cost(exp):"<<model -> cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model, parameter, (int)1e4, 1e-10);

	delete model;

	model = new MultiLinearModel<>(y.size());
	LeastSquare::loadtxt("input.dat", 3, x, y);
	model -> inData(x, y);

	std::cout<<"init cost(linear):"<<model -> cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model, parameter, (int)1e4, 1e-10);

	delete model;

	return 0;
}
