#include <iostream>
#include <math.h>
#include "LeastSquare.h"

// case 1. sine fitting
template <typename Tx = double> 
class SineModel : public LeastSquare::BaseModel<Tx>
{
        using LeastSquare::BaseModel<Tx>::NumDomain;
        using LeastSquare::BaseModel<Tx>::NumParameter;
        using LeastSquare::BaseModel<Tx>::NumDimension;
        using LeastSquare::BaseModel<Tx>::xi;

public:
        explicit SineModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx>(NumDomain_, 4)
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
template <typename Tx = double> 
class MultiLinearModel : public LeastSquare::BaseModel<Tx>
{
        using LeastSquare::BaseModel<Tx>::NumDomain;
        using LeastSquare::BaseModel<Tx>::NumParameter;
        using LeastSquare::BaseModel<Tx>::NumDimension;
        using LeastSquare::BaseModel<Tx>::xi;

public:
        explicit MultiLinearModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx>(NumDomain_, 4, 3)
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


template <typename Tx = double> 
class multivariable : public LeastSquare::BaseModel<Tx>
{
        using LeastSquare::BaseModel<Tx>::NumDomain;
        using LeastSquare::BaseModel<Tx>::NumParameter;
        using LeastSquare::BaseModel<Tx>::NumDimension;
        using LeastSquare::BaseModel<Tx>::xi;

public:
        explicit multivariable(const int NumDomain_, const int NumDimension_)
        : LeastSquare::BaseModel<Tx>(NumDomain_, NumDimension_ + 1, NumDimension_)
        {}

	virtual double modelFunc(const double* p, const int Index) const
	{
		double accum = 0;
		for(int i=1;i<NumParameter;++i)
			accum += p[i]*xi[NumDimension*Index+i-1];
		accum += p[0];
		return accum;
	}
};



// case 3. non-linear fitting
template <typename Tx = double> 
class ExponentialModel : public LeastSquare::BaseModel<Tx>
{
        using LeastSquare::BaseModel<Tx>::NumDomain;
        using LeastSquare::BaseModel<Tx>::NumParameter;
        using LeastSquare::BaseModel<Tx>::NumDimension;
        using LeastSquare::BaseModel<Tx>::xi;

public:
        explicit ExponentialModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx>(NumDomain_, 4, 3)
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
	typedef LeastSquare::BaseModel<double> baseModel;
	const int NumDomain = 20;
	const int NumDimension = 60;

	std::vector<double> x;
	std::vector<double> y;
	baseModel* model;

	double parameter[NumDimension+1];
	for(auto & p : parameter) p = 1.;

	LeastSquare::loadtxt("input3.dat", NumDimension, x, y);

	multivariable<> model1(y.size(), NumDimension);
	model1.inData(x, y);

	std::cout<<"init cost(linear):"<<model1.cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model1, parameter, (int)1e4, 1e-10);


	LeastSquare::loadtxt("input.dat", 1, x, y);

	double p[4] = {1, 4, 10, -3};

        SineModel<> model2(y.size());

	model2.inData(x, y);

	std::cout<<"init cost(linear):"<<model2.cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model2, p, (int)1e4, 1e-10);


	p[0] = 1; p[1] = 4; p[2] = 10; p[3] = -3;

	auto func = [](const double* p, const double* x) -> double
			{
				return p[0]*std::sin(p[1]*x[0] + p[2]) + p[3];
			};

	LeastSquare::FunctionalModel<> model3(func, y.size(), 4);

	model3.inData(x, y);

	LeastSquare::Levenberg_Marquardt(model3, p, (int)1e4, 1e-10);

	return 0;
}
