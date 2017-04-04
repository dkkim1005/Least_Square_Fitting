#include <iostream>
#include <math.h>
#include "LeastSquare.h"


// case 1. sine fitting
class SineModel : public LeastSquare::BaseModelReal
{

public:
        explicit SineModel(const int NumDomain_)
        : LeastSquare::BaseModelReal(NumDomain_, 4)
        {}

	virtual double model_func(const double* parameter, const int Index) const
	{
		return parameter[0]*std::sin(parameter[1]*xi[Index] + parameter[2]) + parameter[3];
	}

};


// case 2. multi-linear fitting
class MultiLinearModel : public LeastSquare::BaseModelReal
{
public:
        explicit MultiLinearModel(const int NumDomain_)
        : LeastSquare::BaseModelReal(NumDomain_, 4, 3)
        {}

	virtual double model_func(const double* parameter, const int Index) const
	{
		return parameter[3]*xi[NumDimension*Index+2] + parameter[2]*xi[NumDimension*Index+1] + parameter[1]*xi[NumDimension*Index] + parameter[0];
	}
};


class multivariable : public LeastSquare::BaseModelReal
{

public:
        explicit multivariable(const int NumDomain_, const int NumDimension_)
        : LeastSquare::BaseModelReal(NumDomain_, NumDimension_ + 1, NumDimension_)
        {}

	virtual double model_func(const double* p, const int Index) const
	{
		double accum = 0;
		for(int i=1;i<NumParameter;++i)
			accum += p[i]*xi[NumDimension*Index+i-1];
		accum += p[0];
		return accum;
	}
};


// case 3. non-linear fitting
class ExponentialModel : public LeastSquare::BaseModelReal
{
public:
        explicit ExponentialModel(const int NumDomain_)
        : LeastSquare::BaseModelReal(NumDomain_, 4, 3)
        {}

	virtual double model_func(const double* parameter, const int Index) const
	{
		double p_x32 = parameter[3] - xi[Index*NumDimension + 2];
		double p_x21 = parameter[2] - xi[Index*NumDimension + 1];
		double p_x10 = parameter[1] - xi[Index*NumDimension + 0];
		return std::exp(-(std::pow(p_x32,2) + std::pow(p_x21,2) + std::pow(p_x10,2))/std::pow(parameter[0],2));
	}
};



int main(int argc, char* argv[])
{
	// real version
	std::cout<<UNDL(FRED("real type fitting"))<<":"<<std::endl;
	const int numParameter = 3;

	auto func = [](const double* p, const double* xi) -> double
		{
			return p[2]*std::pow(xi[0], 0.3) + p[1]*xi[0] + p[0];
		};

	std::vector<double> x;
	std::vector<double> y;

	LeastSquare::loadtxt("fileInput.in", 1, x, y);

	const int numDomain = y.size();

	LeastSquare::FunctionalModelReal model(func, numDomain, numParameter);

	model.in_data(x, y);

	double p[3] = {1,2,3};

	Levenberg_Marquardt(model, p);


	// complex version
	std::cout<<UNDL(FBLU("complex type fitting"))<<":"<<std::endl;
	typedef std::complex<double> dcomplex;
	double comp_para[7] = {1, 1, 1, 1, 1, 1, 1};
	const int compNumParameter = 7;

	auto comp_f = [](const double* p, const dcomplex* xi) -> dcomplex
			{
				return  p[3]*std::pow(xi[0],-p[6]) + 
					p[2]*std::pow(xi[0],-p[5]) + 
					p[1]*std::pow(xi[0],-p[4]) + p[0];
			};

	auto test_f = [](const dcomplex xi) -> dcomplex
			{
				return  0.25*std::pow(xi,-3) +
					0.5*std::pow(xi,-2) + 
					1.*std::pow(xi,-1) + 2.;
			};

#define dc(a,b) std::complex<double>(a,b) 

	std::vector<dcomplex> comp_x = {dc(1,2), dc(3,2), dc(3,4), dc(5,6), dc(1,0)};
	std::vector<dcomplex> comp_y = {test_f(dc(1,2)), test_f(dc(3,2)), test_f(dc(3,4)),
					test_f(dc(5,6)), test_f(dc(1,0))};

	LeastSquare::FunctionalModelComplex modelComplex(comp_f, comp_y.size(), compNumParameter);

	modelComplex.in_data(comp_x, comp_y);

	LeastSquare::Levenberg_Marquardt(modelComplex, comp_para);

	return 0;
}
