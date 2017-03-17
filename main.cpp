#include <iostream>
#include <math.h>
#include "LeastSquare.h"

// case 1. linear fitting
template <typename Tx, typename Ty> 
class LinearModel : public LeastSquare::BaseModel<Tx,Ty>
{
        using LeastSquare::BaseModel<Tx,Ty>::NumDomain;
        using LeastSquare::BaseModel<Tx,Ty>::NumParameter;
        using LeastSquare::BaseModel<Tx,Ty>::NumDimension;
        using LeastSquare::BaseModel<Tx,Ty>::xi;
        using LeastSquare::BaseModel<Tx,Ty>::yi;

public:
        explicit LinearModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, 2)
        {}

        virtual double cost(const double* parameter) const
        {
                double accum = 0;
                for(int i=0; i<NumDomain; ++i)
                        accum += std::pow(yi[i] - (parameter[0]*xi[i] + parameter[1]), 2); 
                return accum/2.;
        }

        virtual double cost(const double* parameter, const int i) const
        {
                return std::pow(yi[i] - (parameter[0]*xi[i] + parameter[1]), 2)/2.;
        }

        virtual void get_jacobian(const double* parameter, double* jacobian) const
        {
                for(int i=0; i<NumDomain; ++i)
                {
                        jacobian[i*NumParameter + 0] = -(yi[i] - (parameter[0]*xi[i] + parameter[1]))*xi[i];
                        jacobian[i*NumParameter + 1] = -(yi[i] - (parameter[0]*xi[i] + parameter[1]));
		}
        }
};


// case 2. non-linear fitting(square function)
template <typename Tx, typename Ty> 
class SquareModel : public LeastSquare::BaseModel<Tx,Ty>
{
        using LeastSquare::BaseModel<Tx,Ty>::NumDomain;
        using LeastSquare::BaseModel<Tx,Ty>::NumParameter;
        using LeastSquare::BaseModel<Tx,Ty>::NumDimension;
        using LeastSquare::BaseModel<Tx,Ty>::xi;
        using LeastSquare::BaseModel<Tx,Ty>::yi;

public:
        explicit SquareModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, 3)
        {}

        virtual double cost(const double* parameter) const
        {
                double accum = 0;
                for(int i=0; i<NumDomain; ++i)
                        accum += std::pow(yi[i] - (parameter[0]*xi[i]*xi[i] 
				+ parameter[1]*xi[i] + parameter[2]), 2); 
                return accum/2.;
        }

        virtual double cost(const double* parameter, const int i) const
        {
                return std::pow(yi[i] - (parameter[0]*xi[i]*xi[i] 
				+ parameter[1]*xi[i] + parameter[2]), 2)/2.;
        }

        virtual void get_jacobian(const double* parameter, double* jacobian) const
        {
                for(int i=0; i<NumDomain; ++i)
                {
                        jacobian[i*NumParameter + 0] = -(yi[i] - (parameter[0]*xi[i]*xi[i] 
							+ parameter[1]*xi[i] + parameter[2]))*xi[i]*xi[i];
                        jacobian[i*NumParameter + 1] = -(yi[i] - (parameter[0]*xi[i]*xi[i] 
							+ parameter[1]*xi[i] + parameter[2]))*xi[i]; 
                        jacobian[i*NumParameter + 2] = -(yi[i] - (parameter[0]*xi[i]*xi[i] 
							+ parameter[1]*xi[i] + parameter[2]));
		}
        }
};


// case 3. linear fitting(multivariable function)
template <typename Tx, typename Ty> 
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

        virtual double cost(const double* parameter) const
        {
                double accum = 0;
                for(int i=0; i<NumDomain; ++i)
                        accum += std::pow(yi[i] - (parameter[3]*xi[i+2] + parameter[2]*xi[i+1]
				+ parameter[1]*xi[i] + parameter[0]), 2); 
                return accum/2.;
        }

        virtual double cost(const double* parameter, const int i) const
        {
                return std::pow(yi[i] - (parameter[3]*xi[i+2] + parameter[2]*xi[i+1]
				+ parameter[1]*xi[i] + parameter[0]), 2)/2.;
        }

        virtual void get_jacobian(const double* parameter, double* jacobian) const
        {
                for(int i=0; i<NumDomain; ++i)
                {
                        jacobian[i*NumParameter + 0] = -(yi[i] - (parameter[3]*xi[i+2] + parameter[2]*xi[i+1]
							+ parameter[1]*xi[i] + parameter[0]))*xi[i+2];

                        jacobian[i*NumParameter + 1] = -(yi[i] - (parameter[3]*xi[i+2] + parameter[2]*xi[i+1]
							+ parameter[1]*xi[i] + parameter[0]))*xi[i+1];

                        jacobian[i*NumParameter + 2] = -(yi[i] - (parameter[3]*xi[i+2] + parameter[2]*xi[i+1]
							+ parameter[1]*xi[i] + parameter[0]))*xi[i];

                        jacobian[i*NumParameter + 3] = -(yi[i] - (parameter[3]*xi[i+2] + parameter[2]*xi[i+1]
							+ parameter[1]*xi[i] + parameter[0]));
		}
        }
};


// case 4. non-linear fitting(multivariable function)
template <typename Tx, typename Ty> 
class MultiNonLinearModel : public LeastSquare::BaseModel<Tx,Ty>
{
        using LeastSquare::BaseModel<Tx,Ty>::NumDomain;
        using LeastSquare::BaseModel<Tx,Ty>::NumParameter;
        using LeastSquare::BaseModel<Tx,Ty>::NumDimension;
        using LeastSquare::BaseModel<Tx,Ty>::xi;
        using LeastSquare::BaseModel<Tx,Ty>::yi;

public:
        explicit MultiNonLinearModel(const int NumDomain_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, 4, 3)
        {}

        virtual double cost(const double* parameter) const
        {
                double accum = 0;
                for(int i=0; i<NumDomain; ++i)
		{
			double x_t32 = parameter[3] - xi[i+2];
			double x_t21 = parameter[2] - xi[i+1];
			double x_t10 = parameter[1] - xi[i];
                        accum += std::pow(yi[i] - exp(-(std::pow(x_t32,2) + std::pow(x_t21,2) 
					+ std::pow(x_t10,2))/parameter[0]), 2); 
		}
                return accum/2.;
        }

        virtual double cost(const double* parameter, const int i) const
        {
		double x_t32 = parameter[3] - xi[i+2];
		double x_t21 = parameter[2] - xi[i+1];
		double x_t10 = parameter[1] - xi[i];

		return  std::pow(yi[i] - exp(-(std::pow(x_t32,2) + std::pow(x_t21,2) + std::pow(x_t10,2))/parameter[0]), 2)/2.; 
        }

        virtual void get_jacobian(const double* parameter, double* jacobian) const
        {
                for(int i=0; i<NumDomain; ++i)
                {
			double x_t32 = parameter[3] - xi[i+2];
			double x_t21 = parameter[2] - xi[i+1];
			double x_t10 = parameter[1] - xi[i];

                        jacobian[i*NumParameter + 0] = -2*(yi[i] - exp(-(std::pow(x_t32,2) 
							+ std::pow(x_t21,2) 
							+ std::pow(x_t10,2))/parameter[0]))*
							exp(-(std::pow(x_t32,2) + std::pow(x_t21,2)
							+ std::pow(x_t10,2))/parameter[0])*
							(std::pow(x_t32,2) + std::pow(x_t21,2) + std::pow(x_t10,2))/std::pow(parameter[0],2);


                        jacobian[i*NumParameter + 1] = 2*(yi[i] - exp(-(std::pow(x_t32,2) 
							+ std::pow(x_t21,2) 
							+ std::pow(x_t10,2))/parameter[0]))*
							exp(-(std::pow(x_t32,2) + std::pow(x_t21,2)
							+ std::pow(x_t10,2))/parameter[0])*
							x_t10/parameter[0];

                        jacobian[i*NumParameter + 2] = 2*(yi[i] - exp(-(std::pow(x_t32,2) 
							+ std::pow(x_t21,2) 
							+ std::pow(x_t10,2))/parameter[0]))*
							exp(-(std::pow(x_t32,2) + std::pow(x_t21,2)
							+ std::pow(x_t10,2))/parameter[0])*
							x_t21/parameter[0];

                        jacobian[i*NumParameter + 3] = 2*(yi[i] - exp(-(std::pow(x_t32,2) 
							+ std::pow(x_t21,2) 
							+ std::pow(x_t10,2))/parameter[0]))*
							exp(-(std::pow(x_t32,2) + std::pow(x_t21,2)
							+ std::pow(x_t10,2))/parameter[0])*
							x_t32/parameter[0];
		}
        }
};


int main(int argc, char* argv[])
{

	typedef LeastSquare::BaseModel<double,double> baseModel;
	const int NumDomain = 20;
	const int NumParameter = 3;

	std::vector<double> x;
	std::vector<double> y;
	baseModel* model;
	double parameter[4] = {1,1,2};

	LeastSquare::loadtxt("input.dat", 3, x, y);

	model = new MultiLinearModel<double, double>(y.size());
	model -> inData(x, y);

	std::cout<<"init cost(linear):"<<model -> cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model, parameter, (int)1e4, 1e-15);
	delete model;


	model = new MultiNonLinearModel<double, double>(y.size());
	model -> inData(x, y);

	std::cout<<"init cost(non-linear):"<<model -> cost(parameter)<<std::endl;

	LeastSquare::Levenberg_Marquardt(model, parameter, (int)1e4, 1e-20);
	delete model;

	return 0;
}
