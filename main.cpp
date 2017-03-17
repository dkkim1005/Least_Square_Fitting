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
        LinearModel(const int NumDomain_, const int NumParameter_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, NumParameter_)
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
        SquareModel(const int NumDomain_, const int NumParameter_)
        : LeastSquare::BaseModel<Tx,Ty>(NumDomain_, NumParameter_)
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


typedef LeastSquare::BaseModel<double,double> baseModel;

int main()
{
	const int NumDomain = 20;
	const int NumParameter = 3;

	std::vector<double> x(NumDomain,0);
	std::vector<double> y(NumDomain,0);
	double parameter[3] = {-10, 13,123};

	for(int i=0;i<NumDomain;++i)
	{
		x[i] = i;
		y[i] = 3*x[i]*x[i] + 123*x[i] + 21;
	}

	baseModel *linear = new LinearModel<double, double>(NumDomain, 2),
		  *square = new SquareModel<double, double>(NumDomain, 3);

	linear -> inData(x, y);
	square -> inData(x, y);

	LeastSquare::Levenberg_Marquardt(linear, parameter, (int)1e4, 1e-15);
	LeastSquare::Levenberg_Marquardt(square, parameter, (int)1e4, 1e-15);

	delete linear;
	delete square;

	return 0;
}
