#include <iostream>
#include <math.h>
#include "LeastSquare.h"

// case 1. linear fitting
template <typename Tx, typename Ty, typename Tp> 
class LinearModel : public LeastSquare::BaseModelClass<Tx,Ty,Tp>
{
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::NumDomain;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::NumParameter;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::NumDimension;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::xi;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::yi;

public:
        LinearModel(const int NumDomain_, const int NumParameter_)
        : LeastSquare::BaseModelClass<Tx,Ty,Tp>(NumDomain_, NumParameter_)
        {}

        virtual double cost(const Tp* parameter) const
        {
                double accum = 0;
                for(int i=0; i<NumDomain; ++i)
                        accum += std::pow(yi[i] - (parameter[0]*xi[i] + parameter[1]), 2); 
                return accum/2.;
        }

        virtual double cost(const Tp* parameter, const int i) const
        {
                return std::pow(yi[i] - (parameter[0]*xi[i] + parameter[1]), 2)/2.;
        }

        virtual void get_jacobian(const Tp* parameter, double* jacobian) const
        {
                for(int i=0; i<NumDomain; ++i)
                {
                        jacobian[i*NumParameter + 0] = -(yi[i] - (parameter[0]*xi[i] + parameter[1]))*xi[i];
                        jacobian[i*NumParameter + 1] = -(yi[i] - (parameter[0]*xi[i] + parameter[1]));
		}
        }
};


// case 2. non-linear fitting(square function)
template <typename Tx, typename Ty, typename Tp> 
class SquareModel : public LeastSquare::BaseModelClass<Tx,Ty,Tp>
{
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::NumDomain;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::NumParameter;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::NumDimension;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::xi;
        using LeastSquare::BaseModelClass<Tx,Ty,Tp>::yi;

public:
        SquareModel(const int NumDomain_, const int NumParameter_)
        : LeastSquare::BaseModelClass<Tx,Ty,Tp>(NumDomain_, NumParameter_)
        {}

        virtual double cost(const Tp* parameter) const
        {
                double accum = 0;
                for(int i=0; i<NumDomain; ++i)
                        accum += std::pow(yi[i] - (parameter[0]*xi[i]*xi[i] 
				+ parameter[1]*xi[i] + parameter[2]), 2); 
                return accum/2.;
        }

        virtual double cost(const Tp* parameter, const int i) const
        {
                return std::pow(yi[i] - (parameter[0]*xi[i]*xi[i] 
				+ parameter[1]*xi[i] + parameter[2]), 2)/2.;
        }

        virtual void get_jacobian(const Tp* parameter, double* jacobian) const
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


typedef LeastSquare::BaseModelClass<double,double,double> baseModel;

int main()
{
	const int NumDomain = 20;
	const int NumParameter = 3;

	std::vector<double> x(NumDomain,0);
	std::vector<double> y(NumDomain,0);
	double parameter[3] = {-213, 1023,123};

	for(int i=0;i<NumDomain;++i)
	{
		x[i] = i;
		y[i] = M_PI*x[i]*x[i] + 123*x[i] + 21;
	}

	baseModel* model = new SquareModel<double, double, double>(NumDomain, NumParameter);
	model -> inData(x, y);

	LeastSquare::Levenberg_Marquardt(model, parameter, (int)1e4, 1e-15);

	delete model;

	return 0;
}
