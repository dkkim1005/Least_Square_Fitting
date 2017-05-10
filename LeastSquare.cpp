#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <list>
#include "LeastSquare.h"

// to compile this code, The Boost-Python(1.64.0 <= ver) is required.
// linking library: -lpython2.x -lboost_python -lboost_numpy -lopenblas -llapack

namespace p = boost::python;
namespace np = boost::python::numpy;

template<typename dType>
class PyModelFunctor
{
public:
	PyModelFunctor(p::object pyFunctor_, int const NumParameter, int const NumXDims)
	:pyFunctor(pyFunctor_), 
	 dtPara(np::dtype::get_builtin<double>()),
	 stridePara(p::make_tuple(sizeof(double))), 
	 shapePara(p::make_tuple(NumParameter)),
	 dtX(np::dtype::get_builtin<dType>()),
	 strideX(p::make_tuple(sizeof(dType))), 
	 shapeX(p::make_tuple(NumXDims))
	{}

	PyModelFunctor(const PyModelFunctor<dType>& model_f_)
	:pyFunctor(model_f_.pyFunctor), 
	 dtPara(model_f_.dtPara),
	 stridePara(model_f_.stridePara), 
	 shapePara(model_f_.shapePara),
	 dtX(model_f_.dtX),
	 strideX(model_f_.strideX), 
	 shapeX(model_f_.shapeX)
	{}

	dType 
	operator()(double const* Parameter, dType const* Xi) const
	{
		np::ndarray pyParameter = np::from_data(Parameter, dtPara, 
							shapePara, stridePara, ownPara);

		np::ndarray pyXi = np::from_data(Xi, dtX, shapeX, strideX, ownXi);

		dType result = p::extract<dType>(pyFunctor(pyParameter, pyXi));

		return result;
	}

	p::object pyFunctor;
	np::dtype const dtPara, dtX;
	p::tuple const stridePara, shapePara, strideX, shapeX;
	p::object ownPara, ownXi;
};


class PyModelReal : public LeastSquare::BaseModelReal
{
public:
	PyModelReal(const PyModelFunctor<double>& model_f_, const int NumSample_,
	const int NumParameter_, const int NumDimension_)
	: BaseModelReal(NumSample_, NumParameter_, NumDimension_), model_f(model_f_),
	_xi_copy(new double [NumDimension_])
	{}

	virtual 
	~PyModelReal() {
		if(_xi_copy != nullptr) { delete [] _xi_copy; }
	}

	virtual double 
	model_func(const double* parameter, const int Index) const
	{
		std::memcpy(_xi_copy, &xi[NumDimension*Index], sizeof(double)*NumDimension);
		return model_f(parameter, _xi_copy);
	}

private:
	double* _xi_copy = nullptr;
	PyModelFunctor<double> const model_f;

};


class PyLeastSquareReal
{
public:
	PyLeastSquareReal(p::object pyFunctor_, int NumParameter_, int NumDimension_)
	: model_f(pyFunctor_, NumParameter_, NumDimension_), NumParameter(NumParameter_), 
	  NumDimension(NumDimension_)
	{}

	void in_data(p::object x, p::object y)
	{
		assert(p::len(x) == p::len(y));
		assert(p::len(x[0]) == NumDimension);
		size_t const sampleSize = p::len(x);

		for(int i=0; i<sampleSize; ++i)
		{
			for(int j=0; j<NumDimension; ++j) 
			{
				double xTemp = p::extract<double>(x[i][j]);
				x_list.push_back(xTemp);
			}

			double yTemp = p::extract<double>(y[i]);

			y_list.push_back(yTemp);
		}


	}

	void fitting(p::object& initialGuess, int iter, double tol, int printlog)
	{
		assert(p::len(initialGuess) == NumParameter);
		size_t const SampleSize = y_list.size();
		std::vector<double> x(x_list.size(), 0), y(y_list.size(), 0);
		int nXIter = 0, nYIter = 0;

		for(auto& xIn : x_list)
		{
			x[nXIter] = xIn;
			nXIter += 1;
		}

		for(auto& yIn : y_list)
		{
			y[nYIter] = yIn;
			nYIter += 1;
		}


		PyModelReal model(model_f, SampleSize, NumParameter, NumDimension);
		model.in_data(x, y);

		std::vector<double> parameter(NumParameter, 0);

		for(int i=0; i<NumParameter; ++i) {
			parameter[i] = p::extract<double>(initialGuess[i]);
		}

		LeastSquare::Levenberg_Marquardt(model, &parameter[0], iter, tol, printlog);

		for(int i=0; i<NumParameter; ++i) {
			 initialGuess[i] = parameter[i];
		}
	}

private:
	PyModelFunctor<double> const model_f;
	int const NumParameter, NumDimension;
	std::list<double> x_list, y_list;

};


class PyModelComplex : public LeastSquare::BaseModelComplex
{

using dcomplex = std::complex<double>;

public:
	PyModelComplex(const PyModelFunctor<dcomplex>& model_f_, const int NumSample_,
	const int NumParameter_, const int NumDimension_)
	: BaseModelComplex(NumSample_, NumParameter_, NumDimension_), model_f(model_f_),
	_xi_copy(new dcomplex [NumDimension_])
	{}

	virtual 
	~PyModelComplex() {
		if(_xi_copy != nullptr) { delete [] _xi_copy; }
	}

	virtual dcomplex
	model_func(const double* parameter, const int Index) const
	{
		std::memcpy(_xi_copy, &xi[NumDimension*Index], sizeof(dcomplex)*NumDimension);
		return model_f(parameter, _xi_copy);
	}

private:
	dcomplex* _xi_copy = nullptr;
	PyModelFunctor<dcomplex> const model_f;
};


class PyLeastSquareComplex
{

using dcomplex = std::complex<double>;

public:
	PyLeastSquareComplex(p::object pyFunctor_, int NumParameter_, int NumDimension_)
	: model_f(pyFunctor_, NumParameter_, NumDimension_), NumParameter(NumParameter_), 
	  NumDimension(NumDimension_)
	{}

	void in_data(p::object x, p::object y)
	{
		assert(p::len(x) == p::len(y));
		assert(p::len(x[0]) == NumDimension);
		size_t const sampleSize = p::len(x);

		for(int i=0; i<sampleSize; ++i)
		{
			for(int j=0; j<NumDimension; ++j) 
			{
				dcomplex xTemp = p::extract<dcomplex>(x[i][j]);
				x_list.push_back(xTemp);
			}

			dcomplex yTemp = p::extract<dcomplex>(y[i]);

			y_list.push_back(yTemp);
		}


	}

	void fitting(p::object& initialGuess, int iter, double tol, int printlog)
	{
		assert(p::len(initialGuess) == NumParameter);
		size_t const SampleSize = y_list.size();
		std::vector<dcomplex> x(x_list.size(), 0), y(y_list.size(), 0);
		int nXIter = 0, nYIter = 0;

		for(auto& xIn : x_list)
		{
			x[nXIter] = xIn;
			nXIter += 1;
		}

		for(auto& yIn : y_list)
		{
			y[nYIter] = yIn;
			nYIter += 1;
		}


		PyModelComplex model(model_f, SampleSize, NumParameter, NumDimension);
		model.in_data(x, y);

		std::vector<double> parameter(NumParameter, 0);

		for(int i=0; i<NumParameter; ++i) {
			parameter[i] = p::extract<double>(initialGuess[i]);
		}

		LeastSquare::Levenberg_Marquardt(model, &parameter[0], iter, tol, printlog);

		for(int i=0; i<NumParameter; ++i) {
			 initialGuess[i] = parameter[i];
		}
	}

private:
	PyModelFunctor<dcomplex> const model_f;
	int const NumParameter, NumDimension;
	std::list<dcomplex> x_list, y_list;
};


BOOST_PYTHON_MODULE(LeastSquare)
{
	np::initialize();

	p::class_<PyLeastSquareReal>("LeastSquareReal", p::init<p::object, int, int>())
	.def("in_data", &PyLeastSquareReal::in_data)
	.def("fitting", &PyLeastSquareReal::fitting);

	p::class_<PyLeastSquareComplex>("LeastSquareComplex", p::init<p::object, int, int>())
	.def("in_data", &PyLeastSquareComplex::in_data)
	.def("fitting", &PyLeastSquareComplex::fitting);
}
