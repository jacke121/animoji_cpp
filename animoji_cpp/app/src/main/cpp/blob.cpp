#include <string>
#include <assert.h>
#include <cstdlib>
#include <iostream>
#include "blob.hpp"

#define INT_MAX 2147483647    /* maximum (signed) int value */
namespace  galaxy {
    bbox::bbox() :x1(0), y1(0), x2(0), y2(0), score(0), array_(NULL){}

    bbox::bbox(int _x1, int _y1, int _x2, int _y2, float _score)
        :x1(_x1), y1(_y1), x2(_x2), y2(_y2), score(_score), array_(NULL){}

    float* bbox::create_array(int n) {
        array_ = std::shared_ptr<float>(new float[n], std::default_delete<float[]>());
        return array_.get();
    }

    float* bbox::array() {
        return array_.get();
    }

    bbox::bbox(const std::shared_ptr<float> &array_) : array_(array_) {}


    Blob::Blob(const int axis1){
		count_ = axis1;
        capacity_ = count_ * sizeof(float);
        data_ = (float*)malloc(capacity_);
        shape_ = { axis1 };
	}

    Blob::Blob(const int axis1, const int axis2){
		assert(axis2 < INT_MAX / axis1);
		count_ = axis1*axis2;
        capacity_ = count_ * sizeof(float);
        data_ = (float*)malloc(capacity_);
        shape_ = { axis1,axis2 };
	}

    Blob::Blob(const int axis1, const int axis2, const int axis3){
		assert(axis2 < INT_MAX / axis1);
		assert(axis3 < INT_MAX / axis2 / axis1);
		count_ = axis1*axis2*axis3;
        capacity_ = count_ * sizeof(float);
        data_ = (float*)malloc(capacity_);
        shape_ = { axis1,axis2,axis3 };
	}

	Blob::Blob(const int axis1, const int axis2, const int axis3,
        const int axis4){
		assert(axis2 < INT_MAX / axis1);
		assert(axis3 < INT_MAX / axis2 / axis1);
		assert(axis4 < INT_MAX / axis3 / axis2 / axis1);
		count_ = axis1*axis2*axis3*axis4;
        capacity_ = count_ * sizeof(float);
        data_ = (float*)malloc(capacity_);
        shape_ = { axis1,axis2,axis3,axis4 };
	}

    Blob::Blob(const Shape& shape){
		count_ = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            assert(shape[i] < (INT_MAX / count_));
            count_ *= shape[i];
		}
        capacity_ = count_ * sizeof(float);
        data_ = (float*)malloc(capacity_);
        shape_.assign(shape.begin(), shape.end());
	}

    void Blob::reshape(const Shape& shape) {
        count_ = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            count_ *= shape[i];
        }
        int capacity = count_*sizeof(float);

        if (capacity != capacity_) {
            capacity_ = capacity;
            free(data_);
            data_ = (float*)malloc(capacity_);
        }
        shape_.assign(shape.begin(), shape.end());
    }

	int Blob::shape(int index) const{
        return shape_[index];
	}

	const Shape& Blob::shape() const {
		return shape_;
	}

	int Blob::num_axes() const { 
		return static_cast<int>(shape_.size());
	}

	Blob::~Blob() { 
        free(data_);
    }

    float* Blob::data() const {
		return data_;
	}
#ifdef _DEBUG
    void Blob::print_data(bool brief) {
        using std::cout;
        using std::endl;
        int n_axes(num_axes());
        int r_print, c_print;
        if (brief) {
            r_print = 5;
            c_print = 3;
        }
        else {
            r_print = INT_MAX;
            c_print = INT_MAX;
        }
        if (n_axes > 2 && shape_[n_axes - 2] < r_print)
            r_print = (r_print / shape_[n_axes - 2] + 1)*shape_[n_axes - 2];

        cout << "Shape: ";
        for (int i = 0; i < n_axes; ++i)
            cout << shape_[i] << " ";
        cout << endl;

        cout << "Data: " << endl;
        if (!data_) {
            cout << "NULL" << endl;
            return;
        }
        std::vector<int> its(n_axes, 0);
        cout << std::string(n_axes, '[');

        int output_col = 0;
        int output_row = 0;
        int shapeinv = shape_[n_axes - 1];
        int i = 0;
        while (i < count_ - 1) {
            int step = 1;
            cout.width(8);
            cout << data()[i] << " ";
            output_col++;
            if (shapeinv > 2 * c_print && output_col == c_print) {
                cout << "..., ";
                step = shapeinv - 2 * c_print + 1;
            }
            int cn = 0;
            its[n_axes - 1] += step;
            i += step;
            for (int j = n_axes - 1; j > 0; --j) {
                if (its[j] >= shape_[j]) {
                    its[j - 1] += 1;
                    cout << "]";
                    output_col = 0;
                    cn++;
                    its[j] = 0;
                }
            }
            if (cn > 0) {
                cout << endl;
                output_row++;
            }
            if (cn > 1) cout << endl;

            if (output_row == r_print) {
                int it = count_ - shapeinv * r_print;
                if (it > i) {
                    i = it;
                    cout << std::string(n_axes - cn, ' ');
                    cout << "...," << endl;
                    its[n_axes - 1] = i;
                    for (int j = n_axes - 1; j > 0; --j) {
                        if (its[j] >= shape_[j]) {
                            its[j - 1] = its[j] / shape_[j];
                            its[j] %= shape_[j];
                        }
                    }
                }
                output_row++;
            }
            cout << std::string(n_axes - cn, ' ');
            cout << std::string(cn, '[');
        }
        cout.width(8);
        if (count_ > 0)	cout << data()[count_ - 1] << " ";
        cout << std::string(n_axes, ']');
        cout << endl << endl;
        return;
    }
#endif
}// namespace galaxy
