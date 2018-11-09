#ifndef BLOB_HPP_
#define BLOB_HPP_
#include <vector>
#include <memory>

namespace  galaxy {
	typedef std::vector<int> Shape;
	class Blob {
    public:
        Blob(): count_(0), capacity_(0), data_(NULL), shape_(0){}
		explicit Blob(const int axis1, const int axis2, const int axis3,
			const int axis4);
		explicit Blob(const int axis1, const int axis2,	const int axis3);
		explicit Blob(const int axis1, const int axis2);
		explicit Blob(const int axis1);
		explicit Blob(const Shape& shape);
        void reshape(const Shape& shape);
		~Blob();

        int shape(int index) const;
		const Shape& shape() const;
		int num_axes() const;
		int count() const { return count_; }
		int capacity() const { return capacity_; }

        float* data() const;
#ifdef _DEBUG
        void print_data(bool brief = true);
#endif
	protected:
		int count_;
		int capacity_;
        float* data_;
		Shape shape_; 
	};
	
	class bbox {
	public:
		bbox();
        bbox(int _x1, int _y1, int _x2, int _y2, float _score);

		bbox(const std::shared_ptr<float> &array_);

		float* create_array(int n=10);
        float* array();
		int x1, y1, x2, y2;
        float score;
    protected:
        std::shared_ptr<float> array_;
	};
} //namespace  galaxy 
#endif //BLOB_HPP_
