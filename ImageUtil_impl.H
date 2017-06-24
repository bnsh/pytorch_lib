#ifndef IMAGE_UTIL_IMPL_H
#define IMAGE_UTIL_IMPL_H

#include <vector>
#include <npp.h>
#include <nppi.h>
#include <npps.h>

template <class T> class scratch {
	public:
		scratch<T>(int sz) : _size(sz), _space(NULL) {
			_space = (T *)nppsMalloc_8u(sizeof(T) * _size);
			fprintf(stderr, "%s: %d: gpumalloc(%lu)\n", __FILE__, __LINE__, sizeof(T) * _size);
		}
		
		~scratch<T>() {
			fprintf(stderr, "%s: %d: gpufree(%lu)\n", __FILE__, __LINE__, sizeof(T) * _size);
			_size = -1;
			nppsFree(_space); _space = NULL;
		}

		int size() const { return(_size); }
		T *space() const { return(_space); }
	private:
		int _size;
		T *_space;

		scratch<T>(const scratch<T>&);
		scratch<T>& operator=(const scratch<T>&);
};

template <class T> void swap_scratch(T *array) {
	T swap = array[0];
	array[0] = array[1];
	array[1] = swap;
}

class filter {
	public:
		filter() { }
		virtual ~filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz) = 0;
	private:
		filter(const filter&);
		filter& operator=(const filter&);
};

class noop_filter : public filter {
	public:
		noop_filter() : filter() { }
		virtual ~noop_filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
	private:
		noop_filter(const noop_filter&);
		noop_filter& operator=(const noop_filter&);
};

class gaussian_blur_filter : public filter {
	public:
		gaussian_blur_filter() : filter(), _kernel(NULL) { }
		virtual ~gaussian_blur_filter() { if (_kernel != NULL) delete _kernel; _kernel = NULL; }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
	private:
		scratch<Npp32f> *_kernel;

		gaussian_blur_filter(const gaussian_blur_filter&);
		gaussian_blur_filter& operator=(const gaussian_blur_filter&);
};

class center_crop_filter : public filter {
	public:
		center_crop_filter() : filter(), szx(-1), szy(-1) { }
		virtual ~center_crop_filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
		int szx, szy;
	private:
		center_crop_filter(const center_crop_filter&);
		center_crop_filter& operator=(const center_crop_filter&);
};

class random_square_crop_filter : public filter {
	public:
		random_square_crop_filter() : filter(), minszx(-1), minszy(-1) { }
		virtual ~random_square_crop_filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
		int minszx, minszy;
	private:
		random_square_crop_filter(const random_square_crop_filter&);
		random_square_crop_filter& operator=(const random_square_crop_filter&);
};

class rotatecrop_filter : public filter {
	public:
		rotatecrop_filter() : filter() { }
		virtual ~rotatecrop_filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
	private:
		rotatecrop_filter(const rotatecrop_filter&);
		rotatecrop_filter& operator=(const rotatecrop_filter&);
};

class scale_filter : public filter {
	public:
		scale_filter() : filter(), scaledx(-1), scaledy(-1) { }
		virtual ~scale_filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
		int scaledx;
		int scaledy;
	private:
		scale_filter(const scale_filter&);
		scale_filter& operator=(const scale_filter&);
};

class horizontalflip_filter : public filter {
	public:
		horizontalflip_filter() : filter() { }
		virtual ~horizontalflip_filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
		int szx, szy;
	private:
		horizontalflip_filter(const horizontalflip_filter&);
		horizontalflip_filter& operator=(const horizontalflip_filter&);
};

class colorwash_filter : public filter {
	public:
		colorwash_filter() : filter() { }
		virtual ~colorwash_filter() { }

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
		int szx, szy;
	private:
		colorwash_filter(const colorwash_filter&);
		colorwash_filter& operator=(const colorwash_filter&);
};

class adjust_contrast_filter : public filter {
	public:
		adjust_contrast_filter() : filter(), _scratchspc(NULL), _mean(new scratch<Npp64f>(3)), _std(new scratch<Npp64f>(3)) { }
		virtual ~adjust_contrast_filter() {
			if (_scratchspc != NULL) delete _scratchspc; _scratchspc = NULL;
			if (_mean != NULL) delete _mean; _mean = NULL;
			if (_std != NULL) delete _std; _std = NULL;
		}

		virtual void operator()(const Npp32f *input, NppiSize inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, NppiSize& outputsz, int& outputlinesz);
		int szx, szy;
	private:
		scratch<Npp8u> *_scratchspc;
		scratch<Npp64f> *_mean;
		scratch<Npp64f> *_std;
		adjust_contrast_filter(const adjust_contrast_filter&);
		adjust_contrast_filter& operator=(const adjust_contrast_filter&);
};

class ImageUtil_impl {
	public:
		ImageUtil_impl();
		virtual ~ImageUtil_impl();

		void transform(bool random, const THCudaTensor *inputbatch, THCudaTensor *transformedbatch, THCudaTensor *conversion_matrix);

	private:
		void random_filter(const Npp32f *input, const NppiSize& inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, const NppiSize& outputsz, int outputlinesz);
		void center_filter(const Npp32f *input, const NppiSize& inputsz, int inputlinesz, THCudaTensor *conversion_matrix, Npp32f *output, const NppiSize& outputsz, int outputlinesz);
		void grab_temporaries(const NppiSize& sz);
		scratch<Npp32f> *_scratch_floats[2]; // We read from 0 and write to 1. (Like fd[0] and fd[1])

		noop_filter noop;

		random_square_crop_filter random_square_crop;
		gaussian_blur_filter gaussian_blur;
		rotatecrop_filter rotatecrop;
		scale_filter scale;
		horizontalflip_filter horizontalflip;
		colorwash_filter colorwash;
		adjust_contrast_filter adjust_contrast;

		scale_filter scaleup_square;
		scale_filter normalized_square;
		center_crop_filter center_crop;

		// BIG THREE
		ImageUtil_impl(const ImageUtil_impl&);
		ImageUtil_impl& operator=(const ImageUtil_impl&);
};

#endif
