#include <assert.h>
#include <TH/TH.h>
#include <THC/THC.h>
#include "ImageUtil_impl.H"
extern "C" {
typedef bool _Bool;
#include "ImageUtil.H"
}

extern THCState *state;

void *ImageUtil() { return new ImageUtil_impl(); }
void ImageUtil_transform(void *v, bool random, const THCudaTensor *inputbatch, THCudaTensor *transformedbatch, THCudaTensor *conversion_matrix) {
	assert(v != NULL);
	ImageUtil_impl *iu = (ImageUtil_impl *)v;
	iu->transform(random, inputbatch, transformedbatch, conversion_matrix);
}

void ImageUtil_destroy(void *v) {
	ImageUtil_impl *iu = (ImageUtil_impl *)v;
	if (iu) delete iu; iu = NULL;
}

void test(void *v, THCudaTensor **masks) {
	fprintf(stderr, "WOOT!\n");
}
