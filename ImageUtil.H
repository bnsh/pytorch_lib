/* From https://cffi.readthedocs.io/en/latest/ref.html?highlight=destructor#ffi-gc
 *
 * 	In these cases, consider writing a wrapper class with custom __enter__() and
 * 	__exit__() methods, allocating and freeing the C data at known points in time,
 * 	and using it in a with statement.
 * Maybe I'll make a
 * class ImageUtil:
 *     def __init__(self):
 *         pass
 *     def __enter__(self):
 *         self.ImageUtil = _ext.ImageUtil()
 *     def __exit__(self):
 *         _ext.ImageUtil_destroy(self.ImageUtil)
 *         self.ImageUtil = None
 *     def transform(bool, inputbatch, maskbatch, transformbatch, probabilitiesbatch):
 *          return _ext.ImageUtil_transform(self.ImageUtil, bool, inputbatch, ...)
 */

void *ImageUtil(void);
void ImageUtil_transform(void *iu, _Bool random, const THCudaTensor *inputbatch, THCudaTensor *transformedbatch, THCudaTensor *conversion_matrix);
void ImageUtil_destroy(void *iu);

void test(void *iu, THCudaTensor **masks);
