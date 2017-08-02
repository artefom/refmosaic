#!/usr/bin/env python3

import argparse
# from scipy import misc
import sys
import glob
import numpy as np
import itertools
import os
import traceback
import tempfile
import time
from sklearn.linear_model import Ridge
from PIL import Image

image_processing_scale = 1

def transform_colors(target, source, scale=0.5):
    X = source.reshape((-1, 3))

    Yr = target.reshape((-1, 3))[:, 0] / 255
    Yg = target.reshape((-1, 3))[:, 1] / 255
    Yb = target.reshape((-1, 3))[:, 2] / 255

    clf = Ridge()

    clf.fit(X, Yr)
    ret_r = clf.predict(X)

    clf.fit(X, Yg)
    ret_g = clf.predict(X)

    clf.fit(X, Yb)
    ret_b = clf.predict(X)

    ret = np.concatenate((ret_r[:, np.newaxis], ret_g[:, np.newaxis], ret_b[:, np.newaxis]), axis=1).reshape(
        source.shape) * 255
    ret = (source * (1 - scale) + ret * scale)
    ret[ret > 255] = 255
    ret[ret < 0] = 0
    ret = ret.astype(np.uint8)
    return ret

def sorted_merge_iterator(merged_arrays, key):
    iterators = []
    bufs = []
    for i in merged_arrays:
        try:
            it = iter(i)
            bufs.append(next(it))
            iterators.append(it)
        except (StopIteration,TypeError):
            pass

    if len(bufs) == 0:
        return

    buf_values = [key(i) for i in bufs]

    while True:
        min_score = np.argmin(buf_values)
        yield bufs[min_score]
        try:
            bufs[min_score] = next(iterators[min_score])
            buf_values[min_score] = key(bufs[min_score])
        except:
            del bufs[min_score]
            del iterators[min_score]
            del buf_values[min_score]

            if len(bufs) == 0:
                raise StopIteration

def image_to_array(img):
    ret_img = np.array(img)
    try:
        ret_img = ret_img[:,:,:3]
    except IndexError as ex:
        arr = ret_img[:,:,np.newaxis]
        ret_img = np.concatenate( (arr,arr,arr), axis=2 )
    return ret_img

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def get_crop_scale(input_shape, target_shape):
    """
    Returns min scale multiplier that allows input_image fill whole target_shape
    :param input_shape:
    :param target_shape:
    :return:
    """
    aspect_input = input_shape[1] / input_shape[0]
    aspect_target = target_shape[1] / target_shape[0]

    # scale input image, so it fits output along this axis
    scale_axis = 1  # fit width to target
    if aspect_input > aspect_target:
        scale_axis = 0  # fit height to target

    scale = target_shape[scale_axis] / input_shape[scale_axis]
    return scale


class memmaped_arraylist_iter:
    def __init__(self, ar_lst, block_size=100000):
        self.last_block = 0
        self.block_size = block_size
        self.ar_lst = ar_lst

        self.current_block = self.get_block(self.last_block)
        if self.current_block is not None:
            self.current_block_iterator = iter(self.current_block)
        else:
            self.current_block = None

        self.last_block += 1

    def get_block(self, block_index):
        beg = self.last_block * self.block_size
        end = beg + self.block_size
        if beg > self.ar_lst.size:
            return None  # Done iterating.
        if end > self.ar_lst.size:
            end = self.ar_lst.size
        return self.ar_lst.at_range(beg, end)

    def __next__(self):

        try:
            try:
                return self.current_block_iterator.__next__()
            except StopIteration:
                pass
            except Exception:
                raise StopIteration

            self.current_block = self.get_block(self.last_block)
            self.current_block_iterator = iter(self.current_block)
            self.last_block += 1

        except Exception:
            raise StopIteration

        return self.current_block_iterator.__next__()


class memmaped_arraylist:
    def __init__(self, capacity, dtype):
        self.size = 0
        self.capacity = capacity

        self.container = np.require(np.memmap(tempfile.mkstemp()[1], dtype=dtype, \
                                              mode="w+", shape=capacity), requirements=['O'])

    #         self.container = np.zeros(capacity,dtype=dtype)

    def extend(self, iterable):
        for i in iterable:
            self.push_back(i)

    def push_back(self, val):
        if self.size >= self.container.shape[0]:
            self.capacity *= 2
            self.container.resize(self.capacity)

        self.container[self.size] = val
        self.size += 1

    def at_range(self, begin, end):
        return self.container[begin:end]

    def at(self, index):
        return self.container[index]

    def pop_back(self):
        self.size -= 1
        return self.at(self.size)

    def sort(self, order):
        self.shrink()
        self.container.sort(order=order)

    def shrink(self):
        self.capacity = self.size
        self.container.resize(self.capacity)

    def __len__(self):
        return self.size

    def __iter__(self):
        return memmaped_arraylist_iter(self)

def crop_image(inputimage : Image, target_size, method=Image.NEAREST):
    """
    Crops image, so it fits into target_size
    returns 2-d np.array
    """

    # inputimage.size dimension order does not match that of np.array(inputimage).shape!!!!

    target_size = np.array(target_size).astype(np.int)
    aspect_input = inputimage.size[0] / inputimage.size[1]
    aspect_target = target_size[1] / target_size[0]

    # input image, so it fits output along this axis
    scale_axis = 1  # fit width to target
    if aspect_input > aspect_target:
        scale_axis = 0  # fit height to target

    scale = target_size[scale_axis] / inputimage.size[1-scale_axis]

    # Resize to make sure one of axes matches target shape
    if scale_axis == 1:

        ret_img = inputimage.resize( (target_size[1],int(round(inputimage.size[1] * scale))), method )
        # ret_img = misc.imresize(inputimage, (int(round(inputimage.shape[0] * scale)), target_size[1]))
    else:
        ret_img = inputimage.resize((int(round(inputimage.size[0] * scale)),target_size[0]), method )
        # ret_img = misc.imresize(inputimage, (target_size[0], int(round(inputimage.shape[1] * scale))))

    crop_min = np.floor((ret_img.size[scale_axis] - target_size[1 - scale_axis]) / 2)

    if crop_min + target_size[1 - scale_axis] > ret_img.size[scale_axis]:
        raise ValueError("Something went wrong")

    crop_width_begin = 0
    crop_height_begin = 0
    crop_width_end = ret_img.size[0]
    crop_height_end = ret_img.size[1]

    if scale_axis == 1:
        crop_height_begin = int(crop_min)
        crop_height_end = int(target_size[0] + crop_min)
        # return ret_img[ int(crop_min): int(target_size[0] + crop_min), :]
    else:
        crop_width_begin = int(crop_min)
        crop_width_end = int(target_size[1] + crop_min)
        # return ret_img[:, int(crop_min): int(target_size[1] + crop_min)]

    return ret_img.crop((crop_width_begin,crop_height_begin,crop_width_end,crop_height_end) )



def dist_abs(img1,img2):
    return np.abs(img1.astype(np.float32)-img2.astype(np.float32)).mean()

def dist2_sqrt(img1,img2):
    return ((img1.astype(np.float32)-img2.astype(np.float32))**2).mean()/100

def generate_fits(target, target_tile_size, tile_shape, tile,
                  tile_data, tile_index,distance_method=dist_abs):
    source = tile_data  # source tile data
    source_size = tile[1:3]  # source size in tiles

    x_ticks = np.arange(0, target_tile_size[1] - source_size[1] + 1).astype(np.int32)
    y_ticks = np.arange(0, target_tile_size[0] - source_size[0] + 1).astype(np.int32)

    xy_pairs = np.array(np.meshgrid(x_ticks, y_ticks), dtype=np.int32).T.reshape(-1, 2)

    ret = []

    for x, y in xy_pairs:
        fit_img = target[y * tile_shape[0]:(y + source_size[0]) * tile_shape[0],
                  x * tile_shape[1]:(x + source_size[1]) * tile_shape[1]]

        # Calculate fit score and put it into INTEGER var
        # so it can sit naturally in integer array

        # score = int( np.round((np.abs(source-fit_img).mean()-np.sqrt(size_weight*source_size[0]*source_size[1])*128)*100) )
        # convert to float to prevent integer overflow

        score = distance_method(source,fit_img)
        ret.append((tile_index, score, x, y))

    return ret


def tiles_from_img(tile_shape, path, max_tiles=5):
    try:

        img = Image.open(path)

        # if image_processing_scale != 1:
        #     img = misc.imresize(img, image_processing_scale)
    except ValueError as ex:
        raise IOError("Could not read image at {}".format(path))

    aspect = np.array([img.size[0] / img.size[1], img.size[1] / img.size[0]])

    tiles_num_max = np.floor([img.size[1] / tile_shape[0], img.size[0] / tile_shape[1]]).astype(np.int)
    max_index = np.argmax(tiles_num_max)

    ret_pool = []

    shapes = set()
    shapes.add((1,1))
    shapes.add((1,2))
    shapes.add((2,1))

    tiles_num = np.array([1, 1], dtype=np.float64)
    ret_pool.append((crop_image(img, tiles_num * tile_shape), tiles_num.astype(np.uint16)))

    if max_tiles > 1:
        tiles_num[max_index] = 2
        ret_pool.append((crop_image(img, tiles_num * tile_shape), tiles_num.astype(np.uint16)))

    for i in np.arange(2, tiles_num_max[max_index] + 1)[::-1]:

        tiles_num[max_index] = i
        tiles_num[1 - max_index] = np.round(i * aspect[max_index])

        if tiles_num[0]*tiles_num[1] > max_tiles:
            continue

        # ret_img = crop_image(img, tiles_num * tile_shape)
        ret_pool.append(( crop_image(img, tiles_num * tile_shape) , tiles_num.astype(np.uint16)))

    return ret_pool

def cache_images( target_image, target_tile_size, image_path_pool, tile_shape,
                 tile_max_size, verbose=True, distance_method=dist_abs):
    """
    Creates array of tiles and sorted array fits.
    tiles are [ (image_id,tile_shape_x,tile_shape_y), ... ]
    fits are [ (tile_id,score,x,y), ... ]
    :param target_image:
    :param target_tile_size:
    :param image_path_pool:
    :param tile_shape:
    :param tile_max_size:
    :param verbose:
    :param size_weight:
    :return: tiles, fits
    """
    tiles = []  # Format tiles[index] = [image_id, tile_shape_x, tile_shape_y]

    # store array of fits for each tile size separately
    fits = [
        memmaped_arraylist(1000000, [('tile_id', 'int32'), ('score', 'float32'), ('x', 'int16'), ('y', 'int16')])
        if i > 0 else None
        for i in range(tile_max_size+1)
    ]

    failed = []

    for i, f in enumerate(image_path_pool):
        if verbose and i % 4 == 0:
            print('Caching {} of {} images ({} failed). {} tiles, {} fits generated' \
                  .format(i, len(image_path_pool), len(failed), len(tiles),
                          np.array([len(i) for i in fits if i is not None]).sum()), end='\n')
        try:
            img_index = i
            tile_ids = []

            for data, ts in tiles_from_img(tile_shape, f, max_tiles=tile_max_size):
                tile_id = len(tiles)

                tile_ids.append(tile_id)

                tiles.append((img_index, ts[0], ts[1]))

                generated_fits = generate_fits(target_image, target_tile_size,
                                          tile_shape, tiles[tile_id], image_to_array(data) , tile_id,
                                          distance_method=distance_method)

                fits[ts[0]*ts[1]].extend(generated_fits)

        except KeyboardInterrupt:  # User interrupt
            if verbose:
                print("\nAborting...")
            break
        except (OSError,IOError) as ex:  # Wrong file
            failed.append(f)
        except (Exception,IndexError) as ex:
            try:
                print("\nUnknown exception while caching images:\"{}\", message: \"{}\"".format(type(ex).__name__, ex))
                exc_info = sys.exc_info()
                failed.append(f)
            finally:
                traceback.print_exception(*exc_info)
                del exc_info
            break

    if len(failed) > 0:
        print('Failed to read images:')
        for i in failed:
            print(i)

    # sort all fits by score
    for i in range(len(fits)):
        if fits[i] is not None:
            fits[i].sort('score')

    return np.array(tiles), fits

def find_best_fits_fill_whole():
    """
    Finds such best fit, that uses all images in input sequence
    :return:
    """

def find_best_fits(target_image_tile_size, tiles, fits, image_repetition=1,
                   use_all = False, num_input_images = 0, size_weight=0):

    if use_all and num_input_images <= 0:
        raise RuntimeError("if use_all == True, num_input_images must be set")

    image = np.ones(target_image_tile_size.astype(np.int32), dtype=np.float64) * -1

    ret = []

    #     print("Evaluating fits")
    #     sizes = -np.prod( tiles[ fits.container['tile_id'] ][:,1:], axis=1 ).astype(np.float32)
    #     scores = fits.container['score'].astype(np.float32)
    #     sizes /= np.percentile(sizes,95)-np.percentile(sizes,5)
    #     scores /= np.percentile(scores,95)-np.percentile(scores,5)
    #     scores = scores*1-size_weight+sizes*size_weight
    #     fits.container['score'] = scores
    #     del scores
    #     del sizes

    #     print("Sorting")
    #     fits.sort('score')

    print("Finding best fits")
    used_images = dict()

    i = 0
    used = 0
    used_doubles = 0

    def put_fit(tile_id, x, y):
        nonlocal used_images
        nonlocal tiles
        nonlocal image_repetition
        nonlocal image
        nonlocal ret
        image_id = tiles[tile_id][0]
        tile_shape = tiles[tile_id][1:3]

        if used_images.get(image_id, 0) >= image_repetition or\
                        image[y:y + tile_shape[0], x:x + tile_shape[1]].max() != -1:
            return False

        image[y:y + tile_shape[0], x:x + tile_shape[1]] = np.random.uniform(5, 10)
        ret.append([tile_id, x, y])
        used_images[image_id] = used_images.get(image_id, 0) + 1

        return True

    if use_all:
        num_double_images = (target_image_tile_size[0] * target_image_tile_size[1]).astype(np.int32) - num_input_images

        ## Fill with doubles
        for tile_id, score, x, y in fits[2]:
            i+=1
            if put_fit(tile_id,x,y):
                used_doubles += 1
                used+=1
                if used_doubles >= num_double_images:
                    break

        # Fill with singles
        for tile_id, score, x, y in fits[1]:
            i+=1
            if put_fit(tile_id,x,y):
                used+=1
                if (image == -1).sum() == 0:
                    break
    else:


        for tile_id, score, x, y in sorted_merge_iterator(fits,key=lambda x: x[1]-tiles[x[0]][0]*tiles[x[0]][1]*128*size_weight ):
            if i % 44263 == 0:
                total = np.array([len(i) for i in fits if i is not None]).sum()
                print('Filling image, {} of {} ({:.0f}%) fits used, filled {:.2f}%'.format(i,
                                                                  total,
                                                                  i / total * 100,
                                100 - (image == -1).sum() / image.size * 100))
                if (image == -1).sum() == 0:
                    break
            i += 1

            if put_fit(tile_id,x,y):
                used+=1

    print()
    print('fits iterated: ', i)
    print('Double images used:',used_doubles)
    print('total images used:', used)
    print('image min: ',image.min())
    print('image max: ',image.max())
    print('image mean:',image.mean())
    return image, np.array(ret)

def get_aspect_ratio(path_to_images) -> float:
    return 1.6

## Deduct tile split
def get_tile_size_num(ref_img_shape, aspect, num_images, res=2) -> ((int, int), (int, int)):
    """
    returns : tile_size, tiles_num
    """
    # image area * num_images  = reference img area
    # x*aspect*x * num_images  = ref_img.shape[0]*ref_img.shape[1]
    # x = np.sqrt( (ref_img_shape[0]*ref_img_shape[1])/num_images/aspect )

    #              height    width
    # tile_shape =    x,   x*aspect,

    # Rule: image ought to have at least as many pixels on smaller side, as sqrt(number of images)*2*res
    # Resize input image to fit this condition

    scale_factor = np.sqrt(num_images) * 2 * res / min(ref_img_shape[0], ref_img_shape[1])
    ref_img_shape = np.floor((ref_img_shape[0] * scale_factor, ref_img_shape[1] * scale_factor)).astype(np.int32)

    x = np.sqrt((ref_img_shape[0] * ref_img_shape[1]) / num_images / aspect)
    tile_shape_base = np.array((x, x * aspect))
    tile_shape = None
    tile_count_min = None
    shape_loss_min = None

    for i in itertools.product((-1, -0.5, 0, 0.5, 1), repeat=2):
        ts_candidate = np.round(tile_shape_base + i).astype(np.int32)
        if ts_candidate[0] <= 0 or ts_candidate[1] <= 0:
            continue

        tile_shape_loss = (ref_img_shape[0] - round(ref_img_shape[0] / ts_candidate[0]) * ts_candidate[0]) ** 2 + \
                          (ref_img_shape[1] - round(ref_img_shape[1] / ts_candidate[1]) * ts_candidate[1]) ** 2

        tile_count = np.prod(
            np.round((ref_img_shape[0] / ts_candidate[0], ref_img_shape[1] / ts_candidate[1])).astype(np.int32))

        if tile_count > num_images:
            if tile_count_min is None or tile_count_min > tile_count or \
                    (tile_count_min == tile_count and tile_shape_loss < shape_loss_min):
                tile_count_min = tile_count
                shape_loss_min = tile_shape_loss
                tile_shape = ts_candidate

    tiles_num = np.round((ref_img_shape[0] / tile_shape[0], ref_img_shape[1] / tile_shape[1]))

    return tile_shape.astype(np.int), tiles_num.astype(np.int)


def assemble_image(tiles_num, tile_shape, paths, tiles, fits, ref_img=None, guess_size=True, \
                   mode='average', max_size=2000, p=5, overflow_error=True,color_correction_scale = 0.5):

    scales = []

    if guess_size:
        print("Guessing image size...")
        #         print("Reading images")
        for i, (tile_id, tile_x, tile_y) in enumerate(fits):
            #             print("\r{} of {}".format(i+1,len(fits)),end='')
            tile = tiles[tile_id]
            tile_tiles_num = tile[1:3]

            img_path = paths[tile[0]]
            #             img = unify_read_image(img_path)
            #         images.append(img)

            try:
                img = Image.open(img_path)
                img_size = np.array([img.size[1],img.size[0]])
                #             img_size = img.shape#(np.random.uniform(500,1900,size=2)/10).astype(np.int32)*10

                target_crop_size = tile_shape * tile_tiles_num
                scale = 1.0 / get_crop_scale(img_size, target_crop_size)
                scales.append(scale)
            except:
                pass

                #         print()

        if mode == 'average':
            final_scale = np.mean(scales)
        elif mode == 'percentille':
            final_scale = np.percentile(scales, p)
        elif mode == 'min':
            final_scale = np.min(scales)
        elif mode == 'max':
            final_scale = np.max(scales)
        else:
            print("Uknown output mode: {}".format(mode))
            exit()
    else:
        final_scale = (max_size / tiles_num[0]) / tile_shape[0]

    final_image_size = tile_shape * final_scale * tiles_num
    if (final_image_size[0] > max_size):
        final_scale = (max_size / tiles_num[0]) / tile_shape[0]

    final_image_size = tile_shape * final_scale * tiles_num
    if (final_image_size[1] > max_size):
        final_scale = (max_size / tiles_num[1]) / tile_shape[1]

    final_tile_size = (tile_shape * final_scale).astype(np.int32)
    final_image_size = (final_tile_size * tiles_num).astype(np.int32)
    final_image_mb_size = int(final_image_size[0]) * int(final_image_size[1]) * 3 / 1024 / 1024

    if final_image_mb_size > 1024 * 2:
        if overflow_error:
            raise ValueError(
                "Final image size is over 2 GB ({:.0f}MB), it may not fit into memory".format(final_image_mb_size))
        else:
            print("Warning! Final image is too large ({:.0f}MB)".format(final_image_mb_size))
            raise ValueError("Nope")
    print('Final image size: {} {} ({:.2f} MB)'.format(*final_image_size, final_image_mb_size))

    if color_correction_scale > 0:
        ret = image_to_array( ref_img.resize((final_image_size[1],final_image_size[0])) )
    else:
        ret = np.zeros((*final_image_size, 3), dtype=np.uint8)

    # ret = image_to_array( ref_img.resize((final_image_size[1],final_image_size[0])) )

    fits_final_paths = np.array([paths[tiles[tile_id][0]] for tile_id, tile_x, tile_y in fits])
    fits_sorted_ids = np.array(sorted(list(range(len(fits))), key=lambda i: fits_final_paths[i]))
    fits = fits[np.array(sorted(list(range(len(fits))), key=lambda i: fits_final_paths[i]))]
    del fits_final_paths
    del fits_sorted_ids

    cached_image = None
    cached_image_path = None

    def get_image(path):
        nonlocal cached_image
        nonlocal cached_image_path
        if cached_image_path is not None and cached_image_path == path:
            return cached_image
        del cached_image
        cached_image = Image.open(path)
        cached_image_path = path
        return cached_image

    for i, (tile_id, tile_x, tile_y) in enumerate(fits):
        if i%4 == 0:
            print("Baking images to output {} of {}".format(i + 1, len(fits)))
        tile = tiles[tile_id]
        tile_tiles_num = tile[1:3]
        img_path = paths[tile[0]]
        img = get_image(img_path)

        crop_size = tile_tiles_num * final_tile_size

        img_cropped = image_to_array( crop_image(img, crop_size,method=Image.ANTIALIAS) )

        if color_correction_scale > 0:
            color_ref = ret[tile_y * final_tile_size[0]:(tile_y + tile_tiles_num[0]) * final_tile_size[0],
            tile_x * final_tile_size[1]:(tile_x + tile_tiles_num[1]) * final_tile_size[1], :]
            img_cropped = transform_colors(color_ref,img_cropped,color_correction_scale)

        ret[tile_y * final_tile_size[0]:(tile_y + tile_tiles_num[0]) * final_tile_size[0],
        tile_x * final_tile_size[1]:(tile_x + tile_tiles_num[1]) * final_tile_size[1], :] = img_cropped

    return ret

def get_final_fits_exact(ref_img_path, input_images, res=5, final_image_size_mode = 'average',
                         result_image_max_size=2000,distance_method=dist_abs,
                         color_correction_scale=0.5):

    ref_img_original = Image.open(ref_img_path)
    ref_img = ref_img_original

    tile_shape, tiles_num = get_tile_size_num((ref_img.size[1],ref_img.size[0]), get_aspect_ratio(input_images), len(input_images), res=res)

    ref_img = image_to_array( crop_image(ref_img, tile_shape * tiles_num) )

    tiles, fits = cache_images(ref_img, tiles_num, input_images, tile_shape, tile_max_size=2,distance_method=distance_method)

    fit_img, fits_final = find_best_fits(tiles_num, tiles, fits, use_all=True, num_input_images=len(input_images))

    result = assemble_image(tiles_num,tile_shape,input_images,tiles,fits_final,
                            mode=final_image_size_mode, max_size=result_image_max_size, ref_img=ref_img_original,
                            color_correction_scale=color_correction_scale)

    return result

def get_final_fits(ref_img_path, input_images, res=5, output_collage_images_num=None, size_weight=0.0,
                   horizontal_tiles=None, vertical_tiles=None, max_patch_size=9,
                   final_image_size_mode='average',result_image_max_size=2000,
                   distance_method=dist_abs, image_repetition=1,color_correction_scale=0.5):


    if output_collage_images_num is None:
        output_collage_images_num = len(input_images)

    ref_img_original = Image.open(ref_img_path)
    ref_img = ref_img_original

    if (horizontal_tiles is not None and vertical_tiles is not None) and\
            (horizontal_tiles > 0 and vertical_tiles > 0):
        tiles_num = np.array( (vertical_tiles,horizontal_tiles) )
        tile_shape = np.floor( (ref_img.size[1]/tiles_num[0],ref_img.size[0]/tiles_num[1]) ).astype(np.int)
        if tile_shape[0] == 0 or tile_shape[1] == 0:
            print('tile dimensions ({}) should be >= 0, possibly invalid tiles_num: {}'.format(tile_shape,tiles_num))
    else:
        tile_shape, tiles_num = get_tile_size_num((ref_img.size[1],ref_img.size[0]), get_aspect_ratio(input_images),
                                              output_collage_images_num, res=res)
    print( 'Image splitted into ({} x {}) tiles'.format(*tiles_num) )
    ref_img = image_to_array( crop_image(ref_img, tile_shape * tiles_num) )
    tiles, fits = cache_images(ref_img, tiles_num, input_images, tile_shape, tile_max_size=max_patch_size,
                               distance_method=distance_method)
    fit_img, fits_final = find_best_fits(tiles_num,tiles,fits,size_weight=size_weight,image_repetition=image_repetition)

    result = assemble_image(tiles_num,tile_shape,input_images,tiles,fits_final,
                            mode=final_image_size_mode,max_size=result_image_max_size,ref_img=ref_img_original,
                            color_correction_scale=color_correction_scale)

    return result


def main():


    parser = argparse.ArgumentParser(description='Image mosaicing by reference')
    parser.add_argument('ref_file', metavar='ref_file', type=str, nargs=1,
                        help='reference image')
    parser.add_argument('input_files', metavar='input_files', type=str, nargs='+',
                        help='paths, determines input files for mosaicing. '
                             'you can provide text file with path on each line as input file')
    parser.add_argument('output_file', metavar='output_file', type=str, nargs=1,
                        help='file to which output will be written')
    parser.add_argument('--useall', dest='useall',type=int,default=1,
                        help='1 or 0, Should final image contain exactly all input images? (default=1)')
    parser.add_argument('--reps',dest='reps',type=int,default=1,
                        help='maximum number of image repetition. only avaliable when --useall=0 (default = 1)')
    parser.add_argument('--max_patch_size',dest='max_patch_size',type=int,default=9,
                        help='maximum area of patches. only avaliable when --useall=0.'
                             ' (use 1 to force all output images same size) (default=9)')
    parser.add_argument('--size_weight',dest='size_weight',type=int,default=0.0,
                        help='higher values will yield more bigger images in output. '
                             'only avaliable when --useall=0 (default=0)')
    parser.add_argument('--vtiles',dest='vtiles',type=int,default=0,
                        help='number of vertical tiles (use with --htiles). only avaluable when --useall=0')
    parser.add_argument('--htiles',dest='htiles',type=int,default=0,
                        help='number of horisontal tiles (use with --vtiles). only avaliable when --useall=0')
    parser.add_argument('--max_images',dest='max_images',type=int,default=2000,
                        help='when useall is 0, restricts maximum number of images in output (default=2000)')
    parser.add_argument('--cc_scale', dest='cc_scale', type=float, default=0.2,
                        help='scale of color correction, from 0 to 1 (default = 0.2)')
    parser.add_argument('--resolution',dest='resolution',type=int,default=5,
                        help='resolution of distance measuring and tile fitting, higher values may '
                             'lead to bigger memory consumption and computition times (default=5)')
    parser.add_argument('--output_max_size',dest='output_max_size',type=int,default=2000,
                        help='maximum width or height of output image (default=2000)')
    parser.add_argument('--output_resolution',dest='output_res',type=str,default='average',
                        help='Method to deduct resolution of output image, but not bigger than output_max_size.'
                             ' Use one of: "average"-resolution is average of tiles resolution,'
                             ' "max"-maximum of tiles resolution, "min"-min of tiles resolution')
    parser.add_argument('--dist',dest='distance_method',type=str,default='abs',
                        help='abs or sqrt - methods to measure tile fitness. (default=abs)'
                             'abs - absolute mean error, '
                             'sqrt - squared mean error.')


    args = parser.parse_args()

    # images with higher area may socre higher in final image, set this parameter
    # to control number of large images in output collage
    size_weight = args.size_weight

    # maximum number of tiles which one patch may contain in final image
    max_patch_size = args.max_patch_size

    # maximum number of images to be used in collage (used when fit is non-exact)
    # flot for image fraction, int for absolute value
    max_num_images = args.max_images

    # number of vertical tiles in final image
    tiles_vertical = args.vtiles
    # number of horisontal tiles in final image
    tiles_horisontal = args.htiles

    # fit resolution, bigger value, better fit
    resolution = args.resolution

    use_all = bool( args.useall )

    result_image_resolution = args.output_res

    result_image_max_size = args.output_max_size

    distance_method_str = args.distance_method

    image_repetition = args.reps

    color_correction_scale = args.cc_scale

    if distance_method_str == 'abs':
        distance_method = dist_abs
    elif distance_method_str == 'sqrt':
        distance_method = dist2_sqrt
    else:
        print('Uknown distance metric: {}'.format(distance_method_str))
        print('Use one of following: "abs", "sqrt"')
        exit(1)

    # parsing input files
    if len(args.input_files) == 1:
        try:
            Image.open(args.input_files[0])
        except OSError as ex:
            with open(args.input_files[0],'r') as f:
                files_str = f.readlines()

                files = []

                for f in files_str:
                    f = f.strip()
                    if len(f) > 0:
                        for f in glob.glob(f.strip()):
                            if os.path.exists(f):
                                files.extend( glob.glob(f.strip()) )

                args.input_files = files

    # print(
    #     'size_weight:', size_weight, '\n',
    #     'max_patch_size:', max_patch_size, '\n',
    #     'max_num_images:', max_num_images, '\n',
    #     'tiles_vertical:', tiles_vertical, '\n',
    #     'tiles_horisontal:', tiles_horisontal, '\n',
    #     'resolution:', resolution, '\n',
    #     'use_all:', use_all, '\n',
    #     'result_image_resolution:', result_image_resolution, '\n',
    #     'result_image_max_size:', result_image_max_size, '\n',
    #     'distance_method:', distance_method, '\n'
    # )
    # a = None
    # input(a)

    # if we should use all input images
    if use_all:
        img_final = get_final_fits_exact(args.ref_file[0],args.input_files,res=resolution,
                                         final_image_size_mode=result_image_resolution,
                                         result_image_max_size=result_image_max_size,
                                         distance_method=distance_method,
                                         color_correction_scale=color_correction_scale)
    else:
        img_final = get_final_fits(args.ref_file[0],args.input_files,res=resolution,
                                   vertical_tiles=tiles_vertical,horizontal_tiles=tiles_horisontal,
                                   output_collage_images_num=max_num_images,size_weight=size_weight,
                                   final_image_size_mode=result_image_resolution,
                                   result_image_max_size=result_image_max_size,
                                   max_patch_size=max_patch_size,
                                   distance_method=distance_method,image_repetition=image_repetition,
                                   color_correction_scale=color_correction_scale)

    Image.fromarray(img_final.astype(np.uint8), 'RGB').save(args.output_file[0])
    # arr = result[:, :, np.newaxis].astype(np.float64)
    #
    #
    # arr = (arr-np.min(arr))/(np.max(arr))*255
    #
    # print('arr_min: ',arr.min())
    # print('arr_max: ',arr.max())
    # print('arr_mean:',arr.mean())
    # print('arr_percentille5:',np.percentile(arr,5))
    # print('arr_percentille95:',np.percentile(arr,95))
    # arr = np.round(arr)
    # arr[arr>=255] = 255
    # arr[arr<=0] = 0
    # arr = arr.astype(np.uint8)
    # ret_img = np.concatenate((arr, arr, arr), axis=2).astype(np.uint8)
    #
    # print('final_image_min: ',ret_img.min())
    # print('final_image_max: ',ret_img.max())
    # print('final_image_mean:',ret_img.mean())
    #
    # Image.fromarray(ret_img,'RGB').save(args.output_file[0])

if __name__ == '__main__':
    main()


