import numpy.random as NPR
import numpy as np
import copy

# ***** GENERATING Simple DATA SETS for MACHINE LEARNING *****

# Generate all bit vectors of a given length (num_bits).
def gen_all_bit_vectors(num_bits):
    def bits(n):
        s = bin(n)[2:]
        return [int(b) for b in '0'*(num_bits - len(s))+s]
    return [bits(i) for i in range(2**num_bits)]

# Convert an integer to a bit vector of length num_bits, with prefix 0's as padding when necessary.
def int_to_bits(i,num_bits):
    s = bin(i)[2:]
    return [int(b) for b in '0' * (num_bits - len(s)) + s]

def all_ints_to_bits(num_bits):
    return [int_to_bits(i) for i in range(2**num_bits)]

# Convert an integer k to a sparse vector in which all bits are "off" except the kth bit.  Note: this
# is zero-based, so the one-hot vector for 0 is 10000..., and for 1 is 010000..

def int_to_one_hot(int,size,off_val=0, on_val=1,floats=False):
    if floats:
        off_val = float(off_val); on_val = float(on_val)
    if int < size:
        v = [off_val] * size
        v[int] = on_val
        return v



# Generate all one-hot vectors of length len
def all_one_hots(len, floats=False):
    return [int_to_one_hot(i,len,floats=floats) for i in range(len)]

# bits = list of 1's and 0's
def bits_to_str(bits): return ''.join(map(str,bits))
def str_to_bits(s): return [int(c) for c in s]

# ****** VECTOR SHIFTING ******
# Shift a vector right (dir=1) or left (dir= -1) and any number of spaces (delta).

def shift_vector(v,dir=1,delta=1):
    dx = dir*delta; vl = len(v)
    v2 = v.copy()
    for i in range(vl):
        j = (i + dx) % vl
        v2[j] = v[i]
    return v2

# Given one shift command (dir + delta), provide MANY examples using randomly-generated initial vectors.
def gen_vector_shift_cases(vlen,count, dir=1,delta=1, density=0.5):
    cases = []
    for i in range(count):
        v = gen_dense_vector(vlen,density=density)
        v2 = shift_vector(v,dir=dir,delta=delta)
        cases.append((v,v2))
    return cases

# ****** RANDOM VECTORS of Chosen Density *****

# Given a density (fraction), this randomly places onvals to produce a vector with the desired density.
def gen_dense_vector(size, density=.5, onval=1, offval=0):
    a = [offval] * size
    indices = np.random.choice(size,round(density*size),replace=False)
    for i in indices: a[i] = onval
    return a

def gen_random_density_vectors(count,size,density_range=(0,1)):
    return [gen_dense_vector(size,density=np.random.uniform(*density_range)) for c in range(count)]

# ****** LINES (horiz and vert) in arrays *********

# This produces either rows or columns of values (e.g. 1's), where the bias controls whether or not
# the entire row/column gets filled in or not just some cells. bias=1 => fill all.  Indices are those of the
# rows/columns to fill.  This is mainly used for creating data sets for classification tasks: horiz -vs- vertical
# lines.

def gen_line_array(dims,indices,line_item=1,background=0,columns=False,bias=1.0):
    a = np.array([background]*np.prod(dims)).reshape(dims)
    if columns: a = a.reshape(list(reversed(dims)))
    for row in indices:
        for j in range(a.shape[1]):
            if np.random.uniform(0, 1) <= bias: a[row,j] = line_item
    if columns: a = a.transpose()
    return a


# ****** ML CASE GENERATORS *****
# A ML "case" is a vector with two elements: the input vector and the output (target) vector.  These are the
# high-level functions that should get called from ML code.  They invoke the supporting functions above.

# The simplest autoencoders use the set of one-hot vectors as inputs and target outputs.

def gen_all_one_hot_cases(len, floats=False):
    return [[c,c] for c in all_one_hots(len,floats=floats)]

# This creates autoencoder cases for vector's with any density of 1's (specified by density_range).
def gen_dense_autoencoder_cases(count,size,dr=(0,1)):
    return [[v,v] for v in gen_random_density_vectors(count,size,density_range=dr)]

# Produce a list of pairs, with each pair consisting of a num_bits bit pattern and a singleton list containing
# the parity bit: 0 => an even number of 1's, 1 => odd number of 1's.  When double=True, a 2-bit vector is the
# target, with bit 0 indicating even parity and bit 1 indicating odd parity.

def gen_all_parity_cases(num_bits, double=True):
    def parity(v): return sum(v) % 2
    def target(v):
        if double:
            tg = [0,0].copy()
            tg[parity(v)] = 1
            return tg
        else: return [parity(v)]

    return [[c, target(c)] for c in gen_all_bit_vectors(num_bits)]

# This produces "count" cases, where features = random bit vectors and target = a one-hot vector indicating
# the number of 1's in the feature vector(default) or simply the count label.  Note that the target vector is one bit
# larger than the feature vector to account for the case of a zero-sum feature vector.

def gen_vector_count_cases(num,size,drange=(0,1),random=True,poptarg=True):
    if random: feature_vectors = gen_random_density_vectors(num,size,density_range=drange)
    else: feature_vectors = gen_all_bit_vectors(size)
    if poptarg:
        targets = [int_to_one_hot(sum(fv),size+1) for fv in feature_vectors]
    else: targets = [sum(fv) for fv in feature_vectors]
    return [[fv,targ] for fv,targ in zip(feature_vectors,targets)]

def gen_all_binary_count_cases(size,poptarg=True): return gen_vector_count_cases(None,size,random=False,poptarg=poptarg)

# Generate cases whose feature vectors, when converted into 2-d arrays, contain either one or more horizontal lines
# or one or more vertical lines.  The argument 'min_opens' is the minimum number of rows (for horizontal lines) or
# columns (for vertical lines) that must NOT be filled. The class is then simply horizontal (0) or vertical (1).
# A bias of 1.0 insures that no noise will be injected into the lines, and thus classification should be easier.

def gen_random_line_cases(num_cases,dims,min_lines=1,min_opens=1,bias=1.0, mode='classify',
                          line_item=1,background=0,flat=True,floats=False):
    def gen_features(r_or_c):
        r_or_c = int(r_or_c)
        size = dims[r_or_c]
        count = np.random.randint(min_lines,size-min_opens+1)
        return gen_line_array(dims,indices=np.random.choice(size,count,replace=False), line_item=line_item,
                              background=background,bias=bias,columns =(r_or_c == 1))
    def gen_case():
        label = np.random.choice([0,1]) # Randomly choose to use a row or a column
        features = gen_features(label)
        if flat: features = features.flatten().tolist()
        if mode == 'classify':  # It's a classification task, so use 2 neurons, one for each class (horiz, or vert)
            target = [0]*2
            target[label] = 1
        elif mode == 'auto':  target = copy.deepcopy(features)  # Autoencoding mode
        else: target = [float(label)]  # Otherwise, assume regression mode.
        return (features, target)

    if floats:
        line_item = float(line_item); background = float(background)
    return [gen_case() for i in range(num_cases)]

# ********** SEGMENT VECTORS **********
# These have groups/segments/blobs of "on" bits interspersed in a background of "off" bits.  The key point is that we can
# specify the number of segments, but the sizes are chosen randomly.

def gen_segmented_vector(vectorsize,numsegs,onval=1,offval=0):
    if vectorsize >= 2*numsegs - 1:  # Need space for numsegs-1 gaps between the segments
        vect = [offval] * vectorsize
        if numsegs <= 0: return vect
        else:
            min_gaps = numsegs - 1 ;
            max_chunk_size = vectorsize - min_gaps; min_chunk_size = numsegs
            chunk_size = NPR.randint(min_chunk_size,max_chunk_size+1)
            seg_sizes = gen_random_pieces(chunk_size,numsegs)
            seg_start_locs = gen_segment_locs(vectorsize,seg_sizes)
            for s0,size in zip(seg_start_locs,seg_sizes): vect[s0:s0+size] = [onval]*size
            return vect

#  Randomly divide chunk_size units into num_pieces units, returning the sizes of the units.
def gen_random_pieces(chunk_size,num_pieces):
    if num_pieces == 1: return [chunk_size]
    else:
        cut_points = list(NPR.choice(range(1,chunk_size),num_pieces-1,replace=False)) # the pts at which to cut up the chunk
        lastloc = 0; pieces = []; cut_points.sort() # sort in ascending order
        cut_points.append(chunk_size)
        for pt in cut_points:
            pieces.append(pt-lastloc)
            lastloc = pt
        return pieces

def gen_segment_locs(maxlen,seg_sizes):
    locs = []; remains = sum(seg_sizes); gaps = len(seg_sizes) - 1; start_min = 0
    for ss in seg_sizes:
        space = remains + gaps
        start = NPR.randint(start_min,maxlen - space + 1)
        locs.append(start)
        remains -= ss; start_min = start + ss + 1; gaps -= 1
    return locs

# This is the high-level routine for creating the segmented-vector cases.  As long as poptargs=True, a
# population-coded (i.e. one-hot) vector will be created as the target vector for each case.

def gen_segmented_vector_cases(vectorlen,count,minsegs,maxsegs,poptargs=True):
    cases = []
    for c in range(count):
        numsegs = NPR.randint(minsegs,maxsegs+1)
        v = gen_segmented_vector(vectorlen,numsegs)
        case = [v,int_to_one_hot(numsegs,maxsegs-minsegs+1)] if poptargs else [v,numsegs]
        cases.append(case)
    return cases

def segment_count(vect,onval=1,offval=0):
    lastval = offval; count = 0
    for elem in vect:
        if elem == onval and lastval == offval: count += 1
        lastval = elem
    return count

# This produces a string consisting of the binary vector followed by the segment count surrounded by a few symbols
# and/or blanks.  These strings are useful to use as labels during dendrogram plots, for example.
def segmented_vector_string(v,pre='** ',post=' **'):
    def binit(vect): return map((lambda x: 1 if x > 0 else 0), vect)
    return ''.join(map(str, binit(v))) + pre + str(segment_count(v)) + post
