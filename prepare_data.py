# loads imagenet and writes it into one massive binary file

import os
import sys
import argparse
import numpy as np
from tensorpack.dataflow import *
from tensorpack.dataflow.serialize import _reset_df_and_get_size, LMDBSerializer
from tensorpack.utils.serialize import dumps, loads
import tqdm
import lmdb
import socket
import pdb
import platform
hostname = socket.gethostname()

class LMDBSplitSaver():
    """
    Serialize a Dataflow to multiple lmdb databases, where the keys are indices and values
    are serialized datapoints.

    You will need to ``pip install lmdb`` to use it.
    """
    @staticmethod
    def save(df, paths, N, write_frequency=1000):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output path. Must be an lmdb file.
            write_frequency (int): the frequency to write back data to disk.
                A smaller value reduces memory usage.
        """
        assert isinstance(df, DataFlow), type(df)
        map_size = 1099511627776 * 2 if platform.system() == 'Linux' else 128 * 10**6
        dbs = []
        txns = []
        all_slice_keys = [ [] for i in range(N) ]
        size = _reset_df_and_get_size(df)
        slice_sizes = [ 0 for i in range(N) ]
        
        for path in paths:
            assert not os.path.isfile(path), "LMDB file {} exists!".format(path)
            
            # It's OK to use super large map_size on Linux, but not on other platforms
            # See: https://github.com/NVIDIA/DIGITS/issues/206
            db = lmdb.open(path, subdir=False,
                           map_size=map_size, readonly=False,
                           meminit=False, map_async=True)    # need sync() at the end
            dbs.append(db)
            # LMDB transaction is not exception-safe!
            # although it has a context manager interface
            txns.append(db.begin(write=True))
            
        # put data into lmdb, and doubling the size if full.
        # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
        def put_or_grow(db, txn, key, value):
            try:
                txn.put(key, value)
                return txn
            except lmdb.MapFullError:
                pass
            txn.abort()
            curr_size = db.info()['map_size']
            new_size = curr_size * 2
            print("Doubling LMDB map_size to {:.2f}GB".format(new_size / 10**9))
            db.set_mapsize(new_size)
            txn = db.begin(write=True)
            txn = put_or_grow(db, txn, key, value)
            return txn

        with tqdm.tqdm(total=size) as pbar:
            idx = -1
            db = None
            
            for idx, dp in enumerate(df):
                slice_idx = idx % N
                db  = dbs[slice_idx]
                txn = txns[slice_idx]
                
                slice_keys = all_slice_keys[slice_idx]
                
                txn = put_or_grow(db, txn, u'{:08}'.format(idx).encode('ascii'), dumps(dp))
                slice_sizes[slice_idx] += 1
                key = u'{:08}'.format(idx).encode('ascii')
                slice_keys.append(key)

                pbar.set_postfix(s=str(slice_sizes))
                pbar.update()
                if (slice_sizes[slice_idx] + 1) % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)
                txns[slice_idx] = txn

            print("Finished reading %d data points" %(idx+1))
            
            for i in range(N):
                db = dbs[i]
                txns[i].commit()
                
                slice_keys = all_slice_keys[i]
                with db.begin(write=True) as txn:
                    txn = put_or_grow(db, txn, b'__keys__', dumps(slice_keys))

                print("Flushing '%s' (%d keys) ..." %((paths[i]), len(slice_keys)) )
                db.sync()
                
        for db in dbs:
            db.close()

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--n', default=1, type=int,
                        help='number of slices (default: 1)')
    parser.add_argument('--check-only', action='store_true',
                        help='check lmdb data consistency only')
    parser.add_argument('--root', default=None, type=str,
                        help='root dir of the images')
                        
    args = parser.parse_args()
    return args      
              
if __name__ == '__main__':
    args = parse_args()
    if args.root:
        os.environ['IMAGENET'] = args.root
    else:
        collab_hostnames = [ 'collabai31-desktop', 'workstation', 'workstation2' ]
        if hostname == 'm10':
            os.environ['IMAGENET'] = '/data/home/shaohua/data.imagenet'
        elif hostname in collab_hostnames:
            os.environ['IMAGENET'] = '/data/shaohua'
        else:
            print("Unknown hostname '%s'. Please specify 'IMAGENET' manually." %hostname)
            exit(0)
    
    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def get_data(self):
            for fname, label in super(BinaryILSVRC12, self).__iter__():
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]
    imagenet_path = os.environ['IMAGENET']
        
    for name in ['train', 'val']: # ['test']
        ds0 = BinaryILSVRC12(imagenet_path, name)
        ds1 = MultiProcessRunnerZMQ(ds0, nr_proc=1)
        # dftools.dump_dataflow_to_lmdb(ds1, os.path.join(imagenet_path,'ILSVRC-%s.lmdb'%name))
        if args.n == 1:
            paths = [os.path.join(imagenet_path,'ILSVRC-%s.lmdb'%name)]
        else:
            paths = [ os.path.join(imagenet_path,'ILSVRC-%s-%d.lmdb'%(name, i)) for i in range(args.n) ]

        if not args.check_only:            
            if args.n == 1:
                LMDBSerializer.save(ds1, paths[0])
            else:
                print("Saving to %d files:\n%s\n" %(args.n, "\n".join(paths)))
                LMDBSplitSaver.save(ds1, paths, args.n)
                
        orig_total_img_count = len(ds0)
        lmdb_total_img_count = 0
        for i in range(args.n):
            ds = LMDBSerializer.load(paths[i], shuffle=False)
            lmdb_total_img_count += len(ds)
            
        print("'%s' orig: %d, lmdb: %d." %(name, orig_total_img_count, lmdb_total_img_count), end=' ')
        if orig_total_img_count != lmdb_total_img_count:
            print("Mismatch!")
            pdb.set_trace()
        else:
            print("Matched!")
            