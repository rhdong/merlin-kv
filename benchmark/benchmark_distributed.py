import tensorflow.compat.v1 as tf
import horovod.tensorflow as hvd
import itertools, time, os
from merlin_kv import tensorflow as mkv
from tabulate import tabulate

tf.compat.v1.disable_v2_behavior()


def make_partions_for_lookup(ids):
  if ids.shape.rank != 1:
    raise ValueError(
        f'Expecting rank of ids shape to be 1, but get {ids.shape}.')
  mask = tf.constant(0x7fffffff, tf.int64)
  ids_int32 = tf.cast(tf.bitwise.bitwise_and(ids, mask), tf.int32)
  mask = ids_int32 % hvd.size()
  relocs = []
  gather_indices = []
  for i in range(hvd.size()):
    idx = tf.where(tf.math.equal(mask, i))
    gather_indices.append(idx)
    relocs.append(tf.gather(ids, idx))
  sizes = tf.stack([tf.size(r) for r in relocs], axis=0)
  relocs_tensor = tf.concat(relocs, axis=0)
  relocs_tensor = tf.squeeze(relocs_tensor, axis=1)
  flat_reloc_ids, remote_sizes = hvd.alltoall(relocs_tensor, splits=sizes)
  return flat_reloc_ids, remote_sizes, gather_indices


def make_partions_for_insert(ids, embs):
  if ids.shape.rank != 1:
    raise ValueError(
        f'Expecting rank of ids shape to be 1, but get {ids.shape}.')
  mask = tf.constant(0x7fffffff, tf.int64)
  ids_int32 = tf.cast(tf.bitwise.bitwise_and(ids, mask), tf.int32)
  mask = ids_int32 % hvd.size()
  relocs_ids = []
  relocs_embs = []
  gather_indices = []
  for i in range(hvd.size()):
    idx = tf.where(tf.math.equal(mask, i))
    gather_indices.append(idx)
    relocs_ids.append(tf.gather(ids, idx))
    relocs_embs.append(tf.gather(embs, idx))
  sizes = tf.stack([tf.size(r) for r in relocs_ids], axis=0)
  relocs_ids_tensor = tf.squeeze(tf.concat(relocs_ids, axis=0), axis=1)
  relocs_embs_tensor = tf.squeeze(tf.concat(relocs_embs, axis=0), axis=1)
  flat_reloc_ids, remote_sizes = hvd.alltoall(relocs_ids_tensor, splits=sizes)
  flat_reloc_embs, _ = hvd.alltoall(relocs_embs_tensor, splits=sizes)
  return flat_reloc_ids, flat_reloc_embs, remote_sizes, gather_indices


def stitch(ids, lookup_result, remote_sizes, gather_indices, dim):
  lookup_result, _ = hvd.alltoall(lookup_result, splits=remote_sizes)

  recover_shape = tf.concat((tf.shape(ids, out_type=tf.int64), (dim,)), axis=0)
  gather_indices = tf.concat(gather_indices, axis=0)

  lookup_result = tf.scatter_nd(gather_indices, lookup_result, recover_shape)

  return lookup_result


def one_test(dim, items_num, device, test_times, maxval):
  sess_config = tf.ConfigProto(intra_op_parallelism_threads=0,
                               inter_op_parallelism_threads=0)
  sess_config.allow_soft_placement = True
  sess_config.gpu_options.allow_growth = True
  sess_config.log_device_placement = False

  with tf.Session(config=sess_config) as sess:
    with tf.device(device):
      ids = tf.random.uniform([items_num],
                              minval=0,
                              maxval=maxval,
                              dtype=tf.int64,
                              seed=None,
                              name=None)
      ids = tf.reshape(ids, shape=[
          items_num,
      ])
      embs = tf.constant([[0.0] * dim] * items_num)

      kv = mkv.get_variable("tf_benchmark",
                            tf.int64,
                            tf.float32,
                            devices=[device],
                            initializer=0.0,
                            dim=dim)

      ids_partitions, remote_sizes, gather_indices = make_partions_for_lookup(
          ids)
      lookup_result = kv.lookup(ids_partitions)
      lookup_result, _ = hvd.alltoall(lookup_result, splits=remote_sizes)
      recover_shape = tf.concat((tf.shape(ids, out_type=tf.int64), (dim,)),
                                axis=0)
      gather_indices = tf.concat(gather_indices, axis=0)
      lookup_op = tf.scatter_nd(gather_indices, lookup_result, recover_shape)

      # for insert

      ids_partitions, embs_partitions, remote_sizes, gather_indices = make_partions_for_insert(
          ids, embs)

      insert_op = kv.upsert(ids_partitions,
                            embs_partitions,
                            allow_duplicated_keys=False)
      size_op = kv.size()
    sess.run(ids)
    start_time = time.process_time()
    for _ in range(test_times):
      sess.run(ids)

    # Used to correct random_keys production time.
    random_time = (time.process_time() - start_time) / test_times
    sess.run(ids)
    start_time = time.process_time()
    for _ in range(test_times):
      sess.run(insert_op)
    insert_time = (time.process_time() - start_time) / test_times - random_time
    start_time = time.process_time()
    for _ in range(test_times):
      sess.run(lookup_op)
    lookup_time = (time.process_time() - start_time) / test_times - random_time
    table_size = sess.run(size_op)
    sess.close()
  tf.reset_default_graph()
  return insert_time, lookup_time, table_size / 1000


# 避免rehash
hvd.init()

device_id = str(hvd.local_rank())
os.environ['TF_HASHTABLE_INIT_SIZE'] = "33554432"
os.environ['CUDA_VISIEBLE_DEVICES'] = device_id
test_list = []
for dim, test_times, items_num in \
    itertools.product(
      [128],  [20, ], [1024 * 1024, ]):
  # [8, 64, 128],  [20, ], [1024, 16384, 131072, 1048576]):
  maxval = 0xFFFFFFFFFFFF

  upsert_gpu, lookup_gpu, size_gpu = one_test(dim, items_num,
                                              '/GPU:{}'.format(device_id),
                                              test_times, maxval)
  test_list.append([
      dim,
      items_num,
      test_times,
      "{:.3f}".format(items_num / (upsert_gpu * 1e6)),
      "{:.3f}".format(items_num / (lookup_gpu * 1e6)),  #size_cpu, size_gpu
  ])

print("Unit of throughput: Million KV-pairs/s.")
headers = [
    'dim',
    'keys num',
    'test_times',
    'upsert',
    'lookup',
]
print(tabulate(test_list, headers, tablefmt="github"))
