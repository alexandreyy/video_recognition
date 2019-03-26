import tensorflow as tf
import numpy as np
import cv2

import sys
import os
import threading
import glob


def video_to_array(video_path, size=256, fps=25):
    frames = []
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_FPS, fps)

    while True:
        available, frame = video.read()
        
        if not available:
            break
        
        rows, cols = frame.shape[:-1]
        ratio = float(rows) / cols

        rows, cols = (int(ratio*size), size) if ratio > 1.0 else (size, int(size/ratio))
        
        frames += [cv2.resize(frame, (rows, cols))]

    return np.stack(frames, 0)


def create_tf_records(records_name, videos, labels, thread_id=0, img_size=256):
    writer = tf.python_io.TFRecordWriter(records_name)

    tf.reset_default_graph()

    video_size = tf.placeholder(dtype=tf.int32, shape=(), name='video_size')
    video_input = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='image_input')
    jpegs = tf.map_fn(lambda image: tf.image.encode_jpeg(image, 'rgb', 95), video_input, dtype=tf.string)

    sess = tf.Session()

    for i in range(len(videos)):
        video = video_to_array(videos[i], 256)
        video_binary = sess.run(jpegs, {video_input: video})

        feature = {'video': tf.train.Feature(bytes_list=tf.train.BytesList(value=video_binary)),
                   'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        if not i % 100:
            print('Thread {} - processed {} videos'.format(thread_id, i))
            sys.stdout.flush()

    print('Thread {} - finished {} videos'.format(thread_id, len(videos)))

    writer.close()
    sys.stdout.flush()


def create_dataset_tf_records(videos, labels, records_name, num_shards=4, num_threads=4, seed=27001):
    shuffle_idx = np.arange(0, len(videos), dtype=np.int)
    np.random.seed(seed)
    np.random.shuffle(shuffle_idx)

    videos = [videos[i] for i in shuffle_idx]
    labels = [labels[i] for i in shuffle_idx]
    
    spacing = np.linspace(0, len(videos), num_threads + 1).astype(np.int)
    ranges = []

    for i in range(len(spacing) - 1):
        ranges += [(spacing[i], spacing[i+1])]

    coord = tf.train.Coordinator()
    threads = []

    for i in range(num_shards):
        args = (records_name + '{}-of-{}.tfrecord'.format(i, num_shards),
                videos[ranges[i][0]:ranges[i][1]],
                labels[ranges[i][0]:ranges[i][1]],
                i)

        thr = threading.Thread(target=create_tf_records, args=args)
        thr.start()

        threads.append(thr)

    coord.join(threads)
    sys.stdout.flush()



def load_from_record(records_pattern, num_workers=1, worker_id=0, shuffle_buffer=None,
                     repeat=True, num_threads=4, batch_size=4, preprocess=lambda x: x):
    features = {'video': tf.FixedLenSequenceFeature([], tf.string, allow_missing=True),
                'label': tf.FixedLenFeature([], tf.int64)}

    def parse_dataset(example):
        parsed_example = tf.parse_single_example(example, features)

        label = parsed_example['label']

        video = tf.reshape(parsed_example['video'], [-1])
        video = tf.map_fn(lambda frame: tf.image.decode_jpeg(frame), video, dtype=tf.uint8)

        return video, label


    dataset = tf.data.Dataset.list_files(records_pattern)

    if num_workers > 1:
        dataset = dataset.shard(num_workers, worker_id)
        #dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=1, block_length=1)

    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=4)
    dataset = dataset.map(parse_dataset)

    if repeat:
        dataset = dataset.repeat()

    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)

    dataset = dataset.map(preprocess, num_parallel_calls=4)
    
    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset


def read_hmdb_dataset(data_path, split_path=None, val_split=0.2, split_number=1, num_classes=51):
        data_format = ".txt"
        split_suffix = "_test_split{}{}".format(split_number, data_format)

        train_videos, val_videos, test_videos = [], [], []
        train_labels, val_labels, test_labels = [], [], []

        if not split_path:
            split_path = os.path.join(data_path, "testTrainMulti_7030_splits")

        for _, _, files in os.walk(split_path):
            i = -1
            for file in files:
                if int(file[-len(data_format)-1]) != split_number:
                    continue

                if i >= num_classes - 1:
                    break

                i += 1
                folder = file[:-len(split_suffix)]

                with open(os.path.join(split_path, file), 'r') as f:
                    for l in f:
                        l = l.split()
                        split = int(l[1])
                        full_file_name = os.path.join(data_path, folder, l[0])

                        if split == 1:
                            train_videos += [full_file_name]
                            train_labels += [i]
                        elif split == 2:
                            test_videos += [full_file_name]
                            test_labels += [i]

        num_val = round(val_split * len(train_videos))

        val_videos, val_labels = train_videos[:num_val], train_labels[:num_val]
        train_videos, train_labels = train_videos[num_val:], train_labels[num_val:]

        return (train_videos, train_labels), (val_videos, val_labels), (test_videos, test_labels)


def get_data_size(data_pattern):
    return sum(sum(1 for _ in tf.python_io.tf_record_iterator(data_file)) for data_file in glob.glob(data_pattern))



if __name__ == '__main__':
    hmdb_data = read_hmdb_dataset("../../data/hmdb_base", val_split=0.0, num_classes=25)

    for name, (videos, labels) in zip(['train', 'val', 'test'],  hmdb_data):
        if(len(videos)):
            create_dataset_tf_records(videos, labels, '../../data/hmdb/hmdb_' + name, num_shards=4, num_threads=4)
                                                                            

    #test_tf_records('test/hmdb_train.tfrecord')