import tensorflow as tf

# input_song_ids: [batch_size]
# target_feature_sequences: [batch_size, sequence_length, num_features]
def build(config, num_songs, input_song_ids, target_feature_sequences):
    batch_size = tf.shape(target_feature_sequences)[0]
    sequence_length = tf.shape(target_feature_sequences)[1]

    with tf.device("/cpu:0"):
        song_embedding_table = tf.get_variable(
            'song_embedding_table',
            [num_songs, config['embedding_size']],
        )

        # [batch_size, embedding_size]
        song_embeddings = tf.nn.embedding_lookup(song_embedding_table, input_song_ids)

    start_input = tf.get_variable('start_input', [1,  1, config['input_size']])

    input_W = tf.get_variable('input_W', [config['num_features'], config['input_size']])
    input_b = tf.get_variable('input_b', [config['input_size']])

    decoder_rnn = tf.contrib.rnn.BasicLSTMCell(config['hidden_size'])
    decoder_initial_state = decoder_rnn.zero_state(batch_size, tf.float32)

    output_W = tf.get_variable('output_W', [config['hidden_size'], config['num_features']])
    output_b = tf.get_variable('output_b', [config['num_features']])

    # [batch_size, sequence_length, embedding_size]
    tiled_song_embeddings = tf.tile(
        tf.expand_dims(song_embeddings, 1),
        [1, sequence_length, 1],
    )

    # [batch_size, num_features, input_size]
    tiled_input_W = tf.tile(
        tf.expand_dims(input_W, 0),
        [batch_size, 1, 1],
    )

    # [batch_size, sequence_length-1, input_size]
    inputs = (tf.matmul(target_feature_sequences, tiled_input_W) + input_b)[:, :-1, :]

    # [batch_size, sequence_length, input_size]
    sequence_inputs = tf.concat(
        [tf.tile(start_input, [batch_size, 1, 1]), inputs],
        axis=1
    )

    # [batch_size, sequence_length, embedding_size+input_size]
    decoder_inputs = tf.concat([tiled_song_embeddings, sequence_inputs], axis=2)

    # [batch_size, sequence_length, hidden_size]
    decoder_outputs, _decoder_state = tf.nn.dynamic_rnn(
        decoder_rnn,
        decoder_inputs,
        initial_state=decoder_initial_state,
    )

    # [batch_size, hidden_size, num_features]
    tiled_output_W = tf.tile(
        tf.expand_dims(output_W, 0),
        [batch_size, 1, 1],
    )

    # [batch_size, sequence_length, num_features]
    feature_outputs = tf.matmul(decoder_outputs, tiled_output_W) + output_b

    return feature_outputs
