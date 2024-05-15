import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa


def segment_audio_data(audio_data, segment_num):
    '''
    Gets the segments ranges
    
    Returns: an array with tuples, where the tuple consists of the start and end of the segment
    '''

    segment_size = audio_data.shape[0]//segment_num

    segment_ranges = []
    for i in range(segment_num):
        segment_ranges.append((i*segment_size, (i+1)*segment_size))
    segment_ranges[-1] = (segment_ranges[-1][0], audio_data.shape[0]+1)
    
    return segment_ranges


def visualize_audio_segments(audio, sr, segment_ranges):
    fig = plt.figure(figsize=(15, 5))

    # Plot the entire audio waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr, color='b')
    plt.title('Original Audio')

    # Plot each segment in a different color
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio, sr=sr, alpha=0.5, color='b')  # Plot the entire audio as a background

    for start, end in segment_ranges:
        plt.axvspan(start/sr, end/sr, color='r', alpha=0.3)  # Highlight each segment

    plt.title('Audio Segments')
    plt.tight_layout()
    plt.show()
    return fig


def get_features(y, sr, segment_ranges):
    '''
    Extracts features from a segment of the audio data
    Examples of features: MFCC, tempo, spectral centroid, etc.
    '''
    df = pd.DataFrame()

    for i, segment_range in enumerate(segment_ranges):
        
        # chroma_stft
        chroma_stft = librosa.feature.chroma_stft(y=y[segment_range[0]:segment_range[1]], sr=sr)
        df.loc[i, 'chroma_stft_mean'] = chroma_stft.mean()
        df.loc[i, 'chroma_stft_var'] = np.var(chroma_stft, axis=1).mean()
        
        # chroma_rms
        chroma_rms = librosa.feature.rms(y=y[segment_range[0]:segment_range[1]])
        df.loc[i, 'rms_mean'] = chroma_rms.mean()
        df.loc[i, 'rms_var'] = np.var(chroma_rms, axis=1).mean()
        
        # spectral_centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y[segment_range[0]:segment_range[1]])
        df.loc[i, 'spectral_centroid_mean'] = spectral_centroid.mean()
        df.loc[i, 'spectral_centroid_var'] = np.var(spectral_centroid, axis=1).mean()
        
        # spectral_bandwidth
        chroma_bandwidth = librosa.feature.spectral_bandwidth(y=y[segment_range[0]:segment_range[1]])
        df.loc[i, 'chroma_bandwidth_mean'] = chroma_bandwidth.mean()
        df.loc[i, 'chroma_bandwidth_var'] = np.var(chroma_bandwidth, axis=1).mean()
        
        # rolloff
        chroma_rolloff = librosa.feature.spectral_rolloff(y=y[segment_range[0]:segment_range[1]])
        df.loc[i, 'chroma_rolloff_mean'] = chroma_rolloff.mean()
        df.loc[i, 'chroma_rolloff_var'] = np.var(chroma_rolloff, axis=1).mean()
        
        # zero_crossing_rate
        chroma_zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y[segment_range[0]:segment_range[1]])
        df.loc[i, 'chroma_zero_crossing_rate_mean'] = chroma_zero_crossing_rate.mean()
        df.loc[i, 'chroma_zero_crossing_rate_var'] = np.var(chroma_zero_crossing_rate, axis=1).mean()

        # Harmonics and perceptual mean
        chroma_harmony, chroma_perceptr = librosa.effects.hpss(y=y[segment_range[0]:segment_range[1]])
        df.loc[i, 'chroma_harmony_mean'] = chroma_harmony.mean()
        df.loc[i, 'chroma_harmony_var'] = chroma_harmony.var()
        df.loc[i, 'chroma_perceptr_mean'] = chroma_perceptr.mean()
        df.loc[i, 'chroma_perceptr_var'] = chroma_perceptr.var()

        # tempo
        df.loc[i, 'chroma_tempo'], _ = librosa.beat.beat_track(y=y[segment_range[0]:segment_range[1]], sr=sr)
    
        # mfcc
        n_mfcc = 10
        mfcc = librosa.feature.mfcc(y=y[segment_range[0]:segment_range[1]], sr=sr, n_mfcc=n_mfcc)
        for mfcc_idx in range(n_mfcc):
            df.loc[i, 'mffc_mean'+str(mfcc_idx+1)] = mfcc[mfcc_idx].mean()
            df.loc[i, 'mffc_var'+str(mfcc_idx+1)] = mfcc[mfcc_idx].var()

    return df    


def split_data(X, Y, split_precentages:dict):
    '''Splits data into train, validation and test sets'''

    if sum(split_precentages.values()) != 1.0:
        raise ValueError('split precenteges together do not give 1.0!')
    
    # finds the split points for the sets
    data_size = X.shape[0]
    v_point = int(data_size * split_precentages['train'])
    t_point = v_point + int(data_size * split_precentages['valid'])

    # splits the dataset
    X_train, Y_train = X[:v_point], Y[:v_point]
    X_valid, Y_valid = X[v_point:t_point], Y[v_point:t_point]
    X_test, Y_test = X[t_point:], Y[t_point:]

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


def scale(X_train, X_valid, X_test):
    '''Scales the values by the maximum value'''

    maximum = X_train.max()
    X_train = X_train/maximum
    X_valid = X_valid/maximum
    X_test = X_test/maximum

    return X_train.astype(np.float32), X_valid.astype(np.float32), X_test.astype(np.float32)


def unroll_sequence(X, Y):
    '''
    Creates unrolled data from segmented/sequenced data.
    Y value (genre) is repeated for each segment.
    '''
    
    if X.ndim != 3:
        raise ValueError(f'X only has {X.ndim} dimensions, while 3 were expected')

    row_num = X.shape[0]
    segment_num = X.shape[1]
    feature_num = X.shape[2]
    
    X_unrolled = X.reshape(row_num*segment_num, feature_num)
    Y_unrolled = np.repeat(Y, segment_num, axis=0)

    return X_unrolled, Y_unrolled