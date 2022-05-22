"""

file use to evaluation
To run this: 
from Audio_identification_brute import fingerprintBuilder()
from Audio_identification_brute import audioIdentification()

Result will be saved in input directory
"""



import numpy as np
import librosa
import glob
import pickle
from scipy import ndimage
from tqdm import tqdm
import os
from scipy import ndimage
from collections import Counter






def get_matching_function(database_peaks, query_peaks, tol_freq=1, tol_time=1):
    """
    input : 
    database_peaks :  constellation map of a  database file
    query_peaks:  constellation map of a query file

    output:
    Return total number of matches for a pair of database and query file

    """
   
    L = database_peaks.shape[1]
    N = query_peaks.shape[1]
    M = L - N
    out = np.zeros(L)
    for m in range(M + 1):
        C_D_crop = database_peaks[:, m:m+N]
        C_Q_exp = ndimage.maximum_filter(query_peaks, size=(2*tol_freq+1, 2*tol_time+1),mode='constant')
        matched = np.logical_and(C_Q_exp, C_D_crop)
        matched_pts = matched.sum()
        out[m] = matched_pts
    max_arg = np.argmax(out)
    maximum_value = out.max()

    return maximum_value


def get_spectrogram(fn_wav, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
    """
    input:
    fn_wav : wav file of audio input
    Fs, N, H : parameters to calculate spectrogram

    output:
    return spectrogram
    /* other features can also be extracted */
     
    """

    x, Fs = librosa.load(fn_wav, Fs)
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    # X = librosa.feature.melspectrogram(x, n_fft=N, hop_length=H, win_length=N, window='hanning'
    # X = librosa.feature.chroma_stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')

    Y = np.abs(X[:bin_max, :frame_max])
  
    return Y

def get_peaks(D, dist_freq=11, dist_time=5, thresh=0.01):
    """
    input:
    D : spectrogram document of file
    dist_freq : neigborhood value for frequency dimension
    dist_time:  neigborhood value for time dimension

    output:
    constellation map for a given spectrogram
    
    """
    result = ndimage.maximum_filter(D, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
    Cmap = np.logical_and(D == result, result > thresh)
    return Cmap


def fingerprintBuilder(Database_files_path,output_path):
    """
    input:
    Database_files_path : path of database file folder [ path should end with "/"]
    output_path : output path to save fingerprints of database files

    output:
    save fingerprints to the given output path

    """

    Database_files = glob.glob(Database_files_path + "*")
    path = output_path
    if not os.path.exists(path):
        os.makedirs(path)
    for i,file in tqdm(enumerate(Database_files)):
        database_fingerprint = {}
        file_name = os.path.basename(file)
        spec = get_spectrogram(file)
        peaks = get_peaks(spec)
        database_fingerprint[file_name] = peaks
        pickle.dump(database_fingerprint, open(path+ file_name+".pkl","wb"))
    print("Database fingerprinting Done...")

def audioIdentification(query_path,database_fingerprints,output_text):
    """
    input:
    query_path : path of query file folder [ path should end with "/"]
    database_fingerprints : path of saved database fingerprints [path should end with "/"]
    output_text : output path to save the text file of result
    
    """

    query_files = glob.glob(query_path+"*")
    database_keys = glob.glob(database_fingerprints +"*")
    scores = {}
    for file in query_files[:50]:
        spec = get_spectrogram(file)
        peaks = get_peaks(spec)
        query_peaks = peaks
        scores[file] = {}
        for i,keys in tqdm(enumerate(database_keys)):
            database_key = pickle.load(open(keys,"rb"))
            data_emb =  [*database_key.values()][0]
            database_peaks = data_emb
            maximum_matched_pts = get_matching_function(database_peaks, query_peaks, tol_freq=3, tol_time=2)
            scores[file][[*database_key.keys()][0]] = maximum_matched_pts

   
    final_output = []
    for key in scores.keys():
        query_file_name = os.path.basename(key)
        all_out = []
        out = dict(Counter(scores[key]).most_common(3))
        out = [*out.keys()]
        all_out.append(query_file_name)
        all_out.extend(out)
        final_output.append(all_out)
    final_output = np.vstack(final_output)
    np.savetxt(output_text, final_output,delimiter='  ',fmt='%s')




    

# if __name__ == "__main__":
    
#     Database_files = "/Users/harshita/Documents/MODULES/Music_informatics/Assignment/Assignment_2/Datasets/database_recordings/"
#     query_files = "/Users/harshita/Documents/MODULES/Music_informatics/Assignment/Assignment_2/Datasets/query_recordings/"
#     fingerprintBuilder(Database_files,"./check/")
#     audioIdentification(query_files,"./check/","output_text.txt")
    
