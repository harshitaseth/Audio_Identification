
import librosa
import glob
import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from collections import Counter


def get_spectrogram(path_database):
    """
    input:
    fn_wav : wav file of audio input
    Fs, N, H : parameters to calculate spectrogram

    output:
    return spectrogram
    /* other features can also be extracted */
     
    """
    y, sr = librosa.load(path_database)
    # Compute and plot STFT spectrogram
    D = np.abs(librosa.stft(y,n_fft=2048,window='hann',win_length=1024,hop_length=1024))
    #D = np.abs(librosa.feature.melspectrogram(x, n_fft=N, hop_length=H, win_length=N, window='hanning'))
    #D = np.abs(librosa.feature.chroma_stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning'))
    return D


def get_peaks(D, dist_freq=20, dist_time=10, thresh=0.01):
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

def get_hash_database(peaks):
    """
    input: constellation map of database file
    output: return the inverted list 
    """
    thresh = 30
    hash_size =50
    rows = peaks.shape[0]//hash_size
    col = peaks.shape[1]//hash_size
    hash_document = np.zeros((rows,col))
    hash_document_1 = np.zeros((rows,col))
    hash_dict = {}
    for i in range(rows):
        for j in range(col):
            hash_document_1[i][j] = (peaks[i*hash_size:i*hash_size+hash_size,j*hash_size:j*hash_size+hash_size]).sum()
            if (hash_document_1[i][j] > thresh ):
                hash_document[i][j] = 1
            else:
                hash_document[i][j] = 0

    for i in range(hash_document.shape[0]):
        list_ = np.argwhere(hash_document[i] > 0)
        if(len(list_) > 0 ):
            hash_dict[i] = list_[:,0]
        else:
            hash_dict[i] = []
    
    return hash_dict

def get_hash_query(peaks):
    """
    input: constellation map of query file
    output : return hashmap of query file
    """

    thresh = 30
    hash_size =50
    rows = peaks.shape[0]//hash_size
    col = peaks.shape[1]//hash_size
    hash_query = np.zeros((rows,col))
    coordinates = peaks.astype(int)
    for i in range(rows):
        for j in range(col):
            hash_query [i][j] = (coordinates[i*hash_size:i*hash_size+hash_size,j*hash_size:j*hash_size+hash_size]).sum()
            if (hash_query[i][j] > thresh):
                hash_query[i][j] = 1
    query_list = []
    for i in range(hash_query.shape[1]):
        rows = np.argwhere(hash_query[:,i] > 0)
        for row in rows:
            if (hash_query[int(row)][i] > 0):
                query_list.append([i,row[0]])
    return query_list

def get_matches(query_hash_list,hash_dict):
    """
    input:
    query_hash_list : hashmap of query file
    hash_dict : inverted list of database file

    output:
    return total number of matches for a given pair of query and database file


    """
    inverted_list = []
    query_list = query_hash_list
    for quer in query_list:
        h = quer[1]
        n = quer[0]
        hash_dict_out = hash_dict[h]
        out = []
        for keys in hash_dict_out:
            out.append(keys - n)
        inverted_list.append(out)
    out = np.zeros((20,20))
    for i in range(len(inverted_list)):
        list_ = inverted_list[i]
        for j in list_:
            if j < 0:
                continue
            out[i,j] = 1

    matches = []
    for i in range(out.shape[1]):
        matches.append(out[:,i].sum())
    return matches

    
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
        hash_dict = get_hash_database(peaks)
        database_fingerprint[file_name] = hash_dict
        pickle.dump(database_fingerprint, open(path+"/embs_database_"+str(i)+".pkl","wb"))

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
    for file in query_files:
        spec = get_spectrogram(file)
        peaks = get_peaks(spec)
        query_hash_list = get_hash_query(peaks)
        scores[file] = {}
        for i,keys in tqdm(enumerate(database_keys)):
            database_key = pickle.load(open(keys,"rb"))
            data_emb =  [*database_key.values()][0]
            hash_dict_database = data_emb
            out = get_matches(query_hash_list,hash_dict_database)
            scores[file][[*database_key.keys()][0]] = sum(out)

   
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
    
